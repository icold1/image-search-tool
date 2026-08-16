"""核心管线：模型加载、建索引、查询。

与 UI 完全解耦：既可以被 QThread 包装（GUI 模式），
也可以被脚本/冒烟测试直接同步调用。
"""
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

from core import config as cfg
from core import model as ml
from core import scanner
from core import thumbnailer

_skip_log = cfg.DATA_DIR / "skipped.log"


def log_skip(path: str, exc: Exception) -> None:
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_skip_log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{path}\t{exc}\n")
    except Exception:
        pass


def ensure_model(model_name: str = "ViT-L-14", device: str = "auto",
                 message: Optional[Callable[[str], None]] = None) -> None:
    if not ml.is_loaded():
        if message:
            message("正在加载模型（首次运行需下载权重约 1.6GB，请耐心等待）...")
        ml.load_model(model_name, device, str(cfg.MODEL_DIR))


def build_index(store, image_dir: str, model_name: str = "ViT-L-14",
                device: str = "auto", batch_size: int = 64,
                thumb_size: int = 256,
                thumbs_dir: Optional[str] = None,
                ocr_enabled: bool = True,
                progress: Optional[Callable[[int, int, float], None]] = None,
                message: Optional[Callable[[str], None]] = None,
                stop=None) -> dict:
    """增量建索引（按 mtime 对比），stop 为可选的 threading.Event。

    阶段一：向量嵌入（GPU）；阶段二：OCR 文字识别（CPU，仅处理未识别过的图）。
    返回统计 dict: total / added / updated / skipped / deleted / ocr_done。
    """
    tdir = thumbs_dir or str(cfg.THUMB_DIR)
    ensure_model(model_name, device, message)
    known = store.known_map()
    files, deleted = scanner.diff_scan(image_dir, known)
    for p in deleted:
        store.remove_by_path(p)
    if deleted and message:
        message(f"已清理磁盘上不存在的图片记录 {len(deleted)} 条")

    total = len(files)
    done = added = updated = skipped = 0
    t0 = time.time()
    for i in range(0, total, batch_size):
        if stop is not None and stop.is_set():
            store.save_vectors()
            if message:
                message("已暂停，进度已保存")
            break
        batch_files = files[i:i + batch_size]
        items = []
        for f in batch_files:
            try:
                img = thumbnailer.load_image(f.path)
                items.append((f, img))
            except Exception as e:
                skipped += 1
                log_skip(f.path, e)
        if items:
            vecs = ml.encode_images([img for _, img in items], batch_size)
            for (f, img), v in zip(items, vecs):
                try:
                    thumb = thumbnailer.save_thumbnail(
                        img, tdir, f.path, thumb_size)
                    fmt = (img.format or Path(f.path).suffix.lstrip(".")).upper()
                    row = store.row_by_path(f.path)
                    if row is None:
                        store.append(f.path, f.mtime, f.size,
                                     img.width, img.height, fmt, thumb, v)
                        added += 1
                    else:
                        store.update(row, f.path, f.mtime, v)
                        updated += 1
                except Exception as e:
                    skipped += 1
                    log_skip(f.path, e)
        done += len(batch_files)
        if progress:
            elapsed = time.time() - t0
            eta = (elapsed / max(done, 1)) * (total - done)
            progress(done, total, eta)
    store.save_vectors()

    # ---------- 阶段二：OCR 文字识别（增量） ----------
    ocr_done = 0
    if ocr_enabled:
        pending = store.rows_missing_ocr()
        if pending:
            try:
                from core import ocr as ocrm
                from core import textindex
            except Exception as e:
                log_skip("ocr-init", e)
                pending = []
            if pending:
                if message:
                    message(f"开始文字识别 OCR（共 {len(pending)} 张，CPU 推理较慢，可边识别边搜索）...")
                ti = textindex.get_index(store)
                n = len(pending)
                t_ocr = time.time()
                for i, (row_id, path) in enumerate(pending):
                    if stop is not None and stop.is_set():
                        store.save_vectors()
                        if message:
                            message("已暂停，进度已保存")
                        break
                    try:
                        img = thumbnailer.load_image(path)
                        text = ocrm.extract_text(img)
                        store.set_ocr(row_id, text or "")
                        if text:
                            ti.add(row_id, text)
                        ocr_done += 1
                    except Exception as e:
                        log_skip(path, e)
                    if progress and ((i + 1) % 5 == 0 or i + 1 == n):
                        elapsed = time.time() - t_ocr
                        eta = (elapsed / max(i + 1, 1)) * (n - i - 1)
                        progress(i + 1, n, eta)
        store.save_vectors()
    return {"total": total, "added": added, "updated": updated,
            "skipped": skipped, "deleted": len(deleted), "ocr_done": ocr_done}


def search(store, query_text: str, k: int, model_name: str = "ViT-L-14",
           device: str = "auto") -> List[tuple]:
    """文本查询，返回 [(meta_dict, clip_score, text_score|None)]。

    排序规则（文字优先 + 语义综合）：
      1. OCR 文字命中者排最前：按文字分降序、语义分降序；
      2. 其余按语义相似度降序补足到 k。
    支持空格/逗号分隔多词：语义侧各词向量取平均后检索。
    """
    from core import searcher, textindex
    ensure_model(model_name, device)
    parts = [p for p in re.split(r"[\s,，]+", query_text.strip()) if p]
    feats = ml.encode_text(parts if parts else [query_text])
    q = feats.mean(axis=0, keepdims=True)
    with store.lock():  # 序列化并发查询与索引写入，保护 sqlite 连接与向量矩阵
        mask = store.valid_mask()
        hits = searcher.search(q, store.vectors, mask, k)

        clip_map = {}
        clip_order = []
        for row_idx, score in hits:
            meta = store.get_meta(row_idx + 1)
            if meta and Path(meta["path"]).exists():
                clip_map[meta["id"]] = (meta, score)
                clip_order.append(meta["id"])

        out = []
        used = set()

        # 文字命中（优先，不受语义 top-k 限制）
        ti = textindex.get_index(store)
        for row_id, tscore in ti.search(query_text, limit=k):
            if row_id in clip_map:
                meta, clip = clip_map[row_id]
            else:
                meta = store.get_meta(row_id)
                if not meta or not Path(meta["path"]).exists():
                    continue
                if row_id - 1 < store.vectors.shape[0]:
                    clip = float(store.vectors[row_id - 1] @ q[0])
                else:
                    clip = 0.0
            out.append((meta, clip, tscore))
            used.add(meta["id"])

        # 语义结果补足
        for rid in clip_order:
            if rid not in used:
                meta, clip = clip_map[rid]
                out.append((meta, clip, None))

        return out[:k]
