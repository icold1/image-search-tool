"""核心管线：模型加载、建索引、caption、查询。

与 UI 完全解耦：既可以被 QThread 包装（GUI 模式），
也可以被脚本/冒烟测试直接同步调用。
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

from core import config as cfg
from core import model as ml
from core import scanner
from core import thumbnailer
from core.searcher import SearchCancelled  # 再导出，worker 统一从这里引用

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
    """确保 CLIP 已加载（经 model_manager 统一管理，支持空闲卸载）。"""
    from core import model_manager
    mgr = model_manager.ModelManager.get()
    idle = int(cfg.load_config().get("idle_unload_seconds", 0))
    mgr.slot("clip", idle_seconds=idle).configure(
        lambda: ml.load_model(model_name, device, str(cfg.MODEL_DIR)),
        unload_fn=ml.unload)
    if not ml.is_loaded():
        if message:
            message("正在加载模型（首次运行需下载权重约 1.6GB，请耐心等待）...")
        mgr.acquire("clip")
    else:
        mgr.touch("clip")


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
                    # 内容变更过的文件（路径已存在）必须重建缩略图，
                    # 否则展示的是过期缩略图
                    force_thumb = store.row_by_path(f.path) is not None
                    thumb = thumbnailer.save_thumbnail(
                        img, tdir, f.path, thumb_size, force=force_thumb)
                    fmt = (img.format or Path(f.path).suffix.lstrip(".")).upper()
                    row = store.row_by_path(f.path)
                    if row is None:
                        store.append(f.path, f.mtime, f.size,
                                     img.width, img.height, fmt, thumb, v)
                        added += 1
                    else:
                        store.update(row, f.path, f.mtime, f.size,
                                     img.width, img.height, fmt, thumb, v)
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
           device: str = "auto", stop=None) -> List[tuple]:
    """文本查询（三路融合），返回 [(meta, score, text_score, desc|None)]。

    - meta: 图片元数据 dict
    - score: 展示分（OCR 命中=CLIP 相似度；融合结果=RRF 融合分）
    - text_score: OCR 文字命中分（None 表示未命中文字）
    - desc: caption 向量命中信息 {"score": float, "line": str}（None 表示未命中）

    排序规则（保持现状语义：文字优先）：
      1. OCR 文字命中者排最前（按文字分降序）；
      2. 其余按 RRF 融合分（CLIP + caption 两路）降序补足到 k；
         caption 索引缺失/禁用时自动降级为纯 CLIP（与旧行为一致）。

    stop: 可选 threading.Event，置位时抛 SearchCancelled（取消在途查询）。
    """
    from core import searcher, textindex
    if stop is not None and stop.is_set():
        raise SearchCancelled()
    ensure_model(model_name, device)
    parts = [p for p in re.split(r"[\s,，]+", query_text.strip()) if p]
    feats = ml.encode_text(parts if parts else [query_text])
    if stop is not None and stop.is_set():
        raise SearchCancelled()
    q = feats.mean(axis=0, keepdims=True)

    conf = cfg.load_config()
    fus = conf.get("fusion") or {}
    cand_n = max(k, int(fus.get("candidate_n", 100)))
    w_clip = float(fus.get("w_clip", 1.0))
    w_caption = float(fus.get("w_caption", 0.6))
    rrf_k = float(fus.get("rrf_k", 60))

    with store.lock():  # 序列化并发查询与索引写入，保护 sqlite 连接与向量矩阵
        mask = store.valid_mask()

        # ---- 路 1：CLIP 全局语义 ----
        clip_map = {}
        clip_rank = {}
        for rank, (row_idx, score) in enumerate(
                searcher.search(q, store.vectors, mask, cand_n)):
            meta = store.get_meta(row_idx + 1)
            if meta and Path(meta["path"]).exists():
                clip_map[meta["id"]] = (meta, float(score))
                clip_rank[meta["id"]] = rank
        clip_order = list(clip_map)

        # ---- 路 2：caption 文本向量（失败/缺失自动降级，不影响主路）----
        desc_rank = {}  # row_id -> (rank, score, hit_line)
        if conf.get("caption_search_enabled", True):
            try:
                from core import captionindex, embed
                dim = int(conf.get("embed_dim", 1024))
                ci = captionindex.get_index(store)
                if ci.ensure_loaded(dim):
                    eq = embed.query_vector(
                        query_text,
                        conf.get("embed_model", "Qwen/Qwen3-Embedding-0.6B"),
                        conf.get("device", "auto"), dim,
                        model_dir=str(cfg.MODEL_DIR))
                    if stop is not None and stop.is_set():
                        raise SearchCancelled()
                    for rank, (rid, cscore, line) in enumerate(
                            ci.search(eq, cand_n)):
                        desc_rank[rid] = (rank, cscore, line)
            except SearchCancelled:
                raise
            except Exception as e:
                log_skip("caption-search", e)

        # ---- RRF 融合打分 ----
        fused = {}
        for rid, rank in clip_rank.items():
            fused[rid] = fused.get(rid, 0.0) + w_clip / (rrf_k + rank + 1)
        for rid, (rank, _s, _l) in desc_rank.items():
            fused[rid] = fused.get(rid, 0.0) + w_caption / (rrf_k + rank + 1)

        out = []
        used = set()

        # ---- OCR 文字命中（保持现状：置顶优先，不受融合分限制）----
        ti = textindex.get_index(store)
        for row_id, tscore in ti.search(query_text, limit=k, stop=stop):
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
            out.append((meta, clip, tscore, None))
            used.add(meta["id"])

        # ---- 融合结果补足 ----
        ids = sorted(set(clip_order) | set(desc_rank),
                     key=lambda r: -fused.get(r, -1.0))
        for rid in ids:
            if rid in used or len(out) >= k:
                continue
            if rid in clip_map:
                meta, clip = clip_map[rid]
            else:
                meta = store.get_meta(rid)
                if not meta or not Path(meta["path"]).exists():
                    continue
                if rid - 1 < store.vectors.shape[0]:
                    clip = float(store.vectors[rid - 1] @ q[0])
                else:
                    clip = 0.0
            if rid in desc_rank:
                _r, cscore, line = desc_rank[rid]
                desc = {"score": cscore, "line": line}
            else:
                desc = None
            out.append((meta, float(fused.get(rid, clip)), None, desc))

        # ---- 精排（可选）：对融合结果前 N 张做交叉编码重排 ----
        if conf.get("rerank_enabled", False):
            try:
                out = _rerank_results(out, query_text, conf, stop)
            except SearchCancelled:
                raise
            except Exception as e:
                log_skip("rerank", e)

        return out[:k]


def _doc_for_rerank(meta: dict, desc) -> str:
    """精排用的文档文本：描述命中行 > caption 内容 > OCR 文本。"""
    if desc is not None:
        return str(desc.get("line", ""))[:300]
    cap = meta.get("caption")
    if cap:
        try:
            data = json.loads(cap)
            content = data.get("content") or data.get("background") or ""
            if content:
                return str(content)[:300]
        except Exception:
            pass
    return str(meta.get("ocr") or "")[:300]


def _rerank_results(out: List[tuple], query_text: str, conf: dict,
                    stop=None) -> List[tuple]:
    """用 Qwen3-Reranker 对融合结果重排；OCR 命中保持置顶不动。"""
    from core import reranker
    top_n = int(conf.get("rerank_top_n", 30))
    if len(out) < 2 or top_n < 2:
        return out
    ocr_part = [r for r in out if r[2] is not None]      # 文字命中，保持原序
    rest = [r for r in out if r[2] is None]
    targets = rest[:top_n]
    docs = [_doc_for_rerank(meta, desc) for meta, _s, _ts, desc in targets]
    if not docs or not any(docs):
        return out
    reranker.ensure_reranker(
        conf.get("rerank_model", "Qwen/Qwen3-Reranker-0.6B"),
        conf.get("device", "auto"), str(cfg.MODEL_DIR))
    if stop is not None and stop.is_set():
        raise SearchCancelled()
    scores = reranker.rerank(query_text, docs,
                             batch_size=int(conf.get("rerank_batch", 16)))
    ranked = sorted(zip(scores, targets), key=lambda kv: -kv[0])
    new_rest = [t for _, t in ranked] + rest[top_n:]
    return ocr_part + new_rest


def _resource_warning() -> str:
    """caption 前预检：内存/显存紧张时返回提示（供 message 输出）。"""
    msgs = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        free_gb = vm.available / 1024**3
        if free_gb < 2.0:
            msgs.append(f"系统可用内存仅 {free_gb:.1f}GB"
                        f"（共 {vm.total/1024**3:.1f}GB），建议关闭部分程序")
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW)
        used, total = (int(x) for x in out.stdout.strip().split(","))
        free_mb = total - used
        if free_mb < 6800:
            msgs.append(f"显存可用仅 {free_mb}MB（caption 峰值需约 6300MB），"
                        f"请关闭悬浮球/ComfyUI 等占用显存的程序")
    except Exception:
        pass
    return "；".join(msgs)


def build_captions(store, model_name: Optional[str] = None,
                   device: str = "auto", batch_size: int = 1,
                   include_failed: bool = False, force: bool = False,
                   only_paths: Optional[List[str]] = None,
                   progress: Optional[Callable[[int, int, float], None]] = None,
                   message: Optional[Callable[[str], None]] = None,
                   stop=None) -> dict:
    """后台图像描述（增量 + 断点续传 + 逐图失败隔离）。

    每张完成立即写库（不学参考项目"结尾一次性写文件"的丢失风险），
    中断后重跑自动跳过已完成的行。
    only_paths: 仅处理这些绝对路径（试跑/验证用），None 表示全部。
    返回统计 dict。
    """
    from core import captioner, model_manager
    conf = cfg.load_config()
    model_name = model_name or conf.get("caption_model",
                                        "Qwen/Qwen3-VL-2B-Instruct")
    device = device or conf.get("caption_device", "auto")
    if not conf.get("caption_enabled", True) and not force:
        if message:
            message("caption 功能已在配置中关闭，跳过（可用 --force 强制执行）")
        return {"total": 0, "done": 0, "ok": 0, "failed": 0}

    mgr = model_manager.ModelManager.get()
    # 腾显存给 VLM（8GB 卡）：卸载嵌入/精排模型；caption_keep_clip=False 时连 CLIP 也卸
    mgr.unload("embed")
    mgr.unload("reranker")
    if not conf.get("caption_keep_clip", True):
        mgr.unload("clip")

    try:
        cap = captioner.make_captioner(
            model_name, device, str(cfg.MODEL_DIR),
            max_new_tokens=int(conf.get("caption_max_new_tokens", 600)),
            mmproj_path=conf.get("caption_mmproj") or None,
            llama_bin=conf.get("caption_llama_bin") or None,
            ctx_size=int(conf.get("caption_ctx_size", 8192)))
    except Exception as e:
        raise RuntimeError(f"caption 模型加载失败: {e}") from e

    warn = _resource_warning()
    if warn and message:
        message(f"资源预检：{warn}")

    pending = store.rows_missing_caption(include_failed)
    if only_paths:
        only = {os.path.abspath(p) for p in only_paths}
        pending = [(rid, p) for rid, p in pending if p in only]
    total = len(pending)
    if message:
        message(f"开始图像描述（共 {total} 张，本地 VLM 较慢，可暂停续传）...")
    done = ok = failed = 0
    t0 = time.time()
    try:
        for i, (row_id, path) in enumerate(pending):
            if stop is not None and stop.is_set():
                if message:
                    message("已暂停，进度已保存（可随时重跑续传）")
                break
            try:
                img = thumbnailer.load_image(path)
                data = cap.describe(img)
                store.set_caption(row_id, json.dumps(data, ensure_ascii=False))
                ok += 1
            except Exception as e:
                failed += 1
                store.set_caption(row_id, None, status="failed",
                                  error=str(e)[:500])
                log_skip(path, e)
            done += 1
            if progress and (done % 5 == 0 or done == total):
                elapsed = time.time() - t0
                eta = (elapsed / max(done, 1)) * (total - done)
                progress(done, total, eta)
    finally:
        # VLM 批次结束必须卸载（8GB 卡：caption 阶段后腾出显存给检索模型）；
        # GGUF 后端同时停掉 llama-server 进程
        mgr.unload("vlm")
        if hasattr(cap, "unload"):
            try:
                cap.unload()
            except Exception:
                pass
    return {"total": total, "done": done, "ok": ok, "failed": failed}


def build_caption_vectors(store, embed_model: Optional[str] = None,
                          dim: Optional[int] = None,
                          device: str = "auto",
                          progress: Optional[Callable[[int, int, float],
                                                      None]] = None,
                          message: Optional[Callable[[str], None]] = None,
                          stop=None) -> dict:
    """把已有 caption 构建为字段化行向量矩阵（可重跑，整表替换）。"""
    from core import captionindex, embed, model_manager
    conf = cfg.load_config()
    embed_model = embed_model or conf.get("embed_model",
                                          "Qwen/Qwen3-Embedding-0.6B")
    dim = int(dim or conf.get("embed_dim", 1024))
    mgr = model_manager.ModelManager.get()
    idle = int(conf.get("idle_unload_seconds", 0))
    mgr.slot("embed", idle_seconds=idle).configure(
        lambda: embed.load(embed_model, device, str(cfg.MODEL_DIR)),
        unload_fn=embed.unload)
    if not embed.is_loaded():
        if message:
            message("正在加载文本嵌入模型（首次运行需下载权重约 1.2GB）...")
        mgr.acquire("embed")
    ci = captionindex.get_index(store)
    stats = ci.build(embed, dim, embed_model=embed_model,
                     progress=progress, message=message, stop=stop)
    if message:
        message(f"caption 向量构建完成：{stats['images']} 张 / "
                f"{stats['lines']} 行")
    return stats
