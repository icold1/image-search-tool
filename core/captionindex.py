"""caption 文本向量索引：字段化多行向量 + 两段式检索。

存储（吸取 image-features 教训，全部二进制 + 原子写）：
- data/caption_vecs.npy ：M 行 x dim 的 fp16 归一化行向量，
  按 row_id 升序连续分组（每图 1~12 行）。
- data/caption_agg.npy ：N 行 x dim 的 fp16 每图 max-pool 聚合向量
  （N = 有 caption 的图片数）。
- SQLite caption_lines 表：row_id -> line_count（分组边界）。

检索（两段式，查询速度优先）：
1. 聚合矩阵全量打分，取 top-candidates（默认 200）；
2. 仅对候选做行级精确 max 重排，返回 (row_id, score, 命中行文本)。
"""
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

MAX_LINES_PER_IMAGE = 12


def caption_to_lines(caption: dict) -> List[str]:
    """把 caption dict 转成带字段前缀的检索行（≤12 行）。"""
    lines: List[str] = []

    def push(line: str, cap: int = 400) -> None:
        line = " ".join(line.split())
        if line:
            lines.append(line[:cap])

    push(f"背景: {caption.get('background', '')}")
    push(f"颜色: {caption.get('colors', '')}")
    push(f"画风: {caption.get('style', '')}")
    push(f"内容: {caption.get('content', '')}")
    for person in (caption.get("people") or [])[:2]:
        if not isinstance(person, dict):
            continue
        desc = person.get("description", "")
        action = person.get("action", "")
        name = person.get("special_name", "")
        push(f"人物: {desc} {action}".rstrip())
        if name:
            push(f"人物标识: {name}")
    for obj in (caption.get("objects") or [])[:2]:
        if not isinstance(obj, dict):
            continue
        push(f"物体: {obj.get('name', '')} {obj.get('description', '')} "
             f"{obj.get('location', '')}".rstrip())
    te = caption.get("text_elements", "")
    if te:
        push(f"文字元素: {te}", cap=300)
    return lines[:MAX_LINES_PER_IMAGE]


class CaptionIndex:
    def __init__(self, store):
        self._store = store
        self._agg: Optional[np.ndarray] = None       # (N, dim) float32
        self._lines: Optional[np.ndarray] = None     # (M, dim) fp16
        self._texts: List[str] = []                  # 与行矩阵一一对应
        self._row_ids: List[int] = []
        self._offsets: List[int] = []                # 每图在行矩阵中的起始行
        self._counts: List[int] = []

    # ---------- 生命周期 ----------
    def is_ready(self) -> bool:
        return self._agg is not None

    def _paths(self) -> Tuple[str, str]:
        from core import config as cfg
        return str(cfg.CAPTION_VEC_PATH), str(cfg.CAPTION_AGG_PATH)

    def ensure_loaded(self, dim: int) -> bool:
        """按需加载矩阵；维度不符或数据缺失返回 False（调用方降级）。"""
        if self.is_ready():
            return self._agg.shape[1] == dim
        vec_path, agg_path = self._paths()
        import os
        if not (os.path.exists(vec_path) and os.path.exists(agg_path)):
            return False
        try:
            stored_dim = self._store.get_meta_value("caption_dim")
            if stored_dim is not None and int(stored_dim) != int(dim):
                return False
            mapping = self._store.caption_line_map()
            if not mapping:
                return False
            lines = np.load(vec_path)
            agg = np.load(agg_path)
            if lines.shape[1] != dim or agg.shape[1] != dim:
                return False
            if lines.shape[0] != sum(mapping.values()):
                return False
            if agg.shape[0] != len(mapping):
                return False
            self._lines = lines
            self._agg = agg.astype(np.float32)
            self._row_ids = sorted(mapping)
            self._counts = [mapping[r] for r in self._row_ids]
            offsets, pos = [], 0
            for c in self._counts:
                offsets.append(pos)
                pos += c
            self._offsets = offsets
            # 从 DB 的 caption JSON 重建行文本（与 build 时同一生成规则，确定性）
            self._texts = self._rebuild_texts()
            if len(self._texts) != lines.shape[0]:
                # caption 内容与矩阵不一致（caption 被外部改动）-> 判为需重建
                self._lines = self._agg = None
                self._row_ids = self._counts = self._offsets = []
                return False
            return True
        except Exception:
            return False

    def _rebuild_texts(self) -> List[str]:
        """按矩阵分组顺序（row_id 升序）从 DB 重建命中行文本。"""
        texts: List[str] = []
        cap_by_row = dict(self._store.iter_captioned_rows())
        for r in self._row_ids:
            raw = cap_by_row.get(r)
            if raw is None:
                continue
            try:
                caption = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            texts.extend(caption_to_lines(caption))
        return texts

    # ---------- 构建 ----------
    def build(self, embed, dim: int, embed_model: str = "",
              progress=None, message=None, stop=None) -> dict:
        """对全部有 caption 的记录构建行向量矩阵（可重跑，整表替换）。"""
        from core.searcher import SearchCancelled
        rows = sorted(self._store.iter_captioned_rows())
        texts: List[str] = []
        mapping: Dict[int, int] = {}
        parsed = 0
        for row_id, cap_json in rows:
            try:
                caption = json.loads(cap_json)
            except (json.JSONDecodeError, TypeError):
                continue
            line_list = caption_to_lines(caption)
            if not line_list:
                continue
            mapping[row_id] = len(line_list)
            texts.extend(line_list)
            parsed += 1
        if message:
            message(f"共 {parsed} 张图片有描述，{len(texts)} 行文本待嵌入...")
        if not texts:
            return {"images": 0, "lines": 0}

        batch = 32
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch):
            if stop is not None and stop.is_set():
                raise SearchCancelled()
            vecs.append(embed.encode(texts[i:i + batch], is_query=False,
                                     batch_size=batch, dim=dim))
        if progress:
            progress(len(texts), len(texts), 0.0)
        matrix = np.concatenate(vecs, axis=0).astype(np.float32)
        # 每图 max-pool 聚合（两段式检索的快路径）
        row_ids = sorted(mapping)
        agg_parts = []
        pos = 0
        for r in row_ids:
            c = mapping[r]
            agg_parts.append(matrix[pos:pos + c].max(axis=0, keepdims=True))
            pos += c
        agg = np.concatenate(agg_parts, axis=0)

        vec_path, agg_path = self._paths()
        self._store.save_array(vec_path, matrix.astype(np.float16))
        self._store.save_array(agg_path, agg.astype(np.float16))
        self._store.set_caption_lines(mapping)
        self._store.set_meta("caption_dim", str(dim))
        self._store.set_meta("embed_model", embed_model or "embed")
        self._store.set_meta("caption_built_at",
                             time.strftime("%Y-%m-%d %H:%M:%S"))
        # 热加载到内存，构建完立即可查
        self._agg = agg.astype(np.float32)
        self._lines = matrix.astype(np.float16)
        self._texts = texts
        self._row_ids = row_ids
        self._counts = [mapping[r] for r in row_ids]
        self._offsets, pos2 = [], 0
        for c in self._counts:
            self._offsets.append(pos2)
            pos2 += c
        return {"images": parsed, "lines": len(texts)}

    # ---------- 检索 ----------
    def search(self, query_vec: np.ndarray, top_k: int,
               candidates: int = 200) -> List[Tuple[int, float, str]]:
        """两段式检索：聚合 top-candidates -> 行级精确 max 重排。

        返回 [(row_id, score, 命中行文本)] 按分数降序。
        """
        if not self.is_ready():
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        scores = self._agg @ q                       # 快路径
        n = self._agg.shape[0]
        if n == 0:
            return []
        cand_n = min(candidates, n)
        idx = np.argpartition(-scores, cand_n - 1)[:cand_n]
        best: List[Tuple[int, float, str]] = []
        for i in idx:
            start, cnt = self._offsets[i], self._counts[i]
            block = self._lines[start:start + cnt].astype(np.float32) @ q
            j = int(np.argmax(block))
            best.append((self._row_ids[i], float(block[j]),
                         self._texts[start + j]))
        best.sort(key=lambda kv: -kv[1])
        return best[:top_k]

    def count_images(self) -> int:
        return len(self._row_ids)


_index_cache: Optional[CaptionIndex] = None
_index_store_id = None


def get_index(store) -> CaptionIndex:
    """与 store 绑定的单例索引（store 实例更换时自动重建）。"""
    global _index_cache, _index_store_id
    if _index_cache is None or _index_store_id != id(store):
        _index_cache = CaptionIndex(store)
        _index_store_id = id(store)
    return _index_cache
