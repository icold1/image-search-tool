"""OCR 文本倒排索引：让查询词优先命中图片中的文字。

词元：jieba 中文分词 + 2 位以上英文/数字串。
打分：命中词元数 / 查询词元数（0~1）；整段查询作为子串出现在
OCR 文本中时额外 +0.5。
"""
import logging
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jieba

jieba.setLogLevel(logging.WARNING)

_index_cache: Optional["TextIndex"] = None
_index_store_id = None


class TextIndex:
    def __init__(self, store):
        self._store = store
        self._inv: Dict[str, Set[int]] = defaultdict(set)
        self._texts: Dict[int, str] = {}
        self._row_tokens: Dict[int, Set[str]] = {}
        self._rebuild()

    # ---------- 构建 ----------
    def _rebuild(self):
        self._inv.clear()
        self._texts.clear()
        self._row_tokens.clear()
        for row_id, text in self._store.iter_ocr_rows():
            self.add(int(row_id), text)

    def add(self, row_id: int, text: str):
        self.remove(row_id)
        if not text:
            return
        toks = self._tokenize(text)
        self._texts[row_id] = text
        self._row_tokens[row_id] = toks
        for t in toks:
            self._inv[t].add(row_id)

    def remove(self, row_id: int):
        for t in self._row_tokens.pop(row_id, ()):
            s = self._inv.get(t)
            if s:
                s.discard(row_id)
                if not s:
                    self._inv.pop(t, None)
        self._texts.pop(row_id, None)

    # ---------- 检索 ----------
    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        toks = {w.strip().lower() for w in jieba.cut(text) if w.strip()}
        toks.update(m.lower() for m in re.findall(r"[A-Za-z0-9]{2,}", text))
        return toks

    def search(self, query: str, limit: int = 100,
               stop=None) -> List[Tuple[int, float]]:
        """检索。stop 为可选 threading.Event，置位时抛 SearchCancelled。"""
        from core.searcher import SearchCancelled
        qtoks = self._tokenize(query)
        if not qtoks:
            return []
        scores = defaultdict(float)
        for t in qtoks:
            for rid in self._inv.get(t, ()):
                scores[rid] += 1.0 / len(qtoks)
        q = query.strip().lower()
        if q:
            checked = 0
            for rid, text in self._texts.items():
                checked += 1
                if stop is not None and (checked % 4096 == 0) and stop.is_set():
                    raise SearchCancelled()
                if q in text.lower():
                    scores[rid] += 0.5
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(rid, round(score, 3)) for rid, score in ranked[:limit]]

    def count(self) -> int:
        return len(self._texts)


def get_index(store) -> TextIndex:
    """获取与 store 绑定的单例索引（store 实例更换时自动重建）。"""
    global _index_cache, _index_store_id
    if _index_cache is None or _index_store_id != id(store):
        _index_cache = TextIndex(store)
        _index_store_id = id(store)
    return _index_cache
