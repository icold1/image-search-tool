"""top-k 余弦相似度检索（纯 numpy，10 万级图片足够快）。"""
from typing import List, Tuple

import numpy as np


class SearchCancelled(Exception):
    """查询被取消（worker.stop()/新查询顶替），调用方应静默丢弃。"""


def search(text_feat: np.ndarray, vectors: np.ndarray, mask: np.ndarray,
           k: int = 10) -> List[Tuple[int, float]]:
    """返回 [(0 基行号, 相似度)] 按分数降序。

    text_feat: (dim,) 或 (1, dim) 已归一化
    vectors:   (n, dim) float32 已归一化
    mask:      (n,) bool，False 的行（墓碑）不参与
    """
    if vectors.shape[0] == 0 or not mask.any():
        return []
    q = np.asarray(text_feat, dtype=np.float32).reshape(1, -1)
    sims = (vectors @ q.T)[:, 0]
    sims = sims.copy()
    sims[~mask] = -np.inf
    valid = int(mask.sum())
    k = min(k, valid)
    if k < 1:
        return []
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx if np.isfinite(sims[i])]
