"""索引存储：SQLite 元数据 + npy 向量矩阵。

images 表（id 与 vectors.npy 行号一一对应，行号 = id - 1）:
  id, path(唯一), mtime, size, width, height, format, thumb,
  ocr, caption(预留), indexed_at

线程安全：所有公开方法持 RLock（多个搜索线程会并发访问同一 sqlite 连接，
sqlite3 模块不允许同一连接并发使用）。删除采用"墓碑"机制：删除记录后把
向量行置零并记入 removed 集合；启动时根据 DB 重建墓碑（跨重启有效）。
"""
import sqlite3
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _locked(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class Store:
    def __init__(self, db_path: str, vec_path: str, dim: int):
        self.db_path = db_path
        self.vec_path = vec_path
        self.dim = dim
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        self.vectors = self._load_vectors()
        self.removed: set = set()
        self._rebuild_tombstones()

    def lock(self):
        """对外暴露锁，供读取 vectors 矩阵等非方法路径使用。"""
        return self._lock

    # ---------- 初始化（仅在 __init__ 中调用，无需锁） ----------
    def _create_schema(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS images(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                mtime REAL, size INTEGER,
                width INTEGER, height INTEGER,
                format TEXT, thumb TEXT,
                ocr TEXT, caption TEXT,
                indexed_at TEXT
            )""")
        self._conn.commit()

    def _load_vectors(self) -> np.ndarray:
        if Path(self.vec_path).exists():
            return np.asarray(np.load(self.vec_path), dtype=np.float32)
        return np.zeros((0, self.dim), dtype=np.float32)

    def _rebuild_tombstones(self) -> None:
        """行号在矩阵内但 DB 中已无记录 -> 墓碑行（跨重启持久）。"""
        ids = {r[0] for r in self._conn.execute("SELECT id FROM images")}
        self.removed = {i for i in range(self.vectors.shape[0]) if (i + 1) not in ids}

    # ---------- 写入 ----------
    @_locked
    def append(self, path: str, mtime: float, size: int, width: int, height: int,
               fmt: str, thumb: str, vec: np.ndarray) -> int:
        cur = self._conn.execute(
            "INSERT INTO images(path,mtime,size,width,height,format,thumb,indexed_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (path, mtime, size, width, height, fmt, thumb,
             time.strftime("%Y-%m-%d %H:%M:%S")))
        self._conn.commit()
        row_id = int(cur.lastrowid)
        if self.vectors.shape[0] < row_id:
            pad = np.zeros((row_id - self.vectors.shape[0], self.dim), dtype=np.float32)
            self.vectors = np.concatenate([self.vectors, pad], axis=0)
        self.vectors[row_id - 1] = np.asarray(vec, dtype=np.float32)
        return row_id

    @_locked
    def update(self, row_id: int, path: str, mtime: float, vec: np.ndarray) -> None:
        self._conn.execute("UPDATE images SET mtime=? WHERE id=?", (mtime, row_id))
        self._conn.commit()
        if row_id - 1 < self.vectors.shape[0]:
            self.vectors[row_id - 1] = np.asarray(vec, dtype=np.float32)

    @_locked
    def remove(self, row_id: int) -> None:
        self._conn.execute("DELETE FROM images WHERE id=?", (row_id,))
        self._conn.commit()
        idx = row_id - 1
        if idx < self.vectors.shape[0]:
            self.vectors[idx] = 0.0
        self.removed.add(idx)

    @_locked
    def remove_by_path(self, path: str) -> bool:
        r = self._conn.execute("SELECT id FROM images WHERE path=?", (path,)).fetchone()
        if not r:
            return False
        self.remove(int(r[0]))
        return True

    @_locked
    def save_vectors(self) -> None:
        np.save(self.vec_path, self.vectors)

    # ---------- 读取 ----------
    @_locked
    def known_map(self) -> Dict[str, float]:
        cur = self._conn.execute("SELECT path, mtime FROM images")
        return {r[0]: r[1] for r in cur.fetchall()}

    @_locked
    def row_by_path(self, path: str) -> Optional[int]:
        r = self._conn.execute("SELECT id FROM images WHERE path=?", (path,)).fetchone()
        return int(r[0]) if r else None

    @_locked
    def get_meta(self, row_id: int) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT id,path,mtime,size,width,height,format,thumb,ocr,caption"
            " FROM images WHERE id=?", (row_id,))
        r = cur.fetchone()
        if not r:
            return None
        keys = ["id", "path", "mtime", "size", "width", "height",
                "format", "thumb", "ocr", "caption"]
        return dict(zip(keys, r))

    @_locked
    def valid_mask(self) -> np.ndarray:
        n = self.vectors.shape[0]
        mask = np.ones(n, dtype=bool)
        for r in self.removed:
            if 0 <= r < n:
                mask[r] = False
        return mask

    # ---------- OCR ----------
    @_locked
    def rows_missing_ocr(self) -> list:
        """ocr 为 NULL（尚未识别）的记录 [(id, path)]。"""
        cur = self._conn.execute("SELECT id, path FROM images WHERE ocr IS NULL")
        return [(int(r[0]), r[1]) for r in cur.fetchall()]

    @_locked
    def iter_ocr_rows(self) -> list:
        """有 OCR 文本的记录 [(id, text)]。"""
        cur = self._conn.execute(
            "SELECT id, ocr FROM images WHERE ocr IS NOT NULL AND ocr != ''")
        return [(int(r[0]), r[1]) for r in cur.fetchall()]

    @_locked
    def set_ocr(self, row_id: int, text: str) -> None:
        """写入 OCR 文本；空字符串表示"已识别但无文字"，避免重复处理。"""
        self._conn.execute("UPDATE images SET ocr=? WHERE id=?", (text, row_id))
        self._conn.commit()

    @_locked
    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    @_locked
    def close(self) -> None:
        self.save_vectors()
        self._conn.close()
