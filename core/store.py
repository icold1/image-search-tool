"""索引存储：SQLite 元数据 + npy 向量矩阵。

images 表（id 与 vectors.npy 行号一一对应，行号 = id - 1）:
  id, path(唯一), mtime, size, width, height, format, thumb,
  ocr, caption, status, error, indexed_at

caption_lines 表: row_id 主键 -> line_count（caption 行向量矩阵中该图占用的行数，
  行按 row_id 升序连续存放；矩阵本体在 data/caption_vecs.npy）。
meta 表: key/value —— 记录各模型名/维度/构建时间等版本信息。

线程安全：所有公开方法持 RLock（多个搜索线程会并发访问同一 sqlite 连接，
sqlite3 模块不允许同一连接并发使用）。删除采用"墓碑"机制：删除记录后把
向量行置零并记入 removed 集合；启动时根据 DB 重建墓碑（跨重启有效）。
向量文件写入一律走"临时文件 + os.replace"原子替换，避免中断损坏。
"""
import os
import sqlite3
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        self._migrate_schema()
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

    def _migrate_schema(self) -> None:
        """旧库平滑升级：补齐 caption 阶段需要的列与表。"""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(images)")}
        if "status" not in cols:
            self._conn.execute(
                "ALTER TABLE images ADD COLUMN status TEXT DEFAULT 'ready'")
        if "error" not in cols:
            self._conn.execute("ALTER TABLE images ADD COLUMN error TEXT")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS caption_lines(
                row_id INTEGER PRIMARY KEY, line_count INTEGER NOT NULL)""")
        self._conn.commit()

    def _load_vectors(self) -> np.ndarray:
        if Path(self.vec_path).exists():
            arr = np.asarray(np.load(self.vec_path), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self.dim:
                raise RuntimeError(
                    f"向量维度不匹配：文件 {arr.shape} 与配置模型维度 {self.dim} "
                    f"不一致。请先删除 data/index.db 与 data/vectors.npy 再重建索引")
            return arr
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
    def update(self, row_id: int, path: str, mtime: float, size: int,
               width: int, height: int, fmt: str, thumb: str,
               vec: np.ndarray) -> None:
        """文件内容变更后的全量刷新：元数据 + 缩略图 + 向量。"""
        self._conn.execute(
            "UPDATE images SET mtime=?, size=?, width=?, height=?, format=?,"
            " thumb=? WHERE id=?",
            (mtime, size, width, height, fmt, thumb, row_id))
        self._conn.commit()
        if row_id - 1 < self.vectors.shape[0]:
            self.vectors[row_id - 1] = np.asarray(vec, dtype=np.float32)

    @_locked
    def remove(self, row_id: int) -> None:
        self._conn.execute("DELETE FROM images WHERE id=?", (row_id,))
        self._conn.execute("DELETE FROM caption_lines WHERE row_id=?", (row_id,))
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
        """原子保存：先写临时文件再替换，中断/崩溃不损坏向量库。"""
        self.save_array(self.vec_path, self.vectors)

    @staticmethod
    def save_array(path: str, arr: np.ndarray) -> None:
        # 临时名必须以 .npy 结尾：np.save 对非 .npy 结尾的名字会自动追加后缀，
        # 导致 os.replace 找不到临时文件
        tmp = path + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)

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

    # ---------- caption ----------
    @_locked
    def rows_missing_caption(self, include_failed: bool = False) -> List[Tuple[int, str]]:
        """尚无 caption 的记录 [(id, path)]。

        include_failed=True 时把 status='failed' 的也纳入（重试）。
        """
        if include_failed:
            cur = self._conn.execute(
                "SELECT id, path FROM images WHERE caption IS NULL")
        else:
            cur = self._conn.execute(
                "SELECT id, path FROM images WHERE caption IS NULL"
                " AND status='ready'")
        return [(int(r[0]), r[1]) for r in cur.fetchall()]

    @_locked
    def iter_captioned_rows(self) -> List[Tuple[int, str]]:
        """有 caption 的记录 [(id, caption_json_str)]。"""
        cur = self._conn.execute(
            "SELECT id, caption FROM images"
            " WHERE caption IS NOT NULL AND caption != ''")
        return [(int(r[0]), r[1]) for r in cur.fetchall()]

    @_locked
    def set_caption(self, row_id: int, caption_json: Optional[str],
                    status: str = "ready", error: Optional[str] = None) -> None:
        """写入 caption（JSON 字符串）；失败时 caption_json 为 None 并记录状态。"""
        self._conn.execute(
            "UPDATE images SET caption=?, status=?, error=? WHERE id=?",
            (caption_json, status, error, row_id))
        self._conn.commit()

    @_locked
    def count_captioned(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM images"
            " WHERE caption IS NOT NULL AND caption != ''").fetchone()[0])

    @_locked
    def count_caption_failed(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM images WHERE status='failed'").fetchone()[0])

    # ---------- caption 行向量映射 ----------
    @_locked
    def set_caption_lines(self, mapping: Dict[int, int]) -> None:
        """整表替换 row_id -> line_count 映射（矩阵按 row_id 升序分组）。"""
        self._conn.execute("DELETE FROM caption_lines")
        self._conn.executemany(
            "INSERT INTO caption_lines(row_id, line_count) VALUES(?,?)",
            sorted(mapping.items()))
        self._conn.commit()

    @_locked
    def caption_line_map(self) -> Dict[int, int]:
        cur = self._conn.execute(
            "SELECT row_id, line_count FROM caption_lines ORDER BY row_id")
        return {int(r[0]): int(r[1]) for r in cur.fetchall()}

    # ---------- 元信息 ----------
    @_locked
    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))
        self._conn.commit()

    @_locked
    def get_meta_value(self, key: str) -> Optional[str]:
        """读取 meta 键值表（注意：与按 row_id 取行的 get_meta 区分）。"""
        r = self._conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(r[0]) if r else None

    # ---------- 统计 / 关闭 ----------
    @_locked
    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    @_locked
    def close(self) -> None:
        self.save_vectors()
        self._conn.close()
