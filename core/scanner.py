"""图片文件夹扫描：全量枚举 + 基于 mtime 的增量对比。"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass
class ImageFile:
    path: str
    mtime: float
    size: int


def scan_folder(root: str, exts: Optional[set] = None) -> List[ImageFile]:
    """递归扫描文件夹，返回支持的图片文件（按路径排序，结果稳定）。"""
    exts = {e.lower() for e in (exts or EXTENSIONS)}
    result: List[ImageFile] = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fn in filenames:
            if Path(fn).suffix.lower() not in exts:
                continue
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            result.append(ImageFile(path=full, mtime=st.st_mtime, size=st.st_size))
    result.sort(key=lambda f: f.path)
    return result


def diff_scan(root: str, known: Dict[str, float],
              exts: Optional[set] = None) -> tuple:
    """对比磁盘与已索引记录（path -> mtime）。

    返回 (需要索引的文件列表, 磁盘上已不存在的路径列表)。
    """
    on_disk = scan_folder(root, exts)
    disk_map = {f.path: f for f in on_disk}
    to_index = [f for f in on_disk
                if f.path not in known or known[f.path] != f.mtime]
    deleted = [p for p in known if p not in disk_map]
    return to_index, deleted
