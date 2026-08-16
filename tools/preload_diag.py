"""诊断 preload_qt 每一步的成败。"""
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Data\Projects\Python\image-captioin-and-select")

from core._qt import _env_library_bin  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.find_spec("PySide6")
pyside_dir = Path(spec.submodule_search_locations[0])
shiboken_dir = pyside_dir.parent / "shiboken6"
lib_bin = _env_library_bin()
print("pyside_dir:", pyside_dir)
print("shiboken_dir:", shiboken_dir)
print("lib_bin:", lib_bin)

files = [str(lib_bin / f) for f in ("icudt.dll", "icuuc.dll", "icuin.dll")]
files += [str(shiboken_dir / f) for f in (
    "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll",
    "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll",
    "shiboken6.abi3.dll")]
files += [str(pyside_dir / f) for f in (
    "pyside6.abi3.dll", "Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll")]
for prefix in (sys.prefix, sys.base_prefix):
    files.append(str(Path(prefix) / "python3.dll"))

for f in files:
    exists = Path(f).exists()
    try:
        h = ctypes.WinDLL(f)
        print(f"OK    {Path(f).name}  ({'存在' if exists else '不存在!'})")
    except Exception as e:
        print(f"FAIL  {Path(f).name}  exists={exists}  err={e}")
