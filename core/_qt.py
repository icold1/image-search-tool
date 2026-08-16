"""Qt 环境引导。

ensure_qt_env():
    显式设置 QT_PLUGIN_PATH 指向 PySide6 的 plugins 目录，确保能找到
    platforms/qwindows.dll（本机 conda 环境下 Qt 的插件前缀解析为空，
    不设置会报 "Could not find the Qt platform plugin 'windows'"）。

preload_qt():
    PySide6 6.11.1 的 Qt6Core.dll 缺少 QtCore.pyd 所需符号（错误 127）时
    曾用完整路径预加载依赖闭包绕开冲突 DLL；当前固定使用 6.9.2 后不再
    需要调用，保留备查。
"""
import ctypes
import importlib.util
import os
import sys
from pathlib import Path


def _pyside_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("PySide6")
        return Path(spec.submodule_search_locations[0])
    except Exception:
        return None


def ensure_qt_env() -> None:
    if os.name != "nt":
        return
    d = _pyside_dir()
    if d is None:
        return
    plugins = d / "plugins"
    if plugins.is_dir():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))


def _env_library_bin() -> Path:
    for prefix in (sys.prefix, sys.base_prefix):
        p = Path(prefix) / "Library" / "bin"
        if p.is_dir():
            return p
    return Path(".")


def _load(path: str) -> bool:
    try:
        return bool(ctypes.WinDLL(path))
    except Exception:
        return False


def preload_qt() -> None:
    if os.name != "nt":
        return
    pyside_dir = _pyside_dir()
    if pyside_dir is None:
        return
    shiboken_dir = pyside_dir.parent / "shiboken6"

    lib_bin = _env_library_bin()
    for f in ("icudt.dll", "icuuc.dll", "icuin.dll"):
        _load(str(lib_bin / f))

    for f in ("vcruntime140.dll", "vcruntime140_1.dll",
              "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
              "concrt140.dll"):
        _load(str(shiboken_dir / f))

    for f in ("shiboken6.abi3.dll", "pyside6.abi3.dll",
              "Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll"):
        _load(str((shiboken_dir if f.startswith("shiboken6") else pyside_dir) / f))

    for prefix in (sys.prefix, sys.base_prefix):
        _load(str(Path(prefix) / "python3.dll"))
