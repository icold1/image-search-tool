"""按 Windows DLL 搜索顺序模拟解析，找出"哪个文件缺哪个导出"。

用法: python tools/dep_check.py [appdir]
模拟: 目标 = PySide6\QtCore.pyd，搜索 = DLL_LOAD_DIR + appdir + System32 +
System + Windows + PATH(环境变量)。递归检查到深度 2。
"""
import os
import sys

import pefile

ENV = r"D:\Data\Anaconda\envs\image_caption_and_select"
PYSIDE = ENV + r"\Lib\site-packages\PySide6"
TARGET = PYSIDE + r"\QtCore.pyd"

APP_DIR = sys.argv[1] if len(sys.argv) > 1 else ENV

_exports_cache = {}


def exports(path):
    if path not in _exports_cache:
        try:
            pe = pefile.PE(path, fast_load=True)
            pe.parse_data_directories(
                directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
            _exports_cache[path] = {
                e.name.decode() if e.name else f"ord{e.ordinal}"
                for e in pe.DIRECTORY_ENTRY_EXPORT.symbols}
        except Exception:
            _exports_cache[path] = set()
    return _exports_cache[path]


def imports(path):
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
    out = []
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        syms = [(i.name.decode() if i.name else f"ord{i.ordinal}")
                for i in entry.imports]
        out.append((entry.dll.decode(), syms))
    return out


def search_dirs(app_dir):
    dirs = [PYSIDE, app_dir, r"C:\Windows\System32", r"C:\Windows\System",
            r"C:\Windows"]
    dirs += [p for p in os.environ.get("PATH", "").split(";") if p]
    return dirs


def resolve(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def check(target, dirs, depth, visited):
    if target in visited or depth > 2:
        return
    visited.add(target)
    name = os.path.basename(target)
    for dll, syms in imports(target):
        if dll.lower().startswith("api-ms-win"):
            continue  # API 集合，由系统内核解析
        path = resolve(dll, dirs)
        if path is None:
            print(f"[{name}] 找不到依赖文件: {dll}")
            continue
        exp = exports(path)
        missing = [s for s in syms if s not in exp]
        if missing:
            print(f"[{name}] <- {dll} ({path}) 缺少导出 {len(missing)} 个:")
            for s in missing[:12]:
                print(f"      {s}")
        else:
            print(f"[{name}] <- {dll} OK ({os.path.dirname(path)})")
        check(path, dirs, depth + 1, visited)


print(f"目标: {TARGET}")
print(f"应用目录: {APP_DIR}")
print("=" * 60)
check(TARGET, search_dirs(APP_DIR), 0, set())
