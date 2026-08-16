"""符号级对比：pyside6.abi3.dll 的导入 vs 其两个依赖的导出。"""
import pefile

ENV = r"D:\Data\Anaconda\envs\image_caption_and_select"
PYSIDE = ENV + r"\Lib\site-packages\PySide6"
SHIBOKEN = ENV + r"\Lib\site-packages\shiboken6"
PY3 = ENV + r"\python3.dll"

TARGETS = [
    (PYSIDE + r"\pyside6.abi3.dll", [SHIBOKEN + r"\shiboken6.abi3.dll", PY3]),
    (PYSIDE + r"\QtCore.pyd", [SHIBOKEN + r"\shiboken6.abi3.dll", PY3,
                               PYSIDE + r"\Qt6Core.dll",
                               PYSIDE + r"\pyside6.abi3.dll"]),
]


def exports(path):
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
    return {e.name.decode() if e.name else f"ord{e.ordinal}"
            for e in pe.DIRECTORY_ENTRY_EXPORT.symbols}


def imports_with_dll(path):
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
    out = {}
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        out[entry.dll.decode().lower()] = {
            i.name.decode() if i.name else f"ord{i.ordinal}"
            for i in entry.imports}
    return out


for target, deps in TARGETS:
    print("=" * 60)
    print("目标:", target.split("site-packages")[-1])
    imp = imports_with_dll(target)
    for dep in deps:
        name = dep.split("\\")[-1]
        need = imp.get(name.lower(), set())
        exp = exports(dep)
        missing = need - exp
        print(f"  <- {name}: 需要 {len(need)} 个符号，缺少 {len(missing)} 个")
        for s in sorted(missing)[:25]:
            print("      MISSING:", s)
