"""精确定位 127：对比 msvcp140 两个小版本的导出差异，与 Qt6Core 导入需求求交。"""
import pefile

ENV = r"D:\Data\Anaconda\envs\image_caption_and_select"
QT6CORE = ENV + r"\Lib\site-packages\PySide6\Qt6Core.dll"
MSVCP_OLD = ENV + r"\msvcp140.dll"                    # 14.44.35208 (env 根)
MSVCP_NEW = ENV + r"\Lib\site-packages\shiboken6\msvcp140.dll"  # 14.44.35211
VCRT_OLD = ENV + r"\vcruntime140.dll"
VCRT_NEW = ENV + r"\Lib\site-packages\shiboken6\vcruntime140.dll"


def exports(path):
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
    return {e.name.decode() if e.name else f"ord{e.ordinal}"
            for e in pe.DIRECTORY_ENTRY_EXPORT.symbols}


def imports_from(path, dllname):
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
    out = set()
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        if entry.dll and entry.dll.decode().lower() == dllname.lower():
            out = {i.name.decode() for i in entry.imports if i.name}
    return out


for name, old, new, dll in [
    ("MSVCP140", MSVCP_OLD, MSVCP_NEW, "MSVCP140.dll"),
    ("MSVCP140_1", ENV + r"\msvcp140_1.dll",
     ENV + r"\Lib\site-packages\shiboken6\msvcp140_1.dll", "MSVCP140_1.dll"),
    ("VCRUNTIME140", VCRT_OLD, VCRT_NEW, "VCRUNTIME140.dll"),
    ("VCRUNTIME140_1", ENV + r"\vcruntime140_1.dll",
     ENV + r"\Lib\site-packages\shiboken6\vcruntime140_1.dll", "VCRUNTIME140_1.dll"),
]:
    e_old, e_new = exports(old), exports(new)
    missing = e_new - e_old
    need = imports_from(QT6CORE, dll)
    hit = need & missing
    print(f"{name}: 新版本独有导出 {len(missing)} 个，Qt6Core 需要其中 {len(hit)} 个")
    for s in sorted(hit)[:20]:
        print("   MISSING:", s)
    if not hit:
        print("   （无命中，该 DLL 不是问题）")
