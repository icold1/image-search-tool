"""DLL 诊断：打印当前进程已加载的关键运行时 DLL 的实际路径。"""
import ctypes
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
GetModuleHandleExW = kernel32.GetModuleHandleExW
GetModuleHandleExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR,
                               ctypes.POINTER(wintypes.HMODULE)]
GetModuleHandleExW.restype = wintypes.BOOL
GetModuleFileNameW = kernel32.GetModuleFileNameW
GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
GetModuleFileNameW.restype = wintypes.DWORD
FROM_ADDRESS = 0x4


def loaded_path(dllname: str) -> str:
    h = ctypes.WinDLL(dllname)  # 若未加载会触发加载；关键 DLL 通常已在进程中
    mod = wintypes.HMODULE()
    GetModuleHandleExW(FROM_ADDRESS,
                       ctypes.cast(h._handle, wintypes.LPCWSTR),
                       ctypes.byref(mod))
    buf = ctypes.create_unicode_buffer(2048)
    GetModuleFileNameW(mod, buf, 2048)
    return buf.value


for d in ["ucrtbase.dll", "vcruntime140.dll", "vcruntime140_1.dll",
          "msvcp140.dll", "python312.dll"]:
    try:
        print(f"{d:18s} -> {loaded_path(d)}")
    except Exception as e:
        print(f"{d:18s} -> ERROR {e}")

# 直接加载 Qt6Core.dll 测试
pyside = (r"D:\Data\Anaconda\envs\image_caption_and_select"
          r"\Lib\site-packages\PySide6")
for dll in ["Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll"]:
    try:
        ctypes.WinDLL(pyside + "\\" + dll)
        print(f"{dll:18s} -> direct load OK")
    except Exception as e:
        print(f"{dll:18s} -> direct load FAIL: {e}")
