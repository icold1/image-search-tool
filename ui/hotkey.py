"""全局热键（Windows RegisterHotKey，零额外依赖）。

在独立线程里跑消息循环接收 WM_HOTKEY；注册失败（热键被占用）时静默降级，
悬浮球点击始终可用。
"""
import ctypes
from ctypes import wintypes

from PySide6.QtCore import QThread, Signal

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT = 0x1
MOD_CONTROL = 0x2
MOD_SHIFT = 0x4
MOD_WIN = 0x8

_MODS = {"Ctrl": MOD_CONTROL, "Alt": MOD_ALT, "Shift": MOD_SHIFT, "Win": MOD_WIN}
_VKS = {"Space": 0x20, "`": 0xC0, "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73}


def parse_combo(combo: str):
    """'Ctrl+Shift+Space' -> (modifiers, vk)"""
    mods, vk = 0, None
    for p in combo.replace(" ", "").split("+"):
        if p in _MODS:
            mods |= _MODS[p]
        elif p in _VKS:
            vk = _VKS[p]
        elif len(p) == 1:
            vk = ord(p.upper())
    if vk is None:
        raise ValueError(f"无法解析热键: {combo}")
    return mods, vk


class HotkeyThread(QThread):
    triggered = Signal()

    def __init__(self, combo: str, parent=None):
        super().__init__(parent)
        self._combo = combo
        self._tid = None

    def run(self):
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        try:
            mods, vk = parse_combo(self._combo)
        except ValueError:
            return
        self._tid = kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, 1, mods, vk):
            return  # 被占用则静默禁用，不影响其他功能
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                if msg.message == WM_HOTKEY:
                    self.triggered.emit()
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            user32.UnregisterHotKey(None, 1)

    def stop(self):
        if self._tid and self.isRunning():
            ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
