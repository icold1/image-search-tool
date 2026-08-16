"""入口：悬浮球图片语义搜索工具。

运行:  conda activate image_caption_and_select
        python main.py
"""
import ctypes
import faulthandler
import os
import sys
import time
import warnings
from pathlib import Path

# 静默第三方库的弃用警告（jieba 的 pkg_resources 警告会污染 stderr，
# 导致后台监控误报退出码 1）
warnings.filterwarnings("ignore")

faulthandler.enable()  # 原生崩溃时输出线程栈（诊断收尾阶段崩溃用）

from core._win import ensure_env_dlls
from core._qt import ensure_qt_env
ensure_env_dlls()   # 未激活 conda 环境时也能找到 cuDNN 等 DLL
ensure_qt_env()     # 保证能找到 Qt 平台插件 qwindows.dll

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog

from core import config as cfg
from core import store as st
from core import worker
from ui.floating_ball import FloatingBall
from ui.hotkey import HotkeyThread
from ui.search_panel import SearchPanel
from ui.tray import TrayIcon


def _quit_log(msg: str):
    """退出路径日志（定位退出码 1 / QThreadStorage 警告用）。"""
    try:
        import datetime
        with open(cfg.DATA_DIR / "quit.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')} {msg}\n")
    except Exception:
        pass


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("图片语义搜索")
    app.setQuitOnLastWindowClosed(False)
    # 全局 QToolTip 样式：避免深色系统主题下提示框显示为不可读的黑块
    app.setStyleSheet("QToolTip { background-color: #343b47; color: #eef1f6;"
                      " border: 1px solid #4a5568; padding: 4px 8px; }")

    # 调试用：--auto-quit-ms=N 启动 N 毫秒后自动走正常退出流程
    auto_ms = None
    for a in sys.argv[1:]:
        if a.startswith("--auto-quit-ms="):
            try:
                auto_ms = int(a.split("=", 1)[1])
            except ValueError:
                auto_ms = None

    # 调试用：--auto-search=词 启动 3 秒后自动触发一次查询（配合退出测试）
    auto_search = None
    for a in sys.argv[1:]:
        if a.startswith("--auto-search="):
            auto_search = a.split("=", 1)[1]

    conf = cfg.load_config()
    # 首次运行：选择图片文件夹
    if not conf.get("image_dir") or not Path(conf["image_dir"]).is_dir():
        chosen = QFileDialog.getExistingDirectory(None, "请选择存放图片的文件夹")
        if not chosen:
            sys.exit(0)
        conf["image_dir"] = chosen
        cfg.save_config(conf)

    dim = cfg.MODEL_DIMS.get(conf["model_name"], 768)
    store = st.Store(str(cfg.DB_PATH), str(cfg.VEC_PATH), dim)

    ball = FloatingBall(conf)
    panel = SearchPanel(conf, store)
    tray = TrayIcon(conf)

    def request_quit():
        """统一退出入口：先标记悬浮球为"明确关闭"。

        Qt 退出序列会先隐藏所有顶层窗口、后发 aboutToQuit——
        提前标记可避免退出时的正常隐藏被误记入诊断日志。
        """
        ball.mark_shutdown()
        app.quit()

    if auto_ms is not None:
        QTimer.singleShot(auto_ms, request_quit)
    tray.show()
    tray.set_status(store.count())

    index_worker = [None]

    def start_index():
        if index_worker[0] is not None and index_worker[0].isRunning():
            return
        panel.set_status("准备建立索引...")
        w = worker.IndexWorker(store, conf["image_dir"], conf["model_name"],
                               conf["device"], conf["batch_size"],
                               conf["thumb_size"])
        w.message.connect(panel.set_status)
        w.progress.connect(panel.show_progress)
        w.finished_ok.connect(on_index_done)
        w.failed.connect(on_index_failed)
        index_worker[0] = w
        w.start()

    def on_index_done(stats: dict):
        panel.hide_progress_bar()
        panel.set_status(
            f"索引完成：新增 {stats['added']} · 更新 {stats['updated']}"
            f" · 跳过 {stats['skipped']} · 清理 {stats['deleted']}"
            f" · OCR {stats.get('ocr_done', 0)}"
            f" · 共 {store.count()} 张可搜索")
        tray.set_status(store.count())

    def on_index_failed(msg: str):
        panel.hide_progress_bar()
        panel.set_status("索引出错，详见控制台")
        print("[索引错误]", msg)

    def on_topk_changed(k: int):
        conf["top_k"] = k
        cfg.save_config(conf)
        tray.set_topk(k)

    def on_ball_visibility(v: bool):
        ball.set_explicit_hidden(not v)
        ball.setVisible(v)
        if v:
            ball.ensure_visible()
        if not v:
            panel.hide()

    def on_remove(path: str):
        from core import textindex
        rid = store.row_by_path(path)
        if store.remove_by_path(path):
            if rid is not None:
                textindex.get_index(store).remove(rid)
            store.save_vectors()
            panel.set_status("已从索引移除该图片")

    ball.toggled.connect(lambda: panel.toggle_near(ball.frameGeometry()))
    ball.position_changed.connect(
        lambda: panel.follow_ball(ball.frameGeometry()))
    ball.refresh_requested.connect(start_index)
    ball.topk_changed.connect(on_topk_changed)
    ball.quit_requested.connect(request_quit)
    panel.remove_requested.connect(on_remove)

    tray.toggle_panel.connect(lambda: panel.toggle_near(ball.frameGeometry()))
    tray.ball_visibility_changed.connect(on_ball_visibility)
    tray.refresh_requested.connect(start_index)
    tray.topk_changed.connect(on_topk_changed)
    tray.quit_requested.connect(request_quit)

    hotkey = HotkeyThread(conf.get("hotkey", "Ctrl+Shift+Space"))
    hotkey.triggered.connect(lambda: panel.toggle_near(ball.frameGeometry()))
    hotkey.start()

    def on_app_state(state):
        """应用回到前台时兜底恢复悬浮球。

        悬浮球是 Qt.Tool 置顶窗口：某些 Windows 场景（打开外部程序/
        文件对话框导致本应用失焦）下可能被系统隐藏，且不会自动恢复。
        用户没在托盘关掉它时，回到前台就重新显示。
        """
        if state == Qt.ApplicationState.ApplicationActive:
            ball.ensure_visible()

    app.applicationStateChanged.connect(on_app_state)

    def on_quit():
        _quit_log("on_quit 开始")
        ball.mark_shutdown()   # 兜底：退出销毁窗口属正常隐藏，不记诊断日志
        ball.save_position()
        cfg.save_config(conf)
        tray.hide()
        if index_worker[0] is not None and index_worker[0].isRunning():
            index_worker[0].stop()
            index_worker[0].wait(3000)
        if index_worker[0] is not None:
            index_worker[0].deleteLater()
            index_worker[0] = None
        _quit_log("索引线程已处理")
        panel.shutdown_workers()  # 等待在途查询线程结束，再关数据库
        _quit_log("搜索线程已等待")
        hotkey.stop()
        hotkey.wait(1500)
        hotkey.deleteLater()
        _quit_log("热键线程已处理")
        store.close()
        _quit_log("store 已关闭")
        # 处理 deleteLater，让 QThread 对象在事件循环结束前完成析构，
        # 避免 Qt 在进程收尾阶段崩溃（QThreadStorage 警告 + 退出码 1）
        app.processEvents()

    app.aboutToQuit.connect(on_quit)

    ball.show()
    if store.count() == 0:
        # 首次运行：显示面板展示索引进度
        panel.toggle_near(ball.frameGeometry())
        panel.set_status("首次运行：正在建立索引（首次需下载模型，约 1.6GB）")
        start_index()
    elif conf.get("ocr_enabled", True) and store.rows_missing_ocr():
        # 升级后自动补齐 OCR 文字识别（后台进行，不打断使用）
        start_index()

    if auto_search is not None:
        queries = [q for q in auto_search.split("|") if q]
        for idx, q in enumerate(queries):
            def _auto_search(qq=q):
                _quit_log(f"调试：自动查询「{qq}」")
                panel.edit.setText(qq)
                panel.query()
            QTimer.singleShot(3000 + idx * 1500, _auto_search)
    code = app.exec()
    _quit_log(f"exec 返回 {code}，TerminateProcess 直接终止（跳过 DLL 卸载）")
    # 所有数据已在 on_quit 保存。本机在进程收尾（ExitProcess 的 DLL 卸载
    # 阶段）存在与 UI 交互相关的静默崩溃，TerminateProcess 不执行任何
    # DLL 卸载回调，彻底绕开该阶段。
    # 注意：必须显式声明参数类型——ctypes 默认把句柄按 32 位整数传递，
    # 64 位下 GetCurrentProcess 的伪句柄被截断导致调用静默失败。
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.TerminateProcess.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    _quit_log(f"TerminateProcess handle=0x{handle:x} code={code}")
    ok = kernel32.TerminateProcess(handle, ctypes.c_uint(code))
    _quit_log(f"TerminateProcess 返回 {ok}（正常不会走到这行）")
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
