"""查询面板：输入框（1s 防抖）+ 结果网格 + 拖拽文件到微信。"""
import os
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import QMimeData, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDrag, QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
                               QLineEdit, QMenu, QProgressBar, QScrollArea,
                               QVBoxLayout, QWidget)

from core.worker import SearchWorker

THUMB = 150            # 网格单元宽
PANEL_WIDTH = 560
PANEL_HEIGHT = 460
COLS = 3               # 每行数量


def open_in_explorer(path: str):
    subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])


def copy_file_to_clipboard(path: str):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(path)])
    QApplication.clipboard().setMimeData(mime)


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} 秒"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分 {int(seconds % 60)} 秒"
    return f"{seconds / 3600:.1f} 小时"


class ResultThumb(QWidget):
    """单个结果缩略图：单击打开原图，按住拖到微信，右键菜单。"""
    open_requested = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, meta: dict, score: float, text_score=None, desc=None,
                 parent=None):
        super().__init__(parent)
        self.path = meta["path"]
        self.score = score
        self.text_score = text_score
        self.desc = desc          # {"score": float, "line": str} | None
        pm = (QPixmap(meta["thumb"])
              if meta.get("thumb") and Path(meta["thumb"]).exists()
              else QPixmap())
        if pm.isNull():
            pm = QPixmap(THUMB - 10, THUMB - 10)
            pm.fill(QColor("#4a5160"))
        self._pm = pm.scaled(THUMB - 10, THUMB - 10,
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setFixedSize(THUMB, self._pm.height() + 22)
        self._name = Path(self.path).name
        self._press = None
        self._dragging = False

    def sizeHint(self):
        return self.size()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        x = (self.width() - self._pm.width()) // 2
        p.drawPixmap(x, 4, self._pm)
        # 文字命中角标（左上角）
        if self.text_score is not None:
            old_font = p.font()
            badge = QRectF(x + 2, 6, 18, 16)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#E8890C"))
            p.drawRoundedRect(badge, 4, 4)
            p.setPen(QColor("white"))
            f = p.font()
            f.setPixelSize(11)
            p.setFont(f)
            p.drawText(badge, Qt.AlignCenter, "文")
            p.setFont(old_font)
        # 描述（caption）命中角标（右上角）
        if self.desc is not None:
            old_font = p.font()
            badge = QRectF(x + self._pm.width() - 20, 6, 18, 16)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#3E9B5D"))
            p.drawRoundedRect(badge, 4, 4)
            p.setPen(QColor("white"))
            f = p.font()
            f.setPixelSize(11)
            p.setFont(f)
            p.drawText(badge, Qt.AlignCenter, "述")
            p.setFont(old_font)
        # 文件名（省略号截断）
        name = self.fontMetrics().elidedText(self._name, Qt.ElideMiddle,
                                             self.width() - 8)
        p.setPen(QColor("#c6cdd8"))
        p.drawText(QRectF(4, self.height() - 17, self.width() - 8, 15),
                   Qt.AlignCenter, name)

    # ---------- 单击 / 拖拽 ----------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press = event.globalPosition().toPoint()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press is not None and (event.buttons() & Qt.LeftButton):
            delta = event.globalPosition().toPoint() - self._press
            if (not self._dragging
                    and delta.manhattanLength() > QApplication.startDragDistance()):
                self._dragging = True
                self._start_drag()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self._dragging:
            self.open_requested.emit(self.path)
        self._press = None
        super().mouseReleaseEvent(event)

    def _start_drag(self):
        """以文件形式拖出（微信聊天窗接受文件拖入）。"""
        drag = QDrag(self)
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(self.path)])
        drag.setMimeData(mime)
        drag.setPixmap(self._pm)
        drag.setHotSpot(self._pm.rect().center())
        drag.exec(Qt.DropAction.CopyAction)
        self._dragging = False

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        act_open = menu.addAction("打开原图")
        act_copy = menu.addAction("复制文件（可去微信 Ctrl+V 粘贴）")
        act_dir = menu.addAction("打开所在文件夹")
        act_rm = menu.addAction("从索引移除")
        chosen = menu.exec(event.globalPos())
        if chosen == act_open:
            os.startfile(self.path)
        elif chosen == act_copy:
            copy_file_to_clipboard(self.path)
        elif chosen == act_dir:
            open_in_explorer(self.path)
        elif chosen == act_rm:
            self.remove_requested.emit(self.path)


class SearchPanel(QWidget):
    remove_requested = Signal(str)   # 转发到主程序处理

    def __init__(self, config: dict, store, parent=None):
        super().__init__(parent)
        self._conf = config
        self._store = store
        self._gen = 0
        self._search_worker = None
        self._workers = []  # 持有运行中的线程引用，防止被 GC 提前析构
        self.setWindowFlags(Qt.FramelessWindowHint
                            | Qt.WindowStaysOnTopHint | Qt.Tool)
        # 不设置 WA_TranslucentBackground：圆角外的透明像素会让鼠标事件
        # 穿透到桌面（滚轮落到其他窗口）。面板为不透明窗口。
        self.setAttribute(Qt.WA_StyledBackground)
        self.setObjectName("panelRoot")
        # 宽度固定，高度自适应：无结果时紧凑（仅输入框），有结果时展开
        self.setFixedWidth(PANEL_WIDTH)
        self._last_ball_geom = None
        self.setStyleSheet("""
            QWidget#panelRoot {
                background-color: #262b33;
                border: 1px solid #3c434d;
            }
            QFrame#resultsFrame {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid #3c434d;
                border-radius: 6px;
            }
            QLineEdit {
                background-color: #343b47;
                color: #eef1f6;
                border: 1px solid #4a5568;
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 14px;
                selection-background-color: #4a7dff;
            }
            QLineEdit:focus { border: 1px solid #4a7dff; }
            QLabel#statusLabel { color: #98a3b5; font-size: 12px; }
            QProgressBar {
                background: transparent; border: none; height: 6px;
                border-radius: 3px; text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4a7dff; border-radius: 3px;
            }
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
            QScrollBar::handle:vertical {
                background: #4a5568; border-radius: 4px; min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText("输入中文描述，如：海边 日落（空格分隔多词）")
        self.edit.setClearButtonEnabled(True)
        root.addWidget(self.edit)

        self.status = QLabel("就绪", self)
        self.status.setObjectName("statusLabel")
        root.addWidget(self.status)

        self.bar = QProgressBar(self)
        self.bar.setTextVisible(False)
        self.bar.hide()
        root.addWidget(self.bar)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.container = QWidget(self.scroll)
        self.container.setStyleSheet("background: transparent;")
        self.grid = QGridLayout(self.container)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(8)
        self.grid.setColumnStretch(COLS, 1)
        self.scroll.setWidget(self.container)
        # 结果区套一个半透明框：视觉有边界，且整块区域都参与滚轮事件
        self.results_frame = QFrame(self)
        self.results_frame.setObjectName("resultsFrame")
        frame_layout = QVBoxLayout(self.results_frame)
        frame_layout.setContentsMargins(4, 4, 4, 4)
        frame_layout.addWidget(self.scroll)
        root.addWidget(self.results_frame, 1)

        # 1s 防抖
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.query)
        self.edit.textChanged.connect(lambda _t: self._timer.start())
        self.edit.returnPressed.connect(self.query)

        # 初始为紧凑模式：不显示图片区
        self._set_results_mode(False)

    # ---------- 查询 ----------
    def _set_results_mode(self, show: bool):
        """有结果显示图片区并撑开面板；无结果收缩为仅输入框。"""
        self.results_frame.setVisible(show)
        if show:
            self.setFixedHeight(PANEL_HEIGHT)
        else:
            # 解除固定高度，收缩到内容实际高度（含进度条等）
            self.setMinimumHeight(0)
            self.setMaximumHeight((1 << 24) - 1)
            self.adjustSize()
        if self._last_ball_geom is not None and self.isVisible():
            self._reposition(self._last_ball_geom)

    def query(self):
        self._timer.stop()
        text = self.edit.text().strip()
        if not text:
            self.clear_results()
            self._set_results_mode(False)
            self.set_status("就绪")
            return
        k = int(self._conf.get("top_k", 10))
        self._gen += 1
        gen = self._gen
        # 取消在途旧查询：避免过期 GPU/CPU 工作继续占用资源
        if self._search_worker is not None:
            try:
                self._search_worker.cancel()
            except RuntimeError:
                pass
        self.set_status("搜索中...")
        w = SearchWorker(self._store, text, k,
                         self._conf["model_name"], self._conf["device"],
                         version=gen,
                         parent=self)  # 挂父对象：即使 Python 引用丢失，C++ 对象也不会在运行中被析构
        w.results_ready.connect(
            lambda res, ms, v: self._on_results(res, ms, v))
        w.failed.connect(lambda msg, v: self._on_failed(msg, v))
        # 线程结束后安全销毁 C++ 对象，避免"QThread 运行中被析构"崩溃
        w.finished.connect(w.deleteLater)
        self._search_worker = w
        self._workers.append(w)
        w.start()
        self._cleanup_workers()  # 先 start 再清理，避免误删刚创建的 worker

    def _on_results(self, results, ms, gen):
        if gen != self._gen:
            return  # 过期结果，忽略
        self.clear_results()
        n_text = 0
        n_desc = 0
        for i, (meta, score, text_score, desc) in enumerate(results):
            if text_score is not None:
                n_text += 1
            if desc is not None:
                n_desc += 1
            w = ResultThumb(meta, score, text_score, desc, self.container)
            w.open_requested.connect(os.startfile)
            w.remove_requested.connect(self.remove_requested)
            self.grid.addWidget(w, i // COLS, i % COLS)
        self._set_results_mode(bool(results))
        if results:
            extra = f" · 文字命中 {n_text} 张" if n_text else ""
            if n_desc:
                extra += f" · 描述命中 {n_desc} 张"
            self.set_status(f"找到 {len(results)} 张{extra}"
                            f" · {(ms / 1000):.2f} 秒 · 单击打开，拖到微信发送")
        else:
            self.set_status("未找到结果，试试更通用的词")

    def _on_failed(self, msg, gen):
        if gen != self._gen:
            return
        self.set_status("查询出错，详见控制台")
        print("[查询错误]", msg)

    def _cleanup_workers(self):
        """移除已结束的线程引用（C++ 对象可能已被 deleteLater 销毁）。"""
        alive = []
        for w in self._workers:
            try:
                if w.isRunning():
                    alive.append(w)
            except RuntimeError:
                continue
        self._workers = alive

    def shutdown_workers(self):
        """退出前等待所有搜索线程结束。"""
        for w in list(self._workers):
            try:
                if w.isRunning():
                    w.wait(3000)
            except RuntimeError:
                pass
        self._workers = []

    def clear_results(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.hide()           # 立即不可见，避免 deleteLater 延迟期间残留
                w.setParent(None)  # 立即脱离容器（隐藏状态下不会闪现独立窗口）
                w.deleteLater()
        # 强制容器重算尺寸，让滚动条范围立即归零
        self.container.adjustSize()
        self.container.updateGeometry()
        self.scroll.verticalScrollBar().setValue(0)

    # ---------- 状态与进度 ----------
    def set_status(self, text: str):
        self.status.setText(text)

    def show_progress(self, done: int, total: int, eta: float):
        self.bar.show()
        self.bar.setRange(0, max(total, 1))
        self.bar.setValue(done)
        self.set_status(f"处理中 {done}/{total} · 预计剩余 {fmt_eta(eta)}"
                        f" · 可继续搜索已索引部分")
        # 紧凑模式下进度条出现需要重新适配高度
        if not self.results_frame.isVisible():
            self._set_results_mode(False)

    def hide_progress_bar(self):
        self.bar.hide()
        if not self.results_frame.isVisible():
            self._set_results_mode(False)

    # ---------- 窗口行为 ----------
    def _screen_at(self, geom):
        """取悬浮球所在屏幕（多显示器下跟随球所在屏）。"""
        screen = QGuiApplication.screenAt(geom.center())
        return screen or QGuiApplication.primaryScreen()

    def _reposition(self, ball_geom, log=False):
        """面板水平居中于悬浮球：优先正下方（8px 间隙），放不下则正上方，
        仍放不下则贴屏幕边缘（避免面板不可见）。"""
        area = self._screen_at(ball_geom).availableGeometry()
        gap = 8
        x = ball_geom.center().x() - self.width() // 2
        x = max(area.left() + gap, min(x, area.right() - self.width() - gap))
        # 下方优先
        y = ball_geom.bottom() + gap
        if y + self.height() > area.bottom() - gap:
            # 空间不足：翻到正上方
            y = ball_geom.top() - self.height() - gap
        # 仍越界则贴边
        if y < area.top() + gap:
            y = area.top() + gap
        elif y + self.height() > area.bottom() - gap:
            y = area.bottom() - self.height() - gap
        self.move(x, y)
        if log:
            self._log_position(ball_geom, area)

    def _log_position(self, ball_geom, area):
        """定位调试日志：记录球/屏幕/面板三方坐标，便于定位错位问题。"""
        try:
            import time
            from core import config as cfg
            h = self.windowHandle()
            hp = (f" handle=({h.position().x()},{h.position().y()})"
                  if h else "")
            with open(cfg.DATA_DIR / "position.log", "a",
                      encoding="utf-8") as f:
                f.write(
                    f"{time.strftime('%H:%M:%S')} "
                    f"ball=({ball_geom.x()},{ball_geom.y()},"
                    f"{ball_geom.width()}x{ball_geom.height()}) "
                    f"area=({area.x()},{area.y()},"
                    f"{area.width()}x{area.height()}) "
                    f"panel={self.width()}x{self.height()} "
                    f"want=({self.x()},{self.y()}){hp}\n")
        except Exception:
            pass

    def open_near(self, ball_geom):
        self._last_ball_geom = ball_geom
        self._reposition(ball_geom, log=True)
        self.show()
        self.raise_()
        self.activateWindow()
        self.edit.setFocus()
        self.edit.selectAll()

    def toggle_near(self, ball_geom):
        if self.isVisible():
            self.hide()
            return
        self.open_near(ball_geom)

    def follow_ball(self, ball_geom):
        """悬浮球拖动时面板实时跟随（仅移动，不抢焦点）。"""
        self._last_ball_geom = ball_geom
        if self.isVisible():
            self._reposition(ball_geom)

    def wheelEvent(self, event):
        # 面板任意位置的滚轮都转发给结果滚动区（半透明框、空白区域同样有效）
        self.scroll.wheelEvent(event)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.hide()
        else:
            super().keyPressEvent(event)
