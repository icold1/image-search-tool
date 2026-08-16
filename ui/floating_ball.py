"""屏幕置顶悬浮球：可拖动、单击呼出查询面板、右键菜单。"""
import traceback
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QActionGroup, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QApplication, QMenu, QWidget

SIZE = 52
TOP_K_OPTIONS = (5, 10, 20, 50)

_HIDE_LOG = (Path(__file__).resolve().parent.parent / "data" / "ball_hide.log")


def _log_hide(reason: str) -> None:
    """悬浮球被隐藏时的诊断日志（排查"神秘消失"问题）。"""
    try:
        import datetime
        _HIDE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_HIDE_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now().strftime('%H:%M:%S.%f')}] "
                    f"{reason}\n")
            f.write("".join(traceback.format_stack(limit=12)))
    except Exception:
        pass


class FloatingBall(QWidget):
    toggled = Signal()               # 单击
    position_changed = Signal()      # 拖动过程中位置变化
    refresh_requested = Signal()     # 菜单：刷新索引
    topk_changed = Signal(int)       # 菜单：修改显示数量
    quit_requested = Signal()        # 菜单：退出

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._conf = config
        self.setWindowFlags(Qt.FramelessWindowHint
                            | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(SIZE, SIZE)
        self._press_global = None
        self._moved = False
        self._hover = False
        self._explicit_hidden = False   # 用户通过托盘明确关闭时置 True
        self.setToolTip("图片语义搜索\n单击：打开查询框（或 Ctrl+Shift+空格）\n"
                        "右键：刷新索引 / 显示数量 / 退出")
        self._place(config)

    # ---------- 位置 ----------
    def _place(self, config):
        screen = QApplication.primaryScreen().availableGeometry()
        pos = config.get("ball_pos")
        x = (pos[0] if isinstance(pos, list) and pos[0] is not None
             else screen.right() - SIZE - 16)
        y = (pos[1] if isinstance(pos, list) and pos[1] is not None else 8)
        # 钳制到屏幕范围内：上次保存的位置可能因分辨率/多显示器变化而越界，
        # 越界会让悬浮球"消失"
        x = max(screen.left() + 4, min(x, screen.right() - SIZE - 4))
        y = max(screen.top() + 4, min(y, screen.bottom() - SIZE - 4))
        self.move(x, y)

    def save_position(self):
        self._conf["ball_pos"] = [self.x(), self.y()]

    def set_explicit_hidden(self, hidden: bool):
        """记录"用户明确隐藏"状态，与被动消失区分开。"""
        self._explicit_hidden = hidden

    def mark_shutdown(self):
        """退出流程标记：窗口销毁触发的 hideEvent 属正常，不再记日志。"""
        self._explicit_hidden = True

    def ensure_visible(self):
        """兜底恢复：非用户主动隐藏却不可见时重新显示并置顶。"""
        if not self._explicit_hidden and not self.isVisible():
            _log_hide("ensure_visible：检测到非主动隐藏，自动恢复显示")
            self.show()
            self.raise_()

    def hideEvent(self, event):
        if not self._explicit_hidden:
            _log_hide("hideEvent：悬浮球被隐藏（非用户主动，"
                      f"explicit_hidden={self._explicit_hidden}）")
        super().hideEvent(event)

    # ---------- 绘制 ----------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        base = QColor("#4A7DFF") if not self._hover else QColor("#2F5FD9")
        grad = QRadialGradient(QPointF(SIZE * 0.35, SIZE * 0.3), SIZE * 0.8)
        grad.setColorAt(0.0, QColor(base.red(), base.green(), base.blue(), 240))
        grad.setColorAt(1.0, QColor(base.red(), base.green(), base.blue(), 200))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawEllipse(QRectF(2, 2, SIZE - 4, SIZE - 4))
        pen = QPen(QColor("white"), 3.2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(15, 15, 15, 15))          # 放大镜镜片
        p.drawLine(QPointF(27.5, 27.5), QPointF(37, 37))  # 手柄

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    # ---------- 交互 ----------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_global = event.globalPosition().toPoint()
            self._moved = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_global is not None and (event.buttons() & Qt.LeftButton):
            delta = event.globalPosition().toPoint() - self._press_global
            if (not self._moved
                    and delta.manhattanLength() > QApplication.startDragDistance()):
                self._moved = True
            if self._moved:
                self.move(self.pos() + delta)
                self._press_global = event.globalPosition().toPoint()
                self.position_changed.emit()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self._moved:
            self.toggled.emit()
        self._press_global = None
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        act_refresh = menu.addAction("刷新索引")
        sub = menu.addMenu("显示数量")
        grp = QActionGroup(sub)
        grp.setExclusive(True)
        for k in TOP_K_OPTIONS:
            act = sub.addAction(str(k))
            act.setCheckable(True)
            act.setChecked(int(self._conf.get("top_k", 10)) == k)
            grp.addAction(act)
        menu.addSeparator()
        act_quit = menu.addAction("退出")
        chosen = menu.exec(event.globalPos())
        if chosen == act_refresh:
            self.refresh_requested.emit()
        elif chosen == act_quit:
            self.quit_requested.emit()
        elif chosen is not None and chosen.text().isdigit():
            self.topk_changed.emit(int(chosen.text()))
