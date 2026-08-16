"""屏幕置顶悬浮球：可拖动、单击呼出查询面板、右键菜单。"""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QActionGroup, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QApplication, QMenu, QWidget

SIZE = 52
TOP_K_OPTIONS = (5, 10, 20, 50)


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
        self.move(x, y)

    def save_position(self):
        self._conf["ball_pos"] = [self.x(), self.y()]

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
