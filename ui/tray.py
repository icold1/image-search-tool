"""系统托盘图标：右键菜单管理应用（显示/隐藏、刷新索引、显示数量、退出）。

托盘图标用代码绘制（蓝底放大镜，与悬浮球一致），无需图片资源。
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QActionGroup, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from ui.floating_ball import TOP_K_OPTIONS


def make_tray_pixmap(size: int = 64) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#4A7DFF"))
    p.drawEllipse(2, 2, size - 4, size - 4)
    pen = QPen(QColor("white"), max(3, size // 14))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    r = size * 0.19
    c = size * 0.40
    p.drawEllipse(int(c - r), int(c - r), int(2 * r), int(2 * r))
    p.drawLine(int(c + r * 0.75), int(c + r * 0.75),
               int(size * 0.78), int(size * 0.78))
    p.end()
    return pm


class TrayIcon(QSystemTrayIcon):
    toggle_panel = Signal()             # 单击/双击托盘图标
    ball_visibility_changed = Signal(bool)
    refresh_requested = Signal()
    topk_changed = Signal(int)
    quit_requested = Signal()

    def __init__(self, config: dict, parent=None):
        super().__init__(QIcon(make_tray_pixmap()), parent)
        self._conf = config
        self._topk_acts = {}
        self._build_menu()
        self.setToolTip("图片语义搜索")
        self.activated.connect(self._on_activated)

    def _build_menu(self):
        menu = QMenu()
        self._act_ball = menu.addAction("显示悬浮球")
        self._act_ball.setCheckable(True)
        self._act_ball.setChecked(True)
        self._act_panel = menu.addAction("打开查询框 (Ctrl+Shift+空格)")
        menu.addSeparator()
        self._act_refresh = menu.addAction("刷新索引")
        self._act_status = menu.addAction("已索引: -")
        self._act_status.setEnabled(False)
        sub = menu.addMenu("显示数量")
        grp = QActionGroup(sub)
        grp.setExclusive(True)
        for k in TOP_K_OPTIONS:
            act = sub.addAction(str(k))
            act.setCheckable(True)
            act.setChecked(int(self._conf.get("top_k", 10)) == k)
            grp.addAction(act)
            act.triggered.connect(
                lambda checked=False, kk=k: self.topk_changed.emit(kk))
            self._topk_acts[k] = act
        menu.addSeparator()
        self._act_quit = menu.addAction("退出")
        self.setContextMenu(menu)
        self._menu = menu  # 保持引用，防止被回收

        self._act_ball.toggled.connect(self.ball_visibility_changed)
        self._act_panel.triggered.connect(self.toggle_panel)
        self._act_refresh.triggered.connect(self.refresh_requested)
        self._act_quit.triggered.connect(self.quit_requested)

    def _on_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            self.toggle_panel.emit()

    def set_status(self, count: int):
        self._act_status.setText(f"已索引: {count} 张")
        self.setToolTip(f"图片语义搜索\n已索引: {count} 张")

    def set_topk(self, k: int):
        act = self._topk_acts.get(k)
        if act:
            act.setChecked(True)

    def set_ball_checked(self, visible: bool):
        self._act_ball.setChecked(visible)

    def is_ball_visible(self) -> bool:
        """悬浮球是否应可见（用户意图的权威状态）。"""
        return self._act_ball.isChecked()
