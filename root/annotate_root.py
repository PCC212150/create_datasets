r"""根系标注工具（PyQt5）—— 模型先预测，人只改错的地方。

    python annotate_root.py                 # 打开第一张没做的
    python annotate_root.py --image <名>     # 直接开某张
    python annotate_root.py --prefetch      # 只跑预测填缓存（不开界面）
    python annotate_root.py --selftest      # 无界面自检（见文件末尾）
    python annotate_root.py --no-gpu        # 强制 CPU 预测（打包给别人用时的默认路径）

## 怎么用（两个阶段，按 Enter 来回切）

**阶段 1 · 修掩码**：红色 = 模型预测的根。不对的用蓝笔涂掉、漏的用绿笔补上。
存盘时保留的是 `(红 ∪ 绿) \ 蓝`。
**阶段 2 · 画折线**：根系的长度靠折线记，**存盘时每条折线各生成一个多边形**。
左键点、右键结束一条；`G` 让工具**照着掩码自动起草**（这才是省时间的地方），
起草完再拖控制点微调。`G` 会整条替换现有折线，所以**已经手工调过时会先问一句**
（没调过就直接起草，连按两下不会弹两次）。紫色的轮廓就是"存出来会是什么形状"，带宽用 `,` `.` 或工具栏的数
字框调。

## 键位

    滚轮 / 两指滚动   平移视角（鼠标、触摸板一样；上下左右都行）
    Shift+滚轮     笔刷大小          Ctrl+滚轮  缩放
    Ctrl+滚轮 / +-  缩放（以光标为锚点）   Ctrl+0 适应窗口   Ctrl+1 100%
    中键拖动 / 空格+左键   平移
    TAB            笔刷循环：绿(补) → 蓝(删) → 擦
    T              开关掩码层（只看原图）      Shift+T  只看最终掩码长什么样
    Ctrl+Z / Ctrl+Y   撤销 / 重做（笔刷、调阈值、折线各自记账）
    [ / ]          预测松紧：[ 更粗、] 更细（阈值 −/+，立刻重算红层，已涂的绿蓝不受影响）
    1 / 2 / Enter  切到掩码阶段 / 折线阶段 / 切下一个阶段
    左键/右键/双击  折线：加点 / 结束这条 / 也结束这条
    Alt+点击       在离点击最近的位置**插入一个控制点**（超出两端就接到最近那个端点）
    拖动控制点      直接拖走
    点控制点选中    Delete 删这一个点；X 删**整条**；B 在鼠标处**断成两条**（交叉处用）
    Ctrl+S         保存并跳下一张        PgDn / PgUp  下一张 / 上一张
    F5             重新扫描目录（在资源管理器里删/加了图片之后用）
    D / K          本图不要 stem / 不要 check_background（预测明显错时用，再按恢复）
    Ctrl+R         清空所有修改，回到纯预测（可撤销）      F1 键位帮助

## 产物

    datasets\<名>.json      labelme 格式：root 多边形（**每条折线一个**：轮廓取自掩码，
                            掩码缺的段用带宽补上）+ root 折线（根长）+ stem + check_background
    datasets\<名>.jpg       原图（硬链接过来的，让 datasets\ 直接能训练）
    datasets\overlay\<名>_overlay.jpg   原图 + 最终掩码 + 折线的验收图
    cache\<模型名>\         中间状态（掩码/概率图/折线），删了只是重新预测

**不写 root_model\datasets\** —— 那是训练数据，预测/半自动标注的结果混进去会污染划分。
"""
import argparse
import atexit
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import annotate_core as core      # noqa: E402
import annotate_io as IO          # noqa: E402

from PyQt5 import sip                                            # noqa: E402
from PyQt5.QtCore import (QPoint, QRect, Qt, QThread, QTimer,     # noqa: E402
                          pyqtSignal)
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen      # noqa: E402
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QLabel,        # noqa: E402
                             QListWidget, QListWidgetItem, QMainWindow,
                             QMessageBox, QSpinBox, QToolBar, QVBoxLayout, QWidget)

# 不开 HighDpi：开了 dpr 会变成 1.5，帧缓冲尺寸/鼠标坐标/ROI 换算三处都要乘 dpr，
# 是"鼠标点和笔迹错位"的经典 bug 源。先保证 1:1 全链路正确（字稍糊一点）。
DPR_AWARE = False

DEFAULT_RADIUS = 25.0          # 笔刷半径（原图像素）；原图里根宽约 10px
RADIUS_MIN, RADIUS_MAX = 1.0, 600.0
RADIUS_STEP = 1.15             # 滚轮一格
ZOOM_STEPS = (0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

COLOR_POLY = QColor(255, 255, 0)
COLOR_POLY_CUR = QColor(255, 200, 0)
COLOR_POLY_SEL = QColor(0, 255, 255)   # 被选中的那条（X 会删掉它，所以要一眼看得出来）
COLOR_POLY_PREVIEW = QColor(255, 0, 255)   # 多边形预览的轮廓（存出来长什么样）
COLOR_BOX = QColor(0, 255, 0)
COLOR_HIT = QColor(255, 255, 255)
HIT_RADIUS_PX = 8              # 控制点命中半径（屏幕像素）


class Doc:
    """一张图的全部状态。"""

    def __init__(self, path: Path, img: np.ndarray, model_name: str, cache_dir: Path):
        self.path = Path(path)
        self.stem = self.path.stem
        self.image_name = self.path.name
        self.img = img
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.store = None
        self.prob = None
        self.check_box = None
        self.check_ok = False
        self.root_ok = True
        self.stem_polys = []
        self.low_thresh = IO.DEFAULT_LOW_THRESH
        self.drop_stem = False
        self.drop_check = False
        self.polylines = []            # 已完成的折线 list[list[(x, y)]]
        self.cur_line = []             # 正在画的那条
        # 自上次起草以来折线有没有被**手工改过**（拖点/插点/删/断/手画）。
        # G 会整条替换折线，所以只在"真有手工成果会丢"时才拦一下问一句 ——
        # 第一次起草、或者连着起草两次时不该弹窗打扰。
        self.polylines_edited = False
        self.dirty = False
        # ---- 「每条折线一个多边形」的预览 ----
        self.poly_width = IO.ROOT_POLY_WIDTH_DEFAULT   # 掩码缺处补带的宽度（界面上可调）
        self.poly_builder = None       # 懒建；里面缓存着折线的分区，改带宽时能秒算
        self.polygons = None           # 上一次算出来的多边形（预览 + 保存共用）
        self.poly_info = None
        h, w = img.shape[:2]
        self.vp = core.Viewport(w, h)
        self.builder = None            # 有了 store 才建

    def ensure_builder(self):
        if self.builder is None and self.store is not None:
            self.builder = core.FrameBuilder(self.img, self.store, self.vp)
        return self.builder

    def final_mask(self):
        return self.store.final_mask()


class PredictWorker(QThread):
    """后台跑一张图的预测（含读图）。主线程只装数据，界面不卡。

    线程里**不碰任何 QWidget**（连 QImage 都不造），只 emit 一个 dict：
    跨线程碰 GUI 对象是 Qt 里最常见的崩溃来源。torch 前向本身是单线程串行的，
    不需要锁。
    """

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, model, meta, device, tile, path):
        super().__init__()
        self.model, self.meta, self.device, self.tile, self.path = (
            model, meta, device, tile, path)

    def run(self):
        try:
            t0 = time.time()
            img = IO.image_io.load_rgb(self.path)
            res = IO.predict_one(self.model, img, self.meta, self.device, self.tile)
            res["img"] = img
            res["seconds"] = time.time() - t0
            self.done.emit(res)
        except Exception as e:                       # 线程里抛异常会静默死掉，必须送回主线程
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class Canvas(QWidget):
    """画布：帧缓冲显示 + 全部鼠标/键盘交互。"""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.mode = core.MODE_ADD
        self.radius = DEFAULT_RADIUS
        self.phase = 0                       # 0 = 修掩码，1 = 画折线
        self.space_down = False
        self._dirty = []                     # 待重合成的屏幕矩形
        self._needs_full = True              # 需要整帧重合成（换图/缩放/调阈值）
        self._auto_fit = True                # 窗口尺寸变了要重新适应（手动缩放过就不再自动）
        self._last_pt = None                 # 上一笔刷落点（图像坐标）
        self._pan_last = None
        self._drag_sel = None                # 正在拖的控制点 (line, idx)
        self._pl_snap_done = True            # 本次拖动是否已经压过快照
        self._pl_snap = None
        self._pl_undo, self._pl_redo = [], []
        self._sel = None                     # 选中的控制点
        self._hover = None                   # 鼠标下的控制点
        self._mouse_screen = None
        self._mouse_down = False
        self.show_poly = True                # P 键：多边形预览（存出来长什么样）

    # ---------------- 便捷访问 ----------------
    @property
    def doc(self):
        return self.win.doc

    def _vp(self):
        return self.doc.vp if self.doc else None

    def mark_dirty_rect(self, image_box):
        """图像坐标的 bbox -> 攒进待重画列表（屏幕坐标，向外留 2px）。"""
        vp = self._vp()
        if vp is None or image_box is None:
            return
        y0, x0, y1, x1 = image_box
        sx0, sy0 = vp.image_to_screen(x0, y0)
        sx1, sy1 = vp.image_to_screen(x1, y1)
        self._dirty.append((int(np.floor(sx0)) - 2, int(np.floor(sy0)) - 2,
                            int(np.ceil(sx1)) + 2, int(np.ceil(sy1)) + 2))

    def refresh(self, full=False):
        """请求重画。full=True 表示帧缓冲整个作废（换图/缩放/调阈值/切层）。"""
        if full:
            self._needs_full = True
            self._dirty = []
        self.update()

    # ---------------- 绘制 ----------------
    def paintEvent(self, ev):
        doc = self.doc
        p = QPainter(self)
        if doc is None or doc.store is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignCenter, self.win.empty_text())
            p.end()
            return
        b = doc.builder
        w, h = self.width(), self.height()
        if b.resize_frame(w, h):
            self._needs_full = True                  # 帧缓冲重新分配了，内容全没了
            self._dirty = []
        if self._needs_full:
            b.compose()
            self._needs_full = False
            self._dirty = []
        elif self._dirty:
            for r in self._dirty:
                b.compose(r)
            self._dirty = []
        frame = b.frame
        # **QImage 不持有 numpy 的引用**：frame 必须是常驻成员（它就是 builder.frame），
        # 每次 paintEvent 现造 QImage 只有 0.002ms，不值得缓存。
        qi = QImage(sip.voidptr(frame.ctypes.data), w, h, frame.strides[0],
                    QImage.Format_RGB32)
        p.drawImage(0, 0, qi)
        self._draw_vectors(p)
        p.end()

    def _draw_vectors(self, p):
        """折线/控制点/检查框/笔刷光标 —— 都用 QPainter 矢量画，不光栅化进帧缓冲。"""
        doc, vp = self.doc, self.doc.vp
        p.setRenderHint(QPainter.Antialiasing, True)
        # 检查框（预测的检查范围，绿框）
        if doc.check_box is not None and doc.check_ok and not doc.drop_check:
            x0, y0 = vp.image_to_screen(doc.check_box[0], doc.check_box[1])
            x1, y1 = vp.image_to_screen(doc.check_box[2], doc.check_box[3])
            pen = QPen(COLOR_BOX, 2)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawRect(QRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
        if self.phase == 1:
            # 多边形预览：**存出来到底是什么形状**，在画的时候就能看见
            if self.show_poly and doc.polygons:
                pen = QPen(COLOR_POLY_PREVIEW, 2)
                pen.setCosmetic(True)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                for poly in doc.polygons:
                    if not poly or len(poly) < 3:
                        continue
                    pts = [QPoint(int(x), int(y)) for x, y in
                           (vp.image_to_screen(px, py) for px, py in poly)]
                    p.drawPolygon(*pts)
            for i, line in enumerate(doc.polylines):
                self._draw_line(p, line, COLOR_POLY, active=(self._sel and self._sel[0] == i))
            if doc.cur_line:
                self._draw_line(p, doc.cur_line, COLOR_POLY_CUR, active=True)
                # 最后一点到光标的橡皮筋
                if self._mouse_screen is not None:
                    lx, ly = vp.image_to_screen(*doc.cur_line[-1])
                    pen = QPen(COLOR_POLY_CUR, 1, Qt.DashLine)
                    pen.setCosmetic(True)
                    p.setPen(pen)
                    p.drawLine(QPoint(int(lx), int(ly)), self._mouse_screen)
        # 笔刷圈：**涂的时候也一直画**（2026-10-07 用户要的）—— 按下左键圈就没了的话，
        # 看不见笔刷多大、也看不见笔尖在哪，尤其掩码层关掉（T）时屏幕上完全没有反馈。
        # 只有**平移**时才藏起来：那会儿手按着却并不是在涂，圈跟着跑纯干扰
        # （中键按下、或按住空格 + 左键）。
        # mouseButtons() 是直接问 Qt 的物理状态 —— 万一某次按下事件被别的控件吃掉
        # （先点在工具栏上、再拖着移进画布），我们自己的标志是假的，它骗不了。
        held = QApplication.mouseButtons()
        panning = (self.space_down or self._pan_last is not None
                   or bool(held & Qt.MiddleButton))
        if self.phase == 0 and self._mouse_screen is not None and not panning:
            r = self.radius * vp.zoom          # 不设下限：圈画的就是"按下去会涂多大"，
            col = QColor(*core.MODE_COLORS_BGR[self.mode][::-1])   # BGR -> RGB
            # 先描一圈深色再画本色：涂下去的那块和圈**是同一个颜色**，不描边的话
            # 圈一进到自己刚涂过的区域里就"陷进去"看不见了
            p.setPen(QPen(QColor(25, 25, 25), 4))
            p.drawEllipse(self._mouse_screen, int(r), int(r))
            p.setPen(QPen(col, 2))
            p.drawEllipse(self._mouse_screen, int(r), int(r))

    def _draw_line(self, p, line, color, active=False):
        vp = self.doc.vp
        pts = [QPoint(int(x), int(y)) for x, y in
               (vp.image_to_screen(px, py) for px, py in line)]
        pen = QPen(COLOR_POLY_SEL if active else color, 4 if active else 2)
        pen.setCosmetic(True)
        p.setPen(pen)
        for a, b in zip(pts, pts[1:]):
            p.drawLine(a, b)
        for i, q in enumerate(pts):
            if self._hover == (id(line), i):
                p.setBrush(COLOR_HIT)
                rad = HIT_RADIUS_PX - 1
            elif self._sel and self._sel[0] == id(line) and self._sel[1] == i:
                p.setBrush(QColor(255, 120, 120))
                rad = HIT_RADIUS_PX - 2
            else:
                p.setBrush(color)
                rad = 4
            p.drawEllipse(q, rad, rad)

    # ---------------- 鼠标 ----------------
    def _img_pt(self, pos):
        return self.doc.vp.screen_to_image(pos.x(), pos.y())

    def mousePressEvent(self, ev):
        doc = self.doc
        if doc is None or doc.store is None:
            return
        self._mouse_screen = ev.pos()
        # **任何键按下都置位**（不只是左键）：中键平移、右键结束折线时手也是按着的，
        # 笔刷圈这时候还跟着鼠标就变成了"按下去圈不消失"（掩码层关掉、到处平移看图时
        # 最容易撞上）。这里代表"手在按着"，不是"正在涂"。
        self._mouse_down = True
        if ev.button() == Qt.MiddleButton or (self.space_down and ev.button() == Qt.LeftButton):
            self._pan_last = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if ev.button() == Qt.RightButton:
            if self.phase == 1:
                self._finish_line()
            return
        if ev.button() != Qt.LeftButton:
            return
        x, y = self._img_pt(ev.pos())
        if self.phase == 0:
            doc.store.begin_stroke()
            box = doc.store.paint_segment(x, y, x, y, self.radius, self.mode)
            self._last_pt = (x, y)
            self.mark_dirty_rect(box)          # 用**本段**的 bbox，不是整笔的
            doc.dirty = True
            self.refresh()
            return
        # ---- 阶段 1：折线 ----
        if ev.modifiers() & Qt.AltModifier:
            self._insert_control_point(x, y)
            return
        hit = self._hit_control_point(ev.pos())
        if hit is not None:
            # 快照**推迟到真的拖动了**（见 mouseMoveEvent）：只点一下选中不该占一步撤销，
            # 否则 Ctrl+Z 要先撤掉一串"什么都没改"的选中动作才能退到真正的改动
            self._drag_sel = hit
            self._pl_snap_done = False
            self._sel = hit
            self.refresh()
            return
        self._pl_snapshot()
        doc.cur_line.append((float(x), float(y)))
        self._sel = (id(doc.cur_line), len(doc.cur_line) - 1)
        self.refresh()

    def mouseMoveEvent(self, ev):
        self._mouse_screen = ev.pos()
        doc = self.doc
        if doc is None or doc.store is None:
            return
        if self._pan_last is not None:
            d = ev.pos() - self._pan_last
            self._pan_last = ev.pos()
            self._auto_fit = False
            doc.vp.pan(d.x(), d.y())
            doc.vp.clamp(self.width(), self.height())
            self.win.update_status()
            self.refresh(full=True)
            return
        if self._drag_sel is not None:
            x, y = self._img_pt(ev.pos())
            line = self._line_by_id(self._drag_sel[0])
            if line is not None:
                if not self._pl_snap_done:          # 第一下移动才压快照
                    self._pl_snapshot()
                    self._pl_snap_done = True
                line[self._drag_sel[1]] = (float(x), float(y))
                self.refresh()
            return
        if self._mouse_down and self.phase == 0 and self._last_pt is not None:
            x, y = self._img_pt(ev.pos())
            box = doc.store.paint_segment(self._last_pt[0], self._last_pt[1], x, y,
                                          self.radius, self.mode)
            self._last_pt = (x, y)
            # **只重画这一段**（paint_segment 返回的就是本段的 bbox）。
            # 千万不要清 store.stroke_bbox —— 它要攒够整笔的范围，抬笔时算差分补丁用；
            # 清了就会变成"只撤销最后一小段"。
            self.mark_dirty_rect(box)
            doc.dirty = True
            self.refresh()
            return
        if self.phase == 1:
            h = self._hit_control_point(ev.pos())
            if h != self._hover:
                self._hover = h
                self.refresh()
        self.refresh()                            # 笔刷光标要跟着走

    def mouseReleaseEvent(self, ev):
        doc = self.doc
        self._mouse_down = False
        if self._pan_last is not None:
            self._pan_last = None
            self.setCursor(Qt.ArrowCursor)
            return
        if doc is None or doc.store is None:
            return
        if self._drag_sel is not None:
            moved = self._pl_snap_done          # 只点一下没拖动 = 只是选中，不算改动
            self._drag_sel = None
            self._pl_snap_done = True
            if moved:
                self.win.set_dirty(True)
            self.win.update_status()
            self.refresh()
            return
        if self.phase == 0 and self._last_pt is not None:
            self._last_pt = None
            box = doc.store.end_stroke()          # 压撤销栈在这一步里做
            doc.builder.invalidate()
            if box is not None:
                self.mark_dirty_rect(box)
                self.win.set_dirty(True)
            self.win.update_status()
            self.refresh()

    def wheelEvent(self, ev):
        """滚轮/两指滚动的分工（2026-10-07 用户定）：

            滚轮 / 触摸板两指   ->  平移视角（上下左右）
            Shift+滚轮          ->  笔刷大小
            Ctrl+滚轮           ->  缩放

        **不再区分设备**：早先版本想靠"带不带 pixelDelta / 一格够不够 120"猜是触摸板
        还是鼠标，猜错就得加开关兜底 —— 而"滚轮就是平移"对两种设备都自然（和浏览器、
        图片查看器一致），鼠标用户想改笔刷大小按 Shift 就是了，没有猜错的可能。
        """
        doc = self.doc
        if doc is None:
            return
        pd, ad = ev.pixelDelta(), ev.angleDelta()
        if pd.isNull() and ad.isNull():
            return
        mod = ev.modifiers()
        if mod & Qt.ControlModifier:
            self._zoom(ad.y() > 0 or pd.y() > 0, ev.pos())
            return
        if mod & Qt.ShiftModifier:
            # 横向分量也要认：有些触摸板/驱动在按住 Shift 时会把竖滚**换成横滚**
            # （那是给浏览器的"Shift+滚轮=横向滚动"习惯），只读 y 的话这些人按 Shift
            # 滚轮会毫无反应。
            d = ad.y() or ad.x() or pd.y() or pd.x()
            if d:
                self.radius = float(np.clip(
                    self.radius * (RADIUS_STEP if d > 0 else 1 / RADIUS_STEP),
                    RADIUS_MIN, RADIUS_MAX))
                self.win.update_status()
                # 提示里带一句半径：阶段 2 的状态栏不显示半径，不然改完没有任何反馈
                self.win.flash(f"笔刷半径 {self.radius:.0f}px"
                               f"（原图像素；滚轮=缩放，Shift+滚轮=笔刷）")
                self.refresh()
            return
        dx = float(pd.x() if not pd.isNull() else ad.x() * 0.5)
        dy = float(pd.y() if not pd.isNull() else ad.y() * 0.5)
        # 方向跟"滚动"一致：往下滑 = 往下看（视角往下走），和浏览器/图片查看器一样；
        # 不是"内容跟着手指走"那种拖拽手感。
        doc.vp.pan(dx, dy)
        doc.vp.clamp(self.width(), self.height())
        self.win.update_status()
        self.refresh(full=True)

    def leaveEvent(self, ev):
        self._mouse_screen = None
        self.refresh()

    # ---------------- 缩放/平移 ----------------
    def _zoom(self, zoom_in, anchor=None):
        doc = self.doc
        if doc is None:
            return
        a = anchor or QPoint(self.width() // 2, self.height() // 2)
        if doc.vp.zoom_at(1.25 if zoom_in else 0.8, a.x(), a.y(),
                          self.width(), self.height()):
            self._auto_fit = False
            self.win.update_status()
            self.refresh(full=True)

    def zoom_to(self, z):
        doc = self.doc
        if doc is None:
            return
        c = QPoint(self.width() // 2, self.height() // 2)
        doc.vp.zoom_at(z / doc.vp.zoom, c.x(), c.y(), self.width(), self.height())
        self._auto_fit = False
        self.win.update_status()
        self.refresh(full=True)

    def fit(self):
        doc = self.doc
        if doc is None:
            return
        doc.vp.fit(self.width(), self.height())
        self._auto_fit = True
        self.win.update_status()
        self.refresh(full=True)

    def resizeEvent(self, ev):
        """窗口尺寸变了：还没手动缩放过就重新适应窗口（打开图片时画布尺寸往往还没定下来，
        不这么做第一眼看到的比例是错的）。"""
        super().resizeEvent(ev)
        if self.doc is not None and self.doc.store is not None:
            if self._auto_fit:
                self.fit()
            else:
                self.refresh(full=True)

    # ---------------- 键盘 ----------------
    def event(self, e):
        # **TAB 必须在这里拦**：QWidget::event() 默认把 Tab 当"焦点切换"吃掉，
        # keyPressEvent 里根本收不到。
        if e.type() == e.KeyPress and e.key() in (Qt.Key_Tab, Qt.Key_Backtab):
            if self.phase == 0:
                i = core.MODE_ORDER.index(self.mode)
                self.mode = core.MODE_ORDER[(i + 1) % len(core.MODE_ORDER)]
                self.win.update_status()
                self.refresh()
            return True
        return super().event(e)

    def keyPressEvent(self, ev):
        k, mod = ev.key(), ev.modifiers()
        if k == Qt.Key_Space:
            self.space_down = True
            return
        if k == Qt.Key_Escape:
            if self.doc and self.doc.cur_line:
                self._pl_snapshot()
                self.doc.cur_line = []
                self.refresh()
            return
        if k == Qt.Key_Backspace and self.phase == 1:
            doc = self.doc
            if doc.cur_line:
                self._pl_snapshot()
                doc.cur_line.pop()
                self.refresh()
            return
        if k == Qt.Key_Delete:
            # Delete = 删选中的**那一个点**；加 Shift 或按 X = 删**整条**折线
            if mod & Qt.ShiftModifier:
                self._delete_selected_line()
            else:
                self._delete_selected()
            return
        if k == Qt.Key_X and self.phase == 1:
            self._delete_selected_line()
            return
        if k == Qt.Key_B and self.phase == 1:
            self._split_selected_line()
            return
        if k in (Qt.Key_Return, Qt.Key_Enter):
            self.win.set_phase(1 - self.phase)
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Space:
            self.space_down = False
            self.refresh()
        super().keyReleaseEvent(ev)

    # ---------------- 折线：控制点 ----------------
    def _line_by_id(self, line_id):
        for line in self.doc.polylines + ([self.doc.cur_line] if self.doc.cur_line else []):
            if id(line) == line_id:
                return line
        return None

    def _hit_control_point(self, pos):
        doc = self.doc
        if doc is None:
            return None
        for line in doc.polylines + ([doc.cur_line] if doc.cur_line else []):
            for i, (x, y) in enumerate(line):
                sx, sy = doc.vp.image_to_screen(x, y)
                if abs(sx - pos.x()) <= HIT_RADIUS_PX and abs(sy - pos.y()) <= HIT_RADIUS_PX:
                    return (id(line), i)
        return None

    def _finish_line(self):
        doc = self.doc
        if not doc.cur_line:
            return
        self._pl_snapshot()
        if len(doc.cur_line) >= 2:
            doc.polylines.append(doc.cur_line)
        doc.cur_line = []
        self._sel = None
        self.win.set_dirty(True)
        self.refresh()

    def _insert_control_point(self, x, y):
        """Alt+点击：插到**离点击最近的那一段**里。

        一个规则管所有情况（用户 2026-10-06 选定）：落在某段中间就插进那两个控制点之间；
        超出两端时 t 被夹到 0/1，自然退化成"接到最近的那个端点"（这时插在最前/最后，
        而不是插在两点之间 —— 否则折线会从端点出去绕一圈又回来，不是"接线"是"打了个折"）。
        """
        doc = self.doc
        lines = doc.polylines + ([doc.cur_line] if doc.cur_line else [])
        best = None
        for line in lines:
            seg, t, dist, pt = core.nearest_on_polyline(line, x, y)
            if best is None or dist < best[0]:
                best = (dist, line, seg, t)
        if best is None:
            return
        _, line, seg, t = best
        self._pl_snapshot()
        if seg == 0 and t <= 1e-6:
            line.insert(0, (float(x), float(y)))
            self._sel = (id(line), 0)
        elif seg == len(line) - 2 and t >= 1 - 1e-6:
            line.append((float(x), float(y)))
            self._sel = (id(line), len(line) - 1)
        else:
            line.insert(seg + 1, (float(x), float(y)))
            self._sel = (id(line), seg + 1)
        self.win.set_dirty(True)
        self.refresh()

    def _delete_selected(self):
        doc = self.doc
        if not self._sel:
            return
        line = self._line_by_id(self._sel[0])
        if line is None:
            return
        self._pl_snapshot()
        idx = self._sel[1]
        if 0 <= idx < len(line):
            line.pop(idx)
        if len(line) < 2:
            # 一条折线至少要 2 个点，剩 0/1 个就整条撤掉
            if line is doc.cur_line:
                doc.cur_line = []
            else:
                doc.polylines = [x for x in doc.polylines if x is not line]
        self._sel = None
        self._hover = None
        self.win.set_dirty(True)
        self.refresh()

    def _split_selected_line(self):
        """在**离鼠标最近的位置**把选中的那条折线断成两条。

        为什么需要它：`G` 起草的折线是照着掩码骨架走的，而骨架在**根系交叉处**会
        顺着其中一条继续 —— 那多半已经不是原来那条根了。项目原本的标注习惯就是在
        交叉处断开重画（见 root_model/config.py 的「续接片段」那段），所以这里给一刀。

        用法：点一下折线上的控制点选中它（整条变青色）→ 把鼠标放到要断的地方 → `B`。
        切点落在段中间会**插入一个新控制点**，两条新线共用它，所以几何形状一点不变
        （不会裂开一道缝）。
        """
        doc = self.doc
        if doc is None:
            return
        line = self._line_by_id(self._sel[0]) if self._sel else None
        if line is None or len(line) < 2:
            self.win.flash("先点一下折线上的控制点选中它（整条变青色），"
                           "再把鼠标放到要断开的地方按 B")
            return
        if self._mouse_screen is None:
            x, y = line[0]
        else:
            x, y = self._img_pt(self._mouse_screen)
        seg, t, _dist, cut = core.nearest_on_polyline(line, x, y)
        n, eps = len(line), 0.02
        if seg == 0 and t <= eps:
            self.win.flash("切点落在最头上 —— 那里已经是端点了，不用切")
            return
        if seg == n - 2 and t >= 1 - eps:
            self.win.flash("切点落在最尾上 —— 那里已经是端点了，不用切")
            return
        cut = (float(cut[0]), float(cut[1]))
        if t <= eps:                       # 正好切在 p[seg] 这个控制点上
            left, right = list(line[:seg + 1]), list(line[seg:])
        elif t >= 1 - eps:                 # 正好切在 p[seg+1] 上
            left, right = list(line[:seg + 2]), list(line[seg + 1:])
        else:                              # 段中间：插一个新控制点，两条线共用
            left = list(line[:seg + 1]) + [cut]
            right = [cut] + list(line[seg + 1:])
        if len(left) < 2 or len(right) < 2:
            self.win.flash("切点太靠近端点，断出来的线不够两个点")
            return
        self._pl_snapshot()                # 校验都过了才压撤销栈（否则会多出空步）
        if line is doc.cur_line:
            doc.cur_line = []
            doc.polylines.append(left)
            doc.polylines.append(right)
        else:
            i = next((k for k, l in enumerate(doc.polylines) if l is line), None)
            if i is None:
                return
            doc.polylines[i:i + 1] = [left, right]
        self._sel = self._hover = None
        self.win.set_dirty(True)
        self.refresh()
        self.win.flash(f"断开成两条（{len(left)} 点 + {len(right)} 点）· Ctrl+Z 可撤回")

    def _delete_selected_line(self):
        """删除**整条**折线（连它的所有控制点一起）。

        专门为自动起草准备：起草出来的折线经常整条都是错的（掩码里多出来一团、
        或者那根本不是根），一个点一个点按 Delete 太慢。选中一个控制点按 `X`
        就把整条删掉，一次撤销能撤回。
        """
        doc = self.doc
        if doc is None:
            return
        line = self._line_by_id(self._sel[0]) if self._sel else None
        if line is None:
            self.win.flash("先点一下折线上的控制点选中它，再按 X 删整条"
                           "（Delete 是删单个点）")
            return
        self._pl_snapshot()
        n = len(line)
        if line is doc.cur_line:
            doc.cur_line = []
        else:
            doc.polylines = [x for x in doc.polylines if x is not line]
        self._sel = self._hover = None
        self.win.set_dirty(True)
        self.refresh()
        self.win.flash(f"删掉整条折线（{n} 个点，剩 {len(doc.polylines)} 条）· Ctrl+Z 可撤回")

    def sel_index(self):
        """选中的是第几条折线（1 起，没选中返回 0）。用于状态栏提示。"""
        if not self._sel:
            return 0
        for i, line in enumerate(self.doc.polylines if self.doc else []):
            if id(line) == self._sel[0]:
                return i + 1
        return 0

    # ---- 折线的撤销（与掩码的撤销分开记：两者是不同性质的操作）----
    def _pl_state(self):
        return ([list(l) for l in self.doc.polylines], list(self.doc.cur_line))

    def _pl_snapshot(self):
        self._pl_undo.append(self._pl_state())
        del self._pl_undo[:-100]
        self._pl_redo.clear()
        # 所有**手工**改动（加点/结束/插点/删点/删整条/断开/清空/拖动）都会走到这里，
        # 起草也会 —— 所以 draft 那边压完快照要立刻把它清回 False。
        if self.doc is not None:
            self.doc.polylines_edited = True

    def _pl_restore(self, st):
        doc = self.doc
        doc.polylines = [list(l) for l in st[0]]
        doc.cur_line = list(st[1])
        self._sel = self._hover = None

    def poly_undo(self):
        if not self._pl_undo:
            return False
        self._pl_redo.append(self._pl_state())
        self._pl_restore(self._pl_undo.pop())
        self.win.set_dirty(True)
        self.refresh()
        return True

    def poly_redo(self):
        if not self._pl_redo:
            return False
        self._pl_undo.append(self._pl_state())
        self._pl_restore(self._pl_redo.pop())
        self.win.set_dirty(True)
        self.refresh()
        return True

    def clear_polylines(self):
        doc = self.doc
        if not doc.polylines and not doc.cur_line:
            return
        self._pl_snapshot()
        doc.polylines, doc.cur_line = [], []
        self._sel = self._hover = None
        self.win.set_dirty(True)
        self.refresh()

    def draft_needs_confirm(self):
        """起草前要不要问一句 = **真有手工成果会被替换掉**。

        只有折线、但那是上一次起草的原样结果时，不问 —— 连按两下 `G` 不该弹两次窗。
        单独抽成方法是为了能自检：offscreen 环境下弹 QMessageBox 会永久阻塞，
        所以"弹不弹"这个判断必须能脱离弹窗单独验。
        """
        doc = self.doc
        return bool(doc is not None and doc.polylines_edited
                    and (doc.polylines or doc.cur_line))

    def draft_polylines(self):
        """`G`：照着**当前最终掩码**自动起草折线（骨架 -> 分链，实测 142ms）。

        这是这个工具最省时间的一步：32 张图每张 8~10 根，纯手画是大头。
        起草结果进的就是折线自己的撤销栈，误按一下 Ctrl+Z 就回去了。
        """
        doc = self.doc
        if doc is None or doc.store is None:
            return
        if self.draft_needs_confirm():
            n = len(doc.polylines)
            r = QMessageBox.question(
                self, "自动起草折线？",
                f"当前有 {n} 条折线，其中**你手工调整过**的内容会被全部替换掉"
                f"（拖过的控制点、插过/删过/断开的点、手画的线都不在了）。\n\n"
                f"起草完立刻按 Ctrl+Z 可以整体撤回。要继续吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                self.win.flash("已取消起草（折线没动）")
                return
        from common.skeleton_stats import analyze_mask_ex
        import config as vcfg
        mask = doc.final_mask() > 0
        if not mask.any():
            QMessageBox.information(self, "没有根", "当前掩码是空的，先修掩码或调预测松紧。")
            return
        t0 = time.time()
        st = analyze_mask_ex(mask, spur=vcfg.PRED_SPUR_LENGTH,
                             min_len=vcfg.MIN_ROOT_LENGTH, with_paths=True)
        paths = [list(map(tuple, p)) for p in st["paths"] if len(p) >= 2]
        self._pl_snapshot()
        doc.polylines = paths
        doc.cur_line = []
        doc.polylines_edited = False   # 刚起草完 = 还没被人动过（_pl_snapshot 会置 True）
        self._sel = self._hover = None
        self.win.set_dirty(True)
        self.win.set_phase(1)          # 起草出来的是折线，直接切到折线阶段接着改
        self.win.flash(f"自动起草 {len(paths)} 条折线（总长 {st['total']:.0f}px，"
                       f"{time.time() - t0:.2f}s）—— 拖控制点微调，Ctrl+Z 可整体撤回")
        self.refresh()


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("根系标注工具")
        self.doc = None
        self.images = []
        self.idx = -1
        self.model = self.meta = self.device = None
        self.tile = 0
        self.model_name = "selftest" if args.selftest else ""
        self.cache_dir = IO.cache_dir(self.model_name)
        self.done = set()
        self.worker = None
        self.pending_idx = None

        self.canvas = Canvas(self)
        self.listw = QListWidget()
        self.listw.setMaximumWidth(300)
        self.listw.setFocusPolicy(Qt.NoFocus)      # 键盘全给画布，别被列表抢焦点
        self.listw.itemClicked.connect(self._on_list_click)

        # 状态栏与这几个成员**必须早于工具栏建**：工具栏里的带宽旋钮一 setRange/setValue
        # 就会发 valueChanged -> set_poly_width -> flash()，而 flash 要写 status2。
        # 顺序反了就是"启动时 AttributeError"。
        self.status = QLabel()
        self.status.setFont(QFont("Consolas", 9))
        self.status2 = QLabel("")
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.status2)
        # 折线带宽是**工具级**设置（不是每张图一份）：调一次，之后每张图都按它来 ——
        # 否则每换一张图就得重调一遍。存在 settings.json 里，下次启动还记得。
        self.poly_width = float(IO.load_settings().get("poly_width",
                                                       IO.ROOT_POLY_WIDTH_DEFAULT))
        self.width_box = None                        # 带宽旋钮（_make_toolbar 里创建）
        # 多边形预览的防抖定时器：拖折线时每一步都重算是跑不动的（一次 100~300ms），
        # 停手 150ms 之后再算 —— 体感上还是"跟着动"
        self._poly_timer = QTimer(self)
        self._poly_timer.setSingleShot(True)
        self._poly_timer.setInterval(150)
        self._poly_timer.timeout.connect(self._recompute_polygons)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.setCentralWidget(central)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._make_dock())
        self._make_toolbar()

        self._load_model()
        self._scan()
        self.resize(1560, 980)

    # ---------------- 模型 / 图片列表 ----------------
    def _load_model(self):
        if self.args.selftest:
            return
        import torch
        pths, names = IO.resolve_model(self.args.model)
        use_cuda = torch.cuda.is_available() and not self.args.no_gpu
        self.device = torch.device("cuda" if use_cuda else "cpu")
        t0 = time.time()
        self.model, metas = IO.ckpt.load_models(pths, self.device)
        self.meta = metas[0]
        self.model_name = names[0]
        self.tile = IO.ckpt.infer_tile(metas, self.args.size)
        if self.args.size and self.tile == 0:
            self.meta = dict(self.meta, size=self.args.size)
        self.cache_dir = IO.cache_dir(self.model_name)
        print(f"模型 {self.model_name} | 设备 {self.device} | 输入长边 "
              f"{self.tile or self.meta.get('size')} | 加载 {time.time() - t0:.1f}s")
        self._warmup()

    def _warmup(self):
        """先跑一次极小的假前向：CUDA 的初始化/算法选择要 ~1s，
        放在第一张真图上会让用户以为卡死了。"""
        try:
            import torch
            dummy = np.zeros((64, 64, 3), np.uint8)
            with torch.no_grad():
                IO.predict_mod._forward_prob(
                    self.model, IO.image_io.to_model_input(dummy).to(self.device))
        except Exception:
            pass

    def _scan(self):
        # 两个目录都扫：pictures\ 是"还没做的"，datasets\ 是"做完的"
        # （做完的会被移到 datasets，只扫 pictures 的话它们会从列表里消失）
        self.images = IO.list_images(self.args.pictures, self.args.datasets)
        if not self.images:
            QMessageBox.critical(self, "没有图片",
                                 f"{self.args.pictures or IO.PICTURES_DIR} 里没有图片")
            sys.exit(1)
        self.done = IO.done_stems(self.args.datasets)
        self.listw.clear()
        for p in self.images:
            it = QListWidgetItem(self._item_text(p.stem))
            self.listw.addItem(it)
        self._update_dock_title()
        first = 0
        if self.args.image:
            want = Path(self.args.image).stem
            hit = [i for i, p in enumerate(self.images) if p.stem == want]
            if not hit:
                QMessageBox.critical(self, "找不到图", f"没有名为 {want} 的图片")
                sys.exit(1)
            first = hit[0]
        else:
            und = [i for i, p in enumerate(self.images) if p.stem not in self.done]
            first = und[0] if und else 0
        self.open_image(first)

    # ---------------- 打开图片 ----------------
    def _resolve_path(self, idx):
        """这张图**现在**在哪 —— 打开前必须确认一次。

        列表是启动时扫出来的，之后磁盘上的变化它不知道：图可能被我们自己移到了
        `datasets\\`（另一个窗口、或者上一次运行移的），也可能被人手工挪走/删掉。
        找不到别硬读 —— 读下去就是 FileNotFoundError 一路崩到控制台（2026-10-07 用户踩到）。
        """
        p = self.images[idx]
        if p.exists():
            return p
        alt = Path(self.args.datasets or IO.DATASETS_DIR) / p.name
        if alt.exists():
            self.images[idx] = alt          # 挪到 datasets 了，跟着走
            return alt
        return None

    def _load_or_report(self, path):
        """读图；读不了就说清楚原因，别把异常冒成 traceback。"""
        try:
            return IO.image_io.load_rgb(path)
        except Exception as e:
            QMessageBox.critical(
                self, "读不了这张图",
                f"{path}\n\n{type(e).__name__}: {e}\n\n"
                f"文件被删掉、被挪走、或者损坏都会这样。\n"
                f"按 F5 重新扫描目录，列表就会跟磁盘对上。")
            return None

    def rescan(self):
        """F5：重新扫两个目录，把列表刷新成磁盘上的真实情况。

        在资源管理器里删了/加了图片之后用它 —— 不用重启工具。
        """
        cur = self.doc.stem if self.doc is not None else None
        self.images = IO.list_images(self.args.pictures, self.args.datasets)
        self.done = IO.done_stems(self.args.datasets)
        self.listw.clear()
        for p in self.images:
            self.listw.addItem(QListWidgetItem(self._item_text(p.stem)))
        self._update_dock_title()
        if not self.images:
            self.doc = None
            self.canvas.refresh(full=True)
            self.flash("两个目录里都没有图片了")
            return
        hit = [i for i, p in enumerate(self.images) if p.stem == cur]
        self.open_image(hit[0] if hit else 0)
        self.flash(f"已重新扫描：共 {len(self.images)} 张 · 已完成 {len(self.done)} · "
                   f"还剩 {len(self.images) - len(self.done)}")

    def open_image(self, idx, force=False):
        if not force and not self._confirm_leave():
            return
        path = self._resolve_path(idx)
        if path is None:
            QMessageBox.warning(
                self, "这张图不在了",
                f"{self.images[idx]}\n\n磁盘上找不到它（被删了？）。\n"
                f"按 F5 重新扫描目录，它就会从列表里消失。")
            return
        self.idx = idx
        self.listw.setCurrentRow(idx)
        self.setWindowTitle(f"根系标注工具 — {path.name}  ({idx + 1}/{len(self.images)})")
        cached = None
        if not self.args.no_cache:
            cached = IO.load_cache(self.cache_dir, path.stem, self.model_name)
        if cached is not None:
            t0 = time.time()
            img = self._load_or_report(path)
            if img is None:                 # 读不了：已经弹过提示了，别往下走
                return
            self._install(path, img, cached, f"缓存 {time.time() - t0:.2f}s")
        else:
            self._start_predict(idx)

    def _start_predict(self, idx):
        if self.args.selftest:
            self._install_selftest(idx)
            return
        # 同时只允许一个预测在跑：两个并发前向会抢显存，而且结果互相覆盖
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(10000)
        self.pending_idx = idx
        self.canvas.doc = None
        self.canvas.refresh(full=True)
        self.status.setText(f"正在预测 {self.images[idx].name} …")
        self.worker = PredictWorker(self.model, self.meta, self.device, self.tile,
                                    self.images[idx])
        self.worker.idx = idx
        self.worker.done.connect(self._on_predicted)
        self.worker.failed.connect(self._on_predict_failed)
        self.worker.start()

    def _on_predict_failed(self, msg):
        self.status.setText("预测失败")
        QMessageBox.critical(self, "预测失败", msg)

    def _on_predicted(self, res):
        idx = getattr(self.worker, "idx", None)
        if idx is None or idx != self.pending_idx:
            return                      # 用户在预测期间换了图，这份结果作废
        self.pending_idx = None
        self._install(self.images[idx], res["img"], res,
                      f"预测 {res.get('seconds', 0):.2f}s（{self.device}）")

    def _install(self, path, img, pred, note=""):
        """把（缓存或预测）结果装成当前的 Doc 并显示。"""
        doc = Doc(path, img, self.model_name, self.cache_dir)
        doc.store = core.MaskStore(pred["pred"],
                                   pred.get("add"), pred.get("dele"))
        doc.prob = pred["prob"]
        meta = pred.get("meta") or {}
        doc.check_box = pred.get("check_box", meta.get("check_box"))
        doc.check_ok = pred.get("check_ok", meta.get("check_ok", False))
        doc.root_ok = pred.get("root_ok", True)
        doc.stem_polys = pred.get("stem_polys", meta.get("stem_polys", [])) or []
        doc.low_thresh = pred.get("low_thresh", meta.get("low_thresh", IO.DEFAULT_LOW_THRESH))
        doc.drop_stem = bool(meta.get("drop_stem", False))
        doc.drop_check = bool(meta.get("drop_check", False))
        doc.polylines = [list(map(tuple, l)) for l in meta.get("polylines", [])]
        # 带宽用**工具级**那个值，不读缓存里那张图自己的 —— 用户要的是"调一次、
        # 之后每张图都按它来"。缓存里那份只是当时存盘时的记录。
        doc.poly_width = self.poly_width
        if self.width_box is not None:
            self.width_box.blockSignals(True)
            self.width_box.setValue(int(round(self.poly_width)))
            self.width_box.blockSignals(False)
        doc.ensure_builder()
        doc.vp.fit(self.canvas.width() or 1200, self.canvas.height() or 800)
        doc.dirty = False
        self.doc = doc
        self.canvas._dirty = []
        self.canvas._needs_full = True
        self.canvas._auto_fit = True
        self.canvas._pl_undo, self.canvas._pl_redo = [], []
        self.canvas.phase = 0
        self.canvas._sel = self.canvas._hover = None
        self.canvas.setFocus()
        self.canvas.refresh(full=True)
        self._recompute_polygons()          # 一打开就能看到"会存成什么样"
        self.update_status()
        if note:
            self.flash(f"{path.name}  {note}")

    def _install_selftest(self, idx):
        """--selftest 用：不加载模型，造一张确定性的假预测。"""
        path = self.images[idx]
        img = self._load_or_report(path)
        if img is None:
            return
        h, w = img.shape[:2]
        pred = np.zeros((h, w), np.uint8)
        for i in range(4):                       # 4 条"根"
            y = int(h * 0.35 + i * h * 0.1)
            pred[y:y + 24, int(w * 0.2):int(w * 0.75)] = 255
        # prob=None：自检没有真概率图，调松紧的按键会明确说"这张图没有缓存的概率图"，
        # 而不是拿一张假的 64x64 概率图去插值成 5472x3648 把红层清空
        self._install(path, img, {"pred": pred, "prob": None,
                                  "check_box": (100, 100, w - 100, h - 100),
                                  "check_ok": True, "stem_polys": [],
                                  "low_thresh": IO.DEFAULT_LOW_THRESH}, "自检假预测")

    # ---------------- 保存 ----------------
    def save(self, go_next=True):
        doc = self.doc
        if doc is None:
            return False
        # 新的多边形口径是"**每条折线一个**"，所以一张有根却没折线的图存出来会是
        # **零个 root 多边形** = 被当成"这张图没有根"的负样本。这种事不能悄无声息地发生。
        if not doc.polylines and doc.final_mask().any():
            r = QMessageBox.warning(
                self, "还没有画折线",
                "root 多边形是**每条折线一个**：现在一条折线都没有，存出来这张图的 root "
                "标注会是空的（训练时会当成「这张图没有根」的负样本）。\n\n"
                "要回去画折线吗？（按 G 可以照着当前掩码自动起草）",
                QMessageBox.Cancel | QMessageBox.Save, QMessageBox.Cancel)
            if r != QMessageBox.Save:
                return False
        t0 = time.time()
        try:
            out = IO.save_annotation(
                doc.stem, doc.image_name, doc.img.shape, doc.final_mask(),
                doc.polylines, doc.stem_polys, doc.check_box,
                drop_stem=doc.drop_stem, drop_check=doc.drop_check,
                src_image=doc.path, datasets_dir=self.args.datasets,
                poly_width=doc.poly_width, builder=doc.poly_builder,
                move_pictures=not self.args.keep_pictures)
            meta = IO.meta_for_save(self.model_name, doc.img.shape, doc.stem,
                                    doc.image_name, doc.low_thresh, doc.check_box,
                                    doc.check_ok, doc.stem_polys, doc.drop_stem,
                                    doc.drop_check, doc.polylines, prob=doc.prob,
                                    poly_width=doc.poly_width)
            IO.save_cache(doc.cache_dir, doc.stem, doc.store, meta)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"{e}\n{traceback.format_exc()}")
            return False
        doc.dirty = False
        # 原图可能刚被从 pictures\ 移到 datasets\ 了，手里的路径要跟着换 ——
        # 不换的话下一次保存会拿一个已经不存在的源路径去链接，直接报错。
        new_path = out.get("image_path")
        if new_path is not None and Path(new_path) != doc.path:
            doc.path = Path(new_path)
            if 0 <= self.idx < len(self.images):
                self.images[self.idx] = doc.path
        self.done.add(doc.stem)
        self._remark(doc.stem)
        pi = out.get("poly_info") or {}
        extra = ""
        if pi.get("n_empty"):
            extra += f" | **{pi['n_empty']} 条折线没得到多边形**"
        if pi.get("n_dropped_blocks"):
            extra += (f" | 丢掉 {pi['n_dropped_blocks']} 块没有折线经过的掩码"
                      f"（{pi['dropped_area']} px）")
        self.flash(f"已保存 {doc.stem}：root 多边形 {out['n_root_poly']}（每条折线一个）"
                   f" + 折线 {out['n_polyline']} | {' + '.join(out['saved'])} | "
                   f"{time.time() - t0:.1f}s{extra}")
        if go_next:
            nxt = self.idx + 1
            if nxt < len(self.images):
                self.open_image(nxt)
            else:
                self.flash("已经是最后一张了")
        return True

    # ---------------- 折线 -> 多边形（预览与保存共用同一份） ----------------
    def schedule_polygons(self):
        """请求重算多边形（防抖）。任何改了掩码或折线的地方都会调到这里。"""
        if self.doc is not None and self.doc.store is not None:
            self._poly_timer.start()

    def _recompute_polygons(self, immediate=False):
        doc = self.doc
        if doc is None or doc.store is None:
            return
        if doc.poly_builder is None:
            doc.poly_builder = IO.RootPolygonBuilder(doc.img.shape[:2])
        lines = doc.polylines + ([doc.cur_line] if len(doc.cur_line) >= 2 else [])
        if not lines:
            doc.polygons, doc.poly_info = [], {"n_lines": 0, "n_empty": 0,
                                               "n_dropped_blocks": 0, "dropped_area": 0,
                                               "width": doc.poly_width}
        else:
            t0 = time.time()
            doc.polygons, doc.poly_info = doc.poly_builder.polygons(
                doc.final_mask(), lines, doc.poly_width)
            doc.poly_info["ms"] = (time.time() - t0) * 1000
        self.update_status()
        self.canvas.update()          # 多边形是矢量画的，重画就行，不用重合成帧缓冲

    def set_poly_width(self, value):
        """改折线带宽。**工具级**：当前这张图立刻重算，后面每张图也都按它来，
        并且写进 settings.json —— 下次启动还是这个值。"""
        w = float(np.clip(value, IO.ROOT_POLY_WIDTH_MIN, IO.ROOT_POLY_WIDTH_MAX))
        if abs(w - self.poly_width) < 1e-9:
            return
        self.poly_width = w
        IO.save_settings({"poly_width": w})
        if self.width_box is not None:
            self.width_box.blockSignals(True)
            self.width_box.setValue(int(round(w)))
            self.width_box.blockSignals(False)
        if self.doc is not None:
            self.doc.poly_width = w
            self._recompute_polygons()
        self.flash(f"折线带宽 {w:.0f}px（已记住，后面每张图都用它）"
                   f"｜只影响掩码缺的地方，掩码有根的地方用掩码自己的宽度")

    def set_dirty(self, v=True):
        """标记"有未保存改动"。**顺便触发多边形重算** —— 掩码或折线一变，
        "每条折线一个多边形"的结果就变了。收敛在这一个钩子上，是因为改动的入口太多
        （笔刷/撤销/折线增删改/起草…），散着写一定会漏一处，而漏了的表现是
        "预览显示的形状和存出来的不一样"，最难查。
        """
        if self.doc is not None:
            self.doc.dirty = bool(v)
            if v and self.doc.store is not None:
                self.schedule_polygons()

    def _confirm_leave(self):
        """切图/关窗前拦一道 —— 改了半天被一次误点吃掉是最让人恼火的事。"""
        if self.doc is None or not self.doc.dirty:
            return True
        r = QMessageBox.question(
            self, "还没保存",
            f"{self.doc.stem} 有未保存的修改。要先保存吗？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self.save(go_next=False)
        return True

    # ---------------- 动作 ----------------
    def set_phase(self, ph):
        self.canvas.phase = int(ph)
        self.canvas._sel = self.canvas._hover = None
        self.update_status()
        self.canvas.refresh()
        self.flash("阶段 2 · 画折线：左键加点、右键结束，G 自动起草，Alt+点击插点、拖控制点微调"
                   if ph == 1 else
                   "阶段 1 · 修掩码：蓝笔删错的、绿笔补漏的，TAB 换笔，滚轮调大小")

    def undo(self):
        if self.canvas.phase == 1 and self.canvas.poly_undo():
            self.flash("撤销折线")
            return
        box = self.doc.store.undo_step() if self.doc else None
        if box is None:
            self.flash("没有可撤销的操作")
            return
        self.doc.builder.invalidate()
        self.canvas.mark_dirty_rect(box)
        self.set_dirty(True)
        self.update_status()
        self.canvas.refresh()

    def redo(self):
        if self.canvas.phase == 1 and self.canvas.poly_redo():
            self.flash("重做折线")
            return
        box = self.doc.store.redo_step() if self.doc else None
        if box is None:
            return
        self.doc.builder.invalidate()
        self.canvas.mark_dirty_rect(box)
        self.set_dirty(True)
        self.update_status()
        self.canvas.refresh()

    def adjust_threshold(self, delta):
        """预测松紧：用缓存的概率图重算红层（几十~两百毫秒），已涂的绿蓝不受影响。"""
        doc = self.doc
        if doc is None or doc.prob is None or getattr(doc.prob, "size", 0) == 0:
            self.flash("这张图没有缓存的概率图，调不了松紧（重新预测一次这张图就有了）")
            return
        new = float(np.clip(doc.low_thresh + delta, IO.THRESH_MIN, IO.THRESH_MAX))
        if abs(new - doc.low_thresh) < 1e-9:
            return
        t0 = time.time()
        mask, warn = IO.mask_from_prob(doc.prob, doc.check_box, doc.check_ok,
                                       doc.img.shape[:2], new)
        old = doc.low_thresh
        doc.low_thresh = new
        doc.store.replace_pred(mask)
        doc.builder.invalidate()
        self.set_dirty(True)
        self.canvas.refresh(full=True)
        self.update_status()
        n = int((mask > 0).sum())
        # 把"往哪个方向"和"粗了多少"直接说出来：这个旋钮的方向靠记忆很容易记反
        # （阈值越低掩码越粗），屏幕上只有红层厚薄的变化，不给数字很难判断。
        self.flash((warn + " | " if warn else "")
                   + f"预测{'更粗' if new < old else '更细'}：阈值 {old:.2f} → {new:.2f}"
                   + f"，根掩码 {n} 像素（越低越粗，0.50 = 只要强响应）· "
                   + f"{time.time() - t0:.2f}s")

    def reset_edits(self):
        doc = self.doc
        if doc is None:
            return
        if QMessageBox.question(self, "重置", "清空所有绿/蓝修改，回到纯预测？（可撤销）") \
                != QMessageBox.Yes:
            return
        doc.store.begin_stroke()
        doc.store.add[:] = 0
        doc.store.dele[:] = 0
        doc.store.stroke_bbox = (0, 0, *doc.img.shape[:2])
        doc.store.end_stroke()
        doc.builder.invalidate()
        self.set_dirty(True)
        self.canvas.refresh(full=True)
        self.update_status()

    def toggle_drop(self, which):
        doc = self.doc
        if doc is None:
            return
        if which == "stem":
            doc.drop_stem = not doc.drop_stem
        else:
            doc.drop_check = not doc.drop_check
        doc.dirty = True
        self.update_status()
        self.flash(f"本图 {'不写' if (doc.drop_stem if which == 'stem' else doc.drop_check) else '写回'}"
                   f" {which} 标注（训练时该通道的损失会自动被屏蔽）")

    def toggle_layers(self, final_only=False):
        doc = self.doc
        if doc is None:
            return
        b = doc.builder
        if final_only:
            b.final_only = not b.final_only
            b.show_layers = True
            self.flash("看最终掩码 (红∪绿)\\蓝 的实际结果" if b.final_only else "回到编辑视图")
        else:
            b.show_layers = not b.show_layers
            self.flash("显示原图 + 掩码" if b.show_layers else "只看原图（T 恢复）")
        b.invalidate()
        self.canvas.refresh(full=True)

    def goto(self, step):
        n = self.idx + step
        if 0 <= n < len(self.images):
            self.open_image(n)

    # ---------------- 界面零件 ----------------
    def _make_dock(self):
        from PyQt5.QtWidgets import QDockWidget
        # 存成成员：保存后要改它的标题（已完成 x / n · 还剩 y）
        self.dock = QDockWidget("图片", self)
        self.dock.setWidget(self.listw)
        self.dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        return self.dock

    def _make_toolbar(self):
        tb = QToolBar()
        tb.setFocusPolicy(Qt.NoFocus)
        self.addToolBar(tb)
        for text, tip, fn in (
                ("保存并下一张", "Ctrl+S", lambda: self.save(True)),
                ("◀ 上一张", "PgUp", lambda: self.goto(-1)),
                ("下一张 ▶", "PgDn", lambda: self.goto(1)),
                ("重新扫描目录", "F5，在资源管理器里删/加了图片之后用",
                 self.rescan),
                ("撤销", "Ctrl+Z", self.undo),
                ("重做", "Ctrl+Y", self.redo),
                ("适应窗口", "Ctrl+0", lambda: self.canvas.fit()),
                ("100%", "Ctrl+1", lambda: self.canvas.zoom_to(1.0)),
                ("修掩码", "1", lambda: self.set_phase(0)),
                ("画折线", "2", lambda: self.set_phase(1)),
                ("自动起草折线", "G，照着当前掩码生成",
                 lambda: self.canvas.draft_polylines()),
                ("删除选中折线", "X，先点一下折线的控制点选中它",
                 lambda: self.canvas._delete_selected_line()),
                ("断开折线", "B，鼠标放在要断的地方（选中的那条会变青色）",
                 lambda: self.canvas._split_selected_line()),
                ("清空折线", "折线全部删掉（可撤销）",
                 lambda: self.canvas.clear_polylines()),
                # 方向：**阈值越低、掩码越粗**（低阈值把弱响应也拉进来，见 mask_from_prob）。
                # 这两个按钮的文案 2026-10-07 之前是反的 —— 实测 0.02→54228px、0.50→27022px，
                # "松"是阈值变小，不是变大。
                ("预测松一点（更粗）", "[ 阈值 −0.02", lambda: self.adjust_threshold(-0.02)),
                ("预测紧一点（更细）", "] 阈值 +0.02", lambda: self.adjust_threshold(+0.02)),
                ("丢弃stem", "D", lambda: self.toggle_drop("stem")),
                ("丢弃check", "K", lambda: self.toggle_drop("check")),
                ("重置修改", "Ctrl+R", self.reset_edits),):
            a = QAction(text, self)
            a.setToolTip(f"{text}（{tip}）" if tip else text)
            a.triggered.connect(fn)
            tb.addAction(a)
            a.setShortcut("")               # 快捷键统一在 keyPressEvent 里处理，免得两套
        tb.addSeparator()
        tb.addWidget(QLabel(" 折线带宽 "))
        self.width_box = QSpinBox()
        self.width_box.setRange(int(IO.ROOT_POLY_WIDTH_MIN), int(IO.ROOT_POLY_WIDTH_MAX))
        self.width_box.setSingleStep(int(IO.ROOT_POLY_WIDTH_STEP))
        self.width_box.blockSignals(True)      # 初始化别触发 valueChanged（会去写 settings）
        self.width_box.setValue(int(round(self.poly_width)))
        self.width_box.blockSignals(False)
        self.width_box.setSuffix(" px")
        self.width_box.setToolTip(
            "折线补成多边形时用的宽度，只作用在**掩码缺了**的地方\n"
            "（掩码有根的地方用掩码自己的宽度）。改完立刻重算，画布上紫色轮廓就是结果。\n"
            "快捷键：, 减 1   . 加 1")
        self.width_box.valueChanged.connect(self.set_poly_width)
        self.width_box.setFocusPolicy(Qt.NoFocus)
        tb.addWidget(self.width_box)
        tb.addSeparator()
        a = QAction("键位帮助", self)
        a.setToolTip("F1")
        a.triggered.connect(self.help_box)
        tb.addAction(a)

    def empty_text(self):
        if self.pending_idx is not None:
            return f"正在预测 {self.images[self.pending_idx].name} …"
        return "没有打开图片"

    def _on_list_click(self, item):
        self.open_image(self.listw.row(item))

    def _item_text(self, stem):
        """列表里一行长什么样：做完的实心点，没做的空心点。"""
        return f" {'●' if stem in self.done else '○'}  {stem}"

    def _update_dock_title(self):
        n, d = len(self.images), len(self.done)
        self.dock.setWindowTitle(f"图片　已完成 {d} / {n}　还剩 {n - d}")

    def _remark(self, stem):
        for i, p in enumerate(self.images):
            if p.stem == stem:
                self.listw.item(i).setText(self._item_text(stem))
                self._update_dock_title()
                return

    def flash(self, msg):
        self.status2.setText(msg)

    def update_status(self):
        doc = self.doc
        if doc is None:
            return
        c = self.canvas
        parts = [
            f"{'修掩码' if c.phase == 0 else '画折线'}",
            core.MODE_NAMES[c.mode] if c.phase == 0 else "折线",
            f"半径 {c.radius:.0f}px" if c.phase == 0 else "",
            f"缩放 {doc.vp.zoom * 100:.0f}%",
            f"阈值 {doc.low_thresh:.2f}",
            f"折线 {len(doc.polylines)} 条" + (f"(在画 {len(doc.cur_line)} 点)"
                                              if doc.cur_line else ""),
            f"选中 #{c.sel_index()}（X 删整条）" if c.sel_index() else "",
            (f"多边形 {sum(1 for q in doc.polygons if q)} 个 / 宽 {doc.poly_width:.0f}px"
             + (f"（丢 {doc.poly_info['n_dropped_blocks']} 块）"
                if doc.poly_info and doc.poly_info.get("n_dropped_blocks") else "")
             ) if doc.polygons is not None else "",
            "不要stem" if doc.drop_stem else "",
            "不要check" if doc.drop_check else "",
            "未保存*" if doc.dirty else "",
        ]
        self.status.setText(" | ".join(x for x in parts if x))

    def keyPressEvent(self, ev):
        k, mod = ev.key(), ev.modifiers()
        ctrl = bool(mod & Qt.ControlModifier)
        shift = bool(mod & Qt.ShiftModifier)
        doc = self.doc
        if k == Qt.Key_S and ctrl:
            self.save(True)
        elif k == Qt.Key_Z and ctrl and shift:
            self.redo()
        elif k == Qt.Key_Z and ctrl:
            self.undo()
        elif k == Qt.Key_Y and ctrl:
            self.redo()
        elif k == Qt.Key_R and ctrl:
            self.reset_edits()
        elif k == Qt.Key_0 and ctrl:
            self.canvas.fit()
        elif k == Qt.Key_1 and ctrl:
            self.canvas.zoom_to(1.0)
        elif k == Qt.Key_F5:
            self.rescan()
        elif k == Qt.Key_PageDown:
            self.goto(1)
        elif k == Qt.Key_PageUp:
            self.goto(-1)
        elif k == Qt.Key_T and shift:
            self.toggle_layers(final_only=True)
        elif k == Qt.Key_T:
            self.toggle_layers()
        elif k == Qt.Key_BracketLeft:
            self.adjust_threshold(-IO.THRESH_STEP)
        elif k == Qt.Key_BracketRight:
            self.adjust_threshold(+IO.THRESH_STEP)
        elif k == Qt.Key_1:
            self.set_phase(0)
        elif k == Qt.Key_2:
            self.set_phase(1)
        elif k == Qt.Key_G:
            self.canvas.draft_polylines()
        elif k == Qt.Key_Comma:
            self.set_poly_width((doc.poly_width if doc else 0) - IO.ROOT_POLY_WIDTH_STEP)
        elif k == Qt.Key_Period:
            self.set_poly_width((doc.poly_width if doc else 0) + IO.ROOT_POLY_WIDTH_STEP)
        elif k == Qt.Key_P:
            self.canvas.show_poly = not self.canvas.show_poly
            self.canvas.update()
            self.flash("多边形预览 开" if self.canvas.show_poly else "多边形预览 关（P 恢复）")
        elif k == Qt.Key_D:
            self.toggle_drop("stem")
        elif k == Qt.Key_K:
            self.toggle_drop("check")
        elif k == Qt.Key_F1:
            self.help_box()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.canvas._zoom(True, None)
        elif k == Qt.Key_Minus:
            self.canvas._zoom(False, None)
        elif doc is not None:
            super().keyPressEvent(ev)

    def help_box(self):
        QMessageBox.information(self, "键位", KEYS_HELP)

    def closeEvent(self, ev):
        if not self._confirm_leave():
            ev.ignore()
            return
        # 线程还在跑就关窗 = "QThread: Destroyed while thread is still running" 崩溃
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(8000)
        ev.accept()


KEYS_HELP = """阶段 1 · 修掩码（红=模型预测的根）
  左键拖动          涂当前笔刷
  TAB               笔刷循环：绿(补漏) → 蓝(删错) → 擦
  滚轮 / 两指滚动   平移视角（鼠标、触摸板一样，上下左右都行）
  Shift+滚轮        笔刷大小（原图像素，状态栏有显示）
  T / Shift+T       关掉掩码层只看原图 / 只看最终掩码的样子
  [ ]               预测松紧：**[ 更粗、] 更细**（阈值越低掩码越粗；立刻重算红层）
  Ctrl+R            清空所有绿蓝修改，回到纯预测（可撤销）

阶段 2 · 画折线（根长靠它记；保存时**每条折线各生成一个多边形**）
  左键              加一个点（第一点自动起一条新折线）
  右键 / 双击       结束当前折线
  Backspace         退掉上一个点        Esc  丢掉正在画的这条
  G                 照着当前掩码自动起草折线（会替换现有折线）
                    已经手工调过折线时会先问一句再动；没调过就直接起草，不打扰
  Alt+点击          在离点击最近的位置插入一个控制点
  拖动控制点        直接拖走
  点控制点选中      Delete 删这一个点    X 或 Shift+Delete 删**整条**（选中的那条会变青色）
  B                 把选中的那条**断成两条**：切点在「离鼠标最近的位置」
                    （交叉处切一刀最常用；切点落在段中间会自动插一个新控制点）
  , / .             折线带宽 -1 / +1 px（工具栏上也有数字框）
  P                 开关多边形预览（紫色轮廓 = 存出来会是什么形状）

通用
  Ctrl+Z / Ctrl+Y   撤销 / 重做         Ctrl+S  保存并跳下一张
  PgDn / PgUp       下一张 / 上一张      F5  重新扫描目录（外部删/加了图片后刷新列表）
  中键拖动 / 空格+左键拖动   平移      Ctrl+滚轮 / + -  缩放
  Ctrl+0 适应窗口   Ctrl+1 100%        F1  这个帮助
  D / K             本图不要 stem / 不要 check_background（再按恢复）"""


# ------------------------------------------------------------------ 无界面模式
def run_prefetch(args):
    """只跑预测填缓存，不开界面。32 张约 1 分钟（GPU）—— 开界面之前先跑一遍，
    之后每张图都是秒开。"""
    import torch
    pths, names = IO.resolve_model(args.model)
    use_cuda = torch.cuda.is_available() and not args.no_gpu
    device = torch.device("cuda" if use_cuda else "cpu")
    model, metas = IO.ckpt.load_models(pths, device)
    meta = metas[0]
    tile = IO.ckpt.infer_tile(metas, args.size)
    cdir = IO.cache_dir(names[0])
    imgs = IO.list_images(args.pictures)
    print(f"模型 {names[0]} | 设备 {device} | 图片 {len(imgs)} 张 | 缓存 {cdir}")
    t0 = time.time()
    for i, p in enumerate(imgs, 1):
        if IO.load_cache(cdir, p.stem, names[0]) is not None:
            print(f"[{i}/{len(imgs)}] {p.stem}: 已有缓存，跳过")
            continue
        t = time.time()
        img = IO.image_io.load_rgb(p)
        res = IO.predict_one(model, img, meta, device, tile)
        store = core.MaskStore(res["pred"])
        IO.save_cache(cdir, p.stem, store,
                      IO.meta_for_save(names[0], img.shape, p.stem, p.name,
                                       res["low_thresh"], res["check_box"],
                                       res["check_ok"], res["stem_polys"],
                                       False, False, [], prob=res["prob"]))
        print(f"[{i}/{len(imgs)}] {p.stem}: 根 {int((res['pred'] > 0).sum())} px, "
              f"茎多边形 {len(res['stem_polys'])}, check {'OK' if res['check_ok'] else '失败'}"
              f" | {time.time() - t:.1f}s", flush=True)
    print(f"全部完成，用时 {time.time() - t0:.1f}s")


def run_selftest(args):
    """无界面自检：跑一遍真实的「预测->涂改->起草->保存->回读」并断言结果。

    直接复用 `annotate_core` 和 `annotate_io`（它们不 import Qt），
    所以这个自检既能在没显示器的地方跑，也确实盖住了出问题时最要命的几段。

    **产物一律写到临时目录**（见 _selftest_outdir）：自检会导出标注、最后还要删掉它们，
    而它用的就是 pictures 里的真实图片 —— 要是写进 datasets，跑一次自检就会把
    **你已经标好的那张**（只要它正好是第一张）覆盖再删掉。
    """
    import json
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'OK ' if cond else '失败'}] {name} {extra}")
        ok = ok and bool(cond)

    print("== 1. 预测（用真模型跑一张）==")
    import torch
    pths, names = IO.resolve_model(args.model)
    use_cuda = torch.cuda.is_available() and not args.no_gpu
    device = torch.device("cuda" if use_cuda else "cpu")
    model, metas = IO.ckpt.load_models(pths, device)
    meta = metas[0]
    tile = IO.ckpt.infer_tile(metas, args.size)
    imgs = IO.list_images(args.pictures)
    path = imgs[0]
    t0 = time.time()
    img = IO.image_io.load_rgb(path)
    res = IO.predict_one(model, img, meta, device, tile)
    print(f"  {path.name}: 根 {int((res['pred'] > 0).sum())} px, "
          f"{time.time() - t0:.2f}s ({device})")
    check("预测出前景", int((res["pred"] > 0).sum()) > 0)
    check("预测的茎/检查框", len(res["stem_polys"]) > 0)

    print("== 2. 阈值重算与默认口径一致 ==")
    m2, warn = IO.mask_from_prob(res["prob"], res["check_box"], res["check_ok"],
                                 img.shape[:2], IO.DEFAULT_LOW_THRESH)
    check("按默认阈值重算 == 原生掩码", bool((m2 == res["pred"]).all()))

    print("== 2b. 预测松紧的方向：阈值越低掩码越粗，且全程单调 ==")
    # 这条钉的是"松/紧"的语义。2026-10-07 发现工具栏两个按钮文案是反的，
    # 而且阈值 0 是"关掉滞回"的特例、会让掩码从最粗直接跳到最细。
    areas = {}
    for t in (IO.THRESH_MIN, 0.10, 0.30, 0.50):
        m, _ = IO.mask_from_prob(res["prob"], res["check_box"], res["check_ok"],
                                 img.shape[:2], t)
        areas[t] = int((m > 0).sum())
    seq = [areas[t] for t in (IO.THRESH_MIN, 0.10, 0.30, 0.50)]
    check("阈值越低掩码越粗（全程单调）",
          all(a > b for a, b in zip(seq, seq[1:])),
          f"阈值 {IO.THRESH_MIN}/{0.10}/{0.30}/{0.50} -> 面积 {seq}")
    check("界面下限不是 0（0 会关掉滞回、突然变最细）", IO.THRESH_MIN > 0,
          f"THRESH_MIN = {IO.THRESH_MIN}")

    print("== 3. 掩码编辑：涂/擦/撤销/重做 ==")
    store = core.MaskStore(res["pred"])
    h, w = img.shape[:2]
    store.begin_stroke()
    store.paint_segment(w * 0.3, h * 0.3, w * 0.5, h * 0.3, 60, core.MODE_ADD)
    store.end_stroke()
    added = int((store.add > 0).sum())
    check("涂绿产生了绿掩码", added > 0, f"{added} px")
    store.begin_stroke()
    store.paint_segment(w * 0.3, h * 0.3, w * 0.4, h * 0.3, 40, core.MODE_DEL)
    store.end_stroke()
    check("涂蓝清掉了重叠的绿", int((store.add > 0).sum()) < added)
    store.begin_stroke()
    store.paint_segment(w * 0.3, h * 0.3, w * 0.5, h * 0.3, 80, core.MODE_ERASE)
    store.end_stroke()
    check("橡皮擦把绿蓝都清掉", not store.add.any() and not store.dele.any())
    store.undo_step()
    check("撤销把绿蓝还原", store.add.any() and store.dele.any())
    store.redo_step()
    check("重做又清干净", not store.add.any() and not store.dele.any())
    final = store.final_mask()
    check("最终掩码 = 预测（绿蓝都空）", bool((final == res["pred"]).all()))

    print("== 4. 折线：起草 / 拖点 / Alt 插点 ==")
    from common.skeleton_stats import analyze_mask_ex
    import config as vcfg
    st = analyze_mask_ex(final > 0, spur=vcfg.PRED_SPUR_LENGTH,
                         min_len=vcfg.MIN_ROOT_LENGTH, with_paths=True)
    lines = [list(map(tuple, p)) for p in st["paths"] if len(p) >= 2]
    check("骨架起草出折线", len(lines) > 0, f"{len(lines)} 条, 总长 {st['total']:.0f}px")
    line = lines[0]
    seg, t, dist, pt = core.nearest_on_polyline(line, pt[0] if False else line[0][0] + 50,
                                                line[0][1] + 50)
    check("最近点计算给出 0~1 的 t", 0.0 <= t <= 1.0, f"t={t:.3f} d={dist:.1f}")
    before = len(line)
    line.insert(seg + 1, (float(pt[0]), float(pt[1])))
    check("插入控制点后点数 +1", len(line) == before + 1)

    print("== 5. 导出 + 回读 ==")
    out_dir = _selftest_outdir()
    args.datasets = str(out_dir)          # 界面那一段（第 6/7 步）也走这个临时目录
    # **自检绝不能把原图从 pictures 移走**：2026-10-07 踩过 —— "保存后移走原图"这个新
    # 逻辑在自检里也生效了，于是自检把用户 pictures\ 里那张真图删了（数据没丢，
    # datasets\ 里那份是同一个 inode；但那是运气，不是设计）。
    args.keep_pictures = True
    out = IO.save_annotation(path.stem, path.name, img.shape, final, lines,
                             res["stem_polys"], res["check_box"], src_image=path,
                             datasets_dir=out_dir, move_pictures=False)
    print("  写出:", out["saved"])
    check("json 写出来了", out["json"].exists())
    check("overlay 在子目录里", (out_dir / IO.OVERLAY_SUBDIR).is_dir())
    # 像素级掩码：训练直接吃的那份（root_model 优先读它，见 dataset.load_mask_png）
    mp = Path(out_dir) / IO.MASK_SUBDIR / f"{path.stem}.png"
    check("像素掩码写在 masks\\ 子目录里", mp.exists())
    if mp.exists():
        from PIL import Image
        _im = Image.open(mp)
        arr = np.asarray(_im)
        check("掩码是单通道黑底白条（只有 root）",
              _im.mode == "L" and set(np.unique(arr)) <= {0, 255}
              and int((arr > 0).sum()) > 0,
              f"mode={_im.mode} 取值 {np.unique(arr)} 白像素 {int((arr > 0).sum())}")
        check("掩码尺寸 = 原图尺寸（原始分辨率，不缩放）",
              (arr.shape[1], arr.shape[0]) == (img.shape[1], img.shape[0]),
              f"{arr.shape[1]}x{arr.shape[0]}")
        # **训练那边真的会用它** —— 这一条才说明掩码没白存
        sys.path.insert(0, str(ROOT.parents[1] / "root_model"))
        import common.dataset as _ds
        _a = _ds.load_annot(str(out_dir), path.stem,
                            (arr.shape[1], arr.shape[0]), verbose=False)
        _px = _ds.load_root_mask_png(_a, (arr.shape[1], arr.shape[0]))
        _m, _v = _ds.build_target_masks(_a, (arr.shape[1], arr.shape[0]), 10)
        # 覆盖后 root 通道应该跟掩码**逐位相同**（同尺寸时不缩放）
        check("root_model 用掩码覆盖了 root 通道（且只动 root）",
              _px is not None and bool((_m[:, :, 0] == _px).all()) and bool(_v.all()),
              f"root {int(_m[:, :, 0].sum())} px / 掩码 {int(_px.sum()) if _px is not None else 0} px")
    sys.path.insert(0, str(ROOT / "_vendor"))
    from common.labelme import parse_other
    from common.gt_mask import draw_polygons_at
    lab = parse_other(out["json"], image_size=(w, h))
    check("回读 root 多边形数 == 导出的", len(lab.roots) == out["n_root_poly"],
          f"{len(lab.roots)}")
    check("回读到的 root 全是多边形", all(lab.root_polygons))
    check("回读 stem / check", len(lab.stems) > 0 and lab.check_rect is not None)
    shapes = json.load(open(out["json"], encoding="utf-8"))["shapes"]
    n_line = sum(1 for s in shapes if s["shape_type"] == "linestrip")
    check("折线也写进 json 了", n_line == out["n_polyline"], f"{n_line} 条")
    check("多边形数 == 折线数", len(lab.roots) == out["n_polyline"],
          f"{len(lab.roots)} 个多边形 / {out['n_polyline']} 条折线")
    back = draw_polygons_at([[tuple(map(float, p)) for p in poly] for poly in lab.roots],
                            (w, h))
    f = final > 0
    cov = float((back & f).sum()) / max(1, f.sum())
    # 现在多边形是"每条折线一个"，只覆盖折线经过的地方 —— 不再是整张掩码的轮廓，
    # 所以拿覆盖率当判据已经不对了（折线到不了的根尖、侧枝本来就不算）。这里只确认
    # 它覆盖了掩码的大部分，真正的规则由 5b 的合成用例逐条验。
    check("回读的多边形覆盖掩码主体 >90%", cov > 0.90, f"{cov * 100:.2f}%")
    import os
    try:
        same = os.stat(path).st_ino == os.stat(out_dir / path.name).st_ino
    except OSError:
        same = False
    check("datasets 里的原图是硬链接", same)

    print("== 5c. 保存后把原图从 pictures 移走（全程在临时目录，不碰真 pictures）==")
    from pathlib import Path as _P
    tmp_pics = _P(out_dir) / "_pics"
    tmp_pics.mkdir(exist_ok=True)
    src_fake = tmp_pics / path.name
    shutil.copy2(path, src_fake)                 # 假装这是"还没做的那张原图"
    IO.save_annotation(path.stem, path.name, img.shape, final, lines,
                       res["stem_polys"], res["check_box"], src_image=src_fake,
                       datasets_dir=out_dir, move_pictures=True)
    check("做完的图从 pictures 里移走（源那份删掉）", not src_fake.exists())
    check("datasets 里那份还在、大小对得上",
          ((_P(out_dir) / path.name).exists()
           and (_P(out_dir) / path.name).stat().st_size == path.stat().st_size))
    # 合并扫描：做完的图从 pictures 消失了，但列表里必须还在（不然想回去改都点不到）
    stems = {p.stem for p in IO.list_images()}
    want = {p.stem for p in Path(IO.PICTURES_DIR).iterdir()
            if p.suffix.lower() in IO.config.IMAGE_EXTS}
    want |= {p.stem for p in Path(IO.DATASETS_DIR).iterdir()
             if p.is_file() and p.suffix.lower() in IO.config.IMAGE_EXTS}
    check("列表 = pictures ∪ datasets（两边都扫）", stems == want,
          f"列表 {len(stems)} 张 / 两边合计 {len(want)} 张")
    # 界面那一路的开关接线：默认要移走，--keep-pictures 才留着
    check("界面默认移走原图（--keep-pictures 才保留）",
          parse_args([]).keep_pictures is False
          and parse_args(["--keep-pictures"]).keep_pictures is True)

    print("== 5b. 折线->多边形的规则（合成用例，逐条对）==")
    # 用一张自己能算准的小图，把三条规则分别钉死：
    #   ① 折线所在的掩码 -> 用**掩码的宽度**（不是带宽）
    #   ② 掩码缺了的地方 -> 用**带宽**补
    #   ③ 没有折线经过的掩码块 -> 丢掉
    from annotate_io import RootPolygonBuilder
    m = np.zeros((600, 900), np.uint8)
    m[200:240, 100:500] = 255            # 一条 40px 宽的"根"
    m[450:520, 700:800] = 255            # 一块没有任何折线经过的掩码
    test_lines = [[(120.0, 220.0), (480.0, 220.0)],      # 沿着那条根
                  [(120.0, 500.0), (500.0, 500.0)]]      # 底下什么都没有
    polys2, info2 = RootPolygonBuilder(m.shape).polygons(m, test_lines, 12)
    check("两条折线 -> 两个多边形", len(polys2) == 2 and all(polys2),
          f"{sum(1 for p in polys2 if p)} 个")
    if all(polys2):
        h0 = max(y for _, y in polys2[0]) - min(y for _, y in polys2[0])
        h1 = max(y for _, y in polys2[1]) - min(y for _, y in polys2[1])
        check("掩码在的地方按掩码宽度（40px）", 32 <= h0 <= 48,
              f"多边形高 {h0:.0f}px（掩码 40 / 带宽 12）")
        check("掩码缺的地方按带宽补（12px）", 8 <= h1 <= 18,
              f"多边形高 {h1:.0f}px（带宽 12）")
    check("没折线经过的掩码块被丢掉", info2["n_dropped_blocks"] == 1,
          f"丢掉 {info2['n_dropped_blocks']} 块 / {info2['dropped_area']} px²")
    # 带宽变化要真的改变结果（界面上的旋钮才有意义）
    _, info_w = RootPolygonBuilder(m.shape).polygons(m, test_lines, 40)
    check("带宽可调（40px 时补出来的带更粗）",
          info_w["width"] == 40, f"实际用宽 {info_w['width']}")

    print("== 6. 界面（offscreen，真开窗口 + 合成鼠标事件）==")
    # 上面几段验的是数据；这一段验的是接线 —— paintEvent/事件处理/保存这条路上
    # 任何一处写错都不会被数据层的自检发现。
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtCore import QEvent, QPointF
    from PyQt5.QtGui import QKeyEvent, QMouseEvent
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    # **offscreen 平台下弹 QMessageBox 会直接段错误**（实测：单独一句
    # QMessageBox.warning(None,...) 就 139，不是本项目的代码问题，是 Qt offscreen
    # 建不出模态窗）。所以先换成"记录下来 + 返回默认值"，自检既不会崩，
    # 还能顺便断言"该提示的地方确实提示了"。
    asked = []

    def _stub(kind, default):
        def f(*a, **k):
            asked.append((kind, a[2] if len(a) > 2 else ""))
            return default
        return staticmethod(f)

    QMessageBox.warning = _stub("warning", QMessageBox.Ok)
    QMessageBox.critical = _stub("critical", QMessageBox.Ok)
    QMessageBox.information = _stub("information", QMessageBox.Ok)
    QMessageBox.question = _stub("question", QMessageBox.No)

    win = MainWindow(args)                     # args.selftest -> 不加载模型，用假预测
    win.resize(1200, 800)
    win.show()
    app.processEvents()
    c = win.canvas
    check("窗口开着、图装上了", c.doc is not None and c.doc.store is not None)

    def mev(kind, x, y, btn=Qt.LeftButton, btns=Qt.LeftButton, mod=Qt.NoModifier):
        return QMouseEvent(kind, QPointF(x, y), btn, btns, mod)

    def _grab():
        app.processEvents()
        im = c.grab().toImage()
        w, h = im.width(), im.height()
        ptr = im.constBits()
        ptr.setsize(im.byteCount())
        return np.frombuffer(ptr, np.uint8).reshape(
            h, im.bytesPerLine() // 4, 4)[:, :w, :3].copy()

    def _ring_px():
        """画面上有几个"笔刷圈色"的像素。圈是用实色笔画空圈，掩码是半透明混出来的，
        所以精确匹配这个颜色只会命中圈本身。"""
        col = np.array(core.MODE_COLORS_BGR[c.mode][::-1], np.uint8)
        return int((np.abs(_grab().astype(int) - col).sum(2) == 0).sum())

    # 笔刷圈：**涂的时候要一直看得见**（用户 2026-10-07 要的），只有平移时才藏。
    # 两个方向都钉住：涂着的时候没了不行，"平移时还跟着跑"也不行。
    c.zoom_to(1.0)
    c.mouseMoveEvent(mev(QEvent.MouseMove, 500, 400, Qt.NoButton, Qt.NoButton))
    n_hover = _ring_px()
    detail = [f"悬停 {n_hover}"]
    ok_ring = n_hover > 20
    # 左键：按下、拖动中、松开，圈都要在（涂痕和圈同色，所以单看圈色仍然只数得到圈）
    c.mousePressEvent(mev(QEvent.MouseButtonPress, 500, 400))
    n_down = _ring_px()
    c.mouseMoveEvent(mev(QEvent.MouseMove, 520, 420, Qt.NoButton, Qt.LeftButton))
    n_drag = _ring_px()
    c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, 520, 420))
    n_up = _ring_px()
    detail.append(f"左键 按下 {n_down}/拖 {n_drag}/松 {n_up}")
    ok_ring = ok_ring and min(n_down, n_drag, n_up) > 20
    win.undo()                                  # 刚涂的那一笔撤掉
    # 中键平移：圈必须藏起来
    c.mousePressEvent(mev(QEvent.MouseButtonPress, 500, 400, Qt.MiddleButton,
                          Qt.MiddleButton))
    n_mid = _ring_px()
    c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, 500, 400, Qt.MiddleButton,
                            Qt.NoButton))
    n_midup = _ring_px()
    detail.append(f"中键平移 {n_mid}/松 {n_midup}")
    check("笔刷圈：涂的时候在、平移时藏、松开回来",
          ok_ring and n_mid == 0 and n_midup > 20, " | ".join(detail))
    c.fit()

    # 涂一笔（走真实的事件处理，不是直接调 store）
    c.mousePressEvent(mev(QEvent.MouseButtonPress, 300, 300))
    for x in range(310, 600, 10):
        c.mouseMoveEvent(mev(QEvent.MouseMove, x, 300, Qt.NoButton, Qt.LeftButton))
    c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, 600, 300))
    check("鼠标拖动涂出了绿掩码", c.doc.store.add.any(),
          f"{int((c.doc.store.add > 0).sum())} px")
    check("掩码被标成未保存", c.doc.dirty)
    c.repaint()
    check("paintEvent + 合成跑通（帧缓冲非全黑）", int(c.doc.builder.frame[:, :, :3].max()) > 0)
    win.undo()
    check("Ctrl+Z 撤销了这一笔", not c.doc.store.add.any())
    win.redo()
    check("Ctrl+Y 重做回来", c.doc.store.add.any())

    win.set_phase(1)
    c.draft_polylines()
    n_lines = len(c.doc.polylines)
    check("G 起草出折线", n_lines > 0, f"{n_lines} 条")

    # 起草的防误触：只在"有手工成果会被替换"时才问（弹窗本身在 offscreen 下会永久阻塞，
    # 所以验的是那个判断，不真弹）
    check("刚起草完不会拦（连按两下 G 不该弹两次）", not c.draft_needs_confirm())
    line = c.doc.polylines[0]
    sx, sy = c.doc.vp.image_to_screen(*line[0])
    c.mousePressEvent(mev(QEvent.MouseButtonPress, int(sx), int(sy)))
    c.mouseMoveEvent(mev(QEvent.MouseMove, int(sx) + 30, int(sy) + 20,
                         Qt.NoButton, Qt.LeftButton))
    c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, int(sx) + 30, int(sy) + 20))
    check("手工拖过控制点后再按 G 会拦一下", c.draft_needs_confirm(),
          f"polylines_edited={c.doc.polylines_edited}")
    win.undo()
    check("撤销手工改动后仍然拦（宁可多问一句）", c.draft_needs_confirm())
    line = c.doc.polylines[0]
    # Alt+点击：插控制点
    mid = line[len(line) // 2]
    sx, sy = c.doc.vp.image_to_screen(mid[0], mid[1])
    n_pt = len(line)
    c.mousePressEvent(mev(QEvent.MouseButtonPress, sx + 30, sy + 30,
                          Qt.LeftButton, Qt.LeftButton, Qt.AltModifier))
    check("Alt+点击插入了控制点", len(c.doc.polylines[0]) == n_pt + 1,
          f"{n_pt} -> {len(c.doc.polylines[0])}")
    # 拖控制点：拖完之后这个点的**图像坐标**应该对应到鼠标松手的位置
    line = c.doc.polylines[0]
    old_pt = tuple(line[0])
    px, py = c.doc.vp.image_to_screen(*old_pt)
    c.mousePressEvent(mev(QEvent.MouseButtonPress, px, py))
    c.mouseMoveEvent(mev(QEvent.MouseMove, px + 40, py + 25, Qt.NoButton, Qt.LeftButton))
    c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, px + 40, py + 25))
    want = c.doc.vp.screen_to_image(px + 40, py + 25)
    got = tuple(line[0])
    # 容差按缩放折算：QMouseEvent.pos() 是**整数**屏幕像素（QPointF 会被取整），
    # 0.5 屏幕像素在 22% 缩放下就是 2.3 图像像素。这不是误差，是鼠标事件本身的粒度。
    tol = 1.0 / c.doc.vp.zoom + 0.5
    check("拖动控制点落到了鼠标位置",
          abs(got[0] - want[0]) < tol and abs(got[1] - want[1]) < tol,
          f"{old_pt} -> ({got[0]:.1f}, {got[1]:.1f}) 期望 ({want[0]:.1f}, {want[1]:.1f}) "
          f"容差 {tol:.1f}px")
    c.repaint()
    check("画折线阶段的 paintEvent 跑通", True)

    # 整条删除（自动起草出来的废线就是这么删的）
    n_lines = len(c.doc.polylines)
    n_undo = len(c._pl_undo)
    line = c.doc.polylines[0]
    qx, qy = c.doc.vp.image_to_screen(*line[0])
    c.mousePressEvent(mev(QEvent.MouseButtonPress, qx, qy))
    c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, qx, qy))
    check("点控制点选中了一条折线", c.sel_index() == 1, f"选中 #{c.sel_index()}")
    check("只点一下选中不占撤销步数", len(c._pl_undo) == n_undo,
          f"撤销栈 {n_undo} -> {len(c._pl_undo)}")
    c.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_X, Qt.NoModifier))
    check("X 删掉整条折线", len(c.doc.polylines) == n_lines - 1,
          f"{n_lines} -> {len(c.doc.polylines)} 条")
    win.undo()
    check("撤销把整条折线恢复", len(c.doc.polylines) == n_lines,
          f"恢复到 {len(c.doc.polylines)} 条")
    c._sel = None
    c.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_X, Qt.NoModifier))
    check("没选中时按 X 不误删", len(c.doc.polylines) == n_lines)


    # 触摸板两指滚动 = 平移视角；真鼠标滚轮维持原样（阶段1 调笔刷）
    from PyQt5.QtCore import QPoint
    from PyQt5.QtGui import QWheelEvent

    def wheel(pix, ang, mod=Qt.NoModifier):
        c.wheelEvent(QWheelEvent(QPointF(500, 400), QPointF(500, 400),
                                 QPoint(*pix), QPoint(*ang), Qt.NoButton, mod,
                                 Qt.ScrollUpdate, False))

    win.set_phase(0)
    c.zoom_to(1.0)          # 先放大：适应窗口时整图都看得见，clamp 会把平移立刻抵消掉
    oy0, r0 = c.doc.vp.oy, c.radius
    wheel((0, -40), (0, -8))                     # 触摸板：带 pixelDelta 的小步长
    check("两指滚动 -> 平移视角（不动笔刷大小）",
          abs(c.doc.vp.oy - oy0) > 1 and abs(c.radius - r0) < 1e-9,
          f"视角 {oy0:.0f}->{c.doc.vp.oy:.0f}，笔刷半径 {r0:.0f}->{c.radius:.0f}")
    oy1 = c.doc.vp.oy
    wheel((0, 0), (0, 120))                      # 鼠标滚轮：一格 120，也是平移
    check("鼠标滚轮也是平移（不区分设备）",
          abs(c.doc.vp.oy - oy1) > 1 and abs(c.radius - r0) < 1e-9,
          f"视角 {oy1:.0f}->{c.doc.vp.oy:.0f}，半径没动")
    oy2, r1 = c.doc.vp.oy, c.radius
    wheel((0, 0), (0, 120), Qt.ShiftModifier)    # Shift+滚轮 = 笔刷大小
    check("Shift+滚轮 -> 笔刷大小（视角不动）",
          abs(c.radius - r1) > 1e-9 and abs(c.doc.vp.oy - oy2) < 1e-9,
          f"半径 {r1:.0f}->{c.radius:.0f}，视角没动")
    wheel((0, 0), (-120, 0), Qt.ShiftModifier)   # 反向也是笔刷（取不到 y 就用 x）
    check("Shift+横滚也能调笔刷（触摸板横滑）", abs(c.radius - r1) < 1e-9,
          f"半径回到 {c.radius:.0f}")
    z0 = c.doc.vp.zoom
    wheel((0, 0), (0, 120), Qt.ControlModifier)
    check("Ctrl+滚轮 = 缩放", abs(c.doc.vp.zoom - z0) > 1e-9,
          f"缩放 {z0:.3f}->{c.doc.vp.zoom:.3f}")
    c.fit()
    win.set_phase(1)        # 下面那些是折线测试，阶段要还回去

    # 断开（在交叉处切一刀）：几何必须严丝合缝，不能裂出一道缝
    n_lines = len(c.doc.polylines)
    line = c.doc.polylines[0]
    mid_seg = max(1, len(line) // 2 - 1)
    if mid_seg < len(line) - 1:
        a, b = line[mid_seg], line[mid_seg + 1]
        cut = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)      # 某一段的中点
        px, py = c.doc.vp.image_to_screen(*cut)
        c.mousePressEvent(mev(QEvent.MouseButtonPress, *c.doc.vp.image_to_screen(*line[0])))
        c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, *c.doc.vp.image_to_screen(*line[0])))
        c.mouseMoveEvent(mev(QEvent.MouseMove, int(px), int(py), Qt.NoButton, Qt.NoButton))
        c.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_B, Qt.NoModifier))
        check("B 把一条断成两条", len(c.doc.polylines) == n_lines + 1,
              f"{n_lines} -> {len(c.doc.polylines)} 条")
        l1, l2 = c.doc.polylines[0], c.doc.polylines[1]
        gap = float(np.hypot(l1[-1][0] - l2[0][0], l1[-1][1] - l2[0][1]))
        check("两段接缝严丝合缝（没有裂开）", gap < 0.51, f"端点间距 {gap:.3f} px")
        check("切开后总长不变",
              abs(core.polyline_length(l1) + core.polyline_length(l2)
                  - core.polyline_length(line)) < 0.6,
              f"{core.polyline_length(line):.1f} -> "
              f"{core.polyline_length(l1) + core.polyline_length(l2):.1f} px")
        win.undo()
        check("撤销把断开还原", len(c.doc.polylines) == n_lines,
              f"回到 {len(c.doc.polylines)} 条")

    # 多边形预览 + 带宽旋钮（这两样连着"存出来是什么形状"）
    win._recompute_polygons()
    check("画布上算出了多边形预览",
          win.doc.polygons is not None and len(win.doc.polygons) == len(win.doc.polylines),
          f"{len(win.doc.polygons or [])} 个 / {len(win.doc.polylines)} 条折线")
    w0 = win.doc.poly_width
    _saved_settings = (IO.SETTINGS_PATH.read_bytes()
                       if IO.SETTINGS_PATH.exists() else None)   # 测完还原，别动用户的偏好
    win.set_poly_width(w0 + 6)
    check("带宽旋钮改得动、并且立刻重算",
          abs(win.doc.poly_width - (w0 + 6)) < 1e-6 and win.doc.polygons is not None,
          f"{w0:.0f} -> {win.doc.poly_width:.0f}px")
    check("旋钮和界面上的数字同步",
          win.width_box is not None and win.width_box.value() == int(round(w0 + 6)),
          f"spinbox = {win.width_box.value() if win.width_box else '无'}")
    check("带宽写进了 settings.json（下次启动还记得）",
          abs(float(IO.load_settings().get("poly_width", -1)) - (w0 + 6)) < 1e-6,
          f"settings = {IO.load_settings()}")
    # （"换个图带宽还在不在"那条放到最后 —— 换图会重载文档，把后面测试依赖的折线清掉）
    c.show_poly = False
    c.repaint()
    check("关掉多边形预览也能画（P 键路径）", True)
    c.show_poly = True
    c.repaint()

    win_stem = win.doc.stem                     # 窗口打开的可能是另一张（第一张已被第 5 步写了）
    win.save(go_next=False)
    check("界面保存出了 json", (out_dir / f"{win_stem}.json").exists())

    # 图被删掉之后点它：2026-10-07 用户报的崩溃（FileNotFoundError 一路冒出来）。
    # 用临时目录造一个"列表里有、磁盘上没有"的条目，别碰真实图片。
    import tempfile as _tf
    tmpd = Path(_tf.mkdtemp(prefix="_gone_", dir=str(IO.CACHE_ROOT)))
    atexit.register(shutil.rmtree, tmpd, True)
    fake = tmpd / "root_FAKE_20250101CK.jpg"   # **不建文件**：模拟"列表里有、磁盘上没有"
    keep_doc = win.doc
    win.set_dirty(False)                       # 别让"未保存"拦在前面
    win.images.append(fake)
    win.idx = len(win.images) - 1
    check("文件不在了：_resolve_path 返回 None（不硬读）",
          win._resolve_path(win.idx) is None)
    n_asked = len(asked)
    win.open_image(win.idx, force=True)        # 这一下以前会崩成 FileNotFoundError
    check("点一张已删除的图不会崩，而且弹了提示、留在原来那张",
          win.doc is keep_doc and len(asked) > n_asked,
          f"当前仍是 {win.doc.stem if win.doc else '无'}，弹了 {len(asked)-n_asked} 个提示")
    # "被挪到 datasets 了"的情况要能自己找回来（我们自己就会把做完的图挪过去）
    moved = Path(args.datasets) / fake.name
    moved.write_bytes(b"x")                    # 这里只验路径解析，不用真图片
    got = win._resolve_path(win.idx)
    check("文件挪到 datasets 了：能自己找回来", got == moved,
          f"解析到 {got.parent.name if got else 'None'}/")
    win.images.pop(win.idx)
    win.doc = keep_doc
    shutil.rmtree(tmpd, ignore_errors=True)
    if moved.exists():
        moved.unlink()
    # 换一张图：带宽应该沿用（"固定到上次选择"）—— 放在最后，换图会重载文档
    win.set_dirty(False)
    w_keep = win.poly_width
    win.open_image(win.idx + 1 if win.idx + 1 < len(win.images) else win.idx - 1,
                   force=True)
    check("换图之后带宽不变（工具级设置，不是每张图一份）",
          abs(win.doc.poly_width - w_keep) < 1e-6,
          f"新图带宽 {win.doc.poly_width:.0f}px（设为 {w_keep:.0f}）")
    win.set_poly_width(IO.ROOT_POLY_WIDTH_DEFAULT)
    if _saved_settings is None:
        IO.SETTINGS_PATH.unlink(missing_ok=True)      # 测完把用户的偏好还原
    else:
        IO.SETTINGS_PATH.write_bytes(_saved_settings)

    # F5 重新扫描：重建列表 + 停在原来那张
    win.set_dirty(False)
    cur_stem = win.doc.stem
    win.rescan()
    check("F5 重新扫描：列表重建、还停在原来那张",
          win.doc is not None and win.doc.stem == cur_stem
          and win.listw.count() == len(win.images),
          f"{win.listw.count()} 行 / {len(win.images)} 张，停在 {win.doc.stem}")
    c.fit()

    print("== 7. 从缓存续做（关掉工具再打开，编辑状态要能回来）==")
    # 这一条是用户最常走的路：32 张图都有缓存，每次打开都是走这里而不是重新预测。
    st0 = core.MaskStore(np.zeros((img.shape[0], img.shape[1]), np.uint8))
    st0.pred[500:600, 500:900] = 255
    st0.begin_stroke()
    st0.paint_segment(700, 700, 900, 700, 40, core.MODE_ADD)
    st0.end_stroke()
    st0.begin_stroke()
    st0.paint_segment(550, 550, 700, 550, 30, core.MODE_DEL)
    st0.end_stroke()
    saved_lines = [[(10.0, 10.0), (20.0, 30.0)], [(40.0, 50.0), (60.0, 70.0)]]
    cache_stem = imgs[0].stem
    IO.save_cache(IO.cache_dir("selftest"), cache_stem, st0,
                  IO.meta_for_save("selftest", img.shape, cache_stem, imgs[0].name,
                                   0.18, [1, 2, 3, 4], True, [[(5.0, 5.0), (6.0, 6.0),
                                                              (7.0, 7.0)]],
                                   True, False, saved_lines,
                                   # 概率图必须给：缓存三件（掩码/概率/元信息）缺一件就整份作废，
                                   # 所以只写掩码的"半份缓存"会被正确地判成没有缓存
                                   prob=np.zeros((16, 16), np.float32)))
    win.open_image(0, force=True)
    d2 = win.doc
    check("缓存的图打开了（没走预测）", d2.stem == cache_stem)
    check("缓存里的三张掩码逐位还原",
          bool((d2.store.pred == st0.pred).all()) and
          bool((d2.store.add == st0.add).all()) and
          bool((d2.store.dele == st0.dele).all()))
    check("折线也还原了", len(d2.polylines) == 2 and
          tuple(d2.polylines[0][0]) == (10.0, 10.0))
    check("丢弃标志 / 检查框 / 阈值还原",
          d2.drop_stem is True and d2.drop_check is False
          and list(d2.check_box) == [1, 2, 3, 4] and abs(d2.low_thresh - 0.18) < 1e-6)
    c.repaint()
    check("接着改还能画（paintEvent 正常）", int(d2.builder.frame[:, :, :3].max()) > 0)

    # 清理：删掉整个临时输出目录（里面的东西只有自检会看），缓存里的 selftest 目录同理。
    # 用 rmtree 而不是逐个删 —— 逐个删就要靠"记住自己写过什么"，而漏记一个名字
    # 就是一个留在别人数据集里的垃圾文件。
    removed = len(list(out_dir.rglob("*")))
    shutil.rmtree(out_dir, ignore_errors=True)
    scd = IO.cache_dir("selftest")
    n_cache = len(list(scd.iterdir())) if scd.exists() else 0
    if scd.exists():
        shutil.rmtree(scd, ignore_errors=True)
    print(f"  自检产物已清理: 临时输出目录 {removed} 项 + cache/selftest {n_cache} 项"
          f"（都没有碰 datasets\\）")
    print("== 自检" + ("通过 ==" if ok else "**失败** =="))
    return 0 if ok else 1


def _selftest_outdir():
    """自检的输出目录：临时目录，跟真实的 datasets\\ 完全无关。

    2026-10-07 差点出事：自检原本是把标注导进 `datasets\\`、跑完再删掉那几个文件，
    而它用的图片就是 `pictures\\` 里的**第一张** —— 那正好是用户开始标注的第一张。
    于是"跑一次自检"= 覆盖并删除用户刚标好的 json。
    """
    import tempfile
    # 放在工具的 cache\ 下、**不放到系统临时目录**：系统临时目录在 C:，而 pictures\ 在 D:，
    # 跨盘时 os.link 会失败并自动退回复制 —— "原图是硬链接"那条断言就永远是假的
    # （不是代码坏了，是测试自己把条件破坏了）。
    IO.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="_selftest_", dir=str(IO.CACHE_ROOT)))
    # 注册兜底清理：自检中途抛异常时走不到最后那段 rmtree，临时目录会一个个攒下来
    # （2026-10-07 排查时就攒了 7 个）。atexit 在内核退出前一定会跑。
    atexit.register(shutil.rmtree, d, True)
    return d


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="根系标注工具：模型预测 -> 人修掩码 -> 画折线 -> 存 labelme",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--image", help="直接打开某张（文件名主干）")
    p.add_argument("--model", help="模型文件夹名/路径，默认取 model\\ 下最新")
    p.add_argument("--size", type=int, help="覆盖输入长边（默认用权重里记录的）")
    p.add_argument("--pictures", help="图片目录（默认 pictures\\）")
    p.add_argument("--datasets", help="标注输出目录（默认 datasets\\）")
    p.add_argument("--prefetch", action="store_true", help="只跑预测填缓存，不开界面")
    p.add_argument("--selftest", action="store_true", help="无界面自检")
    p.add_argument("--backfill-masks", action="store_true",
                   help="给已经存过 json 的图补一份像素级掩码（数据在缓存里，不用重标）")
    p.add_argument("--no-gpu", action="store_true", help="强制 CPU 预测")
    p.add_argument("--no-cache", action="store_true", help="忽略缓存，全部重新预测")
    p.add_argument("--keep-pictures", action="store_true",
                   help="保存后不把原图从 pictures 目录里移走（默认会移走，"
                        "使 pictures 里只剩没做的）")
    return p.parse_args(argv)


def run_backfill_masks(args):
    """给**已经存过 json 的图**补一份像素级掩码（数据在缓存里，不用重新标）。

    为什么需要它：掩码是后来才加的，之前存的图只有多边形 —— 而两者是**两把尺子**
    （实测面积差约 2%），同一批数据里混着用等于一半图一个口径。补一遍就统一了。
    """
    model_name = IO.resolve_model(args.model)[1][0]
    cdir = IO.cache_dir(model_name)
    imgs = IO.list_images(args.pictures, args.datasets)
    done = IO.done_stems(args.datasets)
    print(f"模型 {model_name} | 缓存 {cdir} | 已标 {len(done)} 张")
    n_ok = n_skip = 0
    for p in imgs:
        if p.stem not in done:
            continue
        c = IO.load_cache(cdir, p.stem, model_name)
        if c is None:
            print(f"  {p.stem}: 没有缓存，跳过（这张图重新打开一次再存就能补上）")
            n_skip += 1
            continue
        store = core.MaskStore(c["pred"], c.get("add"), c.get("dele"))
        meta = c["meta"]
        arr = IO.build_mask_png(store.final_mask())      # 黑底白条，只有 root
        fp = IO.save_mask_png(args.datasets or IO.DATASETS_DIR, p.stem, arr)
        print(f"  {p.stem}: 根 {int((arr > 0).sum())} px -> {fp.name}")
        n_ok += 1
    print(f"补了 {n_ok} 张" + (f"，{n_skip} 张没有缓存被跳过" if n_skip else ""))
    return 0


def main(argv=None):
    # 控制台编码：中文 Windows 的 cmd 是 GBK，打不出某些字符（如 '²'）时 print 会直接
    # 抛 UnicodeEncodeError 把程序打断 —— 而那是**打印**的问题，不该弄死一个正在跑的任务。
    # 把编码留着（GBK 下中文显示才正常），只把错误处理换成替换。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    if args.prefetch:
        run_prefetch(args)
        return 0
    if args.selftest:
        return run_selftest(args)
    if args.backfill_masks:
        return run_backfill_masks(args)
    if not DPR_AWARE:
        QApplication.setAttribute(Qt.AA_DisableHighDpiScaling, True)
    app = QApplication(sys.argv[:1])
    win = MainWindow(args)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
