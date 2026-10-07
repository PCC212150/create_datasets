"""标注工具的核心：掩码状态 / 撤销 / 视口 / 帧缓冲合成 —— **本文件不 import Qt**。

拆出来的理由不是"代码整洁"，是**可测**：项目里没有任何 GUI，而"笔刷涂得对不对、
撤销能不能还原、视口坐标换算准不准"这些是纯粹的数据问题，不该只有肉眼盯着窗口才能验。
`annotate_root.py --selftest` 就是直接调这里的类跑无界面自检。

三张掩码全在**原图分辨率**（5472x3648），uint8 0/255：

    pred  模型预测（可被 `[` `]` 调阈值重算）
    add   绿笔：预测漏了、补上
    del   蓝笔：预测错了、删掉

**最终掩码 = (pred | add) & ~del**（`del` 优先级最高）。用户原话是
「保留绿色掩码和白色掩码与蓝色掩码的差集（即属于红色，不属于蓝色）」——
"白色"指代不明，这里按「预测层（红色那层）」理解。

**笔刷三种模式，最后一笔赢**：涂绿会清掉笔下的蓝，涂蓝会清掉笔下的绿，橡皮擦把绿蓝都清掉
（回到纯预测，**不动 pred** —— 预测是模型的东西，要改它只能用阈值）。
不这么做的话，同一像素既在绿里又在蓝里，用户会看到"明明是绿的，存出来却没有"。
"""
import numpy as np

# ---- 笔刷模式 ----
MODE_ADD, MODE_DEL, MODE_ERASE = 0, 1, 2
MODE_ORDER = (MODE_ADD, MODE_DEL, MODE_ERASE)      # TAB 的循环顺序
MODE_NAMES = {MODE_ADD: "补(绿)", MODE_DEL: "删(蓝)", MODE_ERASE: "擦"}
MODE_COLORS_BGR = {MODE_ADD: (0, 200, 0), MODE_DEL: (255, 90, 0), MODE_ERASE: (230, 230, 230)}

# ---- 显示口径（与 inference.py 的 overlay 同一套颜色，方便交叉看）----
ALPHA = 0.45
ALPHA255 = int(round(ALPHA * 255))                 # 定点混合用（0~255）
COLOR_PRED_BGR = (0, 0, 255)
COLOR_ADD_BGR = (0, 200, 0)
COLOR_DEL_BGR = (255, 90, 0)
# 帧缓冲是 QImage.Format_RGB32 的内存序 = B,G,R,A（小端），所以上面的颜色都按 BGR 写。

UNDO_MAX_BYTES = 64 << 20                          # 撤销栈字节上限（超出丢最旧）
UNDO_MAX_OPS = 200                                 # 同时限制步数，免得全是空补丁占位置

_stamp_cache = {}


def disc_stamp(radius: float):
    """半径 radius（图像像素）的实心圆盘模板 → `(bool_st, u8_st)`，都是 (2r+1, 2r+1)。

    两个版本都缓存：盖掩码时直接 `np.maximum(dst, u8)` 不产生临时数组（`bool*255`
    每个章都要分配一次 3600 像素的临时量，一次拖动几百个章就是白烧），
    擦除时用 `bool` 直接做下标。

    按 0.5px 量化缓存 —— 拖动笔刷时半径连续变化，每次重算一个 61x61 的 ogrid
    在 60fps 下是纯浪费。0.5px 的量化在 5472px 的图上肉眼看不出来。
    """
    key = max(1, int(round(radius * 2)))
    st = _stamp_cache.get(key)
    if st is None:
        r = key / 2.0
        n = int(np.ceil(r))
        yy, xx = np.ogrid[-n:n + 1, -n:n + 1]
        b = (xx * xx + yy * yy) <= r * r
        st = (b, (b * 255).astype(np.uint8))
        _stamp_cache[key] = st
    return st


def _stamp_into(mask: np.ndarray, value: int, cx: float, cy: float, radius: float):
    """把圆盘盖进 mask（value=0 表示擦）。返回**实际改动的**整数 bbox (y0,x0,y1,x1)。"""
    st_b, st_u8 = disc_stamp(radius)
    n = st_b.shape[0]
    r = n // 2
    x0, y0 = int(np.floor(cx)) - r, int(np.floor(cy)) - r
    x1, y1 = x0 + n, y0 + n
    h, w = mask.shape
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    sx0, sy0 = max(0, -x0), max(0, -y0)                 # 模板上被裁掉的部分
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    sl_src = (slice(sy0, sy0 + (y1 - y0)), slice(sx0, sx0 + (x1 - x0)))
    dst = mask[y0:y1, x0:x1]
    if value:
        np.maximum(dst, st_u8[sl_src], out=dst)
    else:
        dst[st_b[sl_src]] = 0
    return y0, x0, y1, x1


def _union_box(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


class UndoStack:
    """撤销/重做。一条 `Op` = 一次操作动过的若干张掩码的**差分补丁**。

    为什么不存全图快照：5472x3648 的 uint8 一张 20MB，50 步就是 1GB。
    差分补丁按 bbox 裁剪再 `np.packbits` 压位，一次笔刷通常只有几 KB。
    """

    def __init__(self, max_bytes=UNDO_MAX_BYTES, max_ops=UNDO_MAX_OPS):
        self.max_bytes = int(max_bytes)
        self.max_ops = int(max_ops)
        self._undo, self._redo = [], []
        self._bytes = 0

    def push(self, op):
        if op is None:
            return
        self._undo.append(op)
        self._redo.clear()
        self._bytes += op.nbytes
        while self._undo and (self._bytes > self.max_bytes
                              or len(self._undo) > self.max_ops):
            self._bytes -= self._undo.pop(0).nbytes

    def clear(self):
        self._undo.clear()
        self._redo.clear()
        self._bytes = 0

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    def undo(self, store):
        """回退一步，返回被改动的 bbox（供重绘），没有可回退的返回 None。"""
        if not self._undo:
            return None
        op = self._undo.pop()
        box = op.apply(store, before=True)
        self._redo.append(op)
        return box

    def redo(self, store):
        if not self._redo:
            return None
        op = self._redo.pop()
        box = op.apply(store, before=False)
        self._undo.append(op)
        return box


class Op:
    """一次操作：`parts = [(掩码属性名, bbox, before_packed, after_packed), ...]`。"""

    def __init__(self, parts, label=""):
        self.parts = parts
        self.label = label
        self.bbox = None
        for _, bb, _, _ in parts:
            self.bbox = _union_box(self.bbox, bb)
        self.nbytes = sum(p[2].nbytes + p[3].nbytes for p in parts)

    def apply(self, store, before: bool):
        for name, (y0, x0, y1, x1), pb, pa in self.parts:
            arr = getattr(store, name)
            h, w = y1 - y0, x1 - x0
            packed = pb if before else pa
            vals = np.unpackbits(packed, axis=1, count=w)[:, :w].astype(bool)
            arr[y0:y1, x0:x1] = np.where(vals, 255, 0).astype(np.uint8)
        return self.bbox


class MaskStore:
    """一张图的三张掩码 + 笔刷 + 撤销。所有坐标都是**原图坐标**（float）。"""

    def __init__(self, pred: np.ndarray, add=None, dele=None):
        self.pred = np.ascontiguousarray(pred.astype(np.uint8))
        h, w = self.pred.shape
        self.add = (np.zeros((h, w), np.uint8) if add is None
                    else np.ascontiguousarray(add.astype(np.uint8)))
        self.dele = (np.zeros((h, w), np.uint8) if dele is None
                     else np.ascontiguousarray(dele.astype(np.uint8)))
        self.undo = UndoStack()
        self._snap = None
        self._snap_names = ()
        self.stroke_bbox = None

    # ---------- 基本 ----------
    @property
    def shape(self):
        return self.pred.shape

    def final_mask(self) -> np.ndarray:
        """最终掩码 = (pred | add) & ~del，uint8 0/255。"""
        out = self.pred | self.add
        out &= ~self.dele                            # 0/255 的 uint8 按位取反正好是 255/0
        return out

    def is_dirty(self) -> bool:
        return bool(self.add.any() or self.dele.any())

    # ---------- 笔刷 ----------
    def paint_segment(self, x0, y0, x1, y1, radius, mode):
        """把 (x0,y0)->(x1,y1) 这一段刷上（浮点图像坐标）。返回本段 bbox。

        鼠标事件可能很密（高刷屏每像素一个事件）也可能很疏（快速甩动几帧才一个），
        所以**在两点之间按半径的一半为步长补圆盘**：密的时候只有 1~2 个章，
        疏的时候自动补满，两种极端下都不会画成断续的点。
        """
        if radius < 0.5:
            radius = 0.5
        # 模式决定往哪张掩码盖什么值；0 表示"清掉"
        writes = {MODE_ADD: (("add", 255), ("dele", 0)),
                  MODE_DEL: (("dele", 255), ("add", 0)),
                  MODE_ERASE: (("add", 0), ("dele", 0))}[mode]
        length = float(np.hypot(x1 - x0, y1 - y0))
        n = max(1, int(np.ceil(length / max(radius * 0.5, 1.0))))
        box = None
        for t in np.linspace(0.0, 1.0, n + 1):
            cx = x0 + (x1 - x0) * t
            cy = y0 + (y1 - y0) * t
            for name, val in writes:
                bb = _stamp_into(getattr(self, name), val, cx, cy, radius)
                box = _union_box(box, bb)
        self.stroke_bbox = _union_box(self.stroke_bbox, box)
        return box

    def begin_stroke(self):
        """一次笔画（按下->抬起）开始：留快照，抬起时算差分补丁。"""
        self._snap_names = ("add", "dele")
        self._snap = [getattr(self, n).copy() for n in self._snap_names]
        self.stroke_bbox = None

    def end_stroke(self):
        """结束笔画：有实际改动就**自动**压一个 Op 进撤销栈，返回改动 bbox（没有则 None）。

        **压栈放在这里而不是交给调用方**：调用方是 GUI 的事件处理，漏一次 push 就变成
        "能撤销别的操作、就是撤销不了这一笔"，而且很难发现。空笔迹（点一下没动）
        不压栈 —— 否则 Ctrl+Z 要按好几下才有反应，用户会以为撤销坏了。
        """
        op = self._make_op(self._snap_names, self._snap, self.stroke_bbox, label="笔刷")
        self._snap, self._snap_names, self.stroke_bbox = None, (), None
        if op is not None:
            self.undo.push(op)
            return op.bbox
        return None

    # ---------- 预测层（调阈值 / 重置）----------
    def begin_pred_change(self):
        self._snap_names = ("pred",)
        self._snap = [self.pred.copy()]

    def end_pred_change(self):
        """结束一次预测层改动，压栈并返回改动 bbox（无改动则 None）。理由同 end_stroke。"""
        box = (0, 0, *self.pred.shape)
        op = self._make_op(self._snap_names, self._snap, box, label="预测层")
        self._snap, self._snap_names = None, ()
        if op is not None:
            self.undo.push(op)
            return op.bbox
        return None

    def replace_pred(self, new_pred):
        """换掉整张预测（调阈值 / 重置到纯预测），并**记一次可撤销的操作**。"""
        self.begin_pred_change()
        self.pred = np.ascontiguousarray(new_pred.astype(np.uint8))
        return self.end_pred_change()

    # ---------- 内部 ----------
    def _make_op(self, names, snap, box, label):
        """在 box 内比较快照与当前值，生成差分补丁；无改动返回 None。"""
        if snap is None or box is None:
            return None
        y0, x0, y1, x1 = box
        if y1 <= y0 or x1 <= x0:
            return None
        parts, changed = [], False
        for name, old in zip(names, snap):
            cur = getattr(self, name)[y0:y1, x0:x1]
            ref = old[y0:y1, x0:x1]
            diff = cur != ref
            if not diff.any():
                continue
            changed = True
            parts.append((name, (y0, x0, y1, x1),
                          np.packbits(ref > 0, axis=1),
                          np.packbits(cur > 0, axis=1)))
        if not changed:
            return None
        return Op(parts, label=label)

    # 撤销/重做转发（同一套栈管笔刷与预测层，Ctrl+Z 才是"撤回上一步操作"）
    def undo_step(self):
        return self.undo.undo(self)

    def redo_step(self):
        return self.undo.redo(self)

    def push_op(self, op):
        self.undo.push(op)


class Viewport:
    """屏幕 <-> 原图坐标的换算。**全项目只有这里做这个换算**。

    别的模块（包括 QPainter 画折线、笔刷落点）一律调 `screen_to_image` / `image_to_screen`，
    不许自己写 `* zoom` —— 缩放/平移的 bug 九成都出在"某处漏乘了一个 zoom"。

    映射：screen = (image - origin) * zoom
    """

    MIN_ZOOM = 0.02
    MAX_ZOOM = 8.0

    def __init__(self, img_w, img_h):
        self.img_w, self.img_h = int(img_w), int(img_h)
        self.zoom = 1.0
        self.ox, self.oy = 0.0, 0.0

    # ---------- 换算 ----------
    def screen_to_image(self, sx, sy):
        return self.ox + sx / self.zoom, self.oy + sy / self.zoom

    def image_to_screen(self, ix, iy):
        return (ix - self.ox) * self.zoom, (iy - self.oy) * self.zoom

    def rect_screen_to_image(self, rect):
        x0, y0 = self.screen_to_image(rect[0], rect[1])
        x1, y1 = self.screen_to_image(rect[2], rect[3])
        return x0, y0, x1, y1

    # ---------- 视口操作 ----------
    def fit(self, win_w, win_h):
        """缩放到整图可见（留一点边距，免得贴边看不清）。"""
        z = min(win_w / float(self.img_w), win_h / float(self.img_h))
        self.zoom = float(np.clip(z, self.MIN_ZOOM, self.MAX_ZOOM))
        self.center_on(self.img_w / 2.0, self.img_h / 2.0, win_w, win_h)

    def center_on(self, ix, iy, win_w, win_h):
        self.ox = ix - win_w / 2.0 / self.zoom
        self.oy = iy - win_h / 2.0 / self.zoom
        self.clamp(win_w, win_h)

    def zoom_at(self, factor, sx, sy, win_w, win_h):
        """以屏幕点 (sx, sy) 为锚点缩放 —— 标注工具里这条是命根子：
        放大时想看清的地方必须钉在光标下面，否则每次放大都要重新找位置。"""
        ix, iy = self.screen_to_image(sx, sy)
        new_zoom = float(np.clip(self.zoom * factor, self.MIN_ZOOM, self.MAX_ZOOM))
        if abs(new_zoom - self.zoom) < 1e-9:
            return False
        self.zoom = new_zoom
        self.ox = ix - sx / self.zoom
        self.oy = iy - sy / self.zoom
        self.clamp(win_w, win_h)
        return True

    def pan(self, dx_screen, dy_screen):
        self.ox -= dx_screen / self.zoom
        self.oy -= dy_screen / self.zoom

    def clamp(self, win_w, win_h):
        """别让图完全跑出窗口（图片比窗口小时居中）。"""
        vw, vh = win_w / self.zoom, win_h / self.zoom
        if vw >= self.img_w:
            self.ox = (self.img_w - vw) / 2.0
        else:
            self.ox = min(max(self.ox, 0.0), self.img_w - vw)
        if vh >= self.img_h:
            self.oy = (self.img_h - vh) / 2.0
        else:
            self.oy = min(max(self.oy, 0.0), self.img_h - vh)

    def visible_image_rect(self, win_w, win_h):
        """屏幕上可见区域对应的原图矩形（float，可能超出图外）。"""
        x0, y0 = self.screen_to_image(0, 0)
        x1, y1 = self.screen_to_image(win_w, win_h)
        return x0, y0, x1, y1


class FrameBuilder:
    """把「原图 + 三张掩码」合成进一块窗口大小的帧缓冲。

    **只有一块帧缓冲**（`(h, w, 4)` uint8，BGRX 内存序 = `QImage.Format_RGB32`），
    没有常驻的降采样层：全量重建 1400x900 实测 ~48ms，只在换图/缩放/调阈值时做；
    涂笔刷时只重画笔迹那块的**脏矩形**（<1ms），平移时整块搬 + 只重画新露出的条带。

    掩码降采样用 `INTER_AREA`（面积平均 = 覆盖率，0~255），所以缩小时细根不会整片消失
    （最近邻会）；放大用 `INTER_NEAREST`，边界是"诚实的像素"。原图反过来：
    缩小用 INTER_AREA，放大用 INTER_LINEAR。
    """

    def __init__(self, img: np.ndarray, store: MaskStore, vp: Viewport):
        """img: uint8 (h, w, 3) **RGB**（root_model 全程这个口径）。"""
        self.img = img
        self.store = store
        self.vp = vp
        self.show_layers = True       # T 键
        self.final_only = False       # Shift+T：只看 (红∪绿)\蓝 的最终结果
        self.frame = None             # (h, w, 4) uint8，BGRX；由 resize_frame 维护
        self._final_cache = None      # final_only 模式下的最终掩码缓存（见 invalidate）

    def resize_frame(self, win_w, win_h):
        if (self.frame is None or self.frame.shape[1] != win_w
                or self.frame.shape[0] != win_h):
            self.frame = np.zeros((win_h, win_w, 4), np.uint8)
            self.frame[:, :, 3] = 255
            return True
        return False

    # ---------- 合成 ----------
    # **固定瓦片网格**（不是"按需切一刀"）：部分重采样与整帧重采样的相位不一样，
    # 直接按脏矩形切一刀重画，接缝处会出现一块颜色略有差异的方块 —— 涂笔刷、撤销、
    # 拖动折线时都会闪这种方块。按 256 的固定网格切，任何一个瓦片都只依赖
    # 「自己的屏幕矩形 + 视口」，于是**任意子集重画 == 整帧重画，逐像素相同**。
    TILE = 256

    def compose(self, rect=None):
        """重画 rect（屏幕坐标 (x0,y0,x1,y1)，None = 整窗）。"""
        win_h, win_w = self.frame.shape[:2]
        x0, y0, x1, y1 = (0, 0, win_w, win_h) if rect is None else rect
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(win_w, int(x1)), min(win_h, int(y1))
        if x1 <= x0 or y1 <= y0:
            return
        t = self.TILE
        for ty in range((y0 // t) * t, y1, t):
            for tx in range((x0 // t) * t, x1, t):
                self._compose_tile(tx, ty, min(tx + t, win_w), min(ty + t, win_h))

    def _compose_tile(self, x0, y0, x1, y1):
        tw, th = x1 - x0, y1 - y0
        # 目标屏幕矩形 -> 原图矩形（整数像素，向外取整保证盖满）。
        # 用 **img 的实际形状**裁剪而不是 vp.img_w/h —— 两者本应一致，但万一不一致
        # （比如换图时视口没跟着更新），按实际形状裁最多是画面不对，按视口裁会切出
        # 一个空数组让 cv2 直接抛断言。
        ih, iw = self.img.shape[:2]
        ix0f, iy0f = self.vp.screen_to_image(x0, y0)
        ix1f, iy1f = self.vp.screen_to_image(x1, y1)
        ix0 = max(0, int(np.floor(ix0f)))
        iy0 = max(0, int(np.floor(iy0f)))
        ix1 = min(iw, int(np.ceil(ix1f)))
        iy1 = min(ih, int(np.ceil(iy1f)))
        if ix1 <= ix0 or iy1 <= iy0:
            self.frame[y0:y1, x0:x1, :3] = 0            # 图外：黑
            return

        up = self.vp.zoom >= 1.0
        base = self._resize_rgb_bgr(self.img, (ix0, iy0, ix1, iy1), tw, th,
                                    cv2_interp_name("linear" if up else "area"))
        if not self.show_layers:
            self._blit(base, (x0, y0, x1, y1))
            return

        covs = {}
        if self.final_only:
            # 最终掩码每次现算要 20MB 分配 + 20M 次按位运算（~20ms），而它只在
            # 「按了 Shift+T」和「改了掩码」时才变 —— 缓存住，改动时由 invalidate() 打掉。
            if self._final_cache is None:
                self._final_cache = self.store.final_mask()
            covs["final"] = self._resize_mask(self._final_cache,
                                              (ix0, iy0, ix1, iy1), tw, th, up)
        else:
            for key, arr in (("pred", self.store.pred), ("add", self.store.add),
                             ("del", self.store.dele)):
                covs[key] = self._resize_mask(arr, (ix0, iy0, ix1, iy1), tw, th, up)
        blended = blend_layers(base, covs)
        self._blit(blended, (x0, y0, x1, y1))

    def invalidate(self):
        """掩码改了（笔刷/撤销/调阈值）之后必须调一次 —— 目前只用来清最终掩码缓存。"""
        self._final_cache = None

    def _blit(self, bgr, rect):
        """bgr: (th, tw, 3) uint8，已经是 BGR 序 —— 直接写进帧缓冲的前三通道。"""
        x0, y0, x1, y1 = rect
        self.frame[y0:y1, x0:x1, :3] = bgr

    def _resize_rgb_bgr(self, img, irect, tw, th, interp):
        """原图 ROI -> 窗口大小的 BGR（帧缓冲的内存序）。"""
        ix0, iy0, ix1, iy1 = irect
        sub = img[iy0:iy1, ix0:ix1]
        if not sub.flags["C_CONTIGUOUS"]:
            sub = np.ascontiguousarray(sub)
        small = cv2_resize(sub, (tw, th), interp)
        # RGB -> BGR：在这里做一次的代价只有窗口大小（几 MB），
        # 比让整张 5472x3648 的图常驻两份（RGB 一份 + BGR 一份）省 60MB
        return np.ascontiguousarray(small[:, :, ::-1])

    def _resize_mask(self, mask, irect, tw, th, up):
        ix0, iy0, ix1, iy1 = irect
        sub = mask[iy0:iy1, ix0:ix1]
        if not sub.flags["C_CONTIGUOUS"]:
            sub = np.ascontiguousarray(sub)
        # INTER_AREA 出来的就是 0~255 的覆盖率；放大用 NEAREST 保持边界是原始像素
        return cv2_resize(sub, (tw, th),
                          cv2_interp_name("nearest" if up else "area"))


_INTERP = {}


def cv2_interp_name(name):
    """插值方式名 -> cv2 常量。**延迟 import cv2**：这张模块是纯 numpy 的，
    笔刷/撤销/坐标换算的自检不该被"半秒的 cv2 导入"和它的 DLL 拖累。"""
    v = _INTERP.get(name)
    if v is None:
        import cv2
        v = _INTERP[name] = {"area": cv2.INTER_AREA, "nearest": cv2.INTER_NEAREST,
                             "linear": cv2.INTER_LINEAR}[name]
    return v


def cv2_resize(arr, size, interp):
    import cv2
    return cv2.resize(arr, size, interpolation=interp)


def blend_layers(base_bgr: np.ndarray, covs: dict) -> np.ndarray:
    """把覆盖率图层按 **蓝 > 绿 > 红** 的优先级叠到 base 上（定点混合）。

    base_bgr: (h, w, 3) uint8（BGR）。covs: {'pred'/'add'/'del' 各 (h,w) uint8 覆盖率}
    或 {'final': ...}（只看最终掩码时）。
    优先级与 `MaskStore.final_mask()` 的公式 `(pred|add)&~del` 一致 —— 显示和存盘必须是同一个口径，
    否则会出现"屏幕上看着是红的，存出来那块没了"。
    """
    h, w = base_bgr.shape[:2]
    rgb = np.zeros((h, w, 3), np.uint8)          # 名字叫 rgb，其实存的是 BGR（对齐帧缓冲）
    al = np.zeros((h, w), np.uint8)
    if "final" in covs:
        m = covs["final"] > 0
        rgb[m] = COLOR_PRED_BGR
        al[m] = _alpha_of(covs["final"][m])
    else:
        for key, color in (("pred", COLOR_PRED_BGR),
                           ("add", COLOR_ADD_BGR),
                           ("del", COLOR_DEL_BGR)):
            cov = covs[key]
            m = cov > 0
            if not m.any():
                continue
            rgb[m] = color
            al[m] = _alpha_of(cov[m])            # 后画的覆盖先画的 = 优先级
    a = al[:, :, None].astype(np.uint16)
    out = (base_bgr.astype(np.uint16) * (255 - a) + rgb.astype(np.uint16) * a) >> 8
    return out.astype(np.uint8)


def _alpha_of(cov):
    """覆盖率(0~255) -> 显示 alpha(0~255)：整体乘 ALPHA。"""
    return ((cov.astype(np.uint16) * ALPHA255) // 255).astype(np.uint8)


# ---- 折线 ----
def polyline_length(pts) -> float:
    """折线总长（原图像素）。"""
    if len(pts) < 2:
        return 0.0
    a = np.asarray(pts, float)
    return float(np.hypot(*(np.diff(a, axis=0).T)).sum())


def nearest_on_polyline(pts, x, y):
    """在折线上找离 (x,y) 最近的位置。

    返回 `(seg_index, t, dist, point)`：第 `seg_index` 段上参数 t∈[0,1] 处的点，
    以及该点到 (x,y) 的距离。**插入控制点就用它** —— 用户要的是"按 Alt 点一下，
    新点插到离点击最近的那一段里"，落在段中间就插进那两个控制点之间，
    超出两端自然退化成"接到最近的那个端点"（t 被夹到 0 或 1），一个规则管所有情况。
    """
    p = np.asarray(pts, float)
    if len(p) == 1:
        d = float(np.hypot(p[0, 0] - x, p[0, 1] - y))
        return 0, 0.0, d, p[0].copy()
    best = None
    for i in range(len(p) - 1):
        ax, ay = p[i]
        bx, by = p[i + 1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 <= 1e-9 else ((x - ax) * dx + (y - ay) * dy) / L2
        t = float(min(1.0, max(0.0, t)))
        px, py = ax + t * dx, ay + t * dy
        d = float(np.hypot(px - x, py - y))
        if best is None or d < best[2]:
            best = (i, t, d, np.array([px, py]))
    return best
