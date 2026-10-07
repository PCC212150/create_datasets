"""把 RSML 折线 / labelme 标注画成二值分割掩码（真值 GT）。

统一规范：标注坐标是**原图分辨率**（RSML 控制点、labelme 的 points 都是），
画掩码时先用 `scale_points` 换算到目标画布，再调用 *_at 系列函数。

为什么不「在原图上画好再缩放」（旧做法）：
  - 省掉 2000 万像素级的画布与重采样：实测 0.199s/张 -> 0.001s/张；
  - 细线不会被最近邻降采样漏掉：原图的根线（MASK_LINE_WIDTH px）在 5.34 倍下采样后
    只剩 1~2px，最近邻会让部分段变细甚至断开；换算线宽后在目标画布上直接画就不会。
"""
import numpy as np
from PIL import Image, ImageDraw

from common.rsml_parse import Root


def target_line_width(width: float, from_size, to_size) -> int:
    """原图尺度的线宽 -> 目标画布上的线宽（按短边比例换算，至少 1px）。"""
    w0, h0 = from_size
    w1, h1 = to_size
    s = min(w1 / float(w0), h1 / float(h0))
    return max(1, int(round(width * s)))


def scale_points(points, from_size, to_size) -> list:
    """点列 (x, y) 从 from_size 画布换算到 to_size 画布，返回 [(x, y), ...]（float）。"""
    w0, h0 = from_size
    w1, h1 = to_size
    sx, sy = w1 / float(w0), h1 / float(h0)
    return [(x * sx, y * sy) for x, y in points]


def _canvas(size) -> tuple:
    """size=(w, h) 的黑底画布与画笔。"""
    w, h = int(size[0]), int(size[1])
    img = Image.new("L", (max(w, 1), max(h, 1)), 0)
    return img, ImageDraw.Draw(img)


def _to_bool(img) -> np.ndarray:
    return np.asarray(img, dtype=np.uint8) > 0


def draw_polylines_at(polylines, size, width: int = 1) -> np.ndarray:
    """在 size=(w, h) 画布上画折线（已是画布坐标），返回 bool (h, w)。"""
    img, draw = _canvas(size)
    for pts in polylines:
        if len(pts) < 2:
            continue
        draw.line([tuple(p) for p in pts], fill=255, width=int(width), joint="curve")
    return _to_bool(img)


def draw_polygons_at(polygons, size) -> np.ndarray:
    """填充多边形（茎横截面），已是画布坐标，多个取并集，返回 bool (h, w)。"""
    img, draw = _canvas(size)
    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon([tuple(p) for p in poly], fill=255)
    return _to_bool(img)


def draw_rects_at(rects, size) -> np.ndarray:
    """填充矩形 (x0, y0, x1, y1)（检查范围），已是画布坐标，返回 bool (h, w)。"""
    img, draw = _canvas(size)
    for x0, y0, x1, y1 in rects:
        draw.rectangle([x0, y0, x1, y1], fill=255)
    return _to_bool(img)


# ------------------------- 以下为「原图分辨率」的兼容入口 -------------------------

def draw_mask_from_roots(roots, image_size, width: int = 5) -> np.ndarray:
    """在原图尺寸 (w, h) 画根系二值掩码，返回 bool (h, w)。

    多根重叠处取并集；忽略点数 < 2 的根；越界线段由 PIL 自动裁剪。
    新流程请改用 scale_points + draw_polylines_at（更快且不掉线）。
    """
    return draw_polylines_at([r.points for r in roots if len(r.points) >= 2],
                             image_size, width)


def draw_mask_from_polylines(polyline_groups, image_size, width: int = 5) -> np.ndarray:
    """draw_mask_from_roots 的通用版本：直接给若干折线（每根一段列表）。"""
    roots = [Root(points=pts) for pts in polyline_groups if len(pts) >= 2]
    return draw_mask_from_roots(roots, image_size, width)
