r"""标注工具的 IO 层：预测封装 / 缓存 / 导出 labelme json 与 overlay / 原图硬链接。

**本文件不 import Qt**，也不 import annotate_root —— `--prefetch`（只跑预测填缓存）
和 `--selftest` 都直接调这里，不需要开界面。

## 目录约定（都在本文件所在目录下）

    pictures\           原图（输入）
    model\<模型名>\      权重（输入，默认取最新的）
    cache\<模型名>\      中间状态：掩码 PNG + 概率图 + 元信息（可删，删了只是重新预测）
    datasets\           交付物：<名>.jpg + <名>.json + overlay\<名>_overlay.jpg

**缓存必须按模型名分目录**：掩码是"某个模型在某张图上的输出"，换模型后混用旧缓存
会静默给出错误掩码 —— 那比慢几分钟糟得多。

## 预测口径

一律走 `_vendor` 里冻结的 root_model 实现（`common.predict.predict`），**不复制一份**：
检查范围限定、滞回阈值、跨尺度上采样这些口径只在那边写了一次，工具这边再抄一遍
就等于埋一个"工具里标出来的掩码和训练时算出来的真值不是一套东西"的坑。

`predict()` 同时会把 `prob_target`（**已在概率层清掉检查框之外**的根系概率，模型分辨率）
返回给我们缓存下来 —— 于是「预测松紧」调节不用重跑模型，几十毫秒就能按新阈值重算掩码。
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# 冻结的 root_model 快照。**只从这里 import**（不做"优先找旁边的 root_model"的双路径：
# 那会让"本地跑得好、发给别人却挂"的差异永远不被暴露）。
sys.path.insert(0, str(ROOT / "_vendor"))

import config  # noqa: E402
from common import ckpt, image_io, predict as predict_mod  # noqa: E402
from common.labelme_export import mask_to_polygons, write_labelme_json  # noqa: E402

PICTURES_DIR = ROOT / "pictures"
MODEL_ROOT = ROOT / "model"
CACHE_ROOT = ROOT / "cache"
DATASETS_DIR = ROOT / "datasets"
OVERLAY_SUBDIR = "overlay"          # 见 save_annotation 里的说明：必须在子目录
# 像素级掩码的目录（R=root G=stem B=check，0/255 的 PNG）。
# **必须和 root_model/common/dataset.py 里的 MASK_SUBDIR 一致** —— 训练那边就是按
# `<data_dir>/masks/<名>.png` 找它的。也必须在子目录：`separate_dataset` 按文件名主干分组，
# 放同层会被当成"同名第二张图"；而且它按**组**拷贝，子目录不会被带走 ——
# 所以 separate_dataset 里额外加了一句专门拷 masks\（见那个工具的 readme）。
MASK_SUBDIR = "masks"

# 导出 GT 多边形的参数。**不能用 mask_to_polygons 的默认值**（approx_px=2.0/min_area=400）：
# 实测在真预测掩码上，默认值丢掉 10 个连通块里的 3 个、4.75% 的像素 ——
# 一根 40px 长的细根面积才 ~400px²，正好卡在阈值上被整根丢掉。
# 换成 min_area=100 / approx_px=1.0 后覆盖 98.76%，顶点数 708（labelme 拖得动）。
ROOT_POLY_APPROX_PX = 1.0
ROOT_POLY_MIN_AREA = 100.0

# 预测松紧的调节范围（滞回低阈值）。高阈值固定 0.5（与训练/测试口径一致）。
#
# **阈值越低，掩码越粗**（低阈值把弱响应也拉进来）。实测同一张图：
#     0.02 -> 54228 px（最粗）   0.10 -> 44249 px（项目默认，与训练/测试口径一致）
#     0.30 -> 34282 px          0.50 -> 27022 px（最细 = 只剩 0.5 以上的强响应）
# 所以界面上的"松/紧"= 阈值的低/高，别弄反。
#
# **下限取 0.02 而不是 0**：0 在底层是"关掉滞回"的特例（走纯 0.5 阈值），
# 于是 0.02 -> 0.00 会从**最粗**直接跳到**最细**（54228 -> 27022 px）——
# 用户往回拧一点，掩码突然瘦了一圈，像是坏了。截在 0.02 保证全程单调。
THRESH_MIN, THRESH_MAX, THRESH_STEP = 0.02, 0.60, 0.02

DEFAULT_LOW_THRESH = float(config.PRED_LOW_THRESHOLD)

# 折线加宽成带时用的宽度（原图像素）。只在「掩码缺了」的地方起作用 ——
# 掩码有根的地方，轮廓还是掩码自己的真实宽度。界面上可调（见 annotate_root 的带宽旋钮）。
ROOT_POLY_WIDTH_DEFAULT = float(config.MASK_LINE_WIDTH)      # 10px，与旧折线口径同尺子
ROOT_POLY_WIDTH_MIN, ROOT_POLY_WIDTH_MAX, ROOT_POLY_WIDTH_STEP = 2.0, 80.0, 1.0

# 工具级设置（界面偏好）存这儿。**放在根目录而不是 cache\**：cache 是"可删的中间状态"，
# 而带宽是用户调出来的偏好，不该因为清缓存就丢。文件很小，手动删掉就是回默认值。
SETTINGS_PATH = ROOT / "settings.json"


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_settings(d: dict) -> None:
    """写界面偏好。**合并写**，不是整份覆盖 —— 每个调用方只管自己那几个键。

    整份覆盖踩过：宽带旋钮写 `{"poly_width": w}`、换模型写 `{"model": name}`，
    两边都按"我知道全部内容"来写，结果互相把对方的键抹掉（带宽突然回默认 10、
    或者下次启动换了模型）。合并写之后，加新键也不用再来改每个老的调用点。
    写不进去不能让工具崩（只读目录、盘满都可能）—— 顶多是这次的选择下回不记得。
    """
    try:
        body = load_settings()
        body.update(d)
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=1)
        os.replace(tmp, SETTINGS_PATH)
    except Exception as e:
        print(f"[提示] 界面设置没写进去（{e}），这次的选择下次启动不会记得")


# ---------------------------------------------------------------- 目录与扫描
def list_images(pictures_dir=None, datasets_dir=None) -> list:
    r"""图片清单 = `pictures\` 里的（**还没做的**）∪ `datasets\` 里的（**做完的**）。

    做完的图在保存时会从 `pictures\` 移走（见 save_annotation 的 move_pictures）——
    于是"pictures 里还剩几张 = 还有几张没做"，在资源管理器里一眼就能数。
    代价是两边都得扫：只扫 pictures 的话，标好的图会从工具列表里消失，想回去改都点不到。

    同名的以 `pictures\` 里的为准（说明还没做完，源图还在原处）。
    """
    exts = config.IMAGE_EXTS
    pics = Path(pictures_dir or PICTURES_DIR)
    dst = Path(datasets_dir or DATASETS_DIR)
    found = {}
    for d in (dst, pics):                    # 先 datasets 后 pictures -> 同名的被覆盖成 pictures 那份
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                found[p.stem] = p
    return [found[k] for k in sorted(found)]


def done_stems(datasets_dir=None) -> set:
    """已经导出过标注的图（判定 = `datasets\\<名>.json` 存在）。"""
    d = Path(datasets_dir or DATASETS_DIR)
    if not d.is_dir():
        return set()
    return {p.stem for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".json"}


def resolve_model(model_arg=None):
    """返回 (权重路径列表, 名称列表)；model_arg 为空时取 model\\ 下最新的一版。"""
    return ckpt.resolve_pths(model_arg, root=MODEL_ROOT)


def cache_dir(model_name: str, cache_root=None) -> Path:
    d = Path(cache_root or CACHE_ROOT) / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- 预测
def predict_one(model, img: np.ndarray, meta: dict, device, tile: int = 0):
    """对一张原图跑三类预测，返回可以直接缓存/展示的 dict。

    返回的 `pred` 是 **uint8 0/255**（不是 bool）—— 下游的笔刷、缩放、缓存全按 uint8 走，
    中间少一次转换就少一处不一致。

    顺带把茎多边形算好（`mask_to_polygons` 在全分辨率上要几十毫秒），
    主线程只装数据。
    """
    res = predict_mod.predict(model, img, max_side=meta.get("size") or config.MAX_SIDE,
                              stride=config.STRIDE, device=device, tile=tile)
    masks = res["masks"]
    stem_polys = ()
    if len(masks) > predict_mod.CH_STEM and masks[predict_mod.CH_STEM] is not None:
        stem_polys = [tuple(p) for p in mask_to_polygons(
            masks[predict_mod.CH_STEM], approx_px=ROOT_POLY_APPROX_PX,
            min_area=ROOT_POLY_MIN_AREA)]
    return {
        "pred": (res["mask_counted"].astype(np.uint8) * 255),
        "prob": np.ascontiguousarray(res["prob_target"], dtype=np.float32),
        "check_box": res["check_box"],
        "check_ok": bool(res["check_ok"]),
        "root_ok": bool(res.get("root_ok", True)),
        "stem_polys": stem_polys,
        "low_thresh": DEFAULT_LOW_THRESH,
    }


def mask_from_prob(prob: np.ndarray, check_box, check_ok: bool, out_hw,
                   low_thresh: float, high: float = 0.5):
    """用缓存的概率图 + 新阈值重算根系掩码（不重跑模型）。

    `prob` 是 `predict()` 给的 `prob_target`：**已经在概率层把检查框之外清成 0**，
    所以这里只要做「上采样 -> 滞回阈值 -> 与检查框精确求交」三步，
    与 `predict()` 内部的顺序完全一致（顺序很重要：先清零再阈值，否则框外的弱响应
    会把框内两段连通起来）。

    返回 (掩码 uint8, 警告文本 or None)。
    """
    import torch
    h0, w0 = int(out_hw[0]), int(out_hw[1])
    pt = torch.from_numpy(np.ascontiguousarray(prob))[None, None]
    if low_thresh and low_thresh > 0:
        m = image_io.prob_to_orig_mask_hysteresis(pt, w0, h0, high=high,
                                                  low=float(low_thresh), channel=0)
    else:
        m = image_io.prob_to_orig_mask(pt, w0, h0, threshold=high, channel=0)
    warn = None
    if check_ok and check_box:
        x0, y0, x1, y1 = (int(v) for v in check_box)
        exact = np.zeros((h0, w0), bool)
        exact[y0:y1, x0:x1] = True
        m &= exact
        # 兜底：低阈值把整片检查区淹掉时的保护，与 predict() 里那条同一判据
        # （背景概率有底噪的模型，低阈值会给出"整张全是根"）。手动调松紧时更容易碰到，
        # 这里只报警不拦截 —— 阈值是用户自己拧的，拦下来反而莫名其妙。
        area = float((x1 - x0) * (y1 - y0))
        if area > 0:
            cov = float(m[y0:y1, x0:x1].sum()) / area
            if cov > config.PRED_MAX_ROOT_RATIO:
                warn = (f"根占检查框 {cov:.1%}（正常 1%~5.5%）：阈值可能太松，"
                        f"这个模型的背景概率可能有底噪")
    return (m.astype(np.uint8) * 255), warn


# ---------------------------------------------------------------- 缓存
def cache_paths(cdir: Path, stem: str) -> dict:
    cdir = Path(cdir)
    return {"masks": cdir / f"{stem}_masks.png",
            "prob": cdir / f"{stem}_prob.npy",
            "meta": cdir / f"{stem}_meta.json"}


def load_cache(cdir, stem, model_name: str):
    """读某张图的缓存；三件缺一或模型名对不上就返回 None（当作没缓存，重新预测）。"""
    cp = cache_paths(Path(cdir), stem)
    missing = [k for k, p in cp.items() if not p.exists()]
    if missing:
        # 一张都没做过的图是常态（不吭声）；**做了一半**才是异常，
        # 但要吭声 —— 否则"为什么这张图每次都要重新预测"会完全无从查起。
        if len(missing) < len(cp):
            print(f"[提示] {stem} 的缓存不完整（缺 {', '.join(missing)}），这张图会重新预测")
        return None
    try:
        with open(cp["meta"], encoding="utf-8") as f:
            meta = json.load(f)
        if model_name and meta.get("model") not in (None, model_name):
            return None
        from PIL import Image
        with Image.open(cp["masks"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        pred = np.ascontiguousarray(arr[:, :, 0])
        add = np.ascontiguousarray(arr[:, :, 1])
        dele = np.ascontiguousarray(arr[:, :, 2])
        prob = np.load(cp["prob"])
    except Exception as e:                      # 缓存损坏不该让工具起不来
        print(f"[警告] 缓存读取失败（{stem}）：{e}；将重新预测")
        return None
    return {"pred": pred, "add": add, "dele": dele, "prob": prob, "meta": meta}


def save_cache(cdir, stem, store, meta: dict):
    """写缓存：R=pred G=add B=del 打包成一张 PNG（**直接能肉眼看**，出问题时好排查），
    概率图单独存 .npy（调阈值要用），其余元信息进 json。"""
    from PIL import Image
    cdir = Path(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    cp = cache_paths(cdir, stem)
    packed = np.dstack([store.pred, store.add, store.dele])
    tmp = cp["masks"].with_suffix(".png.tmp")
    # format 必须显式给：PIL 靠扩展名猜格式，`.png.tmp` 它认不出来
    Image.fromarray(packed, "RGB").save(tmp, format="PNG", compress_level=1)
    os.replace(tmp, cp["masks"])                # 原子落盘：写一半崩了不会留下坏缓存
    if meta.get("prob") is not None:
        np.save(cp["prob"], meta["prob"])
    body = {k: v for k, v in meta.items() if k != "prob"}
    with open(cp["meta"], "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 折线 -> 多边形
def _line_fingerprint(polylines) -> str:
    """折线集合的指纹，用来判断"分区要不要重算"。"""
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    for line in polylines:
        h.update(f"{len(line)}|".encode())
        for x, y in line:
            h.update(f"{x:.1f},{y:.1f};".encode())
    return h.hexdigest()


def _bbox_of(line, margin: int, w: int, h: int):
    pts = np.asarray(line, np.float32)
    x0 = max(0, int(np.floor(pts[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - margin)
    x1 = min(w, int(np.ceil(pts[:, 0].max())) + margin)
    y1 = min(h, int(np.ceil(pts[:, 1].max())) + margin)
    return x0, y0, x1, y1


def _sample_polyline(pts, step=2.0):
    """沿折线等距采样，返回 [(x, y, 累计弧长), ...]（含首尾）。"""
    out = [(float(pts[0][0]), float(pts[0][1]), 0.0)]
    acc = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        seg = float(np.hypot(bx - ax, by - ay))
        if seg <= 1e-9:
            continue
        n = max(1, int(np.ceil(seg / step)))
        for k in range(1, n + 1):
            t = k / n
            acc += seg / n
            out.append((ax + (bx - ax) * t, ay + (by - ay) * t, acc))
        # 上面按等分走，实际步长可能略小于 step；用等分足够（只用来分组）
    return out


class RootPolygonBuilder:
    r"""把「掩码 + 折线 + 带宽」变成**每条折线一个**的多边形（保存与实时预览共用同一份实现）。

    口径（用户 2026-10-07 定）：

      · 轮廓来自**修好的掩码**（真实宽度，不是膨胀出来的带），折线只负责"把掩码分成几条"；
      · 碰掩码断开/缺失的段，用宽 `width` 的带补上（保证每条折线一定有多边形）；
      · 掩码里**没有折线经过**的块丢掉 —— 这样"多边形数 == 折线数"才严格成立；
      · 带宽在界面上可调、实时看效果。

    「哪块掩码属于哪条折线」用**最近距离分区**（Voronoi）：把每条折线的中心线画进一张
    降采样的标签图，再用距离变换把每个背景像素判给离它最近的那条折线。

    为什么要降采样：全分辨率（2000 万像素）做距离变换要一两秒、还得吃 160MB，
    而分区线偏几个像素只影响两条根**交界处**的归属；轮廓本身是在**全分辨率**上
    按分区切出来再抽的，所以形状精度不受影响。

    分区只跟折线有关（跟掩码、带宽都无关），所以缓存住 —— 调带宽时只重跑
    "切掩码 + 补带 + 抽轮廓"，这才做得到实时。
    """

    def __init__(self, shape, scale: int = 4):
        self.h, self.w = int(shape[0]), int(shape[1])
        self.scale = max(1, int(scale))
        self._key = None
        self._labels_small = None

    # ---- 分区 ----
    def prepare(self, polylines):
        key = _line_fingerprint(polylines)
        if key == self._key and self._labels_small is not None:
            return
        import cv2
        s = self.scale
        sw, sh = max(1, self.w // s), max(1, self.h // s)
        lab = np.zeros((sh, sw), np.int32)
        for i, line in enumerate(polylines):
            if len(line) < 2:
                continue
            pts = np.round(np.asarray(line, np.float32) / s).astype(np.int32)
            # thickness=2：太细的线在 1/4 尺度上可能整条消失（短折线尤其容易），
            # 那样这条折线就分不到任何掩码了
            cv2.polylines(lab, [pts], False, int(i + 1), thickness=2,
                          lineType=cv2.LINE_8)
        if lab.max() > 0:
            from scipy import ndimage
            _, inds = ndimage.distance_transform_edt(lab == 0, return_indices=True)
            lab = lab[tuple(inds)]            # 每个像素 -> 离它最近的那条折线的编号
        self._labels_small = lab.astype(np.uint16)
        self._key = key

    def _labels_local(self, x0, y0, x1, y1):
        """把分区图裁到局部并按需放大（面积只有一条折线的框，很便宜）。"""
        import cv2
        s = self.scale
        sx0, sy0 = x0 // s, y0 // s
        sx1 = min(self._labels_small.shape[1], int(np.ceil(x1 / s)))
        sy1 = min(self._labels_small.shape[0], int(np.ceil(y1 / s)))
        sub = self._labels_small[sy0:max(sy0 + 1, sy1), sx0:max(sx0 + 1, sx1)]
        up = cv2.resize(sub, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        # 放大是按整块复制的，起点会有零点几个像素的偏移；分区线本身只影响交界归属，
        # 这点偏移无所谓（轮廓是在掩码上切的，不来自这里）
        return up

    # ---- 主入口 ----
    def polygons(self, mask, polylines, width, approx_px=None, min_area=None,
                 gap_min=None, close_r=None):
        """返回 (polygons, info)。

        polygons: 长度与「有效折线」相同，逐条对应；**每条是一个列表** ——
                  正常情况下只有一个多边形，但**切片被切成几块时会有多个**
                  （用户 2026-10-08 选定的口径："允许一条折线出多个多边形"）。
                  某条折线什么都没得到时该位是空列表 []。
        info:     统计（空折线数、被丢掉的碎块数/面积、每条的多边形数），给界面提示用。
        """
        approx_px = ROOT_POLY_APPROX_PX if approx_px is None else approx_px
        min_area = ROOT_POLY_MIN_AREA if min_area is None else min_area
        width = max(1, int(round(width)))
        # 门槛设成 0：**任何**没落进自己切片的段都补带。判据已经是逐点的，留下的
        # 都是真缺口 —— 没有「太短所以不补」的道理，那不补的就是丢掉的根。
        gap_min = 0.0 if gap_min is None else gap_min
        close_r = max(2, width // 4) if close_r is None else close_r
        lines = [list(l) for l in polylines if len(l) >= 2]
        info = {"n_lines": len(lines), "n_empty": 0, "n_dropped_blocks": 0,
                "dropped_area": 0, "width": width}
        if not lines:
            return [], info
        self.prepare(lines)
        m = np.ascontiguousarray(mask) > 0
        covered = np.zeros((self.h, self.w), bool)
        polys = []
        for i, line in enumerate(lines):
            group, raster, x0, y0, dropped = self._one(m, line, i + 1, width,
                                                       approx_px, min_area,
                                                       gap_min, close_r)
            polys.append(group)
            if not group:
                info["n_empty"] += 1
            info["n_dropped_blocks"] += dropped
            if raster is not None:
                covered[y0:y0 + raster.shape[0], x0:x0 + raster.shape[1]] |= raster
        info["per_line"] = [len(g) for g in polys]
        # 掩码里没有折线经过的块：数出来提醒用户（它们被丢掉了）
        orphan = m & ~covered
        if orphan.any():
            import cv2
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                orphan.astype(np.uint8), 8)
            big = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                   if stats[i, cv2.CC_STAT_AREA] >= min_area]
            info["n_dropped_blocks"] += len(big)
            info["dropped_area"] += int(sum(big))
        return polys, info

    def _one(self, mask, line, label, width, approx_px, min_area, gap_min, close_r):
        import cv2
        from common.gt_mask import draw_polylines_at
        x0, y0, x1, y1 = _bbox_of(line, margin=width * 2 + 16, w=self.w, h=self.h)
        lw, lh = x1 - x0, y1 - y0
        if lw <= 0 or lh <= 0:
            return None, None, x0, y0, 0
        lab = self._labels_local(x0, y0, x1, y1)
        region = (mask[y0:y1, x0:x1]) & (lab == label)

        # 沿折线找"掩码缺了"的连续段。
        #
        # **判据是逐点**（只看折线正中那一点），而不是"±宽度/2 内有掩码就算有"。
        # 2026-10-07 实测踩到的坑：切片之间常有几像素宽的**窄缝**（多半来自 1/4 尺度
        # 分区图放大后的块状边界），大窗口会把窄缝判成"有掩码"→ 不补带 → 切片连不起来
        # → 最后"只留最大连通块"就把真实的一块根丢了（某图一条线 24540px 只留下 13730px，
        # 表现就是"折线没被自己的多边形包住"）。逐点判据下窄缝会被补掉，多边形成为
        # 一条**沿折线的连续走廊**，从根上保证折线一定在自己的多边形里。
        # 另留 2px 小窗口，免得折线贴着掩码边缘走时被逐像素的锯齿反复判成"缺"。
        pts = np.asarray(line, np.float32)
        r = 2
        samples = _sample_polyline(pts, step=2.0)
        band = np.zeros((lh, lw), bool)
        run = []
        run_start = 0.0
        # 末尾补一个哨兵，好把"折线最后一截也是缺的"这种情况收掉 ——
        # 少了它，一条**整条都缺**的折线（掩码里根本没有这根）就画不出带，
        # 那条折线会得到 None，破坏"每条折线一个多边形"
        for (px, py, arc) in samples + [(np.nan, np.nan, np.inf)]:
            if np.isnan(px):
                if run and arc - run_start >= gap_min:
                    band |= draw_polylines_at(
                        [[(qx - x0, qy - y0) for qx, qy in run]], (lw, lh), width)
                break
            lx, ly = int(round(px)) - x0, int(round(py)) - y0
            w0, w1 = max(0, lx - r), min(lw, lx + r + 1)
            h0, h1 = max(0, ly - r), min(lh, ly + r + 1)
            covered_here = (w1 > w0 and h1 > h0 and region[h0:h1, w0:w1].any())
            if not covered_here:
                if not run:
                    run_start = arc
                run.append((float(px), float(py)))
            elif run:
                if arc - run_start >= gap_min:
                    band |= draw_polylines_at(
                        [[(qx - x0, qy - y0) for qx, qy in run]], (lw, lh), width)
                run = []
        raster = region | band
        if not raster.any():
            return [], None, x0, y0, 0
        # 闭合细缝（比 close_r 还短的那些）
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1,) * 2)
        closed = cv2.morphologyEx(raster.astype(np.uint8), cv2.MORPH_CLOSE, k)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        if n <= 1:
            return [], None, x0, y0, 0
        # **保留所有面积够的连通块**（用户 2026-10-08 选定的口径）。
        # 原来只留最大的那块，结果把真实的一块根丢了：实测某图一条折线丢了 7458 px，
        # 而它离折线只有 12 px（折线是画了的，是多边形没盖住）—— 16 张 GT 合计少 4% 面积。
        # 代价是"一条折线 = 一个多边形"不再是硬约束：切片被切开时会出多个多边形。
        # 只丢真正的小碎片（< min_area，默认 100px² = 10×10px）。
        keep = np.zeros_like(closed, dtype=bool)
        dropped = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep |= (cc == i)
            else:
                dropped += 1
        if not keep.any():
            return [], None, x0, y0, dropped
        local = mask_to_polygons((keep.astype(np.uint8) * 255), approx_px=approx_px,
                                 min_area=min_area)
        group = [[(float(qx) + x0, float(qy) + y0) for qx, qy in poly]
                 for poly in local]
        return group, keep, x0, y0, dropped


# ---------------------------------------------------------------- 导出
def link_image(src: Path, dst: Path):
    """把原图放进 datasets\\：**优先硬链接**（同一分区时瞬间完成、不占额外空间，
    拷到服务器上自动变成普通文件），不支持就退回复制。

    返回实际用的方式（"硬链接"/"复制"），出问题时能一眼看出来。
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return "已存在"
        except OSError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
        return "硬链接"
    except OSError:
        shutil.copy2(src, dst)
        return "复制"


def build_mask_png(final_mask) -> np.ndarray:
    """最终掩码 -> 一张 uint8 (h, w)：**黑底白条，只有 root**（0=背景，255=根）。

    这是**训练直接吃的那份 root 真值**（root_model 优先读它，见 dataset.py 的
    load_root_mask_png）。和 json 里那套多边形的区别：这个是**原始分辨率上的像素**，
    没有"把多边形在目标分辨率上重新栅格化"的量化损失，也没有"没画折线的根被当成
    背景"的问题 —— 它就是你在阶段 1 一笔一笔改出来的那份掩码本身。

    **只存 root**（用户 2026-10-07 定）：root 是人一笔笔改出来的、值得逐像素保真；
    茎和检查框本来就是模型预测的、人没动过，训练时继续走 json 那条路就够了。
    """
    return (np.asarray(final_mask) > 0).astype(np.uint8) * 255


def save_mask_png(datasets_dir, stem, arr) -> Path:
    """把 root 掩码落到 `datasets\\masks\\<名>.png`（单通道 8 位，原子写）。"""
    d = Path(datasets_dir) / MASK_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}.png"
    tmp = p.with_suffix(".png.tmp")
    from PIL import Image
    Image.fromarray(arr, "L").save(tmp, format="PNG", compress_level=1)
    os.replace(tmp, p)
    return p


def make_overlay_jpg(img: np.ndarray, final_mask: np.ndarray, polylines=(),
                     stem_polygons=(), check_box=None, quality: int = 88) -> np.ndarray:
    """原图 + 最终根系掩码(红) + 折线(黄) + 茎(橙) + 检查框(绿) —— 给人看的验收图。

    颜色口径与 `inference.py:make_overlay` 保持一致（根红/茎橙/框绿），
    这样工具里的图和推理输出可以并排比对。折线用**黄色**：它是这张图独有的东西
    （推理输出里没有），黄色在红/橙/绿里也不会混。
    """
    from PIL import Image, ImageDraw
    out = img.copy()
    if stem_polygons:
        _blend_mask(out, _polys_to_mask(stem_polygons, img.shape[1], img.shape[0]),
                    (255, 165, 0))                      # 茎：实心多边形
    _blend_mask(out, final_mask > 0, (255, 0, 0))       # 根：最终掩码
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    lw = max(2, int(min(img.shape[:2]) * 0.0018))
    for pts in polylines:
        if len(pts) >= 2:
            d.line([(float(x), float(y)) for x, y in pts], fill=(255, 255, 0), width=lw,
                   joint="curve")
    if check_box is not None:
        x0, y0, x1, y1 = (float(v) for v in check_box)
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0),
                    width=max(2, int(min(img.shape[:2]) * 0.004)))
    return np.asarray(im)


def _polys_to_mask(polys, w, h):
    from PIL import Image, ImageDraw
    im = Image.new("1", (int(w), int(h)), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        if len(p) >= 3:
            d.polygon([(float(x), float(y)) for x, y in p], fill=1)
    return np.asarray(im, dtype=bool)


def _blend_mask(img: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45):
    """就地混合（与 inference.py 的 overlay 同口径 alpha=0.45）。"""
    if mask is None or not mask.any():
        return
    c = np.asarray(color, np.float32)
    img[mask] = (img[mask].astype(np.float32) * (1 - alpha) + c * alpha).astype(np.uint8)


def save_annotation(stem: str, image_name: str, img_shape, final_mask, polylines,
                    stem_polygons=(), check_box=None, drop_stem=False, drop_check=False,
                    src_image: Path = None, datasets_dir=None,
                    poly_width: float = None, builder=None,
                    move_pictures: bool = True, save_mask: bool = True):
    """写 `datasets\\<名>.json` + `datasets\\overlay\\<名>_overlay.jpg` + 原图。

    json 里的四种形状（labelme 6.x，与既有标注**同格式**，labelme 能直接打开）：
        root  polygon         最终掩码的轮廓 —— 这是**掩码真值**（新口径：根的真实轮廓）
        root  linestrip       用户画的折线 —— 代表**根系长度**，供以后算根长用
        stem  polygon         模型预测的茎（按 D 丢弃则不写）
        check_background rectangle  模型预测的检查框（按 K 丢弃则不写）

    **overlay 必须放子目录**：`datasets\\<名>_overlay.jpg` 会被 `tool\\separate_dataset`
    按文件名主干当成一个**独立的"只有图没有标注"的组**参与划分（它只按 stem 分组），
    也可能被别的扫描器当第二张图；放进子目录后 `iterdir()` 不递归，没人会看见它。

    返回写入的文件列表（供控制台打印）。
    """
    d = Path(datasets_dir or DATASETS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    h, w = int(img_shape[0]), int(img_shape[1])
    # **每条折线一个多边形**（口径见 RootPolygonBuilder）：轮廓来自掩码、缺处用
    # 宽 poly_width 的带补上。不是"把整张掩码抽成一个多边形"。
    b = builder if builder is not None else RootPolygonBuilder(img_shape)
    polys, poly_info = b.polygons(final_mask, polylines,
                                  ROOT_POLY_WIDTH_DEFAULT if poly_width is None
                                  else poly_width)
    # polys 是**逐折线分组**的（一条折线可能出多个多边形，见 polygons 的说明）→ 摊平
    root_polys = [tuple(p) for group in polys for p in group]
    lines = [tuple(p) for p in polylines if len(p) >= 2]
    json_path = d / f"{stem}.json"
    tmp = json_path.with_suffix(".json.tmp")
    write_labelme_json(tmp, image_height=h, image_width=w, polylines=lines,
                       root_polygons=root_polys, image_path=image_name,
                       check_box=None if drop_check else check_box,
                       stem_polygons=() if drop_stem else stem_polygons)
    # write_labelme_json 按输出名推 imagePath，这里必须显式给**原图文件名**：
    # labelme 靠它在 json 同级目录找图，而 datasets\\ 里的图是硬链接过来的那一份。
    os.replace(tmp, json_path)

    saved = [json_path.name]
    # ---- 像素级 root 掩码（训练直接吃的那份）----
    if save_mask:
        mp = save_mask_png(d, stem, build_mask_png(final_mask))
        saved.append(f"{MASK_SUBDIR}/{mp.name}")
    dst_img = None
    if src_image is not None:
        dst_img = d / image_name
        how = link_image(src_image, dst_img)
        saved.append(image_name)
        # ---- 把 pictures\ 里那份移走 ----
        # 「移走」= 删掉源路径那份：datasets\ 里已经是同一份数据了
        # （硬链接 = 同一个 inode；复制 = 独立的一份，早就落盘了）。
        # 这么做的目的只有一个：**pictures 里还剩几张 = 还有几张没做**，
        # 在资源管理器里一眼能数，不用开工具。
        src = Path(src_image)
        if move_pictures and src.parent != d:
            try:
                # 删之前确认 datasets 里那份真的在、且大小一致 —— 宁可这次不删
                if dst_img.exists() and dst_img.stat().st_size == src.stat().st_size:
                    src.unlink()
                    saved.append("pictures 里的原图已移除")
                else:
                    saved.append("[警告] datasets 里的原图不完整，pictures 那份先留着")
            except OSError as e:
                saved.append(f"[警告] 原图没能移除（{e}），pictures 那份还在")
    # 读图的路径要用 **datasets 里那份**：上面刚把源图移走（删掉）了，
    # 再读 src_image 就是 FileNotFoundError —— json 和原图已经写下去了，overlay 却没了，
    # 而且异常会一路冒到界面上弹"保存失败"。dst_img 才是此刻的权威副本。
    read_from = dst_img if dst_img is not None else src_image
    img = image_io.load_rgb(read_from) if read_from is not None else None
    if img is not None:
        ov = make_overlay_jpg(img, final_mask, polylines=lines,
                              stem_polygons=() if drop_stem else stem_polygons,
                              check_box=None if drop_check else check_box)
        ov_dir = d / OVERLAY_SUBDIR
        ov_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        ov_path = ov_dir / f"{stem}_overlay.jpg"
        Image.fromarray(ov).save(ov_path, quality=88)
        saved.append(f"{OVERLAY_SUBDIR}/{ov_path.name}")
    return {"json": json_path, "n_root_poly": len(root_polys), "n_polyline": len(lines),
            "saved": saved, "poly_info": poly_info,
            # 原图现在的真实位置：pictures 里那份可能刚被移走了，
            # 调用方（工具）要拿它更新手里的路径，否则下一次保存就找不到源图了
            "image_path": dst_img if dst_img is not None else src_image}


def meta_for_save(model_name, img_shape, stem, image_name, low_thresh,
                  check_box, check_ok, stem_polys, drop_stem, drop_check, polylines,
                  prob=None, poly_width=None):
    """攒一份缓存用的元信息（与 load_cache 读回的字段一一对应）。"""
    return {
        "model": model_name, "image": image_name, "stem": stem,
        "h": int(img_shape[0]), "w": int(img_shape[1]),
        "low_thresh": float(low_thresh), "check_box": list(check_box) if check_box else None,
        "poly_width": (float(poly_width) if poly_width is not None
                       else float(ROOT_POLY_WIDTH_DEFAULT)),
        "check_ok": bool(check_ok), "stem_polys": [list(map(list, p)) for p in stem_polys],
        "drop_stem": bool(drop_stem), "drop_check": bool(drop_check),
        "polylines": [list(map(list, p)) for p in polylines],
        "prob": prob,
    }
