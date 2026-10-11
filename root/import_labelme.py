r"""把 labelme 里改过的标注**反向导入**回工具（工具才看得见）。

    python import_labelme.py                    # 默认导工具自己的 datasets\
    python import_labelme.py --from <目录> ...  # 也可以从别处导
    python import_labelme.py --dry-run          # 先看会做什么，不动磁盘
    python import_labelme.py --force            # 缓存比 json 新时也强行覆盖

## 为什么需要它

工具打开一张图时，折线/掩码**只从** `cache\<模型>\<stem>_masks.png` + `_meta.json` 读，
**从不读 `datasets\<stem>.json`**（那是工具导出给 labelme 看、给训练用的产物）。
所以你在 labelme 里改完，工具里一点都看不到；**更糟的是**工具里一按 Ctrl+S，
json + `masks\` + overlay 会按缓存重写 —— labelme 的改动就没了。

这个脚本把 json 读回缓存，之后就一致了：工具打开那张图，看到的就是你改过的状态。

## 它怎么写

1. 解析 labelme json（`root` 折线、`root` 多边形、`stem`、`check_background`）；
2. root 掩码：
   - 有 `root` 多边形 -> **按多边形栅格化**（和训练侧同一个栅格化器 `common.gt_mask`，
     口径一致）；
   - 只有折线 -> 按 `config.MASK_LINE_WIDTH`（10px）描线兜底 —— **这就是"膨胀"**，
     只适合"这张图本来就只是折线"的情况。想要真实宽度，用工具过一遍（模型给宽度 +
     折线切）或者另一条"从图里量宽度"的路。
3. 写进 `cache\<当前模型>\`：掩码放**红层**（工具里显示成红色，但它是你的标注，不是预测）、
   折线/`stem`/`check_background` 进 meta（工具会显示 stem 和绿框，存盘时也原样保留）。

## 防覆盖

缓存的 `_meta.json` 比 json **新** -> 跳过并提示：那说明工具里还有比 json 更新的状态
（你在工具里改过但没存），拿 json 反导会把那份盖掉。真要覆盖加 `--force`。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import annotate_core as core                                # noqa: E402
import annotate_io as IO                                    # noqa: E402
import config                                              # noqa: E402
import config as vcfg                                      # noqa: E402  (骨架化阈值)
from common import gt_mask                                  # noqa: E402
from common.labelme import parse_other                      # noqa: E402


def _image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def _read_json(js):
    """直接读 shapes —— 不用 `parse_other` 的"有多边形就只用多边形"取舍。

    工具的缓存里**两样都要**：折线（中心线，负责"根长"和分区）、多边形（真实轮廓，
    栅格化成掩码）。parse_other 在两者同时存在时只给你多边形，折线就丢了。
    口径跟 parse_other 保持一致：折线至少 2 点、多边形至少 3 点、check 取外接矩形。
    """
    d = json.loads(js.read_text(encoding="utf-8"))
    lines, polys, stems, check = [], [], [], None
    for s in d.get("shapes") or []:
        lab = (s.get("label") or "").strip()
        st = (s.get("shape_type") or "").strip()
        pts = [(float(p[0]), float(p[1])) for p in (s.get("points") or [])]
        if lab == "root" and st in ("linestrip", "line") and len(pts) >= 2:
            lines.append(pts)
        elif lab == "root" and st == "polygon" and len(pts) >= 3:
            polys.append(pts)
        elif lab == "stem" and len(pts) >= 3:
            stems.append(pts)
        elif lab == "check_background" and pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            check = (min(xs), min(ys), max(xs), max(ys))
    return lines, polys, stems, check


def _mask_from_json(lines, polys, shape):
    """root 掩码：有多边形就按填充栅格化（真实轮廓）；只有折线时按带宽描（会说明）。"""
    h, w = shape
    if polys:
        # draw_polygons_at 要的是**目标尺度**下的点列 —— 这里就是原图尺度，直接给
        m = gt_mask.draw_polygons_at(polys, (w, h))
        return m.astype(np.uint8) * 255, "多边形栅格化"
    import cv2
    m = np.zeros((h, w), np.uint8)
    for line in lines:
        cv2.polylines(m, [np.round(np.asarray(line, np.float32)).astype(np.int32)],
                      False, 255, thickness=int(config.MASK_LINE_WIDTH),
                      lineType=cv2.LINE_8)
    return m, f"只有折线，按 {int(config.MASK_LINE_WIDTH)}px 带宽描（=膨胀）"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="把 labelme 改过的标注反向导入回工具的缓存",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--from", dest="srcs", nargs="*", default=None,
                    help="源目录（默认：工具自己的 datasets\\）")
    ap.add_argument("--model", default=None, help="写进哪个模型的缓存（默认：界面选的那个）")
    ap.add_argument("--dry-run", action="store_true", help="只列出会做什么，不动磁盘")
    ap.add_argument("--force", action="store_true",
                    help="缓存比 json 新时也覆盖（默认跳过）")
    args = ap.parse_args(argv)

    srcs = [Path(p) for p in (args.srcs or [IO.DATASETS_DIR])]
    model = args.model or IO.load_settings().get("model")
    try:
        model = IO.resolve_model(model)[1][0]
    except SystemExit:
        if args.model:
            # **显式点名了却解析不了 -> 直接报错**：静默回退会把缓存写进"另一个模型"的
            # 目录里，用户在工具里怎么等都看不见（2026-10-11 测试时真踩到）
            raise SystemExit(f"[错误] --model {args.model!r} 解析不了（没有权重？）。"
                             f"先确认 model\\ 下的文件夹名，再重跑。")
        model = IO.resolve_model(None)[1][0]
    cdir = IO.cache_dir(model)
    print(f"源：{', '.join(str(s) for s in srcs)}")
    print(f"写进：{cdir}")

    n_ok = n_skip = n_bad = 0
    for src in srcs:
        for js in sorted(src.glob("*.json")):
            stem = js.stem
            img = next((p for p in src.glob(f"{stem}.*")
                        if p.suffix.lower() in config.IMAGE_EXTS), None)
            if img is None:
                print(f"  [跳过] {stem}：找不到同名原图（掩码要按原图尺寸栅格化）")
                n_bad += 1
                continue
            try:
                lines, polys, stems, check = _read_json(js)
                w, h = _image_size(img)
                # 尺寸对不上就整张错位：宁可跳过（和 parse_other 一个口径）
                raw = json.loads(js.read_text(encoding="utf-8"))
                jw, jh = raw.get("imageWidth"), raw.get("imageHeight")
                if jw and jh and (int(jw), int(jh)) != (w, h):
                    raise ValueError(f"标注尺寸 {int(jw)}x{int(jh)} 与图片 {w}x{h} 不一致")
            except Exception as e:
                print(f"  [跳过] {stem}：{type(e).__name__}: {e}")
                n_bad += 1
                continue
            if not lines and not polys:
                print(f"  [跳过] {stem}：json 里没有 root 折线/多边形（这张大概是负样本）")
                n_skip += 1
                continue
            meta_p = IO.cache_paths(cdir, stem)["meta"]
            if meta_p.exists() and not args.force:
                if meta_p.stat().st_mtime > js.stat().st_mtime + 1:
                    print(f"  [跳过] {stem}：缓存的 meta 比 json 新"
                          f"（{time.strftime('%m-%d %H:%M', time.localtime(meta_p.stat().st_mtime))}"
                          f" > {time.strftime('%m-%d %H:%M', time.localtime(js.stat().st_mtime))}）"
                          f"—— 工具里可能有比 json 更新的东西，确认覆盖加 --force")
                    n_skip += 1
                    continue

            mask, how = _mask_from_json(lines, polys, (h, w))
            npx = int((mask > 0).sum())
            note = ""
            if not lines and polys:
                # 只有多边形没有折线（在 labelme 里画的多边形）：用工具自己那套骨架化
                # 起草折线（= G 的同一份实现），不然这张图在工具里一条折线都没有
                from common.skeleton_stats import analyze_mask_ex
                st = analyze_mask_ex(mask > 0, spur=vcfg.PRED_SPUR_LENGTH,
                                     min_len=vcfg.MIN_ROOT_LENGTH, with_paths=True)
                lines = [list(map(tuple, p)) for p in st["paths"] if len(p) >= 2]
                note = f"；没有折线，骨架化起草了 {len(lines)} 条（= G 那套）"
            print(f"  {'会导' if args.dry_run else '导入'} {stem}: 掩码 {npx} px（{how}）｜"
                  f"折线 {len(lines)} 条｜stem {len(stems)}｜"
                  f"check {'有' if check else '无'}{note}")
            if args.dry_run:
                n_ok += 1
                continue
            store = core.MaskStore(mask)          # 红层 = 这份标注；绿蓝留空
            meta = IO.meta_for_save(
                model, (h, w), stem, img.name, IO.DEFAULT_LOW_THRESH,
                check, check is not None, stems, lines,
                roi_box=check, roi_ok=check is not None)
            # **旧的概率图要删掉**：现在红层是"你的标注"，概率图却是上一次模型预测的 ——
            # 留着的话按一下 `[` `]`（预测松紧）就会拿模型的红层把导入的标注覆盖掉。
            IO.cache_paths(cdir, stem)["prob"].unlink(missing_ok=True)
            IO.save_cache(cdir, stem, store, meta)     # 不传 prob：没有概率图（调不了松紧）
            n_ok += 1

    print(f"\n{'（dry-run）预计导' if args.dry_run else '导'} {n_ok} 张｜"
          f"跳过 {n_skip}｜读不了 {n_bad}")
    if n_ok and not args.dry_run:
        print("去工具里打开这些图就能看到改动（折线/stem/绿框都在，掩码在红层）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
