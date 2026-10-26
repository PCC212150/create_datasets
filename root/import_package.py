r"""把一个「交付包」导进这个工具：拷贝 → 建缓存（**掩码来自包，不跑模型**）。

    python import_package.py --from "C:\Users\21215\Desktop\datasets-C019-3,4"
    python import_package.py --from <包目录> --dry-run
    python import_package.py --from <包目录> --model model_202610082352   # 指定当前模型

## 什么算交付包

`datasets\` 那一层就是**这个工具自己的导出格式**（见 readme「产物」）：

    <包>\datasets\<名>.jpg           原图（硬链接/拷贝过来的那份）
    <包>\datasets\<名>.json          labelme：root 多边形 + root 折线 + stem + check_background
    <包>\datasets\masks\<名>.png     像素级 root 掩码（黑底白条）
    <包>\datasets\overlay\<名>_overlay.jpg

和 import_lines.py 的区别：**折线数据集没有掩码**，所以那边要模型重新预测一版红层；
交付包带着**最终掩码**，所以这里**不跑模型**，直接把包里的掩码灌进缓存。
（--from 给包根或包里的 datasets\ 目录都行，自动认。）

## 缓存怎么灌（口径，2026-10-12 定）

- `add`（**绿**）= 包里的掩码。这是**人确认过的成果**，不是模型预测 —— 放绿不放红：
  工具里"人做的部分"（绿/蓝/折线）在 ①关「预测」清红层 ②换模型搬运 时**都会保住**，
  红层两种情况下都会被模型输出顶掉。打开后看到整块**绿**是正常的，不是出错了。
- `pred`（红）= 全空。`prob` **留着原来缓存里那份**（属于那个模型自己）—— 想看它的建议，
  按 `[` `]` 就能从概率图把红层按出来。
- 折线从包的 json 取 `root` 的 **linestrip**（不是多边形！解析器「有多边形就只用多边形」，
  拿 `parse_other` 的 `lab.roots` 会错拿成多边形）；stem / check 走 `parse_other`。
- 缓存写到**两个地方**：界面当前用的模型（settings.json 里那个；没记就取最新有权重的）
  ＋ **已经存过这张图缓存的模型目录**（不校正的话，切回老模型看到的是一份比标注旧的
  缓存，按一次 Ctrl+S 就把成果覆盖了 —— 2026-10-08 的 C019-3 事故就是这个形状）。

**源包里的一律只读**：只复制，不移动、不删除（2026-10-11 用户定的）。
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402
from PIL import Image                                       # noqa: E402

import annotate_io as IO                                    # noqa: E402
from common.labelme import parse_other                      # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_pkg_root(from_path: Path) -> Path:
    r"""允许给包根（里面是 datasets\）或直接给 datasets\ 那一层。"""
    p = Path(from_path)
    for cand in (p, p / "datasets"):
        if (cand / "masks").is_dir():
            return cand
    sys.exit(f"[错误] {from_path} 里找不到 masks\\ 子目录 —— 不像是交付包"
             f"（只带折线的数据请用 import_lines.py）")


def image_size(path):
    with Image.open(path) as im:
        return im.size                     # (w, h)


def collect(pkg: Path):
    """扫包：每个 json 一个条目，顺带把 json 里的折线/茎/check 摘出来。"""
    items, broken = [], []
    for jp in sorted(pkg.glob("*.json")):
        img = next((p for p in pkg.glob(f"{jp.stem}.*")
                    if p.suffix.lower() in IMAGE_EXTS), None)
        if img is None:
            broken.append((jp, "找不到同名图片"))
            continue
        mp = pkg / IO.MASK_SUBDIR / f"{jp.stem}.png"
        if not mp.exists():
            broken.append((jp, f"没有 {IO.MASK_SUBDIR}\\{jp.stem}.png（纯折线数据请用 "
                               f"import_lines.py）"))
            continue
        try:
            # 折线**必须自己从 shapes 里挑 linestrip**：parse_other 的 lab.roots
            # 在有多边形时只剩多边形（「有多边形就只用多边形」的解析口径）
            body = json.loads(jp.read_text(encoding="utf-8"))
            lines = [s["points"] for s in body.get("shapes", [])
                     if s.get("label") == "root"
                     and s.get("shape_type") in ("linestrip", "line")
                     and len(s.get("points") or []) >= 2]
            n_poly = sum(1 for s in body.get("shapes", [])
                         if s.get("label") == "root"
                         and s.get("shape_type") == "polygon")
            lab = parse_other(jp, image_size=image_size(img), verbose=False)
        except Exception as e:                      # 单张坏不该拦下整批
            broken.append((jp, f"{type(e).__name__}: {e}"))
            continue
        items.append({"stem": jp.stem, "json": jp, "img": img, "mask": mp,
                      "overlay": pkg / IO.OVERLAY_SUBDIR / f"{jp.stem}_overlay.jpg",
                      "lines": [list(map(list, l)) for l in lines],
                      "n_poly": n_poly,
                      "stems": [list(map(list, p)) for p in lab.stems],
                      "check": list(lab.check_rect) if lab.check_rect else None})
    return items, broken


def plan(items, dest: Path):
    keep, taken = [], []
    for it in items:
        exist = [p for p in list(dest.glob(f"{it['stem']}.*"))
                 + list(IO.PICTURES_DIR.glob(f"{it['stem']}.*"))]
        if exist:
            taken.append((it, f"工具里已有 {exist[0].parent.name}\\{exist[0].name}"))
            continue
        keep.append(it)
    return keep, taken


def copy_in(keep, dest: Path, dry: bool):
    (dest / IO.MASK_SUBDIR).mkdir(parents=True, exist_ok=True)
    (dest / IO.OVERLAY_SUBDIR).mkdir(parents=True, exist_ok=True)
    for it in keep:
        pairs = [(it["img"], dest / it["img"].name), (it["json"], dest / it["json"].name),
                 (it["mask"], dest / IO.MASK_SUBDIR / it["mask"].name)]
        if it["overlay"].exists():
            pairs.append((it["overlay"], dest / IO.OVERLAY_SUBDIR / it["overlay"].name))
        for src, dst in pairs:
            if dry:
                print(f"  会拷 {src.name} -> {dst.parent.name}\\")
            else:
                # copy2 保 mtime；**只复制**，源包一律不动
                shutil.copy2(src, dst)


def target_models():
    """缓存要写到哪些模型目录：界面当前模型 + 已经存过这些图缓存的模型目录。"""
    try:
        cur = IO.resolve_model(IO.load_settings().get("model"))[1][0]
    except SystemExit:
        cur = IO.resolve_model(None)[1][0]
    return cur


def seed_cache(keep, model_name, dry):
    """把包里的掩码/折线灌进缓存。pred 清空、掩码进 add（理由见模块 docstring）。"""
    dirs = {model_name}
    for d in sorted(IO.CACHE_ROOT.glob("model_*")):
        if any(IO.cache_paths(d, it["stem"])["meta"].exists() for it in keep):
            dirs.add(d.name)
    # 当前模型排最后打印（让"界面用的那个"最显眼）；顺手按名字排序
    dirs = sorted(dirs)
    n_written = 0
    for name in dirs:
        cdir = IO.cache_dir(name)
        for it in keep:
            cp = IO.cache_paths(cdir, it["stem"])
            # 老缓存里那份 prob 留着（属于这个模型自己）；[ ] 还能把红层按出来
            prob = np.load(cp["prob"]) if cp["prob"].exists() else None
            old = {}
            if cp["meta"].exists():
                try:
                    old = json.loads(cp["meta"].read_text(encoding="utf-8"))
                except Exception:
                    old = {}
            mask = np.asarray(Image.open(it["mask"])) > 127
            w, h = image_size(it["img"])
            if mask.shape != (h, w):
                print(f"  [跳过] {name}/{it['stem']}: 掩码尺寸 {mask.shape} != 原图 {(h, w)}")
                continue
            meta = IO.meta_for_save(
                name, (h, w), it["stem"], it["img"].name,
                old.get("low_thresh", IO.DEFAULT_LOW_THRESH),
                it["check"], True, it["stems"], it["lines"], prob=prob,
                poly_width=old.get("poly_width"),
                roi_box=it["check"], roi_ok=True)
            if dry:
                print(f"  会灌缓存 {name}/{it['stem']}: 绿 {int(mask.sum())}px / "
                      f"折线 {len(it['lines'])} / 茎 {len(it['stems'])} / "
                      f"prob {'留' if prob is not None else '无'}")
                n_written += 1
                continue
            store = SimpleNamespace(pred=np.zeros((h, w), np.uint8),
                                    add=(mask.astype(np.uint8) * 255),
                                    dele=np.zeros((h, w), np.uint8))
            IO.save_cache(cdir, it["stem"], store, meta)
            n_written += 1
    return dirs, n_written


def verify(keep, model_name):
    """回读每个模型目录：最终掩码必须与包里的掩码**逐位相同**，折线数对上。"""
    ok = 0
    for it in keep:
        c = IO.load_cache(IO.cache_dir(model_name), it["stem"], model_name, quiet=True)
        if c is None:
            print(f"  [失败] {it['stem']}: 回读不到缓存")
            continue
        final = (c["pred"] > 0) | (c["add"] > 0)
        final &= ~(c["dele"] > 0)
        pkgmask = np.asarray(Image.open(it["mask"])) > 127
        n_line = len(c["meta"].get("polylines") or [])
        good = bool((final == pkgmask).all()) and n_line == len(it["lines"])
        ok += good
        if not good:
            print(f"  [失败] {it['stem']}: 掩码一致={bool((final == pkgmask).all())} "
                  f"折线 {n_line}/{len(it['lines'])}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="把「交付包」（带 masks\\ 的完整导出）导进本工具：拷贝 + 灌缓存，不跑模型",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--from", dest="src", required=True, help="包目录（或其中的 datasets\\ 子目录）")
    ap.add_argument("--dest", default=None, help="拷到哪（默认工具的 datasets\\）")
    ap.add_argument("--model", default=None, help="界面当前用的模型名（默认按 settings 推）")
    ap.add_argument("--dry-run", action="store_true", help="只列出会做什么，不动磁盘")
    args = ap.parse_args(argv)

    pkg = find_pkg_root(Path(args.src))
    dest = Path(args.dest) if args.dest else IO.DATASETS_DIR
    model_name = args.model or target_models()
    print(f"包目录: {pkg}\n目标: {dest}\n当前模型: {model_name}")

    items, broken = collect(pkg)
    keep, taken = plan(items, dest)
    print(f"\n扫到 {len(items)} 个带掩码的条目 | 可导 {len(keep)} | 工具里已有跳过 {len(taken)}")
    for jp, why in broken:
        print(f"  [读不了] {jp.name}: {why}")
    for it, why in taken:
        print(f"  [已存在] {it['stem']}: {why}")
    # 包自身的口径自检：每条折线一个多边形，数量该 1:1
    odd = [it["stem"] for it in keep if it["n_poly"] != len(it["lines"])]
    if odd:
        print(f"  [注意] 这些图的 root 多边形数与折线数对不上（包里的多边形可能是旧口径）："
              f"{odd}")
    if not keep:
        print("没有可导的，结束。")
        return 0

    print(f"\n== 1/3 拷贝 {len(keep)} 张（图 + json + masks\\ + overlay\\）-> {dest} ==")
    copy_in(keep, dest, args.dry_run)

    print(f"\n== 2/3 灌缓存（掩码进绿层，pred 清空；不跑模型）==")
    dirs, n = seed_cache(keep, model_name, args.dry_run)
    print(f"  写到 {len(dirs)} 个模型目录：{dirs}（共 {n} 份缓存）")

    if args.dry_run:
        print("\n== 3/3 回读自检（--dry-run 跳过）==\n没有任何文件被改动。")
        return 0
    print(f"\n== 3/3 回读自检（对照 {model_name}）==")
    ok = verify(keep, model_name)
    print(f"  逐位一致 {ok}/{len(keep)}")
    print("\n导好了。打开工具：这批图的掩码会显示成**绿色**（= 人确认过的成果，不是模型"
          "预测）；折线/茎/检查框都在。想切回老模型看的话，那份缓存也一并校正过了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
