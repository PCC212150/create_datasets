r"""把别处标好的**折线数据集**导进这个工具：拷贝 -> 建缓存 -> 灌折线。

    python import_lines.py --from <源目录> [<源目录> ...] [--dry-run]
    python import_lines.py --from ... --no-prefetch      # 缓存已经有了就跳过第 2 步
    python import_lines.py --from ... --patch-only       # 只重灌折线，不拷贝不预测

## 为什么需要它

工具打开一张图时，折线**只从** `cache\<模型名>\<stem>_meta.json` 读，**从不读**
`datasets\<stem>.json`。所以别处标好的 json 直接拷进 `datasets\` 是没用的：

    打开 -> 没有缓存 -> 当场重新预测 -> 折线是空的（json 里的折线读不回来）
         -> 这时按 Ctrl+S，那张图的 json 就被覆盖了，旧折线没了

## 它做的三件事（缺一不可）

1. **拷贝**：源目录里的图 + json 扁平拷进 `datasets\`（工具只扫一层目录，不能带子目录）。
   目的里已经有同名 stem 的（`datasets\` 的 json / 图，或 `pictures\` 里那张）**一律跳过
   并报告**——那说明工具里已经有一份（多半更新的）标注了，覆盖它才是事故。
2. **建缓存**：调用工具的 `--prefetch`（走的是同一条预测路径），每张新图一份
   `_masks.png` + `_prob.npy` + `_meta.json`。
3. **灌折线**：把源 json 的 `root` 折线、`stem` 多边形、`check_background` 框分别写进缓存
   meta 的 `polylines` / `stem_polys` / `check_box`。之后再打开，折线和茎、检查框都是原样，
   可以直接改、直接 Ctrl+S。

## 一个必须知道的口径差

**掩码没法从这些 json 还原**——折线数据集里只有折线，没有 root 多边形。所以导进来之后
红色的预测层是**模型现在**预测的样子，和你当初标注时的掩码不是一回事；折线（根长真值）、
茎、检查框是原样搬过来的。想连掩码一起搬，源数据必须是工具自己存的那种
（带 `root` polygon 的 json），而那种图多半本来就有缓存。

源 json 用 `common.labelme.parse_other` 解析 —— 和训练侧**同一个解析器**。
自己再写一遍解析一定会分叉（"少于 2 个点的折线要不要丢"这种细节就开始不一样了）。
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import annotate_io as IO                                   # noqa: E402
import config                                              # noqa: E402  (_vendor 里那个)
from common.labelme import parse_other                     # noqa: E402


def _image_size(path):
    """原图尺寸（只读文件头，不解码）。用来校验 json 里记的尺寸。"""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def collect(src_dirs):
    """源目录 -> [(json, 图, OtherLabels)]；只扫一层。解析不了的单独报出来。"""
    items, broken = [], []
    for d in src_dirs:
        d = Path(d)
        if not d.is_dir():
            broken.append((d, "目录不存在"))
            continue
        for jp in sorted(d.glob("*.json")):
            img = next((p for p in d.glob(f"{jp.stem}.*")
                        if p.suffix.lower() in config.IMAGE_EXTS), None)
            if img is None:
                broken.append((jp, "找不到同名图片"))
                continue
            try:
                lab = parse_other(jp, image_size=_image_size(img), verbose=False)
            except Exception as e:                          # 单张坏不该拦下整批
                broken.append((jp, f"{type(e).__name__}: {e}"))
                continue
            items.append((jp, img, lab))
    return items, broken


def plan(items, dest):
    """分三类：能导的 / 因为折线是空的跳过 / 因为工具里已有同名的跳过。"""
    keep, empty, taken = [], [], []
    for jp, img, lab in items:
        if not lab.roots:
            why = ("只有 root 多边形、没有折线" if lab.root_polygons
                   else "json 里没有 root 折线")
            empty.append((jp, why))
            continue
        # 目的里任何一份同名文件都算占用：json（做过）、图（在 datasets 或 pictures 里）
        exist = [p for p in list(dest.glob(f"{jp.stem}.*"))
                 + list(IO.PICTURES_DIR.glob(f"{jp.stem}.*"))]
        if exist:
            taken.append((jp, f"工具里已有 {exist[0].parent.name}\\{exist[0].name}"))
            continue
        keep.append((jp, img, lab))
    return keep, empty, taken


def copy_in(keep, dest, dry):
    dest.mkdir(parents=True, exist_ok=True)
    for jp, img, _lab in keep:
        for src in (img, jp):
            dst = dest / src.name
            if dry:
                print(f"  会拷 {src} -> {dst}")
            else:
                shutil.copy2(src, dst)        # copy2 保 mtime：图上仍带着"拍摄/标注时间"


def patch_cache(keep, model_name, dry):
    """把源 json 里的折线/茎/检查框灌进缓存 meta。缓存在（第 2 步刚建的）才灌。"""
    cdir = IO.cache_dir(model_name)
    n_ok = n_nocache = 0
    for jp, _img, lab in keep:
        cp = IO.cache_paths(cdir, jp.stem)["meta"]
        if not cp.exists():
            n_nocache += 1
            print(f"  [跳过] {jp.stem}: 没有缓存（先跑一次 --prefetch）")
            continue
        with open(cp, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("polylines"):
            n_nocache += 1
            print(f"  [跳过] {jp.stem}: 缓存里已经有 {len(meta['polylines'])} 条折线，"
                  f"不覆盖")
            continue
        meta["polylines"] = [list(map(list, l)) for l in lab.roots]
        if lab.stems:
            meta["stem_polys"] = [list(map(list, p)) for p in lab.stems]
        if lab.check_rect is not None:
            meta["check_box"] = list(map(float, lab.check_rect))
            meta["check_ok"] = True
        if dry:
            print(f"  会灌 {jp.stem}: 折线 {len(lab.roots)} / "
                  f"茎 {len(lab.stems)} / check {'有' if lab.check_rect else '无'}")
            n_ok += 1
            continue
        tmp = cp.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        os.replace(tmp, cp)                   # 原子落盘：写一半崩了不会留下半份 meta
        print(f"  灌好 {jp.stem}: 折线 {len(lab.roots)} 条 / 茎 {len(lab.stems)} 个 / "
              f"check {'有' if lab.check_rect else '无'}")
        n_ok += 1
    return n_ok, n_nocache


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="把别处标好的折线数据集导进本工具（拷贝 + 建缓存 + 灌折线）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--from", dest="srcs", nargs="+", required=True,
                    help="源目录（可多个）；只扫它们这一层，不带子目录")
    ap.add_argument("--dest", default=None, help="图片拷贝到哪（默认工具的 datasets\\）")
    ap.add_argument("--dry-run", action="store_true", help="只列出会做什么，不动磁盘")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="跳过预测（缓存已经建好了才这么用）")
    ap.add_argument("--patch-only", action="store_true",
                    help="只灌折线：不拷贝、不预测")
    ap.add_argument("--no-gpu", action="store_true", help="预测强制走 CPU")
    ap.add_argument("--model", default=None,
                    help="用哪个模型建缓存（默认：界面上次选的那个，再退回最新）")
    args = ap.parse_args(argv)

    dest = Path(args.dest) if args.dest else IO.DATASETS_DIR
    if not args.patch_only:
        items, broken = collect(args.srcs)
        print(f"扫到 {len(items)} 个带图的 json")
        keep, empty, taken = plan(items, dest)
        print(f"可导 {len(keep)} 张 | 没有折线跳过 {len(empty)} | 工具里已有跳过 {len(taken)}")
        for jp, why in broken:
            print(f"  [读不了] {jp}: {why}")
        for jp, why in empty:
            print(f"  [无折线] {jp.name}: {why}")
        for jp, why in taken:
            print(f"  [已存在] {jp.name}: {why}")
        if not keep:
            print("没有可导的，结束。")
            return 0
        print(f"\n== 1/3 拷贝 {len(keep)} 张图 + json -> {dest} ==")
        copy_in(keep, dest, args.dry_run)
    else:
        items, _broken = collect(args.srcs)
        keep = [(jp, img, lab) for jp, img, lab in items if lab.roots]
        print(f"== 只灌折线：{len(keep)} 张 ==")

    # 预测这一步用工具自己的 --prefetch：口径完全一致，别在这里另起一套。
    # **模型名要和界面一致**：界面上换过模型的话（存在 settings.json 里），
    # 折线得灌进那个模型的缓存 —— 写死"最新那个"会让用户打开工具时看不见折线。
    model_name = args.model or IO.load_settings().get("model")
    try:
        model_name = IO.resolve_model(model_name)[1][0]
    except SystemExit:
        print(f"[提示] settings.json 里记的模型 {model_name!r} 已经不在了，退回最新那个")
        model_name = IO.resolve_model(None)[1][0]
    print(f"用的模型：{model_name}（界面切过模型的话，这里跟的是同一个）")
    if args.dry_run:
        print("\n== 2/3 预跑预测（--dry-run 跳过）==")
    elif args.patch_only or args.no_prefetch:
        print("\n== 2/3 预跑预测（跳过）==")
    else:
        print(f"\n== 2/3 预跑预测（模型 {model_name}）==")
        import annotate_root as ar
        pf = ar.parse_args([])
        pf.no_gpu = args.no_gpu
        pf.model = model_name       # 别让 --prefetch 自己去挑"最新那个"
        ar.run_prefetch(pf)

    print(f"\n== 3/3 把折线灌进缓存（{model_name}）==")
    n_ok, n_bad = patch_cache(keep, model_name, args.dry_run)
    if args.dry_run:
        print(f"（dry-run：预计灌 {n_ok} 张，{n_bad} 张要跳过）")
        print("没有任何文件被改动。")
    else:
        print(f"灌好 {n_ok} 张；跳过 {n_bad} 张。\n"
              f"现在打开工具，这些图的折线/茎/检查框都在，掩码是模型重新预测的。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
