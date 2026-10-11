r"""把 labelme 改过的标注**反向导入**回工具（手动 / 批量用）。

    python import_labelme.py                    # 默认导工具自己的 datasets\
    python import_labelme.py --from <目录> ...  # 也可以从别处导
    python import_labelme.py --dry-run          # 先看会做什么，不动磁盘
    python import_labelme.py --force            # 缓存比 json 新时也强行覆盖

**其实工具启动时会自动做一遍**（`MainWindow._sync_from_labelme`：逐张比
`datasets\<名>.json` 和缓存的修改时间，json 更新就按它更新缓存，并在终端写出来）。
这个脚本留着是为了：批量导别的目录、`--dry-run` 先看一眼、以及 `--force` 强推。

## 它怎么写（和启动自动同步是同一份实现：`annotate_io.sync_from_labelme`）

工具打开一张图时，折线/掩码**只从** `cache\<模型>\<stem>_*` 读，**从不读**
`datasets\<stem>.json`（那是导出给 labelme 看、给训练用的产物）。所以你在 labelme 里改完，
工具里看不到；**更糟的是**工具里一按 Ctrl+S，json + `masks\` + overlay 会按缓存重写，
改动就没了。这个脚本（和启动自动同步）把 json 读回缓存，两边就一致了。

- root 掩码：有 `root` 多边形就按**多边形栅格化**（和训练侧同一个栅格化器）；
  只有折线就按 10px 带宽描（**这就是膨胀**，报告里会写明）；
  只有多边形没有折线时用工具自己那套**骨架化起草**（= `G` 的同一份实现）补折线
- `stem` / `check_background` 也一起导进去 → 工具里会显示橙色的茎和绿框
- 旧的概率图会被**删掉**（否则按 `[` `]` 会拿模型的红层把你导入的标注盖了）

## 防覆盖

缓存的 `_meta.json` 比 json **新** -> 跳过：那说明工具里还有比 json 更新的状态
（你在工具里改过但没存）。真要覆盖加 `--force`。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 顺序有讲究：`config` 在 _vendor\ 里，是 **annotate_io 把它加进 sys.path** 的，
# 所以必须先 import annotate_io（反过来会 ModuleNotFoundError）
import annotate_io as IO                                    # noqa: E402
import config                                              # noqa: E402


def _image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


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
            # 目录里，用户在工具里怎么等都看不见
            raise SystemExit(f"[错误] --model {args.model!r} 解析不了（没有权重？）。"
                             f"先确认 model" + chr(92) + " 下的文件夹名，再重跑。")
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
            if args.dry_run:
                try:
                    lines, polys, stems, check = IO.read_labelme_shapes(js)
                except Exception as e:
                    print(f"  [跳过] {stem}：{type(e).__name__}: {e}")
                    n_bad += 1
                    continue
                if not lines and not polys:
                    print(f"  [跳过] {stem}：json 里没有 root 折线/多边形")
                    n_skip += 1
                    continue
                w, h = _image_size(img)
                mask, how = IO.mask_from_labelme(lines, polys, (h, w))
                print(f"  会导 {stem}: 掩码 {int((mask > 0).sum())} px（{how}）｜"
                      f"折线 {len(lines)} 条｜stem {len(stems)}｜"
                      f"check {'有' if check else '无'}")
                n_ok += 1
                continue
            state, msg = IO.sync_from_labelme(cdir, stem, js, img, model,
                                              force=args.force)
            print(f"  [{'导' if state == 'ok' else '跳过'}] {stem}: {msg}")
            n_ok += state == "ok"
            n_skip += state in ("same", "empty")
            n_bad += state in ("bad", "noimg")

    print(f"\n{'（dry-run）预计导' if args.dry_run else '导'} {n_ok} 张｜"
          f"跳过 {n_skip}｜读不了 {n_bad}")
    if n_ok and not args.dry_run:
        print("去工具里打开这些图就能看到改动（折线/stem/绿框都在，掩码在红层）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
