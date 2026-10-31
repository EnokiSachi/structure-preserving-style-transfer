#!/usr/bin/env python3
"""
Step 1：用已有的 9 个微调解码器 + 预训练解码器，重跑原验证评估并追加独立指标

做什么：
  1. 按原 worker 完全相同的流程，对 3 个种子 × 4 个条件 × 64 个图像对 × 5 档强度评估；
  2. 自检：把重算的 style / pixel_l1 / multi_structure 等 5 项原指标与
     outputs/adain_confirmation_v1/seed_*/per_pair_metrics.csv 逐条比对，
     确认图像映射、预处理和模型加载与原实验一致（不一致会明确报警）；
  3. 同时计算 3 项不参与训练的指标：ssim_luma、edge_f1、lpips_content；
  4. 生成一张不挑图的对比图：种子 17，前 4 个验证内容，alpha=0.75，四个条件并排。

不修改任何原有文件；所有输出写到 evaluation/results/。
3070 Ti 上预计 5–15 分钟。

用法（PowerShell，在项目根目录）：
  .\\.venv\\Scripts\\python.exe evaluation\\step1_independent_eval.py
  # 先快速试跑（只跑种子 17 的 pretrained 和 multi）：
  .\\.venv\\Scripts\\python.exe evaluation\\step1_independent_eval.py --seeds 17 --arms pretrained multi

输出：
  results/seed_*/ext_metrics.csv        逐条结果（留在本机）
  results/figures/arms_alpha075_seed17.png
  results/step1_report.txt
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_common as G  # noqa: E402

CHECK_KEYS = ("style", "content_feature", "pixel_l1", "single_structure", "multi_structure")
ARMS = ("pretrained", "none", "single", "multi")


def verify(rows, orig_rows):
    ref = {(r["arm"], int(r["pair_index"]), round(r["alpha"], 2)): r for r in orig_rows}
    n, ok, missing = 0, 0, 0
    worst = {k: 0.0 for k in CHECK_KEYS}
    for r in rows:
        o = ref.get((r["arm"], int(r["pair_index"]), round(r["alpha"], 2)))
        if o is None:
            missing += 1
            continue
        n += 1
        good = True
        for k in CHECK_KEYS:
            d = abs(r[k] - o[k])
            worst[k] = max(worst[k], d / max(abs(o[k]), 1e-8))
            if d > 1e-4 + 1e-3 * abs(o[k]):
                good = False
        ok += good
    return n, ok, missing, worst


def save_figure(kept, order, arms, path, size=256):
    from PIL import Image, ImageDraw
    from torchvision.transforms.functional import to_pil_image
    cols = ["Content", "Reference", *arms]
    grid = Image.new("RGB", (size * len(cols), (size + 20) * len(order) + 20), "white")
    draw = ImageDraw.Draw(grid)
    for j, name in enumerate(cols):
        draw.text((j * size + 4, 4), name, fill="black")
    for i, key in enumerate(order):
        y = 20 + i * (size + 20)
        content, reference, _ = kept[arms[0]][key]
        draw.text((4, y + 2), f"pair {key[0]}  alpha {key[1]}", fill="black")
        tiles = [content, reference] + [kept[a][key][2] for a in arms]
        for j, t in enumerate(tiles):
            grid.paste(to_pil_image(t[0].clamp(0, 1)), (j * size, y + 18))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=G.DEFAULT_DATA_ROOT)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(G.SEEDS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    t0 = time.monotonic()
    ctx = G.Context(args.data_root, use_lpips=not args.no_lpips)
    print(f"device={ctx.device}  lpips={'on' if ctx.lp is not None else 'off'}", flush=True)

    report = ["# Step 1 报告：独立指标评估与原指标复现自检",
              f"device={ctx.device}  lpips={'on' if ctx.lp is not None else 'off'}  "
              f"canny={G.CANNY}  edge_tol={G.EDGE_TOL}px",
              f"seeds={args.seeds}  arms={args.arms}", ""]
    all_ok = True
    for seed in args.seeds:
        plan = json.loads((G.CONF_DIR / f"seed_{seed}" / "plan.json").read_text(encoding="utf-8"))
        orig = G.read_csv(G.CONF_DIR / f"seed_{seed}" / "per_pair_metrics.csv")
        n_style = len(ctx.val.style_files)
        first_contents = plan["validation_contents"][:4]
        styles = plan["validation_styles"]
        order = [(c * n_style + styles[i % len(styles)], 0.75) for i, c in enumerate(first_contents)]
        keep = set(order) if seed == args.seeds[0] else None

        seed_rows, kept_by_arm = [], {}
        for arm in args.arms:
            ts = time.monotonic()
            decoder = ctx.make_decoder(ctx.arm_state(seed, arm))
            rows, kept = ctx.evaluate(decoder, arm, plan, keep=keep)
            n, ok, missing, worst = verify(rows, orig)
            passed = n > 0 and ok == n and missing == 0
            all_ok &= passed
            line = (f"seed {seed} {arm:<10} rows={len(rows)} matched={n} pass={ok}/{n} missing={missing} "
                    f"worst_rel_diff=" + ", ".join(f"{k}:{v:.1e}" for k, v in worst.items()) +
                    f"  [{'OK' if passed else 'MISMATCH'}]  {time.monotonic() - ts:.0f}s")
            print(line, flush=True)
            report.append(line)
            seed_rows += rows
            if kept:
                kept_by_arm[arm] = kept
            del decoder
            if ctx.device.type == "cuda":
                ctx.torch.cuda.empty_cache()
        G.write_csv(G.RESULTS / f"seed_{seed}" / "ext_metrics.csv", seed_rows)
        if kept_by_arm and len(kept_by_arm) == len(args.arms):
            fig = G.RESULTS / "figures" / f"arms_alpha075_seed{seed}.png"
            save_figure(kept_by_arm, order, list(args.arms), fig)
            report.append(f"figure: {fig.relative_to(G.ROOT)}  (前 4 个验证内容，未挑选)")

    report += ["", "## 各条件、各强度的独立指标均值（未做风格误差匹配，仅作概览）"]
    try:
        import pandas as pd
        frames = [pd.read_csv(G.RESULTS / f"seed_{s}" / "ext_metrics.csv") for s in args.seeds]
        df = pd.concat(frames)
        tab = df.groupby(["alpha", "arm"])[["style", "multi_structure", *G.NEW_KEYS]].mean().round(4)
        report.append(tab.to_string())
        report.append(f"\ncontent edge density (mean): {df['content_edge_density'].mean():.4f}")
    except Exception as e:
        report.append(f"[summary error: {e}]")

    report += ["", "## 自检结论",
               "全部一致：可以继续 Step 2。" if all_ok else
               "存在 MISMATCH：说明映射或预处理与原实验不一致，请检查数据路径与模型文件后再继续。",
               f"total time: {time.monotonic() - t0:.0f}s"]
    out = G.RESULTS / "step1_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\n完成。结果已保存：{out}", flush=True)


if __name__ == "__main__":
    main()
