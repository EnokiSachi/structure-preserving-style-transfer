#!/usr/bin/env python3
"""
Step 2：在"相同风格误差"下比较各条件，并做考虑依赖结构的统计检验

做什么：
  1. 读取 Step 1（以及 Step 3 若已完成）的逐条结果；
  2. 用与 summarize_adain_confirmation.py 逐字相同的插值函数和接受规则，
     在 none 条件 alpha=0.25/0.5/0.75 的风格误差处匹配所有条件；
  3. 复现原论文数字（multi_structure 应得到 0.404124 / 0.418970 等）作为对照检查；
  4. 对原指标与 3 项独立指标做两两比较：
     先把每个 (内容, 参考) 的差值在种子和匹配点上取平均，得到 16×4 矩阵，
     再做交叉 bootstrap（内容、参考分别重采样）95% 置信区间 + 内容层面符号检验。

纯 CPU，几秒钟完成。不需要 GPU，也不读取模型。

用法（PowerShell，在项目根目录）：
  .\\.venv\\Scripts\\python.exe evaluation\\step2_analyze.py

输出：
  results/matched_ext.csv              匹配后的逐点数据（留在本机）
  results/step2_summary.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_common as G  # noqa: E402

BASE_ARMS = ["pretrained", "none", "single", "multi"]
KEYS = ["multi_structure", "single_structure", "content_feature", "pixel_l1", "pooled_intensity_l1",
        "ssim_luma", "edge_f1", "lpips_content"]
INDEPENDENT = {"ssim_luma", "edge_f1", "lpips_content", "pooled_intensity_l1", "pixel_l1"}
COMPARISONS = [("multi", "none"), ("multi", "single"), ("single", "none"), ("none", "pretrained"),
               ("laplacian", "none"), ("multi", "laplacian"),
               ("combo", "none"), ("combo", "multi"), ("combo", "laplacian")]
PAPER = {("pretrained", "multi_structure"): 0.428099, ("none", "multi_structure"): 0.418970,
         ("single", "multi_structure"): 0.415336, ("multi", "multi_structure"): 0.404124}


def load_rows(seeds, base=None):
    base = base or G.RESULTS
    data = {}
    for seed in seeds:
        rows = []
        for path in [base / f"seed_{seed}" / "ext_metrics.csv"] + \
                [base / sub / f"seed_{seed}" / "ext_metrics.csv" for _, sub in G.EXTRA_ARMS]:
            if path.exists():
                rows += G.read_csv(path)
        if rows:
            data[seed] = rows
    return data


def match(data, arms, keys):
    matched, coverage = [], []
    for seed, rows in data.items():
        by = {}
        for r in rows:
            by.setdefault((r["arm"], int(r["pair_index"])), []).append(r)
        pairs = sorted({int(r["pair_index"]) for r in rows if r["arm"] == "none"})
        accepted = 0
        for pair in pairs:
            curves = {arm: by.get((arm, pair), []) for arm in arms}
            if any(len(c) != len(G.ALPHAS) for c in curves.values()):
                continue
            meta = curves["none"][0]
            for alpha in (0.25, 0.5, 0.75):
                target = next(r["style"] for r in curves["none"] if r["alpha"] == alpha)
                values = {arm: {k: G.interpolate(curves[arm], target, k) for k in keys} for arm in arms}
                if any(v is None for d in values.values() for v in d.values()):
                    continue
                accepted += 1
                for arm in arms:
                    matched.append({"seed": seed, "pair_index": pair, "content_id": int(meta["content_id"]),
                                    "style_id": int(meta["style_id"]), "none_alpha": alpha,
                                    "matched_style": target, "arm": arm, **values[arm]})
        coverage.append((seed, accepted, len(pairs) * 3))
    return pd.DataFrame(matched), coverage


def compare(df, a, b, key, n_boot):
    idx = ["seed", "pair_index", "none_alpha"]
    A = df[df.arm == a].set_index(idx)[[key, "content_id", "style_id"]]
    B = df[df.arm == b].set_index(idx)[[key]]
    j = A.join(B, rsuffix="_b", how="inner").dropna()
    if j.empty:
        return None
    j["d"] = j[key] - j[key + "_b"]
    D = j.groupby(["content_id", "style_id"])["d"].mean().unstack("style_id").to_numpy(dtype=float)
    sgn = 1.0 if G.HIGHER_BETTER[key] else -1.0
    lo, hi = G.crossed_bootstrap_ci(D, n_boot)
    cm, sm = np.nanmean(D, axis=1), np.nanmean(D, axis=0)
    mean_b = j[key + "_b"].mean()
    diff = float(np.nanmean(D))
    excludes_zero = (lo > 0) or (hi < 0)
    return dict(comparison=f"{a} vs {b}", metric=key, mean_a=j[key].mean(), mean_b=mean_b,
                diff=diff, ci_low=lo, ci_high=hi,
                rel=100 * diff / mean_b if mean_b else float("nan"),
                better=("a" if sgn * diff > 0 else "b"), ci_excl_0=excludes_zero,
                contents=f"{int((sgn * cm > 0).sum())}/{len(cm)}",
                styles=f"{int((sgn * sm > 0).sum())}/{len(sm)}",
                sign_p=G.sign_test_p(cm))


def fmt_table(rows):
    t = pd.DataFrame(rows)
    out = pd.DataFrame({
        "comparison": t.comparison, "metric": t.metric,
        "mean_a": t.mean_a.map(lambda v: f"{v:.5f}"), "mean_b": t.mean_b.map(lambda v: f"{v:.5f}"),
        "rel_change": t.rel.map(lambda v: f"{v:+.2f}%"),
        "95%CI(diff)": [f"[{lo:+.5f}, {hi:+.5f}]" for lo, hi in zip(t.ci_low, t.ci_high)],
        "better": t.better, "CI_excl_0": t.ci_excl_0.map(lambda v: "yes" if v else "no"),
        "contents": t.contents, "styles": t.styles, "sign_p": t.sign_p.map(lambda v: f"{v:.2g}")})
    return out.to_string(index=False)


def analyze(data, arms, keys, n_boot, title, lines, paper_check=True):
    df, coverage = match(data, arms, keys)
    lines += ["", "=" * 70, f"## {title}", "=" * 70,
              "coverage (seed, accepted, possible): " + ", ".join(f"({s},{a},{p})" for s, a, p in coverage)]
    if df.empty:
        lines.append("no matched points")
        return df
    seed_means = df.groupby(["seed", "arm"])[keys].mean()
    agg = seed_means.groupby("arm").agg(["mean", "std"])
    lines.append("\n### 匹配后均值（三种子均值 ± 种子间 SD）")
    tab = pd.DataFrame(index=[a for a in arms if a in agg.index])
    for k in keys:
        if agg[(k, "mean")].notna().any():
            tab[k] = [f"{agg.loc[a, (k, 'mean')]:.6f}±{agg.loc[a, (k, 'std')]:.6f}" for a in tab.index]
    lines.append(tab.T.to_string())
    checks = [(a, k, v, agg.loc[a, (k, "mean")]) for (a, k), v in PAPER.items() if a in agg.index]
    if paper_check and checks and set(arms) == set(BASE_ARMS):
        lines.append("\n### 与论文数字对照（应基本一致）")
        lines += [f"{a:<10} {k}: paper={v:.6f}  now={m:.6f}  diff={m - v:+.6f}" for a, k, v, m in checks]
    rows = []
    for a, b in COMPARISONS:
        if a in arms and b in arms:
            for k in keys:
                if df[k].notna().any():
                    r = compare(df, a, b, k, n_boot)
                    if r:
                        rows.append(r)
    if rows:
        lines += ["\n### 两两比较（diff = a − b；CI 为交叉 bootstrap；contents/styles = a 更优的内容数/参考数；"
                  "sign_p 为内容层面的符号检验）", fmt_table(rows)]
    per_alpha = []
    for alpha in (0.25, 0.5, 0.75):
        sub = df[df.none_alpha == alpha]
        for k in ("multi_structure", "ssim_luma", "edge_f1", "lpips_content"):
            if "multi" in arms and sub[k].notna().any():
                r = compare(sub, "multi", "none", k, n_boot)
                if r:
                    r["comparison"] = f"multi vs none @none_alpha={alpha}"
                    per_alpha.append(r)
    if per_alpha:
        lines += ["\n### multi vs none 按匹配强度拆分", fmt_table(per_alpha)]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(G.SEEDS))
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    data = load_rows(args.seeds)
    if not data:
        sys.exit("找不到 Step 1 结果：请先运行 step1_independent_eval.py")
    present = set().union(*({r["arm"] for r in rows} for rows in data.values()))
    keys = [k for k in KEYS if any(np.isfinite(r.get(k, np.nan)) for rows in data.values() for r in rows)]
    lines = ["# Step 2 报告：风格误差匹配后的比较",
             f"seeds={sorted(data)}  arms={sorted(present)}  metrics={keys}",
             "独立于训练目标的指标：ssim_luma(↑) edge_f1(↑) lpips_content(↓)；"
             "pixel_l1 / pooled_intensity_l1 也未进入训练但对合理的颜色变化敏感。"]

    base = [a for a in BASE_ARMS if a in present]
    df = analyze(data, base, keys, args.n_boot, "A. 原四个条件（复现原协议）", lines)
    extra = [a for a, _ in G.EXTRA_ARMS if a in present]
    if extra:
        analyze(data, base + extra, keys, args.n_boot,
                f"B. 加入 {' / '.join(extra)}（所有条件共同覆盖的点）", lines)
    if not df.empty:
        df.to_csv(G.RESULTS / "matched_ext.csv", index=False)
    out = G.RESULTS / "step2_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n完成。结果已保存：{out}")


if __name__ == "__main__":
    main()
