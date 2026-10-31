#!/usr/bin/env python3
"""
Step 4：测试集最终评估（预注册，一次性）

为什么：此前所有结论都来自验证集，而验证集参与过方案选择。测试集从未用于调参，
在这里评估一次，才能说结论"在未参与选择的数据上成立"。

规则（运行前写入 results/test/PROTOCOL.json 锁定，之后不改）：
  - 不重新训练：只评估已有模型 —— pretrained、none、single、multi（原实验 9 个解码器）、
    laplacian、combo（各 3 个种子）；所有模型文件记录 SHA-256；
  - 测试对：用原 worker 的 choose() 规则（固定哈希、每个相关组一张）从测试集选
    32 张内容图 × 8 张参考图 = 256 对，标签分别为 'test' 和 'test_styles'；
  - 强度 0/0.25/0.5/0.75/1；指标、风格误差匹配协议、统计方法与 Step 2 完全相同；
  - 结果只跑一次。若 COMPLETE 已存在，脚本拒绝重跑（--allow-rerun 会把旧结果改名保留，
    并在协议中记录重跑次数）。无论结果好坏都如实报告。

3070 Ti 上约 15–20 分钟。

用法（PowerShell，在项目根目录；需先完成 step3 的 laplacian 与 combo）：
  .\\.venv\\Scripts\\python.exe evaluation\\step4_test_eval.py

输出：
  results/test/PROTOCOL.json
  results/test/seed_*/ext_metrics.csv
  results/test/figures/test_arms_alpha075.png
  results/test/test_summary.txt
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_common as G  # noqa: E402
import step2_analyze as A  # noqa: E402
from step1_independent_eval import save_figure  # noqa: E402

TEST = G.RESULTS / "test"
BASE_ARMS = ["pretrained", "none", "single", "multi"]
N_CONTENT, N_STYLE = 32, 8


def checkpoint_path(seed, arm):
    if arm in ("none", "single", "multi"):
        return G.CONF_DIR / f"seed_{seed}" / f"{arm}_decoder.pt"
    return G.extra_checkpoint(seed, arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=G.DEFAULT_DATA_ROOT)
    ap.add_argument("--skip-combo", action="store_true", help="combo 未训练时仍运行（不推荐）")
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    seeds = list(G.SEEDS)
    extra = [a for a, _ in G.EXTRA_ARMS if not (a == "combo" and args.skip_combo)]
    missing = [str(checkpoint_path(s, a)) for a in extra for s in seeds if not checkpoint_path(s, a).exists()]
    if missing:
        sys.exit("缺少模型文件，请先运行 step3（--mode laplacian 和 --mode combo）：\n  " + "\n  ".join(missing))
    arms = BASE_ARMS + extra

    complete = TEST / "COMPLETE.json"
    reruns = 0
    if complete.exists():
        if not args.allow_rerun:
            sys.exit(f"测试集结果已存在（{complete}）。按预注册规则不重跑；确需重跑请加 --allow-rerun。")
        n = 1
        while (G.RESULTS / f"test_previous_{n}").exists():
            n += 1
        TEST.rename(G.RESULTS / f"test_previous_{n}")
        reruns = n

    t0 = time.monotonic()
    ctx = G.Context(args.data_root, use_lpips=not args.no_lpips)
    test = ctx.test_dataset(args.data_root)
    manifest = json.loads((G.ROOT / "configs" / "vangogh_split_v1.json").read_text(encoding="utf-8"))
    split = manifest["splits"]["test"]
    contents = G.choose(split["content"], N_CONTENT, "test")
    styles = G.choose(split["style"], N_STYLE, "test_styles")
    if hasattr(test, "__len__") and len(test) != len(test.content_files) * len(test.style_files):
        sys.exit(f"测试集索引规则与验证集不同（len={len(test)}），请检查数据加载代码。")

    protocol = {
        "purpose": "one-shot held-out test evaluation; protocol fixed before running",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "reruns_before_this": reruns,
        "arms": arms, "seeds": seeds,
        "test_contents": contents, "test_styles": styles,
        "selection": "choose() from run_adain_confirm_worker.py; labels 'test' (32) / 'test_styles' (8)",
        "alphas": list(G.ALPHAS),
        "metrics": {"independent": list(G.NEW_KEYS), "canny": G.CANNY, "edge_tol": G.EDGE_TOL,
                    "lpips": "alex" if ctx.lp is not None else "off"},
        "analysis": "match at none-arm style error (alpha .25/.5/.75), summarize_adain_confirmation interpolation; "
                    "crossed content x reference bootstrap (10000), content-level sign test",
        "checkpoints_sha256": {f"{a}/seed_{s}": G.sha256(checkpoint_path(s, a))
                               for a in arms if a != "pretrained" for s in seeds},
        "pretrained_decoder_sha256": G.sha256(G.ROOT / ".cache" / "adain_qualified" / "decoder.pth"),
    }
    TEST.mkdir(parents=True, exist_ok=True)
    (TEST / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"协议已锁定：{TEST / 'PROTOCOL.json'}  （{len(contents)} 内容 × {len(styles)} 参考）", flush=True)

    n_style = len(test.style_files)
    order = [(c * n_style + styles[i % len(styles)], 0.75) for i, c in enumerate(contents[:4])]
    report = []
    pretrained_rows, kept_by_arm = None, {}
    for seed in seeds:
        seed_rows = []
        for arm in arms:
            ts = time.monotonic()
            if arm == "pretrained" and pretrained_rows is not None:
                rows = pretrained_rows  # 预训练解码器与种子无关，输出逐位相同，只算一次
            else:
                state = (ctx.pretrained_state if arm == "pretrained" else
                         ctx.torch.load(checkpoint_path(seed, arm), map_location="cpu", weights_only=False)["decoder"])
                decoder = ctx.make_decoder(state)
                keep = set(order) if seed == seeds[0] else None
                rows, kept = ctx.evaluate(decoder, arm, None, keep=keep, dataset=test,
                                          contents=contents, styles=styles)
                if kept:
                    kept_by_arm[arm] = kept
                if arm == "pretrained":
                    pretrained_rows = rows
                del decoder
                if ctx.device.type == "cuda":
                    ctx.torch.cuda.empty_cache()
            seed_rows += rows
            line = f"seed {seed} {arm:<10} rows={len(rows)}  {time.monotonic() - ts:.0f}s"
            print(line, flush=True)
            report.append(line)
        G.write_csv(TEST / f"seed_{seed}" / "ext_metrics.csv", seed_rows)

    if len(kept_by_arm) == len(arms):
        save_figure(kept_by_arm, order, arms, TEST / "figures" / "test_arms_alpha075.png")

    data = A.load_rows(seeds, base=TEST)
    keys = [k for k in A.KEYS if any(r.get(k) == r.get(k) for rows in data.values() for r in rows)]
    lines = ["# Step 4 报告：测试集一次性评估（预注册）",
             f"test pairs: {len(contents)} contents × {len(styles)} references; seeds={seeds}; arms={arms}",
             f"lpips={'on' if ctx.lp is not None else 'off'}  reruns_before_this={reruns}",
             "协议见 results/test/PROTOCOL.json；以下统计方法与验证集 Step 2 相同。"]
    A.analyze(data, BASE_ARMS, keys, args.n_boot, "A. 原四个条件", lines, paper_check=False)
    A.analyze(data, arms, keys, args.n_boot, "B. 全部条件（含 laplacian / combo）", lines, paper_check=False)
    lines += ["", "## 运行记录", *report, f"total time: {time.monotonic() - t0:.0f}s"]
    (TEST / "test_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    complete.write_text(json.dumps({"finished": datetime.datetime.now().isoformat(timespec="seconds"),
                                    "rows_per_seed": len(seed_rows)}), encoding="utf-8")
    print(f"\n完成。结果已保存：{TEST / 'test_summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
