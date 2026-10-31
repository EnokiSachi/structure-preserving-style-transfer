#!/usr/bin/env python3
"""
Step 3：Laplacian 结构损失基线（Li et al., "Laplacian-Steered Neural Style Transfer", ACM MM 2017）

公平性设置（与原三组完全相同）：
  - 同一预训练解码器初始化、同一冻结编码器；
  - 直接读取各种子 plan.json 中保存的 train_schedule（同样的 500 个训练配对与顺序）；
  - 同样的 Adam(lr=1e-5)、torch.manual_seed(seed)、内容权重 1、风格权重 10；
  - 只把结构项换成 Laplacian 损失：对 RGB 图做 4×4 和 16×16 平均池化后，
    比较 3×3 Laplacian 响应的均方误差，两个尺度相加。

结构权重 λ：两种损失的数值量级不同，直接取 1 不公平。默认用预训练解码器在
种子 17 前 32 个训练配对上校准，使 λ·L_lap 的平均值等于 L_multi 的平均值（量级匹配），
校准结果写入报告。也可用 --lap-weight 手动指定。

另有 --replicate-check：先用本脚本的训练循环重训 "none" 条件（结构权重 0），
与已保存的 none_decoder.pt 比较，确认训练循环与原实验一致。

输出到 evaluation/results/laplacian/seed_*/（不触碰原实验目录），之后重跑 step2_analyze.py
即可得到五个条件的比较。

用法（PowerShell，在项目根目录）：
  # 1) 先做复现检查 + 快速试跑（约 2–3 分钟）
  .\\.venv\\Scripts\\python.exe evaluation\\step3_laplacian.py --replicate-check --seeds 17 --steps 20 --out evaluation\\results\\laplacian_smoke
  # 2) 正式运行三个种子（3070 Ti 约 10–20 分钟）
  .\\.venv\\Scripts\\python.exe evaluation\\step3_laplacian.py
  # 3) 重新分析
  .\\.venv\\Scripts\\python.exe evaluation\\step2_analyze.py

组合条件：加 --mode combo，输出到 results/combo/

输出：results/<mode>/step3_report.txt
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_common as G  # noqa: E402


def make_laplacian_loss(torch, F, pools):
    kernel = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def loss(generated, content):
        with torch.autocast(device_type=generated.device.type, enabled=False):
            k = kernel.to(generated.device)
            g, c = generated.float(), content.detach().float()
            total = generated.new_zeros((), dtype=torch.float32)
            for p in pools:
                lg = F.conv2d(F.avg_pool2d(g, p), k, groups=3)
                lc = F.conv2d(F.avg_pool2d(c, p), k, groups=3)
                total = total + F.mse_loss(lg, lc)
            return total
    return loss


def train_arm(ctx, plan, seed, steps, struct_fn, weight, label):
    """逐行对应 run_adain_confirm_worker.py 的训练循环，只替换结构项。"""
    torch, F = ctx.torch, ctx.F
    torch.manual_seed(seed)
    decoder = ctx.make_decoder(ctx.pretrained_state)
    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=plan["lr"])
    history, started = [], time.monotonic()
    schedule = plan["train_schedule"][:steps]
    for step, (c, s) in enumerate(schedule, 1):
        content, reference = ctx.load_train_pair(c, s)
        with torch.no_grad():
            cf, sf = ctx.encode(content), ctx.encode(reference)
            target = ctx.adain(cf[-1], sf[-1])
        optimizer.zero_grad(set_to_none=True)
        result = decoder(target)
        rf = ctx.encode(result)
        lc = F.mse_loss(rf[-1], target)
        ls = ctx.style_error(rf, sf)
        le = struct_fn(result, content) if weight else result.sum() * 0
        loss = lc + plan["style_weight"] * ls + weight * le
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite objective")
        loss.backward()
        optimizer.step()
        history.append({"step": step, "content": lc.item(), "style": ls.item(), "structure": le.item()})
        if step % 50 == 0 or step == len(schedule):
            print(f"  {label} seed {seed} {step}/{len(schedule)} elapsed={time.monotonic() - started:.1f}s", flush=True)
    return decoder, history


def calibrate(ctx, lap_loss, n=32):
    torch = ctx.torch
    plan = json.loads((G.CONF_DIR / "seed_17" / "plan.json").read_text(encoding="utf-8"))
    decoder = ctx.make_decoder(ctx.pretrained_state)
    multi, lap = [], []
    with torch.no_grad():
        for c, s in plan["train_schedule"][:n]:
            content, reference = ctx.load_train_pair(c, s)
            cf, sf = ctx.encode(content), ctx.encode(reference)
            result = decoder(ctx.adain(cf[-1], sf[-1]))
            multi.append(ctx.structure_loss(result, content, "multi").item())
            lap.append(lap_loss(result, content).item())
    m, l = sum(multi) / n, sum(lap) / n
    return m / l, m, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=G.DEFAULT_DATA_ROOT)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(G.SEEDS))
    ap.add_argument("--steps", type=int, default=None, help="默认用 plan.json 中的 500")
    ap.add_argument("--pools", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--lap-weight", type=float, default=None, help="默认按量级自动校准")
    ap.add_argument("--replicate-check", action="store_true")
    ap.add_argument("--mode", choices=["laplacian", "combo"], default="laplacian",
                    help="combo = 0.5×多尺度结构损失 + 0.5×λ×Laplacian 损失（事先固定，不调参）")
    ap.add_argument("--out", default=None, help="默认 results/<mode>")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    out = Path(args.out) if args.out else G.RESULTS / args.mode
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ctx = G.Context(args.data_root, use_lpips=not args.no_lpips)
    torch = ctx.torch
    lap_loss = make_laplacian_loss(torch, ctx.F, args.pools)
    report = [f"# Step 3 报告：{args.mode}", f"device={ctx.device}  pools={args.pools}  mode={args.mode}"]

    if args.lap_weight is None:
        lam, m, l = calibrate(ctx, lap_loss)
        report.append(f"lambda (auto, magnitude-matched) = {lam:.6g}   "
                      f"[mean L_multi={m:.6g}, mean L_lap={l:.6g} on 32 train pairs, pretrained decoder]")
    else:
        lam = args.lap_weight
        report.append(f"lambda (manual) = {lam:.6g}")
    print(report[-1], flush=True)

    if args.replicate_check:
        seed = args.seeds[0]
        plan = json.loads((G.CONF_DIR / f"seed_{seed}" / "plan.json").read_text(encoding="utf-8"))
        steps = args.steps or plan["steps"]
        dec, hist = train_arm(ctx, plan, seed, steps, None, 0.0, "replicate-none")
        saved = json.loads((G.CONF_DIR / f"seed_{seed}" / "none_training.json").read_text(encoding="utf-8"))
        loss_diff = max(abs(h["content"] - o["content"]) + abs(h["style"] - o["style"])
                        for h, o in zip(hist, saved))
        line = f"replicate-check seed {seed}, first {steps} steps: max |Δ training loss| vs saved none run = {loss_diff:.2e}"
        if steps == plan["steps"]:
            ref = ctx.arm_state(seed, "none")
            wdiff = max((dec.state_dict()[k].cpu() - ref[k]).abs().max().item() for k in ref)
            line += f"; max |Δ weight| = {wdiff:.2e}"
        line += "  (≤1e-3 视为一致；GPU 上 upsample 反向传播的微小非确定性可能带来 1e-6 量级差异)"
        print(line, flush=True)
        report.append(line)
        del dec

    if args.mode == "combo":
        def struct_fn(result, content):
            return 0.5 * ctx.structure_loss(result, content, "multi") + 0.5 * lam * lap_loss(result, content)
        weight = 1.0
        report.append("combo objective: content + 10*style + [0.5*L_multi + 0.5*lambda*L_lap] (fixed a priori)")
    else:
        struct_fn, weight = lap_loss, lam

    for seed in args.seeds:
        target = out / f"seed_{seed}"
        if (target / "COMPLETE.json").exists():
            report.append(f"seed {seed}: already complete, skipped")
            print(report[-1], flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        plan = json.loads((G.CONF_DIR / f"seed_{seed}" / "plan.json").read_text(encoding="utf-8"))
        steps = args.steps or plan["steps"]
        ts = time.monotonic()
        decoder, history = train_arm(ctx, plan, seed, steps, struct_fn, weight, args.mode)
        torch.save({"decoder": {k: v.cpu() for k, v in decoder.state_dict().items()}, "mode": args.mode,
                    "lambda": lam, "pools": args.pools, "steps": steps, "plan_seed": seed},
                   target / f"{args.mode}_decoder.pt")
        (target / f"{args.mode}_training.json").write_text(json.dumps(history), encoding="utf-8")
        rows, _ = ctx.evaluate(decoder, args.mode, plan)
        G.write_csv(target / "ext_metrics.csv", rows)
        (target / "COMPLETE.json").write_text(json.dumps({"seed": seed, "steps": steps, "lambda": lam,
                                                          "pools": args.pools, "rows": len(rows)}),
                                              encoding="utf-8")
        last = history[-1]
        report.append(f"seed {seed}: steps={steps} final(content={last['content']:.4f}, style={last['style']:.4f}, "
                      f"struct={last['structure']:.6f})  rows={len(rows)}  {time.monotonic() - ts:.0f}s")
        print(report[-1], flush=True)
        del decoder
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()

    report.append(f"total time: {time.monotonic() - t0:.0f}s")
    (out / "step3_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\n完成。结果已保存：{out / 'step3_report.txt'}；"
          f"正式运行后再执行 step2_analyze.py。", flush=True)


if __name__ == "__main__":
    main()
