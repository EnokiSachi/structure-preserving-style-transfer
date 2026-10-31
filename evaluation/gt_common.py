"""
gt_common.py —— 本轮提升脚本的共用模块（只读引用项目代码，不修改任何原文件）

内容：
  - Context：按 run_adain_confirm_worker.py 完全相同的方式加载 AdaIN 编码器/解码器、
    验证集与训练集、结构损失，并提供与原评估逐行一致的 evaluate()，
    在原有 7 项指标之外同时计算 3 项独立指标。
  - 独立指标：ssim_luma（亮度 SSIM，↑）、edge_f1（Canny 边缘 F1，容差 2px，↑）、
    lpips_content（LPIPS-Alex 感知距离，↓）。均在裁剪到 [0,1] 的输出上计算，与内容图比较。
  - 统计：原协议的插值函数（逐字复制）、交叉 bootstrap、内容层面符号检验。

放置位置：<项目根目录>/evaluation/gt_common.py
"""
import copy
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "outputs" / "adain_confirmation_v1"
RESULTS = ROOT / "evaluation" / "results"
DEFAULT_DATA_ROOT = os.environ.get("STYLIZATION_DATA_ROOT", "data/vangogh2photo")
SEEDS = (17, 43, 101)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ORIG_KEYS = ("style", "content_feature", "pixel_l1", "pooled_intensity_l1",
             "single_structure", "multi_structure", "clipped_fraction")
NEW_KEYS = ("ssim_luma", "edge_f1", "lpips_content")
HIGHER_BETTER = {"ssim_luma": True, "edge_f1": True, "lpips_content": False,
                 "multi_structure": False, "single_structure": False, "content_feature": False,
                 "pixel_l1": False, "pooled_intensity_l1": False}
CANNY = dict(sigma=1.5, low_threshold=0.1, high_threshold=0.2)
EDGE_TOL = 2


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 独立指标（numpy，无 torch 依赖）
def luma_np(rgb01):
    """rgb01: HxWx3 或 3xHxW 的 [0,1] 数组 -> HxW float64 亮度"""
    a = np.asarray(rgb01, dtype=np.float64)
    if a.shape[0] == 3 and a.ndim == 3:
        a = np.moveaxis(a, 0, -1)
    a = np.clip(a, 0.0, 1.0)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def edges(y):
    from skimage.feature import canny
    return canny(y, **CANNY)


_DISK = None


def edge_f1(ec, eo, tol=EDGE_TOL):
    from scipy.ndimage import binary_dilation
    global _DISK
    if _DISK is None:
        yy, xx = np.ogrid[-tol:tol + 1, -tol:tol + 1]
        _DISK = (xx * xx + yy * yy) <= tol * tol
    n_c, n_o = int(ec.sum()), int(eo.sum())
    if n_c == 0 and n_o == 0:
        return 1.0
    if n_c == 0 or n_o == 0:
        return 0.0
    p = (eo & binary_dilation(ec, _DISK)).sum() / n_o
    r = (ec & binary_dilation(eo, _DISK)).sum() / n_c
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def ssim_luma(yc, yo):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(yc, yo, data_range=1.0, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))


# ---------------------------------------------------------------- 统计
def interpolate(curve, target, key):
    """逐字复制自 scripts/summarize_adain_confirmation.py，保证匹配协议一致。"""
    curve = sorted(curve, key=lambda r: r["alpha"])
    if any(b["style"] >= a["style"] for a, b in zip(curve, curve[1:])):
        return None
    for a, b in zip(curve, curve[1:]):
        if b["style"] <= target <= a["style"]:
            t = (target - a["style"]) / (b["style"] - a["style"])
            return a[key] + t * (b[key] - a[key])
    return None


def crossed_bootstrap_ci(D, n_boot=10000, seed=0):
    """D: 内容 × 参考 的差值矩阵（可含 NaN）。内容与参考各自独立有放回重采样。"""
    rng = np.random.default_rng(seed)
    nc, ns = D.shape
    ci = rng.integers(nc, size=(n_boot, nc))
    si = rng.integers(ns, size=(n_boot, ns))
    sub = D[ci[:, :, None], si[:, None, :]].reshape(n_boot, -1)
    with np.errstate(all="ignore"):
        boots = np.nanmean(sub, axis=1)
    boots = boots[np.isfinite(boots)]
    if not len(boots):
        return float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def sign_test_p(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return 1.0
    k = int((d > 0).sum())
    m = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return [{k: (v if k == "arm" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


# ---------------------------------------------------------------- 模型与评估（torch）
class Context:
    """与 run_adain_confirm_worker.py 相同的加载流程。"""

    def __init__(self, data_root, use_lpips=True):
        for p in (ROOT / "src", ROOT / "scripts", ROOT / ".cache" / "adain_qualified"):
            s = str(p.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        cache = ROOT / ".cache" / "adain_qualified"
        provenance = json.loads((cache / "sources.json").read_text())
        for name, record in provenance.items():
            if sha256(cache / name) != record["sha256"]:
                raise ValueError("Upstream asset changed: " + name)
        import net
        from function import adaptive_instance_normalization, calc_mean_std
        from stylization.external_split_data import ExternalManifestDataset
        from adain_structure_research import structure_loss
        self.net, self.adain, self.calc_mean_std, self.structure_loss = \
            net, adaptive_instance_normalization, calc_mean_std, structure_loss

        torch.set_num_threads(4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        net.vgg.load_state_dict(torch.load(cache / "vgg_normalised.pth", map_location="cpu", weights_only=True), strict=True)
        net.decoder.load_state_dict(torch.load(cache / "decoder.pth", map_location="cpu", weights_only=True), strict=True)
        layers = list(net.vgg.children())
        self.encoder = torch.nn.Sequential(*layers[:31]).to(self.device).eval().requires_grad_(False)
        self.pretrained_state = copy.deepcopy(net.decoder.state_dict())

        manifest_path = ROOT / "configs" / "vangogh_split_v1.json"
        self.train = ExternalManifestDataset(manifest_path, Path(data_root), "train", 256, 42)
        self.val = ExternalManifestDataset(manifest_path, Path(data_root), "validation", 256, 42)

        self.lp = None
        if use_lpips:
            try:
                import lpips
                self.lp = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
            except Exception as e:  # 常见原因：首次运行需联网下载 AlexNet 权重
                print(f"[警告] LPIPS 加载失败，将跳过该指标：{e}", flush=True)

    # 与原 worker 相同
    def encode(self, x):
        outputs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in (3, 10, 17, 30):
                outputs.append(x)
        return outputs

    def style_error(self, left, right):
        F = self.F
        return sum(F.mse_loss(a, b) for g, s in zip(left, right)
                   for a, b in zip(self.calc_mean_std(g), self.calc_mean_std(s)))

    def make_decoder(self, state):
        d = copy.deepcopy(self.net.decoder)
        d.load_state_dict(state, strict=True)
        return d.to(self.device).eval()

    def arm_state(self, seed, arm):
        if arm == "pretrained":
            return self.pretrained_state
        ck = self.torch.load(CONF_DIR / f"seed_{seed}" / f"{arm}_decoder.pt", map_location="cpu", weights_only=False)
        if ck.get("mode") != arm:
            raise ValueError(f"checkpoint mode mismatch: {ck.get('mode')} != {arm}")
        return ck["decoder"]

    def load_pair(self, dataset_index):
        s = self.val[dataset_index]
        content = ((s["content"] + 1) / 2).unsqueeze(0).to(self.device)
        reference = ((s["style"] + 1) / 2).unsqueeze(0).to(self.device)
        return content, reference

    def load_train_pair(self, c, s):
        content = ((self.train.read_image(self.train.content_files[c]) + 1) / 2).unsqueeze(0).to(self.device)
        reference = ((self.train.read_image(self.train.style_files[s]) + 1) / 2).unsqueeze(0).to(self.device)
        return content, reference

    def test_dataset(self, data_root):
        if not hasattr(self, "_test"):
            from stylization.external_split_data import ExternalManifestDataset
            self._test = ExternalManifestDataset(ROOT / "configs" / "vangogh_split_v1.json",
                                                 Path(data_root), "test", 256, 42)
        return self._test

    def evaluate(self, decoder, arm, plan, keep=None, dataset=None, contents=None, styles=None):
        """逐行复刻原 evaluate()，并追加独立指标。keep: {(pair_index, alpha)} 需要保留的输出图。
        dataset/contents/styles 缺省时使用验证集与 plan 中的验证对。"""
        torch, F = self.torch, self.F
        decoder.eval()
        dataset = dataset if dataset is not None else self.val
        contents = contents if contents is not None else plan["validation_contents"]
        styles = styles if styles is not None else plan["validation_styles"]
        n_style = len(dataset.style_files)
        rows, kept, content_cache = [], {}, {}
        with torch.inference_mode():
            for c in contents:
                for s in styles:
                    index = c * n_style + s
                    sample = dataset[index]
                    content = ((sample["content"] + 1) / 2).unsqueeze(0).to(self.device)
                    reference = ((sample["style"] + 1) / 2).unsqueeze(0).to(self.device)
                    cf, sf = self.encode(content), self.encode(reference)
                    target = self.adain(cf[-1], sf[-1])
                    if c not in content_cache:
                        yc = luma_np(content[0].cpu().numpy())
                        content_cache[c] = (yc, edges(yc), content * 2 - 1)
                    yc, ec, content_m11 = content_cache[c]
                    for alpha in ALPHAS:
                        result = decoder(torch.lerp(cf[-1], target, alpha))
                        rf = self.encode(result)
                        row = {"arm": arm, "pair_index": index, "alpha": alpha,
                               "style": self.style_error(rf, sf).item(),
                               "content_feature": F.mse_loss(rf[-1], cf[-1]).item(),
                               "pixel_l1": F.l1_loss(result, content).item(),
                               "pooled_intensity_l1": F.l1_loss(F.avg_pool2d(result.mean(1, keepdim=True), 16),
                                                                F.avg_pool2d(content.mean(1, keepdim=True), 16)).item(),
                               "single_structure": self.structure_loss(result, content, "single").item(),
                               "multi_structure": self.structure_loss(result, content, "multi").item(),
                               "clipped_fraction": ((result < 0) | (result > 1)).float().mean().item()}
                        clamped = result.clamp(0, 1)
                        yo = luma_np(clamped[0].cpu().numpy())
                        row["ssim_luma"] = ssim_luma(yc, yo)
                        row["edge_f1"] = edge_f1(ec, edges(yo))
                        row["lpips_content"] = (self.lp(clamped * 2 - 1, content_m11).item()
                                                if self.lp is not None else float("nan"))
                        row["content_id"] = c
                        row["style_id"] = s
                        row["content_edge_density"] = float(ec.mean())
                        rows.append(row)
                        if keep and (index, alpha) in keep:
                            kept[(index, alpha)] = (content.cpu(), reference.cpu(), clamped.cpu())
        return rows, kept


def choose(records, count, label):
    """逐字复制自 run_adain_confirm_worker.py 的 choose()：按固定哈希排序、每个相关组只取一张。"""
    order = sorted(range(len(records)),
                   key=lambda i: hashlib.sha256(("42:adain:" + label + ":" + records[i]["pixel_sha256"]).encode()).hexdigest())
    result, groups = [], set()
    for i in order:
        if records[i]["group_id"] not in groups:
            result.append(i)
            groups.add(records[i]["group_id"])
        if len(result) == count:
            return result
    raise ValueError("Insufficient groups")


# 额外条件：(arm 名, 结果子目录)
EXTRA_ARMS = (("laplacian", "laplacian"), ("combo", "combo"))


def extra_checkpoint(seed, arm):
    return RESULTS / arm / f"seed_{seed}" / f"{arm}_decoder.pt"
