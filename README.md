# Structure-Preserving Controllable Style Transfer: A Controlled Evaluation Study

How much does an explicit structure constraint help reference-based style transfer, and does the answer depend on how you measure it?

This repository fine-tunes a pretrained **AdaIN** decoder under different structure constraints. Each constraint is compared **at matched style strength** on held-out metrics, and every claim is confirmed with a **pre-registered, one-shot test-set evaluation**.

<p align="center">
  <img src="evaluation/results/test/figures/test_arms_alpha075.png" width="100%"><br>
  <em>Test set, first four test contents (not cherry-picked), α = 0.75. Columns: content, reference, then each training condition.</em>
</p>

## Key findings

1. **Structure constraints genuinely help, and not only on the metric they optimize.** At matched style error, the multiscale constraint improves three metrics never used in training: SSIM, edge-F1 and LPIPS. The 95% confidence intervals exclude zero, and the direction is consistent across 8/8 held-out reference paintings.
2. **No single structural prior dominates.** The gradient-pyramid loss is best on structural similarity (SSIM). A Laplacian loss ([Li et al., 2017](#references)) is best on perceptual similarity (LPIPS). A pre-specified hybrid lies between them. The ranking is identical on the validation and test sets.
3. **Plain fine-tuning does not explain the gain.** Fine-tuning without a structure term does not improve test-set SSIM over the pretrained decoder; the improvement comes from the structure constraint itself.
4. **Training-aligned metrics overstate the advantage.** The multiscale loss lowers its own structure error by 3.6%, but its gains on independent metrics are 0.3–1.3%.

## Method

- **Backbone:** pretrained AdaIN ([Huang & Belongie, 2017](#references)), using the [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) port. The VGG encoder is frozen and only the decoder is fine-tuned.
- **Strength control:** feature-space interpolation `f_α = (1 − α)·f_c + α·AdaIN(f_c, f_s)` before decoding, not pixel blending.
- **Multiscale structure loss:** signed Sobel gradients of luminance, normalized by magnitude, compared on a 3-level Gaussian pyramid with edge-aware weighting. Scale weights are 0.2 / 0.3 / 0.5 from fine to coarse.
- **Objective:** `L = L_content + 10·L_style + L_structure`. The training budget is identical for every condition (500 Adam updates, lr 1e-5).

| Condition | Structure term |
|---|---|
| `pretrained` | none (untouched pretrained decoder) |
| `none` | fine-tuned, no structure term |
| `single` | normalized Sobel loss, full resolution only |
| `multi` | normalized Sobel loss on a 3-level Gaussian pyramid |
| `laplacian` | Laplacian loss on 4×4 and 16×16 pooled images ([Li et al., 2017](#references)); weight magnitude-matched to `multi` |
| `combo` | 0.5 × multi + 0.5 × laplacian (fixed in advance, not tuned) |

## Evaluation protocol

- **Matched style error.** Comparing at equal α rewards models that simply stylize less. For each content–reference pair, each condition's metric curve is interpolated at the style error the `none` condition reaches at α = 0.25 / 0.5 / 0.75. Points are kept only where every condition's style error is monotone and overlapping, with no extrapolation.
- **Held-out metrics** (computed against the content image, never used in training): SSIM on luminance, Canny edge-F1 with 2 px tolerance, and LPIPS (AlexNet).
- **Statistics.** Pairs share content images and reference images, so they are not independent. Confidence intervals use a **crossed bootstrap** that resamples contents and references separately (10,000 draws). Sign tests are run at the content level.
- **Seeds.** Three training schedules (17, 43, 101), all starting from identical pretrained weights.
- **Reproducibility check.** The evaluation pipeline reproduces the original experiment's 3,840 per-pair records bit-exactly. The baseline training loop reproduces saved training losses to within 4×10⁻⁴.
- **Pre-registered test set.** 32 contents × 8 references (256 pairs) drawn from a split never used for model selection. The selection rule, all 16 model hashes (15 fine-tuned decoders + the pretrained decoder) and the analysis plan were written to [`PROTOCOL.json`](evaluation/results/test/PROTOCOL.json) before running. The evaluation was run once.

## Results (test set, matched style error)

Mean over 3 seeds; 2,292 of 2,304 candidate points retained.

| Condition | Multiscale struct. err ↓ | SSIM ↑ | Edge-F1 ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| pretrained | 0.4404 | 0.3386 | 0.6286 | 0.4104 |
| none | 0.4360 | 0.3367 | 0.6303 | 0.4054 |
| single | 0.4323 | 0.3386 | 0.6309 | 0.4053 |
| **multi** | **0.4202** | **0.3410** | 0.6330 | 0.4042 |
| **laplacian** | 0.4233 | 0.3396 | **0.6337** | **0.4030** |
| combo | 0.4216 | 0.3405 | 0.6336 | 0.4035 |

Relative change vs. `none` on the held-out metrics (test set). Every listed change has a 95% crossed-bootstrap CI excluding zero, except where marked.

| Condition | SSIM | Edge-F1 | LPIPS |
|---|---:|---:|---:|
| single | +0.58% | +0.08% | −0.03% (n.s.) |
| multi | **+1.28%** | +0.43% | −0.30% |
| laplacian | +0.87% | **+0.53%** | **−0.60%** |
| combo | +1.13% | +0.51% | −0.47% |

The validation set (16 × 4 pairs) gives the same ranking and nearly identical effect sizes (multi vs. none: SSIM +1.29%, edge-F1 +0.32%, LPIPS −0.27%). Full tables: [`evaluation/results/step2_summary.txt`](evaluation/results/step2_summary.txt) (validation) and [`evaluation/results/test/test_summary.txt`](evaluation/results/test/test_summary.txt) (test).

## Reproducing

```bash
pip install -r requirements.txt

# 1. Data: vangogh2photo from the CycleGAN repository, extracted so that
#    <DATA>/trainA, trainB, testA, testB exist.
#    See https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
#    The split and duplicate review used here are in configs/vangogh_split_v1.json.
export STYLIZATION_DATA_ROOT=/path/to/vangogh2photo

# 2. Download and hash-verify the upstream AdaIN code and weights
python scripts/qualify_adain.py --data-root $STYLIZATION_DATA_ROOT

# 3. Original three-seed experiment (pretrained / none / single / multi)
python scripts/run_adain_confirmation.py --data-root $STYLIZATION_DATA_ROOT --output outputs/my_run

# 4. Held-out metrics, baselines and the pre-registered test evaluation
python evaluation/step1_independent_eval.py
python evaluation/step3_laplacian.py                 # Laplacian baseline
python evaluation/step3_laplacian.py --mode combo    # hybrid
python evaluation/step2_analyze.py                   # validation analysis
python evaluation/step4_test_eval.py                 # one-shot test evaluation
```

The evaluation scripts read the original experiment from `outputs/adain_confirmation_v1/`. Per-pair metrics, training logs and plans for that experiment are included, so `step2_analyze.py` can be re-run without a GPU.

## Repository layout

```
scripts/                       training, original analysis, inference, demo
  adain_structure_research.py  single- and multiscale structure losses
  run_adain_confirm_worker.py  per-seed training + validation evaluation
src/stylization/               data loading (manifest-based split)
configs/                       data split manifest, release config
outputs/adain_confirmation_v1/ original experiment: plans, logs, per-pair metrics
evaluation/                    held-out metrics, baselines, test evaluation
  results/                     all summaries, per-pair CSVs, test protocol
```

## Limitations

- Small evaluation: 3 training schedules, one artistic domain (Van Gogh), 256 × 256 resolution.
- Effect sizes are small (0.3–1.3% on held-out metrics). They are statistically consistent, but no human preference study was run.
- SSIM and edge-F1 are independent of the training objective but conceptually close to it. LPIPS is the most independent metric, and there the Laplacian baseline is best.
- The Laplacian weight was magnitude-matched rather than tuned; a tuned baseline could perform differently.
- Overlap between the upstream AdaIN pretraining data and this dataset is unknown.

## References

- X. Huang, S. Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.* ICCV 2017.
- S. Li, X. Xu, L. Nie, T.-S. Chua. *Laplacian-Steered Neural Style Transfer.* ACM Multimedia 2017.
- J.-Y. Zhu, T. Park, P. Isola, A. A. Efros. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.* ICCV 2017 (vangogh2photo dataset).
- N. Inoue. *pytorch-AdaIN.* https://github.com/naoto0804/pytorch-AdaIN

## License

Code in this repository: MIT (see [`LICENSE`](LICENSE)). The upstream AdaIN port, its weights and the dataset images are subject to their own terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
