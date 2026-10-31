# Frozen AdaIN confirmation results

Smoke mode: False

Validation only. Linear interpolation at the no-structure arm's per-pair style errors at alpha 0.25/0.5/0.75. All four arms must overlap and have strictly decreasing style error; unmatched or nonmonotonic curves are excluded. No extrapolation. This is an approximation from five strengths, not newly generated style-matched outputs.

Seeds vary training order and reference pairings; all start from identical pretrained weights. Images and styles are reused across seeds. Standard deviations below describe three training schedules, not independent datasets or statistical significance.

| Seed | Matched points / possible |
|---|---|
| 17 | 190 / 192 |
| 43 | 192 / 192 |
| 101 | 192 / 192 |

| Arm | Metric | Mean across seeds | SD across seeds |
|---|---|---:|---:|
| pretrained | multi_structure | 0.428099 | 0.001052 |
| pretrained | single_structure | 0.688465 | 0.000734 |
| pretrained | content_feature | 1.329837 | 0.021666 |
| pretrained | pixel_l1 | 0.151612 | 0.001817 |
| pretrained | pooled_intensity_l1 | 0.108495 | 0.002081 |
| none | multi_structure | 0.418970 | 0.002664 |
| none | single_structure | 0.687166 | 0.003096 |
| none | content_feature | 1.306300 | 0.011819 |
| none | pixel_l1 | 0.150663 | 0.001890 |
| none | pooled_intensity_l1 | 0.107628 | 0.002170 |
| single | multi_structure | 0.415336 | 0.002460 |
| single | single_structure | 0.685027 | 0.003060 |
| single | content_feature | 1.305269 | 0.011688 |
| single | pixel_l1 | 0.150527 | 0.001978 |
| single | pooled_intensity_l1 | 0.107574 | 0.002253 |
| multi | multi_structure | 0.404124 | 0.002053 |
| multi | single_structure | 0.683286 | 0.002989 |
| multi | content_feature | 1.300234 | 0.011435 |
| multi | pixel_l1 | 0.149503 | 0.002057 |
| multi | pooled_intensity_l1 | 0.106693 | 0.002228 |

Structure metrics and VGG metrics share the training objective or encoder. Pooled intensity L1 is an additional non-trained diagnostic, but is sensitive to legitimate color changes and is not perceptual ground truth. Pretraining overlap is unknown. Expanded validation includes the screening pairs. No automatic winner, test claim, novelty claim, or CV improvement percentage is authorized by this report.
