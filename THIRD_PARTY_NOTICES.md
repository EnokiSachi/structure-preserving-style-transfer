# Third-party notices

## pytorch-AdaIN
The AdaIN network definition (`net.py`, `function.py`) and pretrained weights
(`decoder.pth`, `vgg_normalised.pth`) come from https://github.com/naoto0804/pytorch-AdaIN
(commit 47950d0e6656a95a80a4b105c4c0f58d38ef785c), MIT License, Copyright (c) 2017 Naoto Inoue.
They are **not** redistributed here. `scripts/qualify_adain.py` downloads them and verifies
their SHA-256 hashes against `configs/adain_release_v1.json` / the recorded sources.

## Dataset
Experiments use the `vangogh2photo` dataset released with CycleGAN
(Zhu et al., ICCV 2017; https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Dataset images are **not** redistributed. The figure in `evaluation/results/test/figures/`
shows a small number of dataset images and model outputs for research illustration only.

## Model checkpoints
Fine-tuned decoder checkpoints are derived from the pretrained AdaIN decoder and are not
included in the repository. They can be regenerated with the scripts provided.
