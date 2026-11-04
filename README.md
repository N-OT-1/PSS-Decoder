# PSS-Decoder

Coordinated semantic-spatial refinement for remote sensing segmentation with MambaVision-Tiny.

## Progress

- Init
- Initial contribution
- Under review

## Setup

Use Linux and Python 3.10+. Install compatible PyTorch, MMCV 2.0/2.1, `mamba-ssm` and `causal-conv1d`, then run:

```bash
pip install -r requirements.txt
pip install -e .
```

Download the [MambaVision-T checkpoint](https://huggingface.co/nvidia/MambaVision-T-1K) to `checkpoints/mambavision_tiny_1k.pth.tar`.

## Data

```text
data/loveda/{train,val}/{Urban,Rural}/{images_png,masks_png}/
data/vaihingen/{img_dir,ann_dir}/{train,val}/
data/isaid/{train,val}/{images,Semantic_masks}/
```

LoveDA and Vaihingen masks use 0 for void and 1-based class IDs, reduced by their loaders. iSAID masks use IDs 0-15 including background. Vaihingen uses prepared six-class tiles from disjoint source areas with no-boundary labels.

## Usage

```bash
python tools/train.py configs/pss_loveda.py --seed 101
python tools/test.py configs/pss_loveda.py work_dirs/pss_loveda/iter_20000.pth
```

Use `pss_vaihingen.py` or `pss_isaid.py` for the other datasets. Locations can be changed with `--data-root`, `--pretrained` (training) and `--work-dir`. Configurations use 20k updates; evaluation uses the final checkpoint and single-scale mIoU. The decoder implementation is `pss_decoder/decoder.py`.

## License

MambaVision-derived code is subject to the NVIDIA Source Code License-NC. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). Obtain datasets and weights separately.
