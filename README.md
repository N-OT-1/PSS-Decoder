# PSS-Decoder

Coordinated semantic-spatial refinement for remote sensing semantic segmentation with a MambaVision-Tiny encoder.

## Progress

- Init
- Initial contribution
- Under review

## Installation

Use Linux with Python 3.10 or newer. Install PyTorch and a compatible CUDA-enabled MMCV 2.0/2.1 build, Mamba (`mamba-ssm`) and `causal-conv1d` following their upstream instructions. Then install:

```bash
pip install -r requirements.txt
pip install -e .
```

Download the [MambaVision-T ImageNet-1K checkpoint](https://huggingface.co/nvidia/MambaVision-T-1K) to `checkpoints/mambavision_tiny_1k.pth.tar`.

## Data

Use the following prepared directory layouts:

```text
data/loveda/{train,val}/{Urban,Rural}/{images_png,masks_png}/
data/vaihingen/{img_dir,ann_dir}/{train,val}/
data/isaid/{train,val}/{images,Semantic_masks}/
```

LoveDA uses its seven-class training/validation split. Vaihingen uses six-class 512-pixel tiles from disjoint source-area partitions, with no-boundary labels and within-area overlap. iSAID uses semantic class-ID masks with 16 classes including background; validation preserves aspect ratio within 896-by-896 before mapping predictions to native label dimensions. These are separately supervised protocols. See [data preparation](docs/DATA.md) for label conventions and the Vaihingen split.

## Training and evaluation

```bash
python tools/train.py configs/pss_loveda.py --seed 101
python tools/train.py configs/pss_vaihingen.py --seed 101
python tools/train.py configs/pss_isaid.py --seed 101
python tools/test.py configs/pss_loveda.py work_dirs/pss_loveda/iter_20000.pth
python tools/predict.py configs/pss_loveda.py path/to/checkpoint.pth --image path/to/image.png --output outputs/mask.png
```

Use `--data-root`, `--pretrained`, and `--work-dir` to change input/output locations. `--seed 202` and `--seed 303` select the other main-comparison seeds. Each configuration trains for 20k updates with the same classification initialization, optimizer schedule, augmentation and objective. The primary endpoint is `iter_20000.pth`; best-validation checkpoints are separate outputs. Test inference is FP32 and single-scale without augmentation. It reports mIoU and native-pixel Boundary IoU at widths 1, 2 and 4, with width 2 as the primary boundary setting.

## Implementation and comparisons

The main head is `StateDualPathHead` in `pss_decoder/decoder.py`: width 128, global routing and residual scale 0.5. Names and parameter keys are retained for checkpoint compatibility.

Matched configurations in `configs/` include the SegFormer-style Base, UPerNet, progressive refinement, equal routing and feature reclassification. LoveDA branch, allocation, context and information-source controls are under `configs/controls/`. The ACF configuration is a matched-budget operator adaptation, rather than the original authors' complete implementation. The feature-reclassification control retains the branch-classifier tensors to match construction, although those classifiers are inactive in its forward pass. Source replacement controls use stage-constant allocation; they are not single-branch removals.

## License and acknowledgments

Built on [MambaVision](https://github.com/NVlabs/MambaVision), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), and [MMCV](https://github.com/open-mmlab/mmcv). MambaVision-derived code is subject to the NVIDIA Source Code License-NC. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). Datasets and weights must be obtained separately.
