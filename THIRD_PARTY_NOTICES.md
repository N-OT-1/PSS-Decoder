# Third-party notices

`pss_decoder/backbone.py` is derived from NVIDIA MambaVision and retains its original copyright and proprietary notices. The NVIDIA Source Code License-NC is included in LICENSE. Its non-commercial use terms apply to MambaVision-derived material.

This implementation uses MMSegmentation and MMCV. The progressive decoder uses the MMSegmentation decode-head interface; the class-center context operator follows ACFNet. Boundary evaluation uses fixed-width native-pixel inner bands adapted from Boundary IoU (Cheng et al., CVPR 2021), with a semantic void guard. See the manuscript for method attribution.

OpenMMLab dependencies are distributed separately under their applicable licenses. A copy of Apache License 2.0 is included in licenses/Apache-2.0.txt. PyTorch, timm, einops, Mamba, causal-conv1d, NumPy, OpenCV and Pillow retain their respective licenses.

Datasets and model weights are not redistributed. Obtain them from the respective providers under their applicable terms.
