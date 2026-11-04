# Data preparation

## LoveDA

Obtain LoveDA from its [official repository](https://github.com/Junjue-Wang/LoveDA). Keep Urban/Rural subdirectories under train and val. Input masks use 0 for unlabeled pixels and 1 through 7 for valid classes. The dataset loader reduces labels to 0 through 6 and maps unlabeled pixels to 255.

## Vaihingen

Obtain imagery and no-boundary labels from [ISPRS](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx). Prepared images contain three channels; DSM input is not used. Training and validation tiles are 512-by-512 and use disjoint source areas. Within-area tiles overlap and are scored as tiles. The split used in the study is recorded in `vaihingen_split.json`.

Prepared class-ID masks use 0 for void, followed by impervious surfaces, building, low vegetation, tree, car and clutter/background as IDs 1 through 6. MMSegmentation's ISPRSDataset reduces valid labels to 0 through 5 and void to 255. Black pixels in the eroded source annotations remain void. These tile-level validation results are not an official full-scene leaderboard score.

## iSAID

Obtain iSAID from its [official project](https://captain-whu.github.io/iSAID/). Use full-image semantic masks under `train/Semantic_masks` and `val/Semantic_masks`, paired with the images. Masks must already contain class IDs 0 through 15, including background 0; do not reduce zero labels. Use the class order and palette from MMSegmentation's iSAIDDataset. The configurations apply training crops and resize validation images; do not pre-tile the validation split.

Dataset image and annotation files are not included in this repository.
