import os.path as osp
import mmengine
from mmseg.datasets import LoveDADataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class LoveDADatasetRV(LoveDADataset):

    def __init__(self, split='train', regions=('Urban', 'Rural'), img_subdir='images_png', seg_subdir='masks_png', **kwargs):
        self.split = split
        self.regions = tuple(regions)
        self.img_subdir = img_subdir
        self.seg_subdir = seg_subdir
        super().__init__(**kwargs)

    def load_data_list(self):
        img_root = self.data_prefix.get('img_path', None)
        ann_root = self.data_prefix.get('seg_map_path', None)
        if img_root is None or ann_root is None:
            raise ValueError('LoveDADatasetRV requires img_path and seg_map_path in data_prefix')
        _suffix_len = len(self.img_suffix)
        data_list = []
        for region in self.regions:
            img_dir = osp.join(img_root, self.split, region, self.img_subdir)
            ann_dir = osp.join(ann_root, self.split, region, self.seg_subdir)
            if not osp.isdir(img_dir):
                raise FileNotFoundError(f'Image directory not found: {img_dir}. Check data_root and split={self.split}.')
            for img in mmengine.utils.scandir(img_dir, self.img_suffix, recursive=False):
                data_info = dict(img_path=osp.join(img_dir, img))
                seg_map = img[:-_suffix_len] + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        return sorted(data_list, key=lambda x: x['img_path'])
