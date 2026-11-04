from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(args):
    from mmengine.config import Config
    from mmseg.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if getattr(args, 'data_root', None):
        cfg.data_root = args.data_root
        for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            cfg[key].dataset.data_root = args.data_root
    if getattr(args, 'work_dir', None):
        cfg.work_dir = args.work_dir
    return cfg
