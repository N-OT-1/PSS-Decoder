import argparse
from _common import load_config


def main():
    from mmengine.config import DictAction
    from mmengine.runner import Runner
    parser = argparse.ArgumentParser(description='Evaluate a final segmentation checkpoint.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--work-dir')
    parser.add_argument('--data-root')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    cfg = load_config(args)
    cfg.model.backbone.pretrained = None
    cfg.load_from = args.checkpoint
    cfg.resume = False
    Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()
