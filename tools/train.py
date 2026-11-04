import argparse
from _common import load_config


def main():
    from mmengine.config import DictAction
    from mmengine.runner import Runner
    parser = argparse.ArgumentParser(description='Train PSS-Decoder or a matched comparison.')
    parser.add_argument('config')
    parser.add_argument('--work-dir')
    parser.add_argument('--data-root')
    parser.add_argument('--pretrained')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    cfg = load_config(args)
    if args.pretrained:
        cfg.model.backbone.pretrained = args.pretrained
    if args.seed is not None:
        cfg.randomness.seed = args.seed
    cfg.resume = args.resume
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
