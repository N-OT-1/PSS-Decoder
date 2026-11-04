import argparse
from pathlib import Path
from _common import load_config


def main():
    from mmengine.config import DictAction
    from mmseg.apis import init_model, inference_model
    from PIL import Image
    parser = argparse.ArgumentParser(description='Save a class-ID segmentation mask.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    cfg = load_config(args)
    cfg.model.backbone.pretrained = None
    model = init_model(cfg, args.checkpoint, device=args.device)
    result = inference_model(model, args.image)
    labels = result.pred_sem_seg.data.squeeze().cpu().numpy().astype('uint8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(args.output)


if __name__ == '__main__':
    main()
