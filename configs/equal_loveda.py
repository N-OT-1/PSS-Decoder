_base_ = './pss_loveda.py'

model = {'decode_head': {'_delete_': True,
                 'align_corners': False,
                 'channels': 128,
                 'dropout_ratio': 0.1,
                 'in_channels': [80, 160, 320, 640],
                 'in_index': [0, 1, 2, 3],
                 'loss_decode': [{'loss_weight': 1.0,
                                  'type': 'CrossEntropyLoss',
                                  'use_sigmoid': False},
                                 {'eps': 0.001,
                                  'ignore_index': 255,
                                  'loss_name': 'loss_dice',
                                  'loss_weight': 0.4,
                                  'naive_dice': False,
                                  'type': 'PixelMaskedDiceLoss',
                                  'use_sigmoid': True}],
                 'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
                 'num_classes': 7,
                 'residual_scale': 0.5,
                 'routing': 'global',
                 'type': 'UniformRoutingHead'}}
work_dir = 'work_dirs/equal_loveda'
