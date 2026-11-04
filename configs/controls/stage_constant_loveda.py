_base_ = '../pss_loveda.py'

model = dict(decode_head={'type': 'StageConstantRoutingHead'})
work_dir = 'work_dirs/stage_constant_loveda'
