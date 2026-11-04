_base_ = '../pss_loveda.py'

model = dict(decode_head={'type': 'ConfidenceRoutingHead'})
work_dir = 'work_dirs/confidence_loveda'
