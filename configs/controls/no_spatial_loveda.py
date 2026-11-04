_base_ = '../pss_loveda.py'

model = dict(decode_head={'type': 'NoSpatialCorrectionHead'})
work_dir = 'work_dirs/no_spatial_loveda'
