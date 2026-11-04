_base_ = './pss_vaihingen.py'

model = dict(decode_head=dict(type='SimpleFeatureFusionHead'))
work_dir = 'work_dirs/fusion_vaihingen'
