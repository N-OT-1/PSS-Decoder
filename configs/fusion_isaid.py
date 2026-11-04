_base_ = './pss_isaid.py'

model = dict(decode_head=dict(type='SimpleFeatureFusionHead'))
work_dir = 'work_dirs/fusion_isaid'
