_base_ = '../pss_loveda.py'

model = dict(decode_head={'type': 'InitialGuidanceControlHead'})
work_dir = 'work_dirs/initial_guidance_loveda'
