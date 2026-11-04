_base_ = '../pss_loveda.py'

model = dict(decode_head={'type': 'InitialContextConstantHead'})
work_dir = 'work_dirs/initial_context_constant_loveda'
