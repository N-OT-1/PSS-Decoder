_base_ = '../pss_loveda.py'

model = dict(decode_head={'type': 'InitialContextDynamicHead'})
work_dir = 'work_dirs/initial_context_dynamic_loveda'
