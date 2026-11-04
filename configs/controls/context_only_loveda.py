_base_ = '../pss_loveda.py'

model = dict(decode_head={'type': 'SameSourceConstantHead', 'source': 'context'})
work_dir = 'work_dirs/context_only_loveda'
