_base_ = '../pss_loveda.py'

model = dict(decode_head={'type': 'SameSourceConstantHead', 'source': 'shallow'})
work_dir = 'work_dirs/shallow_only_loveda'
