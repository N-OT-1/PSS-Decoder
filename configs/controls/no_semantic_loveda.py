_base_ = '../pss_loveda.py'

model = dict(decode_head={'type': 'NoSemanticCorrectionHead'})
work_dir = 'work_dirs/no_semantic_loveda'
