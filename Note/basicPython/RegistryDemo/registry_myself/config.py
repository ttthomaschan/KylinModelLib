backbone = dict(
    type = 'ResNet',
    depth = 50,
    num_stages = 4,
    out_indices = (0,1,2,3),
    frozen_stages = 1,
    norm_cfg = dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch'
)

