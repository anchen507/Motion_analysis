from .flow_loss import unFlowLoss

class config_p:
    pass
cfg=config_p()

cfg.alpha=10
cfg.occ_from_back=True
cfg.type="unflow"
cfg.w_l1=0.15
cfg.w_scales=[1.0, 1.0, 1.0, 1.0, 0.0]
cfg.w_sm_scales=[1.0, 0.0, 0.0, 0.0, 0.0]
cfg.w_smooth=60.0
cfg.w_ssim=0.85
cfg.w_ternary=0.0
cfg.warp_pad="border"
cfg.with_bk=True

def get_loss():
    if cfg.type == 'unflow':
        loss = unFlowLoss(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return loss
