from .pwclite import PWCLite
class config_m:
    pass
cfg_m=config_m()
cfg_m.n_frames=2
cfg_m.reduce_dense=True
cfg_m.type='pwclite'
cfg_m.upsample=True

def get_model():
    if cfg_m.type == 'pwclite':
        model = PWCLite(cfg_m)
    else:
        raise NotImplementedError(cfg_m.type)
    return model
