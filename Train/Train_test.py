from detectron2.model_zoo import get_config
from detectron2.config import LazyConfig
print(LazyConfig.to_py(get_config('new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py')))

# cfg.model.backbone.bottom_up.ResNet.stages.norm = 'BN'
# cfg.model.backbone.bottom_up.ResNet.stem.norm = 'BN'
# cfg.model.backbone.norm = 'BN'
