from easydict import EasyDict as edict

__C = edict()
cfg = __C


__C.current_epoch = 0
__C.num_epochs = 800

__C.batch_size = 100

__C.lr = 0.05
__C.weight_decay = 0.0001
__C.momentum = 0.9

__C.backbone_arch = "resnet18"
__C.backbone_dim = 2048
__C.backbone_pred_dim = 512


__C.path_to_checkpoint = "checkpoint/"