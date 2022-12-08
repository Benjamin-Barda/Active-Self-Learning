from easydict import EasyDict as edict

__C = edict()
cfg = __C


__C.initial_budget = .01
__C.final_budget = .1

__C.num_epochs = 70

__C.batch_size = 100

__C.lr = 0.05
__C.weight_decay = 0.0001
__C.momentum = 0.9

__C.backbone_arch = "resnet18"
__C.backbone_dim = 2048
__C.backbone_pred_dim = 512

