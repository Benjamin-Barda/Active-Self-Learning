from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Percentage of whole dataset to label at the beginning
__C.initial_budget = .1

# Percentage of whole dataset to label before stop labeling
__C.final_budget = 0.8

# How many sample to add evry cycle
__C.b = .005

# Stage2 training parameters
__C.stage2_bs = 32
__C.stage2_lr = 2
__C.stage2_num_workers = 1
__C.stage2_momentum = .9
__C.stage2_weigth_decay = 0
__C.stage2_num_epochs = 100


#Stage3 training parameters
__C.stage3_bs = 100
__C.stage3_num_workers = 1



# Do not change as this are related to the backbone training
__C.backbone_arch = "resnet18"
__C.backbone_dim = 2048
__C.backbone_pred_dim = 512

