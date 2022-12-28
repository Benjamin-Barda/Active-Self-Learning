import torch
import torch.nn as nn

from torchvision.models import resnet18

class BackboneEncoder(nn.Module) : 

  '''
  Rotation Prediction Pretext Task using Resnet Backbone
  '''

  def __init__(self, 
               num_rotations : int,
               ) -> None:
    super(BackboneEncoder, self).__init__()


    self.resnet = resnet18()

    # Dimension after the avg pool layer
    out_dim = self.resnet.fc.weight.shape[1]

    # Cut fc layer so we can later load state_dict more easyly
    self.resnet.fc = nn.Identity()

    self.rotation_prediction_head = nn.Linear(out_dim, num_rotations, bias=False)

  def forward(self, x) : 

    feat = self.resnet(x)
    return self.rotation_prediction_head(feat)

    



