import torch
import torch.nn as nn




class BackBoneEncoder(nn.Module) : 
  '''
  SimSiam - Simple Siamese https://arxiv.org/pdf/2011.10566.pdf  
  '''

  def __init__(self, base_encoder, dim, pred_dim, in_pretrain = True):

    super(BackBoneEncoder, self).__init__()
    self.in_pretrain = in_pretrain
    self.encoder = base_encoder(num_classes = dim, zero_init_residual=True) 

    self.last_dim = self.encoder.fc.in_features

    self.encoder.fc = nn.Sequential(
        nn.Linear(self.last_dim, self.last_dim, bias = False), 
        nn.BatchNorm1d(self.last_dim),
        nn.Mish(inplace = True),
        nn.Linear(self.last_dim, self.last_dim, bias = False), 
        nn.BatchNorm1d(self.last_dim),
        nn.Mish(inplace=True),
        self.encoder.fc,
        nn.BatchNorm1d(dim, affine=False)
    )
    self.encoder.fc[6].bias.requires_grad = False

    self.predictor = nn.Sequential(
        nn.Linear(dim, pred_dim, bias = False), 
        nn.BatchNorm1d(pred_dim),
        nn.Mish(inplace = True), 
        nn.Linear(pred_dim, dim) 
    )

  def forward(self, x1, x2 = None) : 

    if x2 is None and self.in_pretrain: 
      raise ValueError("Expected 2 images but got 1 -> Are you in train or pretrain")

    if self.in_pretrain: 
      z1 = self.encoder(x1)
      z2 = self.encoder(x2)

      p1 = self.predictor(z1)
      p2 = self.predictor(z2)

      # Detach as the stop gradient operation
      return p1, p2, z1.detach(), z2.detach()
    
    else :
      return self.encoder(x1)



   