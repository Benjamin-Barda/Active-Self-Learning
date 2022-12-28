import torch
import torch.nn as nn



class LinearClassifier(nn.Module): 
    def __init__(self, backbone, num_classes) : 

        super(LinearClassifier, self).__init__()

        self.backbone = backbone

        for _, param in self.backbone.named_parameters(): 
            if param.requires_grad : 
                param.requires_grad = False

        backbone_last_dim = self.backbone.dim

        self.head = nn.Sequential(
            nn.Linear(backbone_last_dim, 10, bias=False), 
        )

    def forward(self, x): 
        x = self.backbone(x)

        return self.head(x)
