from torch import nn 
from torchvision.models import resnet18

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        hidden_dim = out_dim

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False) 
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class prediction_MLP(nn.Module):
    
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x



class SimSiam(nn.Module) : 

    def __init__(self) -> None:
        super(SimSiam, self).__init__()

        self.backbone = resnet18()
        out_dim = self.backbone.fc.weight.shape[1]

        self.backbone.fc = nn.Identity()

        self.projector = projection_MLP(
            out_dim,
            2048
        )

        self.encoder = nn.Sequential(
            self.backbone, 
            self.projector
        )

        self.predictor = prediction_MLP()

    def forward(self, x1, x2 ):

        z1 = self.encoder(x1) 
        z2 = self.encoder(x2) 

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()





if __name__ == "__main__" : 
    model = SimSiam()
    print(1)
