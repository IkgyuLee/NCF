import torch
import torch.nn as nn

from MLP import MLP
from GMF import GMF

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors, num_layers, neumf):
        super(NeuMF, self).__init__()
        """
              num_users   : number of users
              num_items   : number of items
              num_factors : number of predictive factors
              num_layers  : number of hidden layers in MLP Model
              neumf       : True(Fusion MLP&GMF)/False(Only MLP)

        """
        self.MLP = MLP(num_users, num_items, num_factors, num_layers, neumf)
        self.GMF = GMF(num_users, num_items, num_factors, neumf)
        self.predict_layer = nn.Sequential(
                                nn.Linear(num_factors*2, 1),
                                nn.Sigmoid()
                                )

    def forward(self, users, items):
        concat_layer = torch.cat([self.MLP(users, items), self.GMF(users, items)], dim=1)
        output_NeuMF = self.predict_layer(concat_layer)

        return output_NeuMF.view(-1)