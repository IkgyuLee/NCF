import torch
import torch.nn as nn

from MLP import MLP
from GMF import GMF

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors, num_layers, neumf, use_pretrain, pretrained_GMF, pretrained_MLP):
        super(NeuMF, self).__init__()
        """
              num_users   : number of users
              num_items   : number of items
              num_factors : number of predictive factors
              num_layers  : number of hidden layers in MLP Model
              neumf       : True(Fusion MLP&GMF)/False(Only MLP)

        """
        self.use_pretrain = use_pretrain
        self.pretrained_GMF = pretrained_GMF
        self.pretrained_MLP = pretrained_MLP

        self.GMF = GMF(num_users, num_items, num_factors, use_pretrain, neumf, pretrained_GMF)
        self.MLP = MLP(num_users, num_items, num_factors, num_layers, use_pretrain, neumf, pretrained_MLP)
        self.predict_layer = nn.Linear(num_factors*2, 1)
        self.sigmoid = nn.Sigmoid()

        if use_pretrain:
            predict_weight = torch.cat([
                self.pretrained_GMF.predict_layer.weight,
                self.pretrained_MLP.predict_layer.weight], dim=1)
            predict_bias = self.pretrained_GMF.predict_layer.bias + \
                           self.pretrained_MLP.predict_layer.bias
            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * predict_bias)
        else:
            # weight 초기화
            nn.init.normal_(self.predict_layer.weight, mean=0.0, std=0.01)


    def forward(self, users, items):
        concat_layer = torch.cat([self.MLP(users, items), self.GMF(users, items)], dim=1)
        output_NeuMF = self.predict_layer(concat_layer)
        output_NeuMF = self.simoid(output_NeuMF)

        return output_NeuMF.view(-1)