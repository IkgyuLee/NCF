import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_users, num_items, num_factors, num_layers, use_pretrain, neumf, pretrained_MLP=None):
        super(MLP, self).__init__()
        """
              num_users   : number of users
              num_items   : number of items
              num_factors : number of predictive factors
              num_layers  : number of hidden layers in MLP Model
              neumf       : True(Fusion MLP&GMF)/False(Only MLP)
        """
        self.use_pretrain = use_pretrain
        self.neumf = neumf
        self.pretrained_MLP = pretrained_MLP

        # Embedding 객체 생성
        self.user_embedding = nn.Embedding(num_embeddings=num_users,
                                           embedding_dim=num_factors * (2 ** (num_layers - 2)))
        self.item_embedding = nn.Embedding(num_embeddings=num_items,
                                           embedding_dim=num_factors * (2 ** (num_layers - 2)))

        """ Case 'num_layer == 0', Resolving... 
        if num_layers == 0:
          self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_factors * (2 ** (num_layers - 1)))
          self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_factors * (2 ** (num_layers - 1)))
        else:
          self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_factors * (2 ** (num_layers - 2)))
          self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_factors * (2 ** (num_layers - 2)))
        
        MLP_modules = []
        if num_layers > 0:
          input_size = num_factors * (2 ** (num_layers - 1))
          MLP_modules.append(nn.Linear(input_size, input_size))
          MLP_modules.append(nn.ReLU())
        """

        MLP_modules = []
        for i in range(1, num_layers):
          input_size = num_factors * (2 ** (num_layers - i))
          MLP_modules.append(nn.Linear(input_size, input_size // 2))
          MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.neumf == False:
            predict_size = num_factors
            self.predict_layer = nn.Linear(predict_size, 1)
            self.sigmoid = nn.Sigmoid()


        if self.use_pretrain:
            self.user_embedding.weight.data.copy_(
                self.pretrained_MLP.user_embedding.weight)
            self.item_embedding.weight.data.copy_(
                self.pretrained_MLP.item_embedding.weight)
            for layer, pretrained_layer in zip(self.MLP_model, self.pretrained_MLP.MLP_model):
                if isinstance(layer, nn.Linear) and isinstance(pretrained_layer, nn.Linear):
                    layer.weight.data.copy_(pretrained_layer.weight)
                    layer.bias.data.copy_(pretrained_layer.bias)
        else:
            # weight 초기화
            nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)


    def forward(self, users, items):
        # Embdding 해주기
        user_embedded = self.user_embedding(users)
        item_embedded = self.item_embedding(items)

        # user, item tensor concat
        vector = torch.cat([user_embedded, item_embedded], dim=1)

        if self.neumf == False:
            output_MLP = self.MLP_layers(vector)
            prediction = self.predict_layer(output_MLP)
            sigmoid = self.sigmoid(prediction)
            return sigmoid.view(-1)

        else:
            output_MLP = self.MLP_layers(vector)
            return output_MLP