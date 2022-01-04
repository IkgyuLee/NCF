import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_users, num_items, num_factors, num_layers):
        super(MLP, self).__init__()
        """
              num_users   : number of users
              num_items   : number of items
              num_factors  : number of predictive factors
              num_layers  : number of hidden layers in MLP Model
        """
        # Embedding 객체 생성
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

        for i in range(1, num_layers):
          input_size = num_factors * (2 ** (num_layers - i))
          MLP_modules.append(nn.Linear(input_size, input_size // 2))
          MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        predict_size = num_factors
        self.predict_layer = nn.Linear(predict_size, 1)
        self.sigmoid_layer = nn.Sigmoid(())


    def forward(self, users, items):
      # Embdding 해주기
      user_embedded = self.user_embedding(users)
      item_embedded = self.item_embedding(items)

      # weight 초기화
      user_embedded = torch.nn.init.normal_(user_embedded, mean=0.0, std=0.01)
      item_embedded = torch.nn.init.normal_(item_embedded, mean=0.0, std=0.01)

      # user, item tensor concat
      vector = torch.cat([user_embedded, item_embedded], 1)

      output_MLP = self.MLP_layers(vector)
      prediction = self.predict_layer(output_MLP)
      sigmoid = self.sigmoid_layer(prediction)
      #return prediction.view(-1)
      return sigmoid.view(-1)