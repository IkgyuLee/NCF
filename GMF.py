import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors, neumf):
        super(GMF, self).__init__()
        """
              num_users   : number of users
              num_items   : number of items
              num_factors : number of predictive factors
              neumf       : True(Fusion MLP&GMF)/False(Only MLP)
        """
        self.neumf = neumf

        # Embedding 객체 생성
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_factors)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_factors)

        if self.neumf == False:
            predict_size = num_factors
            self.predict_layer = nn.Linear(predict_size, 1)
            self.sigmoid_layer = nn.Sigmoid()

        # weight 초기화
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

    def forward(self, users, items):
      # Embdding 해주기
      user_embedded = self.user_embedding(users)
      item_embedded = self.item_embedding(items)

      if self.neumf == False:
          # element wise product
          output_GMF = user_embedded * item_embedded

          prediction = self.predict_layer(output_GMF)
          sigmoid = self.sigmoid_layer(prediction)
          return sigmoid.view(-1)

      else:
          # element wise product
          output_GMF = user_embedded * item_embedded
          return output_GMF