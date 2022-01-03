import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(GMF, self).__init__()
        """
              num_users   : number of users
              num_items   : number of items
              num_factors  : number of predictive factors
        """
        # Embedding 객체 생성
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_factors)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_factors)
        predict_size = num_factors
        self.predict_layer = nn.Linear(predict_size, 1)

    def forward(self, users, items):
      # Embdding 해주기
      user_embedded = self.user_embedding(users)
      item_embedded = self.item_embedding(items)

      # weight 초기화
      user_embedded = torch.nn.init.normal_(user_embedded, mean=0.0, std=0.01)
      item_embedded = torch.nn.init.normal_(item_embedded, mean=0.0, std=0.01)

      # element wise product
      output_GMF = user_embedded * item_embedded

      prediction = self.predict_layer(output_GMF)
      return prediction.view(-1)