import numpy as np
import torch

def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k, device):
    HR, NDCG = [], []

    for user, item, _ in test_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)
        # 가장 높은 top_k개 선택, value와 index에 대한 tensor를 반환
        _, indices = torch.topk(predictions, top_k)
        # test_loader에서 batch_size만큼 불러온 item tensor중에 해당 상품 index 선택한 tensor를 numpy array로 변환하고 최종 리스트로 변환
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        # 정답값 선택
        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)