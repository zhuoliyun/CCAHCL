import torch
import torch.nn.functional as F
from torch import Tensor

from .logreg import LogReg


def masked_accuracy(logits: Tensor, labels: Tensor, node_ids: Tensor = None, mask_name: str = "Unknown Mask"):
    if len(logits) == 0:
        print(f"No logits for {mask_name}. Accuracy: 0%.")
        return 0

    pred = torch.argmax(logits, dim=1)

    acc = pred.eq(labels).sum() / len(logits) * 100
    return acc.item()


def accuracy(logits: Tensor, labels: Tensor, masks: list[Tensor], all_node_ids: Tensor):
    accs = []
    mask_names = ["train", "validation", "test"] 
    for i, mask in enumerate(masks):
        current_mask_node_ids = all_node_ids[mask]
        
        # 调用 masked_accuracy 并传入 mask 名称，它会直接打印错误节点
        acc = masked_accuracy(logits[mask], labels[mask], current_mask_node_ids, mask_name=mask_names[i])
        accs.append(acc)
        
    return accs


def linear_evaluation(z, labels, masks, lr=0.01, max_epoch=100):
    z = z.detach()
    hid_dim, num_classes = z.shape[1], int(labels.max()) + 1

    classifier = LogReg(hid_dim, num_classes).to(z.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.0)

    for epoch in range(1, max_epoch + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)

        logits = classifier(z[masks[0]])
        loss = F.cross_entropy(logits, labels[masks[0]])
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        classifier.eval()
        logits = classifier(z)
        
        # 创建所有节点ID的张量
        all_node_ids = torch.arange(z.shape[0]).to(z.device) 
        
        # accuracy 函数现在会直接打印错误节点，并只返回准确率
        accs = accuracy(logits, labels, masks, all_node_ids)

    return accs