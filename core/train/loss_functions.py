# core/train/loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AMARFContrastiveLoss(nn.Module):
    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, query_embeddings, passage_embeddings):
        q_norm = F.normalize(query_embeddings, p=2, dim=1)
        p_norm = F.normalize(passage_embeddings, p=2, dim=1)
        
        scores = torch.matmul(q_norm, p_norm.transpose(0, 1)) * self.scale
        
        labels = torch.arange(len(query_embeddings), device=query_embeddings.device)
        
        return self.cross_entropy(scores, labels)
