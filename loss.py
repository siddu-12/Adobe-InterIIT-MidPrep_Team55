import torch
import torch.nn.functional as F


class CustomLoss(torch.nn.Module):
    def _init_(self):
        super(CustomLoss, self)._init_()

    def batched_dpr_loss(self, qi, context_vecs, pos_idxs, neg_idxs):
        batch_size = qi.shape[0]
        embed_size = context_vecs.shape[-1]
        pos_vecs = torch.gather(context_vecs, 1, pos_idxs.unsqueeze(-1).repeat(1, 1, context_vecs.size(-1)))
        qi_pos = torch.repeat_interleave(qi, repeats=pos_vecs.shape[1], dim=1)
        pos_sim = F.cosine_similarity(qi_pos.reshape(-1, embed_size), pos_vecs.reshape(-1, embed_size)).reshape(
            batch_size, pos_vecs.shape[1])
        neg_vecs = torch.gather(context_vecs, 1, neg_idxs.unsqueeze(-1).repeat(1, 1, context_vecs.size(-1)))
        qi_neg = torch.repeat_interleave(qi, repeats=neg_vecs.shape[1], dim=1)
        neg_sim = F.cosine_similarity(qi_neg.reshape(-1, embed_size), neg_vecs.reshape(-1, embed_size)).reshape(
            batch_size, neg_vecs.shape[1])

        pos_lse = torch.logsumexp(pos_sim, dim=-1)
        neg_lse = torch.logsumexp(neg_sim, dim=-1)
        loss = -torch.log(
            torch.exp(pos_lse) / (torch.exp(pos_lse) + torch.exp(neg_lse))
        )

        return loss.mean()
