import torch
from ..basemodel import Recommender
from recstudio.data import dataset

from .rankflow import RankFlow

class ICC(RankFlow):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('ICC')
        parent_parser.add_argument('--retriever', type=str, choices=['SASRec', 'Caser'], default='SASRec')
        parent_parser.add_argument('--ranker', type=str, choices=['DIN', 'BST'], default='DIN')
        parent_parser.add_argument('--negative_count', type=int, default=20, help='number of negative samples')
        return parent_parser


    @staticmethod
    def _get_dataset_class():
        return dataset.SeqDataset


    def _get_loss_func(self):
        class ICCLoss(torch.nn.Module):
            def __init__(self):
                super(ICCLoss, self).__init__()
                self.bce_loss_func = torch.nn.BCELoss()

            def forward(self, model_preds, label):
                indicators = None
                for i in range(model_preds.size(1)):
                    k = torch.tanh(torch.tensor([i])).to(model_preds.device)
                    # k = torch.tensor([i]).to(model_preds.device)
                    indicator = torch.sigmoid(model_preds[:, i] - k).unsqueeze(1)
                    indicators = indicator if indicators is None else torch.cat((indicators, indicator), dim=1)
                weights = torch.cumprod(torch.cat((torch.ones((indicators.size(0), 1, indicators.size(2))).to(indicators.device), indicators), dim=1), dim=1)[:, :-1]
                weights = (1 - indicators) * weights
                score = torch.sum(model_preds * weights, dim=1)
                loss = self.bce_loss_func(score, label)
                return loss
        return ICCLoss()


    def forward(self, batch, self_learning_epoch=True, independent_epoch=False):
        bs = batch[self.fiid].size(0)
        _, neg_item_idx, __, query = self.retriever._sample(
                batch,
                neg = self.neg_count,
            )
        # retriever
        pos_vec = self.retriever.item_encoder(self.retriever._get_item_feat(batch))
        neg_vec = self.retriever.item_encoder(self.retriever._get_item_feat(neg_item_idx))
        pos_score_re = self.retriever.score_func(query, pos_vec)
        neg_score_re = self.retriever.score_func(query, neg_vec)

        # ranker
        pos_score_ra = self.scorer(batch)
        neg_batch = self._generate_neg_batch(batch, neg_item_idx)
        neg_score_ra = self.scorer(neg_batch).view(bs, -1)

        score_re = torch.cat((pos_score_re.view(-1,1), neg_score_re), dim=-1)
        score_ra = torch.cat((pos_score_ra.view(-1,1), neg_score_ra), dim=-1)
        scores = torch.sigmoid(torch.stack((score_re, score_ra), dim=1))
        labels = scores.new_zeros((scores.size(0), scores.size(2)))
        labels[:, 0] = 1
        return scores, labels

    def training_step(self, batch):
        scores, labels = self.forward(batch)
        loss = self.loss_fn(scores, labels)
        return {"loss": loss}
