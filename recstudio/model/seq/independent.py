from .corr import CoRR
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import loss_func
import torch.optim as optim

class Independent(CoRR):

    def _get_optimizers(self):
        opt_re = optim.Adam(
            self.retriever.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])
        opt_ra = optim.Adam(
            self.scorer.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])
        return [{'optimizer': opt_re}, {'optimizer': opt_ra}]

    def current_epoch_optimizers(self, nepoch):
        return self.optimizers

    def _get_dataset_class():
        return dataset.SeqDataset

    def _get_retriever(self, train_data):
        retriever = super()._get_retriever(train_data)
        retriever._init_model(train_data)
        retriever.sampler = sampler.UniformSampler(train_data.num_items, retriever.score_func)
        return retriever

    def _get_loss_func(self):
        return loss_func.BinaryCrossEntropyLoss()

    def forward(self, batch, self_learning_epoch=True, independent_epoch=False):
        bs = batch[self.fiid].size(0)
        loss_1 = self.retriever.training_step(batch)
        pos_score = self.scorer(batch)
        _, neg_id, __ = self.retriever.sampler(bs, self.neg_count, batch[self.fiid])
        neg_batch = self._generate_neg_batch(batch, neg_id)
        neg_score = self.scorer(neg_batch).view(bs, -1)
        loss_2 = self.loss_fn(None, pos_score, None, neg_score, None)
        loss = loss_1 + loss_2
        return loss


    def training_step(self, batch, nepoch):
        loss = self.forward(batch, independent_epoch=True)
        return {'loss': loss}
