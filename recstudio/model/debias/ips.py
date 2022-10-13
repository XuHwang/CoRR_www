import torch
from recstudio.utils import get_model
from recstudio.model import loss_func
from recstudio.model import basemodel
from recstudio.data.dataset import MFDataset
from recstudio.model.module import propensity


class IPS(basemodel.BaseRanker):

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('IPS')
        parent_parser.add_argument("--scorer_model", type=str, default='PMF', help='scorer model for IPS')
        parent_parser.add_argument("--loss_metric", type=str, default='mse', help='loss metric for IPS')
        parent_parser.add_argument("--propensity_estimation", type=str, default='logistic_regression', help='estimation for propensities')
        return parent_parser


    @staticmethod
    def _get_dataset_class():
        return MFDataset

    def _get_scorer(self, train_data):
        scorer_model_class, scorer_model_conf = get_model(self.config['scorer_model'])
        scorer_model_conf.update(self.config)
        scorer_model = scorer_model_class(scorer_model_conf)
        scorer_model._init_model(train_data, drop_unused_field=False)
        if isinstance(scorer_model, basemodel.BaseRetriever):
            scorer_model.sampler = None
            return scorer_model
        elif isinstance(scorer_model, basemodel.BaseRanker):
            return scorer_model._get_scorer(train_data)

    def _get_loss_func(self):
        class IPSLoss(loss_func.PointwiseLoss):
            def __init__(self, loss_metric) -> None:
                super().__init__()
                self.loss_metric = loss_metric.lower()

            def forward(self, weight, pos_score, label):
                if self.loss_metric == 'mse':
                    loss_ = torch.square(pos_score - label)
                elif self.loss_metric == 'mae':
                    loss_ = torch.abs(pos_score - label)
                elif self.loss_metric == 'accuracy':
                    # TODO(@pepsi2222): not differentiable
                    loss_ = (pos_score == label)
                else:
                    raise ValueError(f'{self.loss_metric} is not supportable.')
                return torch.mean(weight * loss_)

        return IPSLoss(self.config['loss_metric'])

    def training_step(self, batch):
        y_h = self.forward(batch)
        if self.config['propensity_estimation'].lower() == "naive_bayes":
            batch_ = batch[self.frating]
        elif self.config['propensity_estimation'].lower() == "logistic_regression":
            batch_ = batch
        weight = 1 / (torch.sigmoid(self.p_model(batch_)['pos_score']) + 1e-7).detach()
        loss = self.loss_fn(weight, **y_h)
        return loss

    def fit(self, train_data, val_data = None, run_mode='light', config = None, unif_data = None, **kwargs):
        self.p_model, required_train = propensity.get_propensity(self.config)
        if self.config['propensity_estimation'].lower() == "naive_bayes":
            self.p_model.fit(train_data, unif_data)
        elif self.config['propensity_estimation'].lower() == "logistic_regression":
            if required_train:
                self.p_model.fit(train_data, run_mode=run_mode, epochs=self.config['p_epochs'])
            else:
                self.logger.info(f"Estimation model loaded from {self.config['estimation_model_path']}")
        super().fit(train_data, val_data, run_mode, config, **kwargs)
