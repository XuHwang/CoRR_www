from typing import Tuple
import torch
from torch.utils.data import Dataset
from recstudio.model.fm.lr import LR
from recstudio.utils import get_model
from recstudio import quickstart

def get_propensity(config) -> Tuple:
    if config['propensity_estimation'].lower() == "naive_bayes":
        return NaiveBayes(), False
    elif config['propensity_estimation'].lower() == "logistic_regression":
        # TODO(@AngusHuang17)

        _, model_conf = get_model('LR')
        for k, v in config.items():
            if k.startswith('p_'):
                model_conf.update({k[2:]: v})
        model_conf['gpu'] = config['gpu']
        model = LR(model_conf)
        required_train = True
        if config['estimation_model_path'] is not None:
            model.load_checkpoint(config['estimation_model_path'])
            required_train = False

        return model, required_train
    else:
        raise ValueError(f"{config['propensity_estimation']} is not supportable.")


class NaiveBayes(torch.nn.Module):
    """Learning propensity by Naive Bayes method.

    Args:
        train_data (Dataset): missing not at random data; for training recommender and propensity
        unif_data (Dataset): missing completely at random data; for training propensity only

    """
    def fit(self, train_data : Dataset, unif_data : Dataset):
        y, y_cnt_given_o = torch.unique(train_data.inter_feat.get_col[train_data.frating], return_counts=True)
        y = y.tolist()
        P_y_given_o = y_cnt_given_o / torch.sum(y_cnt_given_o)
        P_o = train_data.num_inters / train_data.num_users * train_data.num_items

        y_, y_cnt = torch.unique(unif_data.inter_feat.get_col[unif_data.frating], return_counts=True)
        y_ = y_.tolist()
        P_y = y_cnt / torch.sum(y_cnt)

        y_dict = {}
        # TODO: what will happen when one rating score is not in y_ but in y
        # try not to use list but tensor
        for k, v in zip(y, P_y_given_o):
            y_dict[k] = v * P_o / P_y[y_.index(k)]

        self.y_dict = y_dict

    def forward(self, batch):
        p = torch.zeros_like(batch)
        for i, y in enumerate(batch):
            p[i] = self.y_dict[y]
        return p