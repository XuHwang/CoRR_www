import torch
from recstudio.model import basemodel

class MultiTaskBase(basemodel.Recommender):
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.tasks = train_data.config['tasks']    # topk, binary classification, rating prediction
        # tasks is a dict, key: task types; value: labels name

        self.logger.info("Tasks-Labels: " + [f"{k}-{l}" for k,l in self.tasks.items()].join("; "))
        self.set_task_models(train_data)

    def set_task_models(self, train_data):
        self.task_models = None

    
    def forward(self, batch):
        pass

