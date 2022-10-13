from turtle import forward
from responses import activate
import torch
import torch.nn as nn
# import recstudio.model as recmodel
from recstudio.model import basemodel, loss_func, scorer, module

class MMoE(basemodel.Recommender):
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.tasks = train_data.config['tasks']
        self.embedding = module.Embeddings(self.fields, train_data.field2type, {f: train_data.num_values(f) for f in self.fields}, 
            self.embed_dim, self.frating, reduction='mean')
        self.experts = torch.nn.ModuleList([
            module.MLPModule(self.config['expert_mlp_size'], activate_func=self.config['activation'], dropout=self.config['dropout'],\
                bias=True, batch_norm=False)
            for _ in range(self.config['num_experts'])
        ])
        self.gates = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.embed_dim * 2, self.config['num_experts']),
                torch.nn.Softmax(dim=-1)
            ) for _ in range(len(self.tasks))
        ])


    def forward(self, batch):

        pass




    