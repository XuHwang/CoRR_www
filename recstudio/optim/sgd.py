import torch
import torch.optim.sgd
import torch.optim._functional as F
from torch.optim.optimizer import Optimizer, required


class LambdaSGD(Optimizer):
    def __init__(self, model, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.weight_decay = torch.nn.Parameter(torch.DoubleTensor(weight_decay), requires_grad=True)
        self.model = model
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=self.weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(LambdaSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    def step(self, val_batch, closure=None):
        """Performs a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p.data)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            old_p = self.model.state_dict()  # save old parameter for the final update
            F.sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=group['weight_decay'],  # weight_decay here is differentiable parameter
                  momentum=group['momentum'],
                  lr=group['lr'],
                  dampening=group['dampening'],
                  nesterov=group['nesterov'],
                  maximize=group['maximize'],
                  has_sparse_grad=has_sparse_grad,
                  foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        # TODO: use valid batch and 'param_with_grad' to update lambda parameters

        return loss

    def lambda_step(self):
        loss = None
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for k, v in group:
                if k != 'params':
                    if v.grad is not None:
                        params_with_grad.append(v)
                        d_p_list.append(v.grad)
                        if v.grad.is_sparse:
                            has_sparse_grad = True

                        state = self.state[v]
                        if 'momentum_buffer' not in state:
                            momentum_buffer_list.append(None)
                        else:
                            momentum_buffer_list.append(state['momentum_buffer'])

            old_p = self.model.state_dict()  # save old parameter for the final update
            F.sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=0.0,   # here no weight_decay is used
                  momentum=group['momentum'],
                  lr=group['lr'],
                  dampening=group['dampening'],
                  nesterov=group['nesterov'],
                  maximize=group['maximize'],
                  has_sparse_grad=has_sparse_grad,
                  foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        pass


def test(trn_data, val_data, model, optimizer):
    for batch in trn_data:
        loss = model.training_step(batch)
        loss.backward()
        # (assumed) update parameters in model. save the paramters before updating to resume for truly updating.
        optimizer.step()
        # model.zero_grad()     # the grad of model parameters should be saved for next truly updating.
        # Actually, if no other backward() is executed, the grad will hold util the last update is executed. So zero_grad() should be executed after the final update.

        # use assumed updated parameters to get grad of lambda; and there is no grad for model parameters here
        val_loss = model.lambda_training_step(val_batch)
        # TODO: a problem here is that in the lambda_training_step, the loss should only record the gradient of lambda, which means that the parameters of model should be detached.
        # model.eval() will solve the problem
        val_loss.backward()
        optimizer.lambda_step()  # update lambda and update parameters again(resume first and then )
        optimizer.lambda_zero_grad()    # set gradient of lambda to be zero
        optimizer.zero_grad()   # set gradient of parameters of model to be zero


class LambdaOPT(torch.nn.Module):
    def __init__(self, model, lr, weight_decay, optimizer):
        super(LambdaOPT, self).__init__()
        self.lr = lr
        self.model = model
        self.weight_decay = torch.nn.Parameter(torch.tensor(weight_decay), requires_grad=True)
        self.optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=self.weight_decay)

    def forward(self, val_batch):
        old_parameter_dict = self.model.state_dict()    # whether state_dict is a deepcopy() ?
        old_grad = self.copy_grad()

        self.optimizer.step()   # assumed update

        self.optimizer.zero_grad()
        # self.model.eval()
        val_loss = self.model.training_step(val_batch)

        if isinstance(val_loss, dict):
            if val_loss['loss'].requires_grad:
                val_loss['loss'].backward()
        elif isinstance(val_loss, torch.Tensor):
            if val_loss.requires_grad:
                val_loss.backward()

        self._update_lambda()
        self.lambda_zero_grad()

        self.model.load_state_dict(old_parameter_dict)
        self.optimizer.step()   # final update -> F.sgd()

    @torch.no_grad()
    def _update_lambda(self):
        d_lambda = self.weight_decay.grad
        self.weight_decay.add_(d_lambda, alpha=-self.lr)

    def lambda_zero_grad(self, set_to_none: bool = False):
        p_lambda = self.weight_decay
        if p_lambda.grad is not None:
            if set_to_none:
                p_lambda.grad = None
            else:
                if p_lambda.grad.grad_fn is not None:
                    p_lambda.grad.detach_()
                else:
                    p_lambda.grad.requires_grad_(False)
                p_lambda.grad.zero_()
