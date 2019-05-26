__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'


import torch.optim as optim


class BasePolicyOptimizer(object):
    def __init__(self, args, model_dict):
        self.args = args
        self.model_dict = model_dict
        self.optimizer = self.set_optimizer(args)

    def set_optimizer(self, args):
        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model_dict['train'].parameters(), lr=args.lr, momentum=args.momentum)

        elif args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model_dict['train'].parameters(), lr=args.lr, momentum=args.momentum)

        elif args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model_dict['train'].parameters(), lr=args.lr)

        elif args.optimizer == 'swa':
            from toolbox.optimizer.swa import SWA
            self._base_optimizer = optim.SGD(
                self.model_dict['train'].parameters(), lr=args.lr, momentum=args.momentum)
            self.optimizer = SWA(self._base_optimizer, update_interval=10, swap_interval=1000)
        else:
            raise NotImplementedError
        return self.optimizer

    @property
    def model(self):
        return self.model_dict['train']

    def parameters(self):
        return self.model_dict['train'].parameters()

    def put(self, samples):
        raise NotImplementedError

    def ready(self):
        raise NotImplementedError

    def prepare_optimize(self, args):
        pass

    def optimize(self, args):
        raise NotImplementedError

    def after_optimize(self, args):
        pass