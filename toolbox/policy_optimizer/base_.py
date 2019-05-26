


class BasePolicyOptimizer(object):
    def __init__(self, rank, device, args, env, model, mailbox):
        self.rank = rank
        self.max_steps = args.max_steps
        self.mailbox = mailbox

        self.steps_per_generation = args.steps_per_generation
        self.cur_steps = 0
        self.episodes_per_generation = args.episodes_per_generation
        self.cur_episodes = 0

    @staticmethod
    def seed(seed):
        raise NotImplementedError

    @staticmethod
    def make_model(args, env, n_channels, n_states, n_actions):
        raise NotImplementedError

    def update_optimizer(self, args, model):
        raise NotImplementedError

    def ready(self):
        raise NotImplementedError
        
    def put(self, episode):
        raise NotImplementedError

    def step(self, device, args, env, model):
        raise NotImplementedError

    def optimize(self, args, model):
        raise NotImplementedError

    def complete_ratio(self):
        raise NotImplementedError

