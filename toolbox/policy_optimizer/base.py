

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
    def make_model(args, observation_shape, state_shape, action_space):
        raise NotImplementedError

    def step(self, device, args, env, model):
        raise NotImplementedError

    def fit(self, args, model):
        raise NotImplementedError
