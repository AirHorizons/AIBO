
import os
import random
import xmlrpc.client
from collections import deque
from collections import OrderedDict

import cloudpickle as pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tools.base import BasePolicyOptimizer
from const import ReplayBufferType
from const import WorkerId as WID
from const import Msg
from workers import replay_buffer

from IPython import embed


# model
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class Model(torch.nn.Module):
    
    def __init__(self, device, args, env, n_chnnnels, n_states, n_actions):
        super(Model, self).__init__()
        self.eval_interval = args.eval_interval
        self.n_actions = env.n_actions
        self.n_outputs = env.n_outputs

        self.conv1 = nn.Conv2d(env.ob_channels, 16, 6, stride=3, padding=0)
        self.conv2 = nn.Conv2d(16, 32, 6, stride=3, padding=0)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.fc1 = nn.Linear(512, 512)
        self.q_values = nn.Linear(512, self.n_actions)

        self.apply(weights_init)
        for layer, weight in ((self.fc1, 1.0), (self.q_values, 1.0)):
            layer.weight.data = normalized_columns_initializer(
                layer.weight.data, weight)
            layer.bias.data.fill_(0)

        self.epoch = 0
        self.last_action = 0
        # self.value = None
        # self.probs = None
        self.state_value = None
        self.action_values = None

    def visual_embedding(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x_ = self.conv3(x) + x[:, :, 1:-1, 1:-1]
        x = F.relu(x_)
        x = self.conv4(x) + x_[:, :, 1:-1, 1:-1]
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def forward(self, inputs):
        obs, state = inputs
        x = self.visual_embedding(obs)
        return self.q_values(x)

    def reset_buffer(self):
        pass

    def act(self, device, args, inputs, eval_game: bool):
        with torch.no_grad():
            # print(len(inputs))
            obs, state = inputs
            obs = torch.from_numpy(obs).unsqueeze(0).to(torch.float).to(device)
            state = torch.from_numpy(state).unsqueeze(0).to(torch.float).to(device)
            inputs = (obs, state)

            q_values = self(inputs)
            # eplison greedy
            esp = max(0.1, 1. - (2 * self.epoch / args.max_epochs))

            if eval_game:
                action = q_values.argmax(dim=1).item()
            else:
                if random.random() < esp:
                    action = random.randint(0, q_values.shape[1] - 1)
                else:
                    action = q_values.argmax(dim=1).item()

            self.state_value = q_values.mean().item()
            self.action_values = q_values.cpu().numpy()
        return action, 0.0

    def buffer_step(self, env):
        pass

    def to_params(self):
        return [p.data.cpu().numpy() for p in self.parameters()]

    def from_solution(self, device, solution):
        self.epoch = solution.generation
        for param, weight in zip(self.parameters(), solution.params):
            param.data = torch.from_numpy(weight).to(device)

    def save(self):
        pass

    def load(self):
        pass

    def is_eval(self, n_games):
        return n_games % self.eval_interval == 0


# class ReplayMemory(deque):

#     def __init__(self, batch_size, ob_shape, state_len, maxlen):
#         super(ReplayMemory, self).__init__(maxlen=maxlen)
#         self.ob1 = np.zeros((batch_size, *ob_shape))
#         self.s1 = np.zeros((batch_size, state_len))
#         self.a = np.zeros(batch_size)
#         self.ob2 = np.zeros((batch_size, *ob_shape))
#         self.s2 = np.zeros((batch_size, state_len))
#         self.done = np.zeros(batch_size)
#         self.r = np.zeros(batch_size)

#     def add_episode(self, episode):
#         for sample in episode:
#             self.append(sample)

#     def sample(self, sample_size):
#         samples = random.sample(self, sample_size)
#         info = list()
#         for idx, sample in enumerate(samples):
#             self.ob1[idx, :] = sample.get('ob1')
#             self.s1[idx, :] = sample.get('state1')
#             self.a[idx] = sample.get('action')
#             self.ob2[idx, :] = sample.get('ob2')
#             self.s2[idx, :] = sample.get('state2')
#             self.done[idx] = sample.get('done')
#             self.r[idx] = sample.get('reward')
#             info.append(sample.get('info'))

#         return self.ob1, self.s1, self.a, self.ob2, self.s2, self.done, self.r, info

#     def size(self):
#         return len(self)


class ReplayMemory:
    
    def __init__(self, mailbox, buffer_type, rank, tag, batch_size, ob_shape, state_len, maxlen):
        self.mailbox = mailbox
        self.outputs = self.mailbox.get_queue(WID.ReplayBufferHub())
        self.inputs = self.mailbox.get_queue(WID.ReplayBuffer(rank))
        self.buffer_type = buffer_type
        self.rank = rank
        self.tag = tag
        self.maxlen = maxlen

        self.memory = None
        if buffer_type == ReplayBufferType.SimpleDeque:
            self.memory = replay_buffer.SimpleDeque(':memory:', maxlen)
        else:
            self.outputs.put(
                dict(sender=WID.ReplayBuffer(self.rank), to=WID.ReplayBufferHub(), 
                    buffer_type=buffer_type, rank=rank, tag=tag, maxlen=maxlen,
                    msg=Msg.REQ_CREATE_BUFFER))
            resp = self.inputs.get()

        # 버퍼 설정
        self.ob1 = np.zeros((batch_size, *ob_shape))
        self.s1 = np.zeros((batch_size, state_len))
        self.a = np.zeros(batch_size)
        self.ob2 = np.zeros((batch_size, *ob_shape))
        self.s2 = np.zeros((batch_size, state_len))
        self.done = np.zeros(batch_size)
        self.r = np.zeros(batch_size)

    def add_episode(self, episode):
        if self.memory is not None:
            self.memory.put(episode)
        else:
            self.outputs.put(
                dict(sender=WID.ReplayBuffer(self.rank), to=WID.ReplayBufferHub(),
                    buffer_type=self.buffer_type, rank=self.rank, tag=self.tag, maxlen=self.maxlen,
                    msg=Msg.REQ_ADD_SAMPLES, samples=episode))
            resp = self.inputs.get()

    def sample(self, sample_size):
        if self.memory is not None:
            samples = self.memory.get(sample_size)
        else:
            self.outputs.put(
                dict(sender=WID.ReplayBuffer(self.rank), to=WID.ReplayBufferHub(), 
                    buffer_type=self.buffer_type, rank=self.rank, tag=self.tag, maxlen=self.maxlen,
                    msg=Msg.REQ_TRAIN_SAMPLES, sample_size=sample_size))
            resp = self.inputs.get()
            samples = resp['samples']

        info = list()
        for idx, sample in enumerate(samples):
            self.ob1[idx, :] = sample.get('ob1')
            self.s1[idx, :] = sample.get('state1')
            self.a[idx] = sample.get('action')
            self.ob2[idx, :] = sample.get('ob2')
            self.s2[idx, :] = sample.get('state2')
            self.done[idx] = sample.get('done')
            self.r[idx] = sample.get('reward')
            info.append(sample.get('info'))

        return self.ob1, self.s1, self.a, self.ob2, self.s2, self.done, self.r, info

    def size(self):
        if self.memory is not None:
            return self.memory.size()
        else:
            self.outputs.put(
                dict(sender=WID.ReplayBuffer(self.rank), to=WID.ReplayBufferHub(), 
                    buffer_type=self.buffer_type, rank=self.rank, tag=self.tag, maxlen=self.maxlen,
                    msg=Msg.REQ_SIZE))
            resp = self.inputs.get()
            # print(resp)
            return resp['size']


# optimizer
class DQN(BasePolicyOptimizer):

    @staticmethod
    def seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)

    @staticmethod
    def make_model(device, args, env):
        n_channels = env.ob_channels
        n_states = env.state_dims
        n_actions = len(env.actions)
        model = Model(device, args, env, n_channels, n_states, n_actions)
        model.eval_interval = args.eval_interval
        return model.to(device)

    def __init__(self, rank, device, args, env, model, mailbox):
        super(DQN, self).__init__(rank, device, args, env, model, mailbox)

        self.batch_size = args.n_batches
        ob_shape = env.ob_channels, args.ob_height, args.ob_width

        self.memory = ReplayMemory(
            self.mailbox, args.replay_buffer_type, 
            rank, f'{args.log_path}/{args.session_id}/data/dqn_{rank}', 
            args.n_batches, ob_shape, args.n_states, 
            maxlen=args.replay_memory_size)

        self.optimizer = self.update_optimizer(args, model)
        self.loss_dict = OrderedDict(total_loss=0)
        self.value_dict = OrderedDict(value=0)
        self._new_data_added = False

    def update_optimizer(self, args, model):

        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError

        return self.optimizer
        
    def put(self, episode):
        self.memory.add_episode(episode)
        self._new_data_added = True  # 새로운 데이터 추가: ready() -> True

    def ready(self):
        if self._new_data_added:
            # 데이터가 새로 추가되어야만 학습할 수 있음
            if self.memory.size() > self.batch_size:
                return True
        return False

    def step(self, device, args, env, model):
        ob1, s1, a, ob2, s2, done, r, info = self.memory.sample(self.batch_size)
        ob1 = torch.tensor(ob1).to(torch.float).to(device)
        s1 = torch.tensor(s1).to(torch.float).to(device)
        a = torch.tensor(a).to(torch.long).to(device)
        ob2 = torch.tensor(ob2).to(torch.float).to(device)
        s2 = torch.tensor(s2).to(torch.float).to(device)
        done = torch.tensor(done).to(torch.float).to(device)
        r = torch.tensor(r).to(torch.float).to(device)

        q2, _ = model((ob2, s2)).max(dim=1)
        q1 = model((ob1, s1))

        target_q = q1.clone()
        idx = torch.arange(target_q.shape[0])
        target_q[idx, a] = r + args.gamma * (1. - done) * q2
        loss = (target_q - q1) ** 2
        loss = loss.mean()  # sum() ?

        self.value_dict['value'] = q1.max(dim=1)[0].mean().item()
        self.loss_dict['total_loss'] = loss.item()
        grad_dict = self._optimize(args, model, loss)

        self._new_data_added = False  # 새로운 데이터 추가 필요: ready() -> False
        return self.loss_dict, grad_dict

    def _optimize(self, args, model, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        grad_dict = OrderedDict()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_dict[name] = param.grad.data.mean().item()
            else:
                grad_dict[name] = 0.0
        self.optimizer.step()

        return grad_dict

    def buffer_size(self, max_size=100):
        buffer_size = OrderedDict()
        buffer_size['ER'] = max_size * self.memory.size() / self.memory.maxlen
        return buffer_size

