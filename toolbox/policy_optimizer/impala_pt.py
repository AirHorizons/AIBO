

from collections import deque
from collections import OrderedDict
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tools.base import BasePolicyOptimizer

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


class Mode(object):
    NORMAL = 0
    REWARD_PREDICTION = 1
    PIXEL_CONTROL = 2


class Model(torch.nn.Module):

    def __init__(self, args, env, n_chnnnels, n_states, n_actions):
        super(Model, self).__init__()
        self.eval_interval = args.eval_interval
        self.n_actions = env.n_actions
        self.n_outputs = env.n_outputs
        self.n_lstm_cells = 256

        # visual embedding
        self.conv1 = nn.Conv2d(env.ob_channels, 16, 6, stride=3, padding=0)
        # self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 6, stride=3, padding=0)
        # self.pool2 = nn.MaxPool2d(2)
        # self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        # self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        # self.conv4_bn = nn.BatchNorm2d(32)
        # self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        # self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        # self.fc1 = nn.Linear(288, self.n_lstm_cells)
        self.fc1 = nn.Linear(512, self.n_lstm_cells)

        # recurrent processing
        self.lstm = nn.LSTMCell(self.n_lstm_cells + n_states + env.n_keys + 1, 
                                self.n_lstm_cells)

        # policy and baseline
        # num_outputs = n_actions
        self.critic = nn.Linear(self.n_lstm_cells, 1)
        # self.actor = nn.Linear(self.n_lstm_cells, n_actions)
        for out_idx in range(len(env.n_outputs)):
            setattr(self, 'actor{}'.format(out_idx), 
                    nn.Linear(self.n_lstm_cells, env.n_outputs[out_idx]))

        # reward prediction
        self.rp_fc1 = nn.Linear(self.n_lstm_cells * 3 + n_states, 128)
        self.reward = nn.Linear(128, 3)  # positive, negative, neutral

        self.apply(weights_init)

        self.fc1.weight.data = normalized_columns_initializer(
            self.fc1.weight.data, 1.0)
        self.fc1.bias.data.fill_(0)

        self.lstm.weight_ih.data = normalized_columns_initializer(
            self.lstm.weight_ih.data, 1.0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.weight_hh.data = normalized_columns_initializer(
            self.lstm.weight_hh.data, 1.0)
        self.lstm.bias_hh.data.fill_(0)

        self.critic.weight.data = normalized_columns_initializer(
            self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)

        for out_idx in range(len(env.n_outputs)):
            actor = getattr(self, 'actor{}'.format(out_idx))
            actor.weight.data = normalized_columns_initializer(
                actor.weight.data, 0.1)
            actor.bias.data.fill_(0)

        # self.actor.weight.data = normalized_columns_initializer(
        #     self.actor.weight.data, 0.1)
        # self.actor.bias.data.fill_(0)

        self.rp_fc1.weight.data = normalized_columns_initializer(
            self.rp_fc1.weight.data, 1.0)
        self.rp_fc1.bias.data.fill_(0)

        self.reward.weight.data = normalized_columns_initializer(
            self.reward.weight.data, 1.0)
        self.reward.bias.data.fill_(0)
        self.train()

        # act 메소드
        # self._prev_action = torch.zeros(1, env.n_keys)
        # self._prev_r = torch.zeros(1, 1)

        # act buffer
        self.cx = np.zeros((1, self.n_lstm_cells))
        self.hx = np.zeros((1, self.n_lstm_cells))
        self.last_action = 0
        self.prev_a = np.zeros((1, env.n_keys))
        self.prev_r = np.zeros((1, 1))
        # self.value = None
        # self.probs = None
        self.state_value = None
        self.action_values = None

    def visual_embedding(self, obs):
        x = F.relu(self.conv1(obs))
        # x = self.pool1(x)

        # x = F.relu(self.conv2_bn(self.conv2(x)))
        # x = F.relu(self.conv3_bn(self.conv3(x)))
        # x = F.relu(self.conv4_bn(self.conv4(x)))
        
        x = F.relu(self.conv2(x))
        # x = self.pool2(x)

        x_ = self.conv3(x) + x[:, :, 1:-1, 1:-1]
        x = F.relu(x_)

        x = self.conv4(x) + x_[:, :, 1:-1, 1:-1]
        x = F.relu(x)
        # x = F.relu(self.conv5(x) + x[:, :, 1:-1, 1:-1])
        # x = F.relu(self.conv6(x) + x[:, :, 1:-1, 1:-1])

        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def forward(self, inputs, mode=Mode.NORMAL):
        if mode == Mode.NORMAL:
            obs, state, (hx, cx), pa, pr = inputs
            x = self.visual_embedding(obs)
            x = torch.cat((x, state, pa, pr), dim=1)
            hx, cx = self.lstm(x, (hx, cx))
            x = hx

            value = self.critic(x)
            probs = list()
            for out_idx in range(len(self.n_outputs)):
                actor = getattr(self, 'actor{}'.format(out_idx))
                logit = actor(x)
                probs.append(F.softmax(logit, dim=1))

            if len(self.n_outputs) == 1:
                probs = torch.einsum('bi->bi', probs)
            elif len(self.n_outputs) == 2:
                probs = torch.einsum('bi,bj->bij', probs)
            elif len(self.n_outputs) == 3:
                probs = torch.einsum('bi,bj,bk->bijk', probs)
            elif len(self.n_outputs) == 4:
                probs = torch.einsum('bi,bj,bk,bl->bijkl', probs)
            probs = probs.reshape(-1, self.n_actions)
            log_probs = torch.log(probs)

            return value, probs, log_probs, (hx, cx)

        elif mode == Mode.REWARD_PREDICTION:
            ob1, ob2, ob3, state = inputs
            x1 = self.visual_embedding(ob1)
            x2 = self.visual_embedding(ob2)
            x3 = self.visual_embedding(ob3)
            x = torch.cat((x1, x2, x3, state), dim=1)
            x = F.relu(self.rp_fc1(x))
            return self.reward(x)

        elif mode == Mode.PIXEL_CONTROL:
            pass

        else:
            raise NotImplementedError

    def reset_buffer(self):
        self.cx.fill(0)
        self.hx.fill(0)
        self.prev_a.fill(0)
        self.prev_r.fill(0)

    def act(self, device, args, inputs, eval_game):
        with torch.no_grad():
            # obs, state, (hx, cx), pa, pr = inputs
            obs, state = inputs
            hx, cx = self.hx, self.cx
            pa, pr = self.prev_a, self.prev_r

            obs = torch.from_numpy(obs).unsqueeze(0).to(torch.float).to(device)
            state = torch.from_numpy(state).unsqueeze(0).to(torch.float).to(device)
            hx = torch.from_numpy(hx).to(torch.float).to(device)
            cx = torch.from_numpy(cx).to(torch.float).to(device)
            pa = torch.from_numpy(pa).to(torch.float).to(device)
            pr = torch.from_numpy(pr).to(torch.float).to(device)
            inputs_pt = obs, state, (hx, cx), pa, pr

            value, prob, log_prob, (hx, cx) = self(inputs_pt, mode=Mode.NORMAL)
            prob_np = prob.cpu().numpy()
            log_prob_ = log_prob.cpu().numpy()
            self.hx = hx.cpu().numpy()[:]
            self.cx = cx.cpu().numpy()[:]

            action = np.random.choice(prob.shape[1], p=prob_np.squeeze(0))
            self.last_action = action
            action_log_prob = np.take(log_prob_, [1], axis=1)

            self.state_value = value.cpu().mean().item()
            self.action_values = prob.cpu().numpy()

        return action, action_log_prob

    def buffer_step(self, env):
        self.prev_a[:] = env.action_keys[self.last_action]
        self.prev_r[:] = env.reward

    def to_params(self):
        return [p.data.cpu().numpy() for p in self.parameters()]

    def from_solution(self, device, solution):
        for param, weight in zip(self.parameters(), solution.params):
            param.data = torch.from_numpy(weight).to(device)

    def save(self):
        pass

    def load(self):
        pass

    def is_eval(self, n_games):
        return n_games % self.eval_interval == 0


# optimizer
class Impala(BasePolicyOptimizer):

    @staticmethod
    def seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)

    @staticmethod
    def make_model(device, args, env):
        n_channels = env.ob_channels
        n_states = env.state_dims
        n_actions = len(env.action_keys)
        model = Model(args, env, n_channels, n_states, n_actions)
        model.eval_interval = 1
        return model.to(device)

    def __init__(self, rank, device, args, env, model, mailbox):
        super(Impala, self).__init__(rank, device, args, env, model, mailbox)

        self.max_steps = args.max_steps

        height = args.ob_height
        width = args.ob_width
        len_lstm = model.n_lstm_cells
        self.n_batches = args.n_batches
        self.n_steps = args.n_steps

        self.episodes = [[] for _ in range(args.n_batches)]
        self.observations = torch.zeros(args.n_steps + 1, args.n_batches, env.ob_channels, height, width).to(device)
        self.states = torch.zeros(args.n_steps + 1, args.n_batches, env.state_dims).to(device)
        self.actions = torch.LongTensor(args.n_steps, args.n_batches).to(device)
        self.action_log_probs_mu = torch.zeros(args.n_steps, args.n_batches).to(device)

        self.hx = torch.zeros(args.n_steps + 1, args.n_batches, len_lstm).to(device)
        self.cx = torch.zeros(args.n_steps + 1, args.n_batches, len_lstm).to(device)
        self.prev_actions = torch.zeros(args.n_steps + 1, args.n_batches, env.n_keys).to(device)
        self.prev_rewards = torch.zeros(args.n_steps + 1, args.n_batches, 1).to(device)
        self.rewards = torch.zeros(args.n_steps, args.n_batches, 1).to(device)
        self.masks = torch.zeros(args.n_steps, args.n_batches, 1).to(device)

        self.optimizer = self.update_optimizer(args, model)
        self._loss_variable = 0
        self.loss_dict = OrderedDict(loss=0, value_loss=0, policy_loss=0, entropy_loss=0)
        self.value_dict = OrderedDict(value=0)

        # aux. tasks
        self.aux_tasks = list()
        if UnrealRewardPredction.get_coef(args) > 0.0:
            self.aux_tasks.append(UnrealRewardPredction(rank, device, args, env, model, mailbox))

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

    def ready(self):
        if min([len(self.episodes[env_id]) for env_id in range(self.n_batches)]) > self.n_steps:
            return True
        else:
            return False

    def put(self, episode):
        env_id = np.argmin([len(self.episodes[env_id]) for env_id in range(self.n_batches)])
        # self.episodes[env_id] += episode[:]
        self.episodes[env_id] += episode

        for aux_task in self.aux_tasks:
            aux_task.put(episode)

    def step(self, device, args, env, model):
        height = args.ob_height
        width = args.ob_width
        len_lstm = model.n_lstm_cells
        self.n_batches = args.n_batches
        self.n_steps = args.n_steps

        # Section: Value, Policy Func
        reward_list = deque(maxlen=1000)
        for env_id in range(self.n_batches):
            for step in range(self.n_steps):
                # ob1, s1, hx1, cx1, action, a_u, reward, ob2, s2, info2, done = self.episodes[env_id].pop(0)
                sample = self.episodes[env_id].pop(0)
                ob1 = sample.get('ob1')
                s1 = sample.get('state1')
                hx1 = sample.get('hx')
                cx1 = sample.get('cx')
                action = sample.get('action')
                a_u = sample.get('action_log_prob')
                reward = sample.get('reward')
                ob2 = sample.get('ob2')
                s2 = sample.get('state2')
                info2 = sample.get('info')
                done = sample.get('done')

                ob1 = torch.from_numpy(ob1).to(device)
                s1 = torch.from_numpy(s1).to(device)
                hx1 = torch.from_numpy(hx1).to(device)
                cx1 = torch.from_numpy(cx1).to(device)
                reward_list.append(reward)

                self.observations[step, env_id] = ob1
                self.states[step, env_id] = s1
                self.actions[step, env_id] = action
                self.action_log_probs_mu[step, env_id] = torch.tensor(a_u)
                self.rewards[step, env_id] = reward
                self.masks[step, env_id] = 0.0 if done else 1.0
                self.hx[step + 1, env_id] = hx1 * self.masks[step, env_id]
                self.cx[step + 1, env_id] = cx1 * self.masks[step, env_id]
                self.prev_actions[step + 1, env_id, :] = torch.tensor(env.action_keys[action]).to(torch.float)
                self.prev_actions[step + 1, env_id, :] *= self.masks[step, env_id]
                self.prev_rewards[step + 1, env_id] = reward * self.masks[step, env_id]

            ob2 = torch.from_numpy(ob2).to(device)
            s2 = torch.from_numpy(s2).to(device)
            self.observations[self.n_steps, env_id] = ob2
            self.states[self.n_steps, env_id] = s2

        flat_observations = self.observations.reshape(-1, env.ob_channels, height, width)
        flat_states = self.states.reshape(-1, env.state_dims)
        flat_hx = self.hx.reshape(-1, len_lstm)
        flat_cx = self.cx.reshape(-1, len_lstm)
        flat_prev_actions = self.prev_actions.reshape(-1, env.n_keys)
        flat_prev_r = self.prev_rewards.reshape(-1, 1)
        values, probs, log_probs, _ = model(
            (flat_observations, flat_states, (flat_hx, flat_cx), flat_prev_actions, flat_prev_r))
        values = values.view(self.n_steps + 1, self.n_batches, -1)
        # probs = F.softmax(logits, dim=1).view(args.n_steps + 1, args.n_batches, -1)
        # log_probs = F.log_softmax(logits, dim=1).view(args.n_steps + 1, args.n_batches, -1)
        probs = probs.view(self.n_steps + 1, self.n_batches, -1)
        log_probs = log_probs.view(self.n_steps + 1, self.n_batches, -1)
        action_log_probs = log_probs[:-1].gather(2, self.actions.unsqueeze(2))

        # v-trace
        entropies = -(log_probs * probs).sum(-1).unsqueeze(2)
        value_loss = torch.zeros(self.n_batches, 1).to(device)
        policy_loss = torch.zeros(self.n_batches, 1).to(device)
        gae = torch.zeros(self.n_batches, 1).to(device)
        returns = torch.zeros(self.n_batches, 1).to(device)
        returns.copy_(values[-1].data)
        rhos = torch.exp(action_log_probs.data - self.action_log_probs_mu.unsqueeze(2)).data.clamp(max=1.0)
        cs = torch.ones(self.n_batches, 1).to(device)

        for step in reversed(range(self.n_steps)):
            delta = rhos[step] * (self.rewards[step] + args.gamma * values[step + 1].data - values[step].data)
            cs = (cs * rhos[step]).clamp(max=1.0)
            returns = values[step].data + delta + args.lmbd * args.gamma * cs * (returns - values[step + 1].data) * self.masks[step - 1]
            # returns = values[step].data + delta + args.lmbd * args.gamma * cs * (returns - values[step + 1].data) * self.masks[step]
            value_loss = value_loss + 0.5 * (values[step] - returns.data).pow(2)
            # value_loss = value_loss + 0.5 * (self.rewards[step] + args.gamma * values[step + 1].data - values[step]).pow(2)
            if args.no_gae:
                advantages = returns - values[step].data
                policy_loss = policy_loss - action_log_probs[step] * advantages - args.ent_coef * entropies[step].data
            else:
                gae = args.gamma * args.tau * gae + delta
                policy_loss = policy_loss - action_log_probs[step] * gae - args.ent_coef * entropies[step].data

        self.value_dict['value'] = values.mean().item()

        value_loss = value_loss.mean()
        policy_loss = policy_loss.mean()
        entropy_loss = entropies.mean()
        self._loss_variable = args.value_coef * value_loss + args.policy_coef * policy_loss
        self.loss_dict['value_loss'] = value_loss.item()
        self.loss_dict['policy_loss'] = policy_loss.item()
        self.loss_dict['entropy_loss'] = entropy_loss.item()

        loss_dict = dict(
            total_loss=self._loss_variable.item(),
            value_loss=args.value_coef * value_loss.item(),
            policy_loss=args.policy_coef * policy_loss.item(),
            entropy_loss=args.ent_coef * entropy_loss.item())

        for aux_task in self.aux_tasks:
            name = aux_task.loss_name
            if aux_task.ready():
                coef = aux_task.get_coef(args)
                aux_task_loss = aux_task.step(device, args, env, model)
                aux_task_loss = aux_task_loss.mean()
                self._loss_variable += coef * aux_task_loss
                self.loss_dict[name] = aux_task_loss.item()
                loss_dict[name] = coef * aux_task_loss.item()
            else:
                loss_dict[name] = 0.0

        self.loss_dict['total_loss'] = self._loss_variable.item()

        grad_dict = self._optimize(args, model)
        
        # self.hx[0].copy_(self.hx[-1])
        # self.cx[0].copy_(self.cx[-1])

        self.hx[0] = self.hx[-1]
        self.cx[0] = self.cx[-1]

        return loss_dict, grad_dict

    def _optimize(self, args, model):

        self.optimizer.zero_grad()
        self._loss_variable.backward()
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

        max_buff_size = max_size
        for bid, episode in enumerate(self.episodes):
            buffer_size[f'b{bid:03d}'] = len(episode)
            max_buff_size = max(max_buff_size, len(episode))

        for aux_task in self.aux_tasks:
            buffer_size.update(aux_task.buffer_size(max_size=max_buff_size))

        return buffer_size

    def complete_ratio(self):
        return min(1.0, self.cur_steps / self.steps_per_generation)

    def clear(self):
        # clear buffer
        [e.clear() for e in self.episodes]

        # clear buffer in aux tasks
        [at.clear() for at in self.aux_tasks]


class UnrealRewardPredction(BasePolicyOptimizer):

    loss_name = 'reward_pred_loss'

    def __init__(self, rank, device, args, env, model, mailbox):
        super(UnrealRewardPredction, self).__init__(rank, device, args, env, model, mailbox)
        self.device = device
        self.n_batches = args.n_batches
        self.base_reward = env.base_reward
        self.zero_reward_memory = deque(maxlen=args.replay_memory_size)
        self.nonzero_reward_memory = deque(maxlen=args.replay_memory_size)
        self._new_data_added = False

    @staticmethod
    def get_coef(args):
        return getattr(args, 'reward_pred_coef', 0.0)

    def put(self, episode):
        buff = deque(maxlen=4)
        pa, pr = None, 0

        for sample in episode:
            ob1 = sample.get('ob1')
            s1 = sample.get('state1')
            action = sample.get('action')
            reward = sample.get('reward')
            ob2 = sample.get('ob2')

            buff.append(ob1)
            if len(buff) == 4:
                ob1 = np.expand_dims(buff[0], 0)
                ob2 = np.expand_dims(buff[1], 0)
                ob3 = np.expand_dims(buff[2], 0)
                if reward == self.base_reward:
                    self.zero_reward_memory.append([ob1, ob2, ob3, s1, reward])
                else:
                    self.nonzero_reward_memory.append([ob1, ob2, ob3, s1, reward])
            pa, pr = action, reward

        self._new_data_added = True  # 새로운 데이터 추가: ready() -> True

    def ready(self):
        if self._new_data_added:
            # 데이터가 새로 추가되어야만 학습할 수 있음
            if len(self.zero_reward_memory) > self.n_batches // 2 and \
                len(self.nonzero_reward_memory) > self.n_batches // 2:
                return True
        return False

    def step(self, device, args, env, model):

        zero_samples = random.sample(self.zero_reward_memory, self.n_batches // 2)
        nonzero_samples = random.sample(self.nonzero_reward_memory, self.n_batches // 2)
        samples = zero_samples + nonzero_samples

        ob1, ob2, ob3, st, rw = zip(*samples)
        ob1 = torch.from_numpy(np.vstack(ob1)).to(self.device)
        ob2 = torch.from_numpy(np.vstack(ob2)).to(self.device)
        ob3 = torch.from_numpy(np.vstack(ob3)).to(self.device)
        st = torch.from_numpy(np.vstack(st)).to(self.device)
        rwc = torch.zeros(len(rw), 3).to(self.device)
        for idx, r in enumerate(rw):
            if r > self.base_reward:
                rwc[idx, 0] = 1
            elif r < self.base_reward:
                rwc[idx, 1] = 1
            else:
                rwc[idx, 2] = 1

        logit = model((ob1, ob2, ob3, st), mode=Mode.REWARD_PREDICTION)
        log_prob = F.log_softmax(logit, dim=1)
        loss = -(rwc * log_prob).mean()

        self._new_data_added = False  # 새로운 데이터 추가 필요: ready() -> False
        return loss

    def optimize(self):
        pass

    def buffer_size(self, max_size=100):
        buffer_size = OrderedDict()
        value = max_size * len(self.zero_reward_memory) / self.zero_reward_memory.maxlen
        buffer_size['zero'] = value
        value = max_size * len(self.nonzero_reward_memory) / self.nonzero_reward_memory.maxlen
        buffer_size['non-zero'] = value
        return buffer_size

    def clear(self):
        self.zero_reward_memory.clear()
        self.nonzero_reward_memory.clear()

