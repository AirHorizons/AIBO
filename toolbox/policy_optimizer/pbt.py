
import copy
import collections
import datetime
import random
import os
import stat
import sys
import glob
import uuid
import logging
import shutil
import json
import struct
import time
import sqlite3
from collections import namedtuple
from functools import partial

import numpy as np
import torch
# import joblib
from IPython import embed
# pip install sklearn-extensions
# from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor
# from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.preprocessing import StandardScaler

# TODO: elm 사용
# pip install sklearn-extensions
# from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor
# from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
# net = GenELMRegressor(hidden_layer=MLPRandomLayer(n_hidden=10, activation_func='tanh'))
# net.fit(X, y)
# pred = net.predict(X)

from const import PopInit
from const import PopMutation
import config


# logger = logging.getLogger(__name__)
# # logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler())


def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)


class Solution(object):
    def __init__(self, params, args=None, scores=None, loss=None):

        self.id = uuid.uuid1().hex
        self.parent = None
        # self._params = [p.data.cpu().numpy() for p in params]
        self._params = params
        self.hyperparams = []
        self.init_ops = []
        self.mutate_ops = []

        self.rank = None
        self.learner = None
        self.generation = None
        self.depth = 0

        # for ucb selection
        self.n_try = 1
        self.ucb = np.NaN

        # for pred mutation
        self.pred_w = None
        self.pred_bias = None
        self.pred_score = None

        self.loss = loss

        if args is not None:
            names = args.hyperparams.split(',')
            for name in names:
                name = name.strip()
                if len(name) > 0 and hasattr(args, name):
                    self.hyperparams.append(name)

            for name in names:
                setattr(self, '_{}'.format(name), getattr(args, name))
            
            for name in self.hyperparams:
                init_op = getattr(self, 'init_{}'.format(name), None)
                if init_op is not None:
                    self.init_ops.append(init_op)

                inc_op = getattr(self, 'inc_{}'.format(name), None)
                if inc_op is not None:
                    self.mutate_ops.append(inc_op)

                dec_op = getattr(self, 'dec_{}'.format(name), None)
                if dec_op is not None:
                    self.mutate_ops.append(dec_op)
                    
        # for mutation op backup
        self._mutate_op_values = {op: 1. for op in self.mutate_ops}
        self._mutate_op_counts = {op: 1. for op in self.mutate_ops}
        self._last_mutation_op = None
        self._last_mean_score = None
    
        self.scores = [] if scores is None else scores

    def state_dict(self):
        data = self.__dict__
        data['init_ops'] = [op.__name__ for op in data['init_ops']]
        data['mutate_ops'] = [op.__name__ for op in data['mutate_ops']]
        data['_mutate_op_values'] = {op.__name__: v for op, v in data['_mutate_op_values'].items()}
        data['_mutate_op_counts'] = {op.__name__: v for op, v in data['_mutate_op_counts'].items()}
        if data['_last_mutation_op'] is not None:
            data['_last_mutation_op'] = data['_last_mutation_op'].__name__
        return data

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict
        self.init_ops = [getattr(self, op) for op in self.init_ops]
        self.mutate_ops = [getattr(self, op) for op in self.mutate_ops]
        self._mutate_op_values = {getattr(self, op): v for op, v in self._mutate_op_values.items()}
        self._mutate_op_counts = {getattr(self, op): v for op, v in self._mutate_op_counts.items()}
        if state_dict['_last_mutation_op'] is not None:
            self._last_mutation_op = getattr(self, state_dict['_last_mutation_op'])

    def __repr__(self):
        if self.pred_score is None:
            values = ['{}, {}, Score: {:.3f}, #scores: {:3d}, UCB: {:.3f}, n_try: {:d}'.format(
                self.learner, self.generation, self.score, self.n_scores, self.ucb, self.n_try)]
        else:
            values = ['{}, {}, Score: {:.3f}({:.3f}), #scores: {:3d}, UCB: {:.3f}, n_try: {:d}'.format(
                self.learner, self.generation, self.score, self.n_scores, self.pred_score, self.ucb, self.n_try)]

        for name in self.hyperparams:
            value = getattr(self, '_{}'.format(name))
            if type(value) == float:
                values.append('{}: {:8.6f}'.format(name, value))
            elif type(value) == int:
                values.append('{}: {:4d}'.format(name, value))
            else:
                values.append('{}: {}'.format(name, value))
        return ', '.join(values)

    @property
    def score(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return config.PBT.MIN_SCORE

    @property
    def mean_score(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return config.PBT.MIN_SCORE

    @property
    def median_score(self):
        if len(self.scores) > 0:
            return np.median(self.scores)
        else:
            return config.PBT.MIN_SCORE

    @property
    def n_scores(self):
        return len(self.scores)

    @property
    def params(self):
        # return [torch.from_numpy(p) for p in self._params]
        return [p for p in self._params]

    @property
    def vec(self):
        vec = np.zeros(len(self.hyperparams) + 1)
        for idx, hyperparam in enumerate(self.hyperparams):
            vec[idx] = getattr(self, hyperparam)
        vec[-1] = self.generation
        return vec, self.mean_score

    def init(self, init_method=PopInit.NONE, **kwargs):
        if init_method == PopInit.LOG_UNIFORM_SAMPLE:
            for init_op in self.init_ops:
                init_op()
        elif init_method == PopInit.MULTIPLE_MUTATIONS:
            n_mutations = kwargs.get('n_mutations', 1)
            for _ in range(n_mutations):
                mutate_op = random.choice(self.mutate_ops)
                mutate_op()
        elif init_method == PopInit.NONE:
            pass
        else:
            raise NotImplementedError

        other = copy.deepcopy(self)
        other.id = uuid.uuid1().hex
        return other

    def random_mutation(self):
        other = copy.deepcopy(self)
        mutation_op = random.choice(other.mutate_ops)
        mutation_op()
        other._last_mutation_op = mutation_op
        other.parent = self
        other.id = uuid.uuid1().hex
        return other

    def prob_mutation(self):
        probs = np.zeros(len(self.mutate_ops))
        for idx, op in enumerate(self.mutate_ops):
            probs[idx] = self._mutate_op_values[op] / self._mutate_op_counts[op]
        probs = probs / np.sum(probs)

        other = copy.deepcopy(self)
        mutation_op = np.random.choice(other.mutate_ops, p=probs)
        mutation_op()
        other._last_mutation_op = mutation_op
        other.parent = self
        other.id = uuid.uuid1().hex
        return other

    def propagate_op_result(self, pop):
        if self._last_mean_score is None:
            self._last_mean_score = self.mean_score

        if self.mean_score > self._last_mean_score:
            value = 1.0
        elif self.mean_score < self._last_mean_score:
            value = 0.0
        else:
            value = 0.5

        sol = self
        while True:
            if self._last_mutation_op is None:
                break
            sol._mutate_op_values[self._last_mutation_op] += value
            sol._mutate_op_counts[self._last_mutation_op] += 1.
            sol = sol.parent
            if sol is None or sol not in pop:
                break

        self._last_mean_score = self.mean_score
        self._last_mutation_op = None

    def predict_mutation(self, normalizer, model):
        other = copy.deepcopy(self)

        xs = np.zeros((len(self.mutate_ops), len(self.hyperparams) + 1))
        for idx, _ in enumerate(self.mutate_ops):
            other_ = copy.deepcopy(self)
            other_.mutate_ops[idx]()
            xs[idx, :], _ = other_.vec

        ys = model.predict(normalizer.transform(xs))

        self.pred_score = self.pred_w * np.max(ys) + self.pred_bias
        mutation_op = other.mutate_ops[np.argmax(ys)]
        mutation_op()
        other._last_mutation_op = mutation_op
        other.parent = self
        other.id = uuid.uuid1().hex
        return other

    def apply(self, args):
        for name in self.hyperparams:
            setattr(args, name, getattr(self, '_{}'.format(name)))
        return args

    # mutation operators
    def _log_uniform(self, low=1e-10, high=0.1, size=None):
        return np.exp(np.random.uniform(np.log(low), np.log(high), size))

    def _clip(self, value, low, high):
        return max(low, min(high, value))

    @property
    def lr(self):
        return self._lr

    def init_lr(self):
        self._lr = float(self._log_uniform(0.00001, 0.005))

    def inc_lr(self):
        self._lr = self._clip(1.25 * self._lr, 1e-10, 0.1)

    def dec_lr(self):
        self._lr = self._clip(0.8 * self._lr, 1e-10, 0.1)

    @property
    def ent_coef(self):
        return self._ent_coef

    def init_ent_coef(self):
        self._ent_coef = float(self._log_uniform(0.0005, 0.01))

    def inc_ent_coef(self):
        self._ent_coef = self._clip(1.25 * self._ent_coef, 0.0, 0.1)

    def dec_ent_coef(self):
        self._ent_coef = self._clip(0.8 * self._ent_coef, 0.0, 0.1)

    @property
    def value_coef(self):
        return self._value_coef

    def init_value_coef(self):
        self._value_coef = float(self._log_uniform(0.3, 0.7))

    def inc_value_coef(self):
        self._value_coef = self._clip(1.25 * self._value_coef, 0.0, 0.75)

    def dec_value_coef(self):
        self._value_coef = self._clip(0.8 * self._value_coef, 0.0, 0.75)        

    @property
    def frame_repeat(self):
        return self._frame_repeat

    def init_frame_repeat(self):
        self._frame_repeat = int(np.random.uniform(4, 10))

    def inc_frame_repeat(self):
        self._frame_repeat = int(self._clip(self._frame_repeat * 2, 1, 20))

    def dec_frame_repeat(self):
        self._frame_repeat = int(self._clip(self._frame_repeat // 2, 1, 20))

    def add_noise_to_params(self, args, model, sigma=0.0001):
        params = [param for param in model.parameters()]

        noises = list()
        for param in params:
            noises.append(np.random.randn(*param.data.numpy().shape).astype(np.float32))

        new_params = list()
        for param, noise in zip(params, noises):
            new_params.append(param[:] + sigma * torch.tensor(noise))

        for param, weights in zip(model.parameters(), params):
            param.data = weights

        logger.info('add noise to param')
        return args, model
    


class Solution2(object):
    def __init__(self, path):
        self.path = path
        self.args = None
        self.results = list()
        self.fitness = -1e10
        self.cost = 10000
        self.ucb = None
        self.n_mutations = 0
        self.depth = 0

    def save_args(self, args):
        args_path = '{}/args.pkl'.format(self.path)
        try:
            if args is None:
                joblib.dump(self.args, args_path)
            else:
                joblib.dump(args, args_path)
        except OSError as exc:
            # "Input/output error"
            logger.warning('W: save_args, {}'.format(exc))
        except Exception as exc:
            logger.warning('W: save_args, {}'.format(exc))
            # embed(); exit()

    def load_args(self):
        args_path = '{}/args.pkl'.format(self.path)
        try:
            self.args = joblib.load(args_path, mmap_mode='r')
            self.results = getattr(self.args, 'results', list())
            self.fitness = getattr(self.args, 'fitness', None)
            self.cost = getattr(self.args, 'cost', 1.0)
            self.ucb = None
            self.n_mutations = getattr(self.args, 'n_mutations', 0)
            self.depth = getattr(self.args, 'depth', 1)
        except (EOFError, FileNotFoundError, struct.error) as exc:
            logger.warning('W: load_args, {}'.format(exc))
            self.args = None
        except Exception as exc:
            logger.warning('W: load_args, {}'.format(exc))
            import traceback; traceback.print_stack()
            self.args = None
            # embed(); exit()
        return self

    def save_model(self, model):
        model_path = '{}/model.pkl'.format(self.path)
        try:
            joblib.dump(model.state_dict(), model_path)
        except OSError as exc:
            # "Input/output error"
            logger.warning('W: save_args, {}'.format(exc))
        except Exception as exc:
            logger.warning('W: save_model, {}'.format(exc))
            # embed(); exit()

    def load_model(self):
        model_path = '{}/model.pkl'.format(self.path)
        try:
            return joblib.load(model_path, mmap_mode='r')
        except (EOFError, FileNotFoundError, struct.error) as exc:
            logger.warning('W: load_model, {}'.format(exc))
            self.args = None
        except Exception as exc:
            logger.warning('W: load_model, {}'.format(exc))
            import traceback; traceback.print_stack()
            # embed(); exit()
            return None

    def backup(self, backup_path):
        dst_path = os.path.join(backup_path, os.path.basename(self.path))
        try:
            logger.debug('rename')
            shutil.move(self.path, dst_path)
            logger.debug('remove sampled solution: {}'.format(self.path))
        except shutil.Error as exc:
            # No such file or directory
            # Destination path %s already exists
            logger.warning('W: backup, {}'.format(exc))
        except FileNotFoundError as exc:
            logger.warning('W: backup, {}'.format(exc))
        except OSError as exc:  # Directory not empty
            logger.warning('W: backup, {}'.format(exc))
        except Exception as exc:
            logger.warning('W: backup, {}'.format(exc))
            # embed(); exit()

    def delete(self):
        try:
            if os.path.exists(self.path):
                logger.info('rmtree: {}'.format(self.path))
                rmtree(self.path)
        except OSError as exc:
            logger.warning('W: delete, {}'.format(exc))
        except Exception as exc:
            logger.warning('W: delete, {}'.format(exc))

    def update_ucb(self, ucb_c, max_fitness, min_fitness, total_cost):
        self.ucb = self.ucb_func(
            self.fitness, self.cost, ucb_c, max_fitness, min_fitness, total_cost)
        return self

    @staticmethod
    def ucb_func(fitness, cost, ucb_c, max_fitness, min_fitness, total_cost):
        Q = (fitness - min_fitness + 1e-10) / (max_fitness - min_fitness + 1e-10)
        U = ucb_c * np.sqrt((total_cost + 1e-10) / (cost + 1e-10))
        return Q + U
       

def checkpoint(frames_remain):
    stop = False
    if frames_remain <= 0 and len(self.results) >= self.n_results:
        stop = True
        self.generation += 1
    return stop


class Population(object):

    def __init__(self, init_method, selection_method, mutation_method, 
                 path='population', max_size=30, n_results=2, n_eval=1,
                 selection_rate=0.2, survival_rate=0.8, ucb_c=0.25, n_init_mutation=10,
                 conditions=None, clean=False, **kwargs):

        # init method
        self.init_method = init_method

        # selection method
        self.selection_method = selection_method
        self.ucb_c = ucb_c  # 0.25, 1.41421
        self.selection_rate = selection_rate
        self.survival_rate = survival_rate

        # mutation method
        self.mutation_method = mutation_method
        self.n_init_mutations = n_init_mutation

        self.generation = 0
        self.max_size = max_size
        self.n_results = n_results
        self.n_eval = n_eval
        self.path = path

        self.sample_cost = 1.0

        self._solutions = None

        # make dirs
        logger.info('open population: {}'.format(os.path.abspath(path)))
        os.makedirs(self.path, exist_ok=True)

        # records experiment conditions
        if conditions:
            cmd = ' '.join(sys.argv)
            head = """{} <table style="width:100%"> <tr> <th>Key</th> <th>Value</th> </tr>\n""".format(cmd)
            body = ''
            for k in sorted(conditions.__dict__.keys()):
                if k[0] != '_':
                    body += """<tr> <td>{}</td> <td>{}</td> </tr>\n""".format(k, getattr(conditions, k))
            end = """</table>\n"""
            content = head + body + end
            with open(os.path.join(self.path, 'condition.html'), 'wt') as f:
                f.write(content)

        self.init_ops = []
        self.mutation_ops = []
        self.labels = []

        self.results = list()
        self.fitness = None
        self.ucb = None

    def init_population(self, args):

        self._solutions = [Solution() for _ in range(self.max_size)]


    def init_sample(self, sol, model):
        if self.init_method == 'none':
            pass

        elif self.init_method == 'sample':
            # model만 로드
            _, model = self.sample(sol, model)

        elif self.init_method == 'log_uniform_sample':
            _, model = self.sample(sol, model)
            for init_op in self.init_ops:
                sol, model = init_op(sol, model)

        elif self.init_method == 'multiple_mutation':
            sol, model = self.sample(sol, model)
            sol.id = getattr(sol, 'id', None)
            sol.depth = getattr(sol, 'depth', 1)
            sol.log = getattr(sol, 'log', list())

            for _ in range(self.n_init_mutations):
                mutation_op = random.choice(self.mutation_ops)
                sol, model = mutation_op(sol, model)

        return sol, model


    def checkpoint(self, frames_remain):
        stop = False
        if frames_remain <= 0 and len(self.results) >= self.n_results:
            stop = True
            self.generation += 1
        return stop

    def _best_select(self, solutions):
        return max(solutions, key=lambda s: s.fitness)

    def _truncated_select(self, solutions):
        parent_solutions = int(self.selection_rate * len(solutions))
        parent_solutions = max(len(solutions), parent_solutions)
        offspring_solution = random.choice(solutions[:parent_solutions])
        return offspring_solution

    def _prob_select(self, solutions):
        ucb = np.array([s.ucb for s in solutions])
        norm_ucb = (ucb - ucb.min() + 1e-10) / (ucb.max() - ucb.min() + 1e-10)
        probs = norm_ucb / norm_ucb.sum()
        offspring = np.random.choice(range(len(solutions)), p=probs)
        return solutions[offspring]

    def sample(self, args, model, eval_mode=False):
        logger.info('SAMPLE')
        solutions = glob.glob('{}/*'.format(self.path))
        solutions = [Solution(s).load_args() for s in solutions if os.path.isdir(s)]
        solutions = [s for s in solutions if s.args is not None]
        untried_solutions = [s for s in solutions if s.fitness is None]
        solutions = [s for s in solutions if s.fitness is not None]

        if len(untried_solutions) > 0 and eval_mode is False:
            untried_solution = random.choice(untried_solutions)
            logger.info('sample new solution: {}'.format(untried_solution.path))
            args = untried_solution.args
            state_dict = untried_solution.load_model()
            if state_dict is not None:
                model.load_state_dict(state_dict)
                if self.enable_backup:
                    untried_solution.backup(self.backup_path)
                else:
                    untried_solution.delete()

        elif len(solutions) > 0:
            fitness_list = [s.fitness for s in solutions]
            max_fitness, min_fitness = max(fitness_list), min(fitness_list)
            cost_list = [s.cost for s in solutions]
            total_cost = sum(cost_list)

            # update UCB
            [s.update_ucb(self.ucb_c, max_fitness, min_fitness, total_cost) for s in solutions]
            solutions = sorted(solutions, reverse=True, key=lambda s: s.ucb)
            logger.info('sample: # of solutions: {}'.format(len(solutions)))

            if self.selection_method == 'best':
                offspring_solution = self._best_select(solutions)
            elif self.selection_method == 'prob':
                offspring_solution = self._prob_select(solutions)
            elif self.selection_method == 'truncated':
                offspring_solution = self._truncated_select(solutions)

            offspring_solution = self._add_sample_cost(offspring_solution)
            args = offspring_solution.args
            state_dict = offspring_solution.load_model()
            if state_dict is not None:
                model.load_state_dict(state_dict)
            logger.info('sample solution: {}'.format(offspring_solution.path))

        args.id = getattr(args, 'id', None)
        args.cost = 1.0
        args.datetime = datetime.datetime.now().isoformat()
        self.results = list()
        self.fitness = None
        return args, model

    def _add_sample_cost(self, solution):
        # solution = Solution(solution_path).load_args()
        solution.args.cost = getattr(solution.args, 'cost', 1.0) + self.sample_cost
        solution.cost = solution.args.cost
        solution.save_args(None)
        return solution

    def _propagate_op_results(self, args, solutions, results):
        args.op_results = getattr(args, 'op_results', dict())
        args.op_n_tries = getattr(args, 'op_n_tries', dict())
        for op in self.mutation_ops:
            args.op_results.setdefault(op.__name__, 0.0)
            args.op_n_tries.setdefault(op.__name__, 1.0)

        args.op_results[args.op] += results
        args.op_n_tries[args.op] += 1.0

        parent_id = args.parent_id
        updated = [parent_id]
        while True:
            if parent_id is None:
                break

            parent_solutions = [s for s in solutions if s.path.find(parent_id) > 0]
            if len(parent_solutions) == 0:
                break

            parent = parent_solutions[0]
            logger.debug('update parent: {}, op: {} update'.format(parent_id, args.op))
            # parent_path = parent_solution[0].path
            # parent = Solution(parent_path).load_args()
            if parent.args is not None:
                parent.args.op_results = getattr(parent.args, 'op_results', dict())
                parent.args.op_n_tries = getattr(parent.args, 'op_n_tries', dict())
                for op in self.mutation_ops:
                    parent.args.op_results.setdefault(op.__name__, 0.0)
                    parent.args.op_n_tries.setdefault(op.__name__, 1.0)
                # logger.debug('- before: result: {}, n_tries: {}'.format(
                #     parent_args.op_results[args.op], parent_args.op_n_tries[args.op]))
                parent.args.op_results[args.op] += results
                parent.args.op_n_tries[args.op] += 1.0
                # logger.debug('- after: result: {}, n_tries: {}'.format(
                #     parent_args.op_results[args.op], parent_args.op_n_tries[args.op]))
                parent.save_args(None)
                # with open(db_args_path, 'wb') as f:
                #     cloudpickle.dump(parent_args, f)
                parent_id = parent.args.parent_id

                if parent_id in updated:
                    logger.warning('cycle in population')
                    break
                else:
                    updated.append(parent_id)
        return args

    def put(self, args, model):
        logger.info('PUT')
        args.parent_id = getattr(args, 'id', None)
        args.id = uuid.uuid1().hex
        args.fitness = getattr(args, 'fitness', None)
        args.op = getattr(args, 'op', None)
        args.cost = getattr(args, 'cost', 1.0)
        args.depth = getattr(args, 'depth', 0.0) + 1.0
        args.n_mutations = getattr(args, 'n_mutations', 0.0)

        os.makedirs(self.path, exist_ok=True)

        # update fitness
        args.results = self.results[-self.n_eval:]
        # self.fitness = np.mean(self.results)
        prev_fitness = args.fitness
        args.fitness = np.mean(args.results)
        args.f10, args.f25, args.f50, args.f75, args.f90 = np.percentile(
            args.results, [10, 25, 50, 75, 90])
        logger.info('eval: {:.3f}'.format(args.fitness))

        # maintain population size
        solutions = glob.glob('{}/*'.format(self.path))

        solutions = [Solution(s).load_args() for s in solutions if os.path.isdir(s)]
        solutions = [s for s in solutions if s.path is not None]
        untried_solutions = [s for s in solutions if s.fitness is None]
        solutions = [s for s in solutions if s.fitness is not None]
        fitness_list = [s.fitness for s in solutions] + [args.fitness]
        max_fitness, min_fitness = max(fitness_list), min(fitness_list)
        cost_list = [s.cost for s in solutions] + [args.cost]
        total_cost = sum(cost_list)

        # update UCB
        [s.update_ucb(self.ucb_c, max_fitness, min_fitness, total_cost) for s in solutions]
        args.ucb = Solution.ucb_func(
            args.fitness, args.cost, self.ucb_c, max_fitness, min_fitness, total_cost)
        solutions = sorted(solutions, reverse=True, key=lambda s: s.ucb)
        solutions, backups = solutions[:self.max_size], solutions[self.max_size:]
        self._solutions = solutions

        logger.debug('Current ucb: {:.2f}, fitness: {:.2f}, cost: {:.2f}, n_mutations: {}, depth: {}'.format(
            args.ucb, args.fitness, args.cost, args.n_mutations, args.depth))
        for solution in solutions[:30]:
            logger.debug('Pool ucb: {:.2f}, fitness: {:.2f}, cost: {:.2f}, n_mutations: {}, depth: {}'.format(
                solution.ucb, solution.fitness, solution.cost, solution.n_mutations, solution.depth))

        # backup
        for backup in backups:
            if self.enable_backup:
                backup.backup(self.backup_path)
            else:
                backup.delete()

        # evaluate solution
        rank = 0
        for rank in range(len(solutions)):
            if args.ucb >= solutions[rank].ucb:
                break
        rank += 1

        # truncated survive
        logger.debug('Current rank: {} / {}'.format(
            rank, max(1.0, self.survival_rate * self.max_size)))

        if rank > max(1.0, self.survival_rate * self.max_size) or len(untried_solutions) > 0:
            stop_training = True
        else:
            stop_training = False
        logger.info('{} training current solution: {}'.format(
            'stop' if stop_training else 'cont.', args.fitness))

        # update op's results
        if args.op is not None and prev_fitness is not None:
            if args.fitness > prev_fitness:
                args = self._propagate_op_results(args, solutions, 1.0)
            else:
                args = self._propagate_op_results(args, solutions, -1.0)

        # save current solution
        args.datetime = datetime.datetime.now().isoformat()
        args.log = getattr(args, 'log', list())
        args.log.append([self.generation] + [getattr(args, label) for label in self.labels])

        solution_path = '{}/{:.3f}__{}__{}'.format(
            self.path, args.fitness, args.depth, args.id)
        os.makedirs(solution_path, exist_ok=True)

        solution = Solution(solution_path)
        solution.save_args(args)
        solution.save_model(model)

        csv_path = '{}/model.csv'.format(solution_path)
        with open(csv_path, 'wt') as f:
            f.write(','.join(['generation'] + [label for label in self.labels]) + '\n')
            for vs in args.log:
                f.write(','.join([str(v) for v in vs]) + '\n')

        json_path = '{}/model.json'.format(solution_path)
        with open(json_path, 'wt') as f:
            json.dump(args.__dict__, f, sort_keys=True, indent=4)

        logger.info('save current solution: {}'.format(solution_path))

        args.op = None
        self.results = list()
        return args, model, stop_training

    @staticmethod
    def _random_op_choice(args, ops):
        return random.choice(ops)

    def _prob_op_choice(self, args, ops):
        args.op_results = getattr(args, 'op_results', dict())
        args.op_n_tries = getattr(args, 'op_n_tries', dict())
        for op in self.mutation_ops:
            args.op_results.setdefault(op.__name__, 0.0)
            args.op_n_tries.setdefault(op.__name__, 1.0)

        results = args.op_results
        n_tries = args.op_n_tries
        values = [float(results[op.__name__]) / float(n_tries[op.__name__]) for op in ops]
        values = np.array(values)
        norm_values = (values + 1.0) / 2.0
        probs = norm_values / norm_values.sum()

        logger.debug('mutation-op list:')
        for op, prob in zip(self.mutation_ops, probs):
            logger.debug('- op: {}: prob: {:.3f} ({} / {})'.format(
                op.__name__, prob, results[op.__name__], n_tries[op.__name__]))
        selected = np.random.choice(range(len(ops)), p=probs)
        return ops[selected]

    def _pred_op_choice(self, args, ops):
        if len(self._solutions) < 10:  # args.population_size // 4:
            logger.debug('population size is too small')
            return random.choice(ops)
        else:
            def vectorize(args):
                return [args.depth, args.lr, args.ent_coef, args.value_coef, args.frame_repeat], args.fitness

            n_hidden = 1024

            xs, ys = [], []
            for solution in self._solutions:
                x, y = vectorize(solution.args)
                xs.append(x)
                ys.append(y)
            xs = np.array(xs)
            ys = np.array(ys)

            x_scaler = StandardScaler()
            x_scaler.fit(xs)
            ys_ = (ys - min(ys)) / (max(ys) - min(ys))
            
            hidden_layer = MLPRandomLayer(n_hidden=n_hidden, activation_func='tanh')
            net = GenELMRegressor(hidden_layer=hidden_layer)
            xs_ = x_scaler.transform(xs)
            net.fit(xs_, ys_)
            pred = net.predict(xs_)
            pred_error = ((ys_ - pred) ** 2).mean()
            print('pred error: {:5.4f}'.format(pred_error))
            logger.debug('pred error: {:5.4f}'.format(pred_error))
            with open('pred_error.txt', 'at') as f:
                f.write('{:5.4f}\n'.format(pred_error))

            xs = []
            for op in ops:
                args_, _ = op(copy.deepcopy(args), None)
                x, _ = vectorize(args_)
                xs.append(x)
            xs = np.array(xs)

            xs_ = x_scaler.transform(xs)
            ys = net.predict(xs_)

            return ops[np.argmax(ys)]

    def mutate(self, args, model):
        args.datetime = datetime.datetime.now().isoformat()
        if self.mutation_method == 'random':
            mutation_op = self._random_op_choice(args, self.mutation_ops)
        elif self.mutation_method == 'prob':
            mutation_op = self._prob_op_choice(args, self.mutation_ops)
        elif self.mutation_method == 'pred':
            mutation_op = self._pred_op_choice(args, self.mutation_ops)
            
        args, model = mutation_op(args, model)
        args.op = mutation_op.__name__
        args.n_mutations = getattr(args, 'n_mutations', 0.0) + 1.0
        return args, model


# mutation operators
def log_uniform(low=1e-10, high=0.1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


def init_lr(sol, min_val=0.00001, max_val=0.005):
    sol.lr = float(log_uniform(min_val, max_val))
    return sol


def inc_lr(sol, min_val=1e-10, max_val=0.1):
    old_val = sol.lr
    sol.lr = max(min_val, min(max_val, 1.25 * sol.lr))
    # logger.info('inc lr: {:.10f} -> {:.10f}'.format(old_val, sol.lr))
    return sol


def dec_lr(sol, min_val=1e-10, max_val=0.1):
    old_val = sol.lr
    sol.lr = max(min_val, min(max_val, 0.8 * sol.lr))
    # logger.info('dec lr: {:.10f} -> {:.10f}'.format(old_val, sol.lr))
    return sol


def init_ent_coef(sol, model, min_val=0.0005, max_val=0.01):
    sol.ent_coef = float(log_uniform(min_val, max_val))
    return sol, model


def inc_ent_coef(sol, model, min_val=0.0, max_val=0.1):
    sol.ent_coef = max(min_val, min(max_val, 1.25 * sol.ent_coef))
    logger.info('inc ent_coef: {:.10f}'.format(sol.ent_coef))
    return sol, model


def dec_ent_coef(sol, model, min_val=0.0, max_val=0.1):
    sol.ent_coef = max(min_val, min(max_val, 0.8 * sol.ent_coef))
    logger.info('dec ent_coef: {:.10f}'.format(sol.ent_coef))
    return sol, model


def init_value_coef(sol, model, min_val=0.3, max_val=0.7):
    sol.value_coef = float(log_uniform(min_val, max_val))
    return sol, model


def inc_value_coef(sol, model, min_val=0.0, max_val=0.75):
    sol.value_coef = max(min_val, min(max_val, 1.25 * sol.value_coef))
    logger.info('inc value_coef: {:.10f}'.format(sol.value_coef))
    return sol, model


def dec_value_coef(sol, model, min_val=0.0, max_val=0.75):
    sol.value_coef = max(min_val, min(max_val, 0.8 * sol.value_coef))
    logger.info('dec value_coef: {:.10f}'.format(sol.value_coef))
    return sol, model


def init_frame_repeat(sol, model, min_val=4, max_val=10):
    sol.frame_repeat = int(np.random.uniform(min_val, max_val))
    return sol, model


def inc_frame_repeat(sol, model, min_val=1, max_val=8):
    old_val = sol.frame_repeat
    sol.frame_repeat = int(max(min_val, min(max_val, sol.frame_repeat + 1)))
    logger.info('inc frame repeat: {:.10f} -> {:.10f}'.format(old_val, sol.frame_repeat))
    return sol, model


def dec_frame_repeat(sol, model, min_val=1, max_val=8):
    old_val = sol.frame_repeat
    sol.frame_repeat = int(max(min_val, min(max_val, sol.frame_repeat - 1)))
    logger.info('dec frame repeat: {:.10f} -> {:.10f}'.format(old_val, sol.frame_repeat))
    return sol, model


def add_noise_to_params(args, model, sigma=0.0001):
    params = [param for param in model.parameters()]

    noises = list()
    for param in params:
        noises.append(np.random.randn(*param.data.numpy().shape).astype(np.float32))

    new_params = list()
    for param, noise in zip(params, noises):
        new_params.append(param[:] + sigma * torch.tensor(noise))

    for param, weights in zip(model.parameters(), params):
        param.data = weights

    logger.info('add noise to param')
    return args, model

