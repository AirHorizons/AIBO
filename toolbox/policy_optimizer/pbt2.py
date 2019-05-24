
import copy
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
import joblib
from IPython import embed
# pip install sklearn-extensions
from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.preprocessing import StandardScaler


# TODO: elm 사용
# pip install sklearn-extensions
# from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor
# from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
# net = GenELMRegressor(hidden_layer=MLPRandomLayer(n_hidden=10, activation_func='tanh'))
# net.fit(X, y)
# pred = net.predict(X)


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


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
                # shutil.rmtree(self.path)
                rmtree(self.path)
        except OSError as exc:
            logger.warning('W: delete, {}'.format(exc))
        except Exception as exc:
            logger.warning('W: delete, {}'.format(exc))
            # embed(); exit()

    def update_ucb(self, ucb_c, max_fitness, min_fitness, total_cost):
        self.ucb = self.ucb_func(
            self.fitness, self.cost, ucb_c, max_fitness, min_fitness, total_cost)
        return self

    @staticmethod
    def ucb_func(fitness, cost, ucb_c, max_fitness, min_fitness, total_cost):
        Q = (fitness - min_fitness + 1e-10) / (max_fitness - min_fitness + 1e-10)
        U = ucb_c * np.sqrt((total_cost + 1e-10) / (cost + 1e-10))
        return Q + U


class Population(object):

    def __init__(self, init_method, selection_method, mutation_method, 
                 path='population', max_size=30, n_results=2, n_eval=1,
                 selection_rate=0.2, survival_rate=0.8, ucb_c=0.25, n_init_mutation=10,
                 conditions=None, clean=False, backup=False, **kwargs):

        self.node_id = kwargs.get('node_id', -1)
        logger.addHandler(logging.FileHandler('pbt_{}.log'.format(self.node_id)))
        self.max_lock_time = kwargs.get('max_lock_time', 30)

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
        self.enable_backup = backup
        self.backup_path = path + '_backup'

        self.exproation_success_cost = 0.0
        self.exproation_failure_cost = 0.0
        self.sample_cost = 1.0

        self._solutions = list()

        # selection and mutation improvement?
        # self.boltzmann_selection = kwargs.get('boltzmann_selection', False)
        # self.boltzmann_mutation = kwargs.get('boltzmann_mutation', True)
        # self.uniform_search_cost = kwargs.get('uniform_search_cost', False)

        # make dirs
        self.acquire_lock()
        if os.path.exists(path):
            logger.info('open population: {}'.format(os.path.abspath(path)))

            if clean:
                if os.path.exists(self.path):
                    logger.info('rmtree: {}'.format(self.path))
                    paths = glob.glob('{}/*'.format(self.path))
                    paths = [path for path in paths if '_pop_lock-' not in path]
                    for path in paths:
                        try:
                            # 어짜피, 편의기능이니까, 나중에 제거
                            if os.path.isdir(path):
                                rmtree(path)
                            else:
                                os.remove(path)
                        except OSError as exc:
                            logger.warning('pop init: {}'.format(exc))
                            time.sleep(1)
                        # except KeyboardInterrupt as exc:
                        #   embed(); exit()
                os.makedirs(self.path, exist_ok=True)
        else:
            os.makedirs(self.path, exist_ok=True)
            if self.enable_backup:
                os.makedirs(self.backup_path, exist_ok=True)
            logger.info('make population: {}'.format(os.path.abspath(path)))

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
            with open(os.path.join(self.path, 'condition_{}.html'.format(self.node_id)), 'wt') as f:
                f.write(content)

        self.init_ops = [init_lr, init_ent_coef, init_frame_repeat]
        self.mutation_ops = [mutate_lr, mutate_v_coef, mutate_p_coef, mutate_ent_coef]
        self.labels = ['datetime', 'parent_id', 'id', 'fitness', 'depth', 'n_mutations', 'lr', 'ent_coef']

        self.results = list()
        self.fitness = None
        self.ucb = None
        self.release_lock()

    def acquire_lock(self):
        # TODO: mmap 으로 수정
        lock_file = '{}/_pop_lock-{}'.format(self.path, self.node_id)
        print('acquire lock: {}'.format(lock_file))

        os.makedirs(self.path, exist_ok=True)
        lock_files = glob.glob('{}/_pop_lock-*'.format(self.path))

        while True:
            locked_times = []
            for lf in lock_files:
                try:
                    locked_times.append(time.time() - os.path.getmtime(lf))
                except FileNotFoundError:
                    pass
            locked_times = [lt for lt in locked_times if lt < self.max_lock_time]
            if len(locked_times) > 0:
                print('pop locked: {} sec.'.format(min(locked_times)))
                time.sleep(1)
            else:
                break

        with open(lock_file, 'wt'):
            pass

    def release_lock(self):
        lock_file = '{}/_pop_lock-{}'.format(self.path, self.node_id)
        print('release lock: {}'.format(lock_file))

        if os.path.exists(lock_file):
            os.remove(lock_file)

    def checkpoint(self, frames_remain):
        stop = False
        if frames_remain <= 0 and len(self.results) >= self.n_results:
            stop = True
            self.generation += 1
        return stop

    def _best_select(self, solutions):
        # solutions = sorted(solutions, reverse=True, key=lambda s: s.fitness)
        # return solutions[0]
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

    def init_sample(self, args, model):
        if self.init_method == 'none':
            pass
        elif self.init_method == 'sample':
            # model만 로드
            _, model = self.sample(args, model)

        elif self.init_method == 'log_uniform_sample':
            _, model = self.sample(args, model)
            for init_op in self.init_ops:
                args, model = init_op(args, model)

        elif self.init_method == 'multiple_mutation':
            args_, model = self.sample(args, model)
            args.id = getattr(args_, 'id', None)
            args.depth = getattr(args_, 'depth', 1)
            args.log = getattr(args_, 'log', list())

            for _ in range(self.n_init_mutations):
                mutation_op = random.choice(self.mutation_ops)
                args, model = mutation_op(args, model)

        return args, model

    def sample(self, args, model, eval_mode=False):
        self.acquire_lock()
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
        self.release_lock()
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
        self.acquire_lock()
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

        # if self.uniform_search_cost:
        #     args.cost += 1.0
        # else:
        #     if args.fitness > getattr(args, 'fitness', -1e10):
        #         args.cost += self.exproation_success_cost
        #     else:
        #         args.cost += self.exproation_failure_cost

        # maintain population size
        solutions = glob.glob('{}/*'.format(self.path))
        # solutions = list()
        # for solution_file in solution_files:
        #     if os.path.isdir(solution_file):
        #         try:
        #             solutions.append(Solution(solution_file))
        #         except Exception as exc:
        #             pass
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
        self.release_lock()
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


def init_lr(args, model, min_val=0.00001, max_val=0.005):
    args.lr = float(log_uniform(min_val, max_val))
    return args, model


def init_ent_coef(args, model, min_val=0.0005, max_val=0.01):
    args.ent_coef = float(log_uniform(min_val, max_val))
    return args, model


def init_value_coef(args, model, min_val=0.3, max_val=0.7):
    args.value_coef = float(log_uniform(min_val, max_val))
    return args, model


def init_frame_repeat(args, model, min_val=4, max_val=10):
    args.frame_repeat = int(np.random.uniform(min_val, max_val))
    return args, model


def inc_lr(args, model, min_val=1e-10, max_val=0.1):
    old_val = args.lr
    args.lr = max(min_val, min(max_val, 1.25 * args.lr))
    logger.info('inc lr: {:.10f} -> {:.10f}'.format(old_val, args.lr))
    return args, model


def dec_lr(args, model, min_val=1e-10, max_val=0.1):
    old_val = args.lr
    args.lr = max(min_val, min(max_val, 0.8 * args.lr))
    logger.info('dec lr: {:.10f} -> {:.10f}'.format(old_val, args.lr))
    return args, model


def inc_frame_repeat(args, model, min_val=1, max_val=8):
    old_val = args.frame_repeat
    args.frame_repeat = int(max(min_val, min(max_val, args.frame_repeat + 1)))
    logger.info('inc frame repeat: {:.10f} -> {:.10f}'.format(old_val, args.frame_repeat))
    return args, model


def dec_frame_repeat(args, model, min_val=1, max_val=8):
    old_val = args.frame_repeat
    args.frame_repeat = int(max(min_val, min(max_val, args.frame_repeat - 1)))
    logger.info('dec frame repeat: {:.10f} -> {:.10f}'.format(old_val, args.frame_repeat))
    return args, model


def inc_ent_coef(args, model, min_val=0.0, max_val=0.1):
    args.ent_coef = max(min_val, min(max_val, 1.25 * args.ent_coef))
    logger.info('inc ent_coef: {:.10f}'.format(args.ent_coef))
    return args, model


def dec_ent_coef(args, model, min_val=0.0, max_val=0.1):
    args.ent_coef = max(min_val, min(max_val, 0.8 * args.ent_coef))
    logger.info('dec ent_coef: {:.10f}'.format(args.ent_coef))
    return args, model


def inc_value_coef(args, model, min_val=0.0, max_val=0.75):
    args.value_coef = max(min_val, min(max_val, 1.25 * args.value_coef))
    logger.info('inc value_coef: {:.10f}'.format(args.value_coef))
    return args, model


def dec_value_coef(args, model, min_val=0.0, max_val=0.75):
    args.value_coef = max(min_val, min(max_val, 0.8 * args.value_coef))
    logger.info('dec value_coef: {:.10f}'.format(args.value_coef))
    return args, model


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


def mutate_lr(args, min_val=1e-10, max_val=0.1):
    args.lr = random.choice([0.8, 1.2]) * args.lr
    args.lr = min(max_val, args.lr)
    args.lr = max(min_val, args.lr)
    return args


def mutate_v_coef(args, min_val=0.0, max_val=1.0):
    args.v_coef = random.choice([0.8, 1.2]) * args.v_coef
    args.v_coef = min(max_val, args.v_coef)
    args.v_coef = max(min_val, args.v_coef)
    return args


def mutate_p_coef(args, min_val=0.0, max_val=1.0):
    args.p_coef = random.choice([0.8, 1.2]) * args.p_coef
    args.p_coef = min(max_val, args.p_coef)
    args.p_coef = max(min_val, args.p_coef)
    return args


def mutate_ent_coef(args, min_val=0.0, max_val=1.0):
    args.ent_coef = random.choice([0.8, 1.2]) * args.ent_coef
    args.ent_coef = min(max_val, args.ent_coef)
    args.ent_coef = max(min_val, args.ent_coef)
    return args
