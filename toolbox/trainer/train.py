#!/usr/bin/env python

import copy
import datetime
import multiprocessing as mp
import os
import os.path as path
import random
import time
from collections import deque

import visdom
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import env as envs
import initializer
import tools
import utils
from const import (ESP, Msg, PopInit, PopLogShare, PopMode, PopMutation,
                   PopSelection, HEARTBEAT_BUFFER_SIZE)
from const import WorkerId as WID
from const import set_pop_configurations
from env import make_environment
from utils import ascii_plot, keyboard, pbt
from worker import (CloudPickleQueue, Mailbox,
                    actor_task, run_device, run_repeater)
from workers import Learner
from workers.replay_buffer import run_replay_buffer_hub

from IPython import embed


__author__ = 'Hyunsoo Park'
__email__ = 'rex8312@gmail.com'

# from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor
# from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer


def train(args):

    if args.cuda:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
    else:
        os.environ['OMP_NUM_THREADS'] = '16'
        os.environ['MKL_NUM_THREADS'] = '16'

    import numpy as np

    logger = initializer.get_logger(WID.Pop(), args=args)
    error_logger = initializer.get_error_logger(logger.log_path)

    # backup current codebase
    project_name = path.basename(path.abspath('.'))
    dst = path.join(logger.log_data_path, f'{project_name}.zip')
    utils.archive_dir('.', dst)

    # initialization
    start_time = time.time()
    args.policy_optimizer.seed(args.seed)

    device = 'cuda' if args.cuda and utils.cuda_available() else 'cpu'
    logger.info('use device: {}'.format(device))
    logger.table('args', args.__dict__, width=350)

    # pop method를 참조하여 적절한 init, selection, mutation 방법 설정
    args = set_pop_configurations(args)

    # 환경, 모델 초기화
    env = envs.make_environment(args, False, False, init_game=False)
    env.close()

    # Actor + Learner 프로세스 실행
    args.n_actors = min(mp.cpu_count(), args.n_actors)

    processes = list()

    # mailbox 초기화
    mailbox = Mailbox(args.queue_type, args.pop_log_share_method)
    # inputs = mailbox.get_queue(WID.Pop())
    mailbox.get_queue(WID.Logger())
    mailbox.get_queue(WID.Actor())
    # mailbox.get_queue(WID.ReplayBufferHub())
    for rank in range(args.n_learners):
        mailbox.get_queue(WID.Learner(rank))
        # mailbox.get_queue(WID.Control(rank))
        # mailbox.get_queue(WID.ReplayBuffer(rank))

    # actors
    for rank in range(args.n_actors):  # TODO: ip * 100 + rank 필요
        p = mp.Process(
            target=actor_task,
            args=(rank, args, 'cpu', mailbox.get_queue(WID.Actor()), mailbox),
            daemon=True)
        processes.append(p)

    # test_actor
    if args.test_actor:
        logger.info('start test actor')
        p = mp.Process(
            target=actor_task,
            args=(-1, args, 'cpu', mailbox.get_queue(WID.Actor()), mailbox),
            daemon=True)
        processes.append(p)

    # remote actors
    if args.start_relay_server:

        p = mp.Process(
            target=run_device,
            args=(args.repeater_frontend_port, args.repeater_backend_port),
            daemon=True,
        )
        processes.append(p)

        p = mp.Process(
            target=run_repeater,
            args=(
                args.repeater_frontend_port,
                args.repeater_backend_port,
                mailbox.get_queue(WID.Actor()), mailbox),
            daemon=True,
        )
        processes.append(p)

    [p.start() for p in processes]

    # learner
    learners = list()
    for rank in range(args.n_learners):
        learner = Learner(rank, args, device,
                          mailbox.get_queue(WID.Learner(rank)), mailbox)
        learners.append(learner)

    # 리플레이 버퍼
    # p = mp.Process(
    #     target=run_replay_buffer_hub,
    #     args=(mailbox.get_queue(WID.ReplayBufferHub()), mailbox),
    #     daemon=True)
    # processes.append(p)

    # TODO: init population
    pop = []  # 집단
    generation = 0  # 현재 세대
    n_games = 0  # 총 플레이한 게임 수
    n_steps = 0  # 총 스텝 수
    generation_start_time = time.time()  # 학습 시작 시간
    generation_n_games = 0  # 한 세대동안 플레이한 게임 수
    generation_n_steps = 0  # 한 세대동안 스텝 수
    # 최근 점수 목록 [learner 마다 따로 점수 기록]
    recent_scores = [deque(maxlen=args.pbt_n_evals) for _ in range(args.n_learners)]
    current_steps = 0  # 현재 스텝, args.log_share_method에 따라 generation_n_steps와 다를 수 있음
    steps_per_generation = args.steps_per_generation # 한 세대 동안 프레임 수
    heartbeat_recoder = dict()  # heartbeat 정보 저장

    # 모델 초기화
    assert args.n_learners > 0
    for rank in range(args.n_learners):
        model = args.policy_optimizer.make_model(device, args, env)
        sol = pbt.Solution(params=model.to_params(), args=args)
        sol = sol.init(args.pop_init_method)
        args = sol.apply(args)
        sol.generation = generation
        learners[rank].set_solution(sol)

    # 학습 시작
    while True:

        # Learner (local search) 실행
        for learner in learners:
            learner.step()
            
        # Evolution Computation (global search) 실행
        if all([learner.done() for learner in learners]):
            [learner.reset() for learner in learners]
            # if generation_n_steps > steps_per_generation:
            logger.table('args', args.__dict__, width=350)
            logger.info(f'method: {args.pop_method}, seed: {args.seed}, Tag: {args.pop_tag}')

            #----------------
            # Evaluation: 현재 learners 에서 solution 수집
            #-----------------
            children = list()
            for learner in learners:
                sol = learner.get_solution()
                if len(sol.scores) >= args.pbt_n_evals:
                    sol.rank = learner.rank
                    sol.learner = WID.Learner(learner.rank)
                    # solution이 생성될 당시의 세대
                    sol.generation = generation
                    sol.depth += 1
                    sol.propagate_op_result(pop)
                    children.append(sol)
                    logger.info('child: {}'.format(sol))

            if len(pop) >= args.population_size:
                # --------------
                # Selection: 성능이 좋은 모델 선택
                # --------------
                low_perf_learners = list()
                new_solutions = list()

                if args.pop_selection_method == PopSelection.NONE:
                    pass

                elif args.pop_selection_method == PopSelection.TRUNCATED_SELECTION:
                    # population 평균 점수로 정렬
                    pop = sorted(pop, key=lambda sol: sol.score, reverse=True)
                    children = sorted(children, key=lambda sol: sol.score, reverse=True)

                    for child in children[1:]:
                        # 점수 샘플 추출
                        child_scores = np.array(child.scores)[-args.pbt_n_evals//2:]
                        best_scores = np.array(children[0].scores)[-args.pbt_n_evals//2:]

                        # 유의성 검사
                        stat = stats.mannwhitneyu(child_scores, best_scores, alternative='less')
                        if stat.pvalue < 0.01:
                            # sol 교체할 learner 기록
                            low_perf_learners.append((child.rank, np.median(child_scores), stat.pvalue))

                    for _ in low_perf_learners:
                        sol = random.choice(pop)
                        sol.n_try += 1
                        new_solutions.append(copy.deepcopy(sol))

                elif args.pop_selection_method == PopSelection.UCB_SELECTION:
                    # TODO: pop, children 분리 방식에 맞게 업데이트 필요

                    def ucb_func(score, total_try, n_try, ucb_c, max_score, min_score):
                        Q = (score - min_score + 1e-10) / (max_score - min_score + 1e-10)
                        U = np.sqrt((total_try + 1e-10) / (n_try + 1e-10))
                        return Q + ucb_c * U

                    # ucb 값 갱신
                    max_score = max([sol.mean_score for sol in pop + children])
                    min_score = min([sol.mean_score for sol in pop + children])
                    total_try = sum([sol.n_try for sol in pop + children])

                    for sol in pop:
                        sol.ucb = ucb_func(
                            sol.mean_score, total_try, sol.n_try,
                            args.pop_ucb_c, max_score, min_score)

                    # ucb로 정렬
                    pop = sorted(pop, key=lambda sol: sol.ucb, reverse=True)

                    # TODO: 문제 부분!!
                    # selection 공통부분
                    if args.pop_selection_method != PopSelection.NONE:
                        # 저성능 learner 제거, 및 새로운 solution 생성
                        # 이번 세대에 생성된 sol 중에 집단에서 상대적으로 성능이 낮은 sol 검색
                        # split_value = 1
                        for i, sol in enumerate(pop):
                            if i >= args.pop_survival_ratio * args.population_size:
                                if sol.generation == generation:
                                    low_perf_learners.append(sol.rank)

                        logger.info('Stop learning: {}'.format(low_perf_learners))

                        # 찾은 sol 개수 만큼 새로운 sol 생성
                        for _ in range(len(low_perf_learners)):
                            n_parents = int(args.pop_selection_ratio * args.population_size)
                            sol = pop[random.randint(0, n_parents - 1)]
                            sol.n_try += 1
                            new_solutions.append(copy.deepcopy(sol))

                # --------------
                # Mutation: 변형
                # --------------
                logger.info('Select next solution')
                if args.pop_mutation_method == PopMutation.NONE:
                    pass

                elif args.pop_mutation_method == PopMutation.RANDOM:
                    new_solutions = [sol.random_mutation() for sol in new_solutions]

                elif args.pop_mutation_method == PopMutation.BACKUP:
                    new_solutions = [sol.prob_mutation() for sol in new_solutions]

                elif args.pop_mutation_method == PopMutation.PREDICTION:
                    if len(pop) <= args.population_size:
                        new_solutions = [sol.random_mutation() for sol in new_solutions]
                    else:
                        xs, ys = zip(*[sol.vec for sol in pop])
                        xs, ys = np.array(xs), np.array(ys)

                        x_scaler = StandardScaler()
                        x_scaler.fit(xs)

                        # n_hidden_nodes = len(2 * xs[0])
                        # hidden_layer = MLPRandomLayer(
                        #     n_hidden=n_hidden_nodes, activation_func='tanh')
                        # net = GenELMRegressor(hidden_layer=hidden_layer)

                        xs_ = x_scaler.transform(xs)
                        ys_ = (ys - min(ys)) / (max(ys) - min(ys) + ESP)
                        # net.fit(xs_, ys_)
                        reg = LinearRegression().fit(xs_, ys_)

                        for sol in new_solutions:
                            sol.pred_w = max(ys) - min(ys)
                            sol.pred_bias = min(ys)

                        # pred = net.predict(xs_)
                        pred = reg.predict(xs_)
                        pred_error = ((ys_ - pred) ** 2).mean()
                        logger.debug('pred error: {:5.4f}'.format(pred_error))
                        logger.line('loss/prediction error', generation, pred_error)
                        new_solutions = [sol.predict_mutation(
                            x_scaler, reg) for sol in new_solutions]

                # 키보드 이벤트 검사 -> 여기서 수작업으로 파라미터 조작
                if keyboard.event('esc'):
                    print('==== Enter debuge mode ====')
                    embed()

                # learner 에 전송
                pvalues = list()
                for (rank, prev_score, pvalue), sol in zip(low_perf_learners, new_solutions):
                    learners[rank].set_solution(sol)
                    logger.info('replace sol: {}, {: >8.1f}, {: >1.3f} <- {}'.format(
                        WID.Learner(rank), prev_score, pvalue, sol))
                    pvalues.append(pvalue)

                if pvalues:
                    logger.line('p-value', generation, np.mean(pvalues), visdom_env='main')

            # chidren 출력
            # children = sorted(children, key=lambda sol: sol.rank, reverse=False)
            # for i, child in enumerate(children):
            #     logger.info('child: {}'.format(child))

            # 현재 population 출력
            pop = pop + children
            # 평균성능으로 정렬
            pop = sorted(pop, key=lambda sol: sol.score, reverse=True)
            for i, sol in enumerate(pop):
                logger.info('{} -> {}'.format(i, sol))
            # 성능이 나쁜 모델 제거
            pop = pop[:args.population_size]

            try:
                # population에서 가장 좋은 개체 결과 출력
                # if len(pop) > 0:
                #     for name in pop[0].hyperparams:
                #         value = np.mean([getattr(s, name) for s in pop])
                #         logger.line('hyperparams-p/' + name, generation, value)

                # children 하이퍼파라미터 평균
                for child in children:
                    for name in child.hyperparams:
                        value = np.mean([getattr(s, name) for s in pop])
                        logger.line('hyperparams/' + name, generation, value)

                # logger.line('sys/fd_counts', generation, utils.count_open_fds(), use_visdom=True)
                logger.line('sys/memory_usage', generation,
                            utils.get_memory_usage(), visdom_env='sys')
                logger.line('sys/memory_delta', generation,
                            utils.get_memory_usage_delta(), visdom_env='sys')

                avg_game_time = (time.time() - start_time) / n_games
                logger.line('sys/average_game_time', generation,
                            avg_game_time, visdom_env='sys')

                generation_interval = time.time() - generation_start_time
                logger.line('sys/game_time', generation,
                            generation_interval / generation_n_games, visdom_env='sys')
                logger.line('sys/fps', generation, generation_n_steps /
                            generation_interval, average=3, visdom_env='main')
            except Exception as exc:
                logger.error('Logging error {}'.format(exc))
                embed()

            # 남은 frame 초기화
            current_steps = 0

            generation += 1
            generation_start_time = time.time()
            generation_n_games = 0
            generation_n_steps = 0

        # log 작업
        logger_inputs = mailbox.get_mailbox(WID.Logger())
        # while logger_inputs.qsize() > 0 and current_steps <= steps_per_generation:
        while logger_inputs.qsize() > 0:
            resp = logger_inputs.get()
            if resp['msg'] == Msg.SAVE_LOG:
                n_steps += resp['frames']
                generation_n_steps += resp['frames']
                # if args.pop_log_share_method == PopLogShare.OneToOne:
                #     current_steps += resp['frames']
                # elif args.pop_log_share_method == PopLogShare.OneToAll:
                #     current_steps += resp['frames'] // args.n_learners
                # else:
                #     raise NotImplementedError

                learner_rank = resp['rank']
                learner_name = resp['sender']
                scores = resp['scores']
                recent_scores[learner_rank] += scores
                n_games += len(resp['scores'])
                generation_n_games += len(resp['scores'])

                current_time = int(time.time() - start_time)
                n_queued_episodes = np.mean(
                    [mb.qsize() for name, mb in mailbox.mailbox.items()
                     if name.startswith(WID.Learner())])
                msg = [
                    '{}'.format(args.session_id),
                    '{}'.format(datetime.timedelta(seconds=current_time)),
                    '{}'.format(learner_name),
                    'G: {: >4,d}'.format(generation),
                    '{: >9,d} < {: >3.1f}'.format(n_steps, n_queued_episodes),
                    'FPS: {: >4.1f}'.format(n_steps / current_time),
                    'Score: {: >8.1f}, #eval: {: >3d}'.format(
                        np.mean(recent_scores[learner_rank]), len(scores)),
                    'Loss: {: >6.4f}'.format(resp['total_loss'])]
                msg = ', '.join(msg)
                logger.info(msg)

                try:
                    # 평균 점수:
                    # 개별 learner 의 최근 점수 50개(기본값) 의 평균을 구하고,
                    # 그중 최고 점수를 현재 세대의 평가값으로 함
                    # 아직 최근 점수 개수가 부족하면 계산에서 제외한다.
                    valid_scores = np.array([s for s in recent_scores if len(s) == args.pbt_n_evals])
                    if len(valid_scores) > 0:
                        max_score = np.array(valid_scores).mean(axis=1).max(axis=0)
                        score_dict = dict(max_score=max_score)
                        # 개별 learner 점수
                        for rank, scores in enumerate(recent_scores):
                            if len(scores) == args.pbt_n_evals:
                                score_dict[f'learner_{rank}'] = np.mean(scores)
                        logger.line('score', generation, score_dict, visdom_env='main', width=600)
                    logger.line('losses', generation, resp['loss'], visdom_env='main', width=600)
                    logger.line('values', generation, resp['values'], visdom_env='main')
                    logger.bar('gradients', resp['gradients'], visdom_env='main', sort=False)
                    logger.bar('buffer_size', resp['buffer_size'], visdom_env='main', sort=False)
                except Exception as exc:
                    logger.error('Logging error {}'.format(exc))
                    embed()

            elif resp['msg'] == Msg.HEARTBEAT:
                # actor와 learner가 정상적으로 작동하고 있는지 모니터링 하는 용도
                heartbeat = resp['heartbeat']
                heartbeats_recv_time = time.time()
                heartbeat_recoder[heartbeat] = heartbeats_recv_time
                heart_beat_eltime = dict()
                for hb in heartbeat_recoder:
                    heart_beat_eltime[hb] = heartbeats_recv_time - heartbeat_recoder[hb]

                logger.bar('heartbeats', heart_beat_eltime, visdom_env='main', sort=True)

            elif resp['msg'] == Msg.ERROR_INFO:
                error_logger.error(resp['error_info'] + '\n')

        # logger.progressbar(
        #     'Generations',
        #     ['Gen.', 'Frames'],
        #     [generation, current_steps],
        #     [args.max_generations, steps_per_generation], visdom_env='main',)
        logger.progressbar(
            'Generations',
            ['Gen.'] + [WID.Learner(l.rank) for l in learners],
            [generation] + [l.policy_optimizer.cur_steps for l in learners],
            [args.max_generations] + [l.policy_optimizer.steps_per_generation for l in learners], 
            visdom_env='main',)

        logger.line('sys/q_size', generation,
                    mailbox.qsize_dict(), visdom_env='main')

        # PBT 학습 종료
        if generation >= args.max_generations:
            logger.info('==== DONE TRAIN: Gen. {}, Step: {}, Elapsed Time: {} ===='.format(
                generation, n_steps, datetime.timedelta(seconds=time.time() - start_time)))
            utils.log_tools.save_log_to_csv(args.log_path, args.session_id)

            # # 자식 프로세스 전부 강제 종료
            # pid = os.getpid()
            # utils.kill_children_processes(pid)
            # [p.kill() for p in processes]
            # [learner.policy_optimizer.clear() for learner in learners]
            break

