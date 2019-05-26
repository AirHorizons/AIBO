
import datetime
import multiprocessing as mp
import os
import random
import time
from collections import deque

import visdom
from IPython import embed
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
# from worker import run_relay_server
from worker import (CloudPickleQueue, Mailbox, actor_task, learner_task,
                    learner_task_gen, run_device, run_repeater)
from workers.replay_buffer import run_replay_buffer_hub

# mp = mp.get_context('spawn')


# from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor
# from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer


# from model import make_model
# from worker import MsgPackQueue


def simple_train(args, sync_learner):

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
    project_name = os.path.basename(os.path.abspath('.'))
    dst = os.path.join(logger.log_data_path, f'{project_name}.zip')
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
    args.n_actors = min(mp.cpu_count() // args.n_learners, args.n_actors)

    processes = list()

    # mailbox 초기화
    mailbox = Mailbox(args.queue_type, args.pop_log_share_method)
    inputs = mailbox.get_queue(WID.Pop())
    mailbox.get_queue(WID.Logger())
    mailbox.get_queue(WID.Actor())
    mailbox.get_queue(WID.ReplayBufferHub())
    for rank in range(args.n_learners):
        mailbox.get_queue(WID.Control(rank))
        mailbox.get_queue(WID.Learner(rank))
        mailbox.get_queue(WID.ReplayBuffer(rank))

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

    # learner
    if sync_learner:
        learners = list()
        for rank in range(args.n_learners):
            learner_gen = learner_task_gen(rank, args, device,
                                           (mailbox.get_queue(WID.Control(rank)),
                                            mailbox.get_queue(WID.Learner(rank))),
                                           mailbox)
            learners.append(learner_gen)
    else:
        for rank in range(args.n_learners):
            p = mp.Process(
                target=learner_task,
                args=(rank, args, device,
                      (mailbox.get_queue(WID.Control(rank)),
                       mailbox.get_queue(WID.Learner(rank))),
                      mailbox),
                daemon=True)
            processes.append(p)

    # 리플레이 버퍼
    p = mp.Process(
        target=run_replay_buffer_hub,
        args=(mailbox.get_queue(WID.ReplayBufferHub()), mailbox),
        daemon=True)
    processes.append(p)

    [p.start() for p in processes]

    # TODO: init population
    pop = []
    generation = 0
    generation_start_time = time.time()
    generation_n_games = 0
    generation_frames = 0
    n_games = 0
    n_steps = 0
    total_frames = 0
    recent_scores = deque(maxlen=args.pbt_n_evals * args.n_learners)
    frames_per_epoch = args.frames_per_epoch * args.n_learners
    frames = 0
    heartbeat_recoder = dict()
    heartbeat_recoder_update = 0

    # 모델 초기화
    assert args.n_learners > 0
    for rank in range(args.n_learners):
        model = args.policy_optimizer.make_model(device, args, env)
        sol = pbt.Solution(params=model.to_params(), args=args)
        sol = sol.random_init(args.pop_init_method)
        args = sol.apply(args)
        sol.generation = generation
        mailbox.put(dict(sender=WID.Pop(), to=WID.Control(
            rank), msg=Msg.NEW_SOLUTION, solution=sol))

    # 학습 시작
    iteration = 0
    while True:
        if sync_learner:
            for learner_gen in learners:
                next(learner_gen)

        if frames > frames_per_epoch:
            logger.table('args', args.__dict__, width=350)
            logger.info(f'method: {args.pop_method}, Tag: {args.pop_tag}')
            logger.text('pop', generation, f'Generation: {generation}')

            # 모든 learner에게 지금까지 학습한 모델을 요청
            logger.info('Request solution from all learners')
            for rank in range(args.n_learners):
                mailbox.put(dict(sender=WID.Pop(), to=WID.Control(
                    rank), msg=Msg.REQ_SOLUTION))

            if sync_learner:
                resp_list = list()
                for learner_gen in learners:
                    next(learner_gen)
                    if inputs.qsize() > 0:
                        resp = inputs.get()
                        if resp['msg'] == Msg.SAVE_SOLUTION:
                            resp_list.append(resp)

                    if len(resp_list) >= args.n_learners:
                        break

            logger.info('Gather current solutions')

            def proces_resp(resp):
                if resp['msg'] == Msg.SAVE_SOLUTION:
                    sol = resp['solution']
                    sol.learner = resp['sender']
                    sol.generation = generation
                    sol.depth += 1
                    if sol.loss == np.NaN:
                        sol.mean_score = -1e10
                    sol.propagate_op_result(pop)
                    pop.append(sol)

            if sync_learner:
                for resp in resp_list:
                    proces_resp(resp)
            else:
                for _ in range(args.n_learners):
                    resp = inputs.get()
                    proces_resp(resp)

            pop = sorted(pop, key=lambda sol: sol.mean_score, reverse=True)

            # Selection: 성능이 좋은 모델 선택
            learner_to_stop = list()
            new_solutions = list()

            if args.pop_selection_method == PopSelection.NONE:
                pass

            elif args.pop_selection_method == PopSelection.TRUNCATED_SELECTION:

                for i, sol in enumerate(pop):
                    if i + 1 > args.pop_survival_ratio * args.population_size:
                        if sol.generation >= generation:
                            learner_to_stop.append((sol.learner, i))

                logger.info('Stop learning: {}'.format(learner_to_stop))

                for learner, _ in learner_to_stop:
                    max_idx = min(len(pop) - 1,
                                  int(args.pop_selection_ratio * args.population_size) - 1)
                    sol = pop[random.randint(0, max_idx)]
                    new_solutions.append(sol)

            elif args.pop_selection_method == PopSelection.UCB_SELECTION:

                def ucb_func(score, total_try, n_try, ucb_c, max_score, min_score):
                    Q = (score - min_score + 1e-10) / \
                        (max_score - min_score + 1e-10)
                    U = np.sqrt((total_try + 1e-10) / (n_try + 1e-10))
                    return Q + ucb_c * U

                # ucb 값 갱신
                max_score = max([sol.mean_score for sol in pop])
                min_score = min([sol.mean_score for sol in pop])
                total_try = sum([sol.n_try for sol in pop])

                for sol in pop:
                    sol.ucb = ucb_func(sol.mean_score, total_try, sol.n_try,
                                       args.pop_ucb_c, max_score, min_score)

                # ucb로 정렬
                pop = sorted(pop, key=lambda sol: sol.ucb, reverse=True)
                for i, sol in enumerate(pop):
                    if i + 1 > args.pop_survival_ratio * args.population_size:
                        if sol.generation >= generation:
                            learner_to_stop.append((sol.learner, i))

                # progress.write('Stop learning: {}'.format(learner_to_stop))
                logger.info('Stop learning: {}'.format(learner_to_stop))

                for learner, _ in learner_to_stop:
                    max_idx = min(len(pop) - 1,
                                  int(args.pop_selection_ratio * args.population_size) - 1)
                    sol = pop[random.randint(0, max_idx)]
                    sol.n_try += 1
                    new_solutions.append(sol)

            # Mutation: 변형
            logger.info('Select next solution')
            if args.pop_mutation_method == PopMutation.NONE:
                pass

            elif args.pop_mutation_method == PopMutation.RANDOM:
                new_solutions = [sol.random_mutation()
                                 for sol in new_solutions]

            elif args.pop_mutation_method == PopMutation.BACKUP:
                new_solutions = [sol.prob_mutation() for sol in new_solutions]

            elif args.pop_mutation_method == PopMutation.PREDICTION:
                if len(pop) <= args.population_size:
                    new_solutions = [sol.random_mutation()
                                     for sol in new_solutions]
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
                    logger.line('loss/prediction error',
                                generation, pred_error)
                    new_solutions = [sol.predict_mutation(
                        x_scaler, reg) for sol in new_solutions]

            # 키보드 이벤트 검사 -> 여기서 수작업으로 파라미터 조작
            if sync_learner:
                if keyboard.event('esc'):
                    print('==== Enter debuge mode ====')
                    embed()

            # learner에 전송
            for (learner, _), sol in zip(learner_to_stop, new_solutions):
                mailbox.put(dict(sender=WID.Pop(), to=learner,
                                 msg=Msg.NEW_SOLUTION, solution=sol))

            # 성능이 나쁜 모델 제거
            pop = pop[:args.population_size]

            # 현재 population
            for i, sol in enumerate(pop):
                logger.info('{} -> {}'.format(i, sol))

            # 남은 frame 초기화
            frames = 0

            # 결과 출력
            score = np.mean(recent_scores)
            logger.info('Gen: {: >4}, Score: {: >10.3f}'.format(
                generation, score))

            for name in pop[0].hyperparams:
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
            logger.line('sys/fps', generation, generation_frames /
                        generation_interval, average=3, visdom_env='main')

            generation += 1
            generation_start_time = time.time()
            generation_n_games = 0
            generation_frames = 0

        # log 작업
        logger_inputs = mailbox.get_mailbox(WID.Logger())
        while logger_inputs.qsize() > 0 and frames <= frames_per_epoch:
            resp = logger_inputs.get()
            if resp['msg'] == Msg.SAVE_LOG:
                total_frames += resp['frames']
                generation_frames += resp['frames']
                if args.pop_log_share_method == PopLogShare.OneToOne:
                    frames += resp['frames']
                elif args.pop_log_share_method == PopLogShare.OneToAll:
                    frames += resp['frames'] // args.n_learners
                else:
                    raise NotImplementedError

                learner_name = resp['sender']
                scores = resp['scores']
                recent_scores += scores
                n_games += len(resp['scores'])
                generation_n_games += len(resp['scores'])

                current_time = int(time.time() - start_time)
                n_queued_episodes = np.mean(
                    [mb.qsize() for name, mb in mailbox.mailbox.items()
                     if name.startswith(WID.Learner())])
                msg = ['{}'.format(datetime.timedelta(seconds=current_time)),
                       '{}'.format(learner_name),
                       'G: {: >4,d}'.format(generation),
                       '{: >9,d} < {: >3.1f}'.format(
                           total_frames, n_queued_episodes),
                       'Remain: {: >6,d}, FPS: {: >4.1f}'.format(
                           frames_per_epoch - frames, total_frames / current_time),
                       'Score: {: >8.1f}, #eval: {: >3d}'.format(
                           np.mean(recent_scores), len(scores)),
                       'Loss: {: >6.4f}'.format(resp['total_loss'])]
                msg = ', '.join(msg)
                logger.info(msg)

                score_dict = dict(mean_score=np.mean(recent_scores))
                logger.line('score', generation, score_dict,
                            visdom_env='main', width=600)
                logger.line('losses', generation,
                            resp['loss'], visdom_env='main', width=600)
                logger.line('values', generation,
                            resp['values'], visdom_env='main')
                logger.bar('gradients', resp['gradients'],
                           visdom_env='main', sort=False)
                logger.bar('buffer_size',
                           resp['buffer_size'], visdom_env='main', sort=False)

            elif resp['msg'] == Msg.HEARTBEAT:
                # actor와 learner가 정상적으로 작동하고 있는지 모니터링 하는 용도
                heartbeat = resp['heartbeat']
                heartbeats_recv_time = time.time()
                heartbeat_recoder[heartbeat] = heartbeats_recv_time
                heartbeat_recoder_update = (heartbeat_recoder_update + 1) % HEARTBEAT_BUFFER_SIZE
                
                if heartbeat_recoder_update == 0:
                    heart_beat_eltime = dict()
                    for hb in heartbeat_recoder:
                        heart_beat_eltime[hb] = heartbeats_recv_time - heartbeat_recoder[hb]

                    logger.bar('heartbeats', heart_beat_eltime, visdom_env='main', sort=True)

            elif resp['msg'] == Msg.ERROR_INFO:
                error_logger.error(resp['error_info'] + '\n')


        iteration = (iteration + 1) % 5
        if iteration == 0:
            logger.progressbar(
                'Generations',
                ['Gen.', 'Frames'],
                [generation, frames],
                [args.max_epochs, frames_per_epoch], visdom_env='main',)

            logger.line('sys/q_size', generation,
                        mailbox.qsize_dict(), visdom_env='main')

        # PBT 학습 종료
        if generation >= args.max_epochs or n_steps >= args.max_steps:
            logger.info('==== DONE TRAIN: Gen. {}, Step: {}, Elapsed Time: {} ===='.format(
                generation, n_steps, datetime.timedelta(seconds=time.time() - start_time)))
            utils.log_tools.save_log_to_csv(args.log_path, args.session_id)

            # # 자식 프로세스 전부 강제 종료
            # pid = os.getpid()
            # utils.kill_children_processes(pid)
            break
