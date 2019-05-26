
예제 7. QBot
=============

이 예제는 주력부대를 이동시킬 지점을 규칙을 이용해 결정하는 대신,
강화학습으로 학습된 신경망으로 결정하는 예를 보여준다.

**학습 스크립트 실행**

QBot 예제는 강화학습 알고리즘 중 하나인 Q-Learning을 사용한 예제이다.

.. code-block:: bash

   # 학습 실행
   (sc2) ~/sc2minigame $ python -m bots.nc_example_v7.train -vv


다른 콘솔에서 tensorboard를 실행하면 웹브라우저에서 학습 결과를 확인할 수 있다.

.. code-block:: bash

   # tensorboard 실행
   (sc2) ~/sc2minigame $ tensorboard --logdir=../outs_sc2mini

웹브라우저(http://localhost:6006)로 접속하면 현재 학습 상태를 볼 수 있다(:ref:`ref_qbot_tensorboard`).

.. _ref_qbot_tensorboard:
.. figure:: figs/qbot_tensorboard.png
    :figwidth: 600

    Tensorboard 실행화면

학습 결과는 ../{args.out_path}/{args.session_id}에 저장된다.
args.out_path의 기본값은 ../outs_sc2mini이고, args.session_id는 학습이 시작한 시간이 사용된다.

../{args.out_path}/{argssession_id} 폴더에는 게임 로그 뿐아니라,
학습한 신경망 모델, 학습 당시의 코드 등 모든 결과물에 저장된다.

이 예제 코드는 기본으로 주어진 예제 AI 중 가장 강력한 AI인 DropBot을 상대로
강화학습을 한다. 학습 초기에는 거의 10% ~ 20% 수준의 승률을 보이지만,
약 500게임(15시간)을 플레이한 뒤에는 80% 이상의 승률에 도달할 수 있다(:ref:`ref_qbot_eval_game`).

.. _ref_qbot_eval_game:
.. figure:: figs/qbot_win_ratio.png
    :figwidth: 600

    평균 승률 변화

처음 100게임을 플레이하는 동안은 학습을 하지 않고, 랜덤하게 게임을 플레이하며 초기 학습 데이터를 수집했고,
100게임 이후부터 한 게임이 종료할 때마다, 64개 데이터를 샘플링 하여 32번 학습했다.
학습 초기 :math:`\varepsilon` 값은 1.0이고, 32번 학습(한 게임 종료)할 때마다,
최소 :math:`\varepsilon` 값 0.05까지 0.01씩 감소시켰다.
학습 게임에서는 매번 액션을 결정할 때마다, eplsilon의 확률로 랜덤하게 액션을 결정했고,
평가 게임에서는 언제나 최선의 액션을 결정햇다.
학습게임 9게임마다 한번 평가게임을 실행했다.
