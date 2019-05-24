
빠른시작
==========

소개
------

이 플랫폼은 2019년 NC Fellowship Game AI 경진대회 목적으로 구현되었다.
이 플랫폼은 Blizzard의 StarCraft 2, s2client-api [#]_ 그리고 Hannes Karppila의
python-sc2 [#]_ 를 이용해 간단한 실시간 전략 시뮬레이션(Real-Time Strategy) 환경을 제공한다.

StarCraft 2같은 RTS 게임은 건물 건설, 유닛 생산, 업그레이드, 전투 등
다양한 작업을 수행해야 하기 때문에, AI를 구현하려면 매우 많은 작업이 필요하다.
그래서 문제의 규모를 줄이기 위해 자원 수집, 건물 건설, 유닛 생산 같은 작업을 고려할 필요 없는,
전투에 집중한 새로운 경진대회 환경을 구축했다.

.. note::

   현재 사용하고 있는 게임 규칙이나 유닛 속성은 경진대회 운영에 문제가 될 경우 일부 변경될 수 있음

이 플랫폼은 python-sc2를 기반으로 구현되었기 때문에, python-sc2의 설치방법,
사용법을 거의 그대로 공유한다. 이 문서에서 부족한 부분은, python-sc2 혹은
그 기반이 되는 s2client-api를 참조할 수 있다.


설치
-----

이 플랫폼은 Winodws 10과 Ubuntu 16.04 환경에서 개발 및 테스트 했다.
python-sc2가 macOS 환경도 지원하기 때문에 macOS에서도 문제가 없을 것으로 추측되지만,
테스트 해보지는 않았다.

플랫폼을 설치하는 과정은 세 단계로 구성된다.

1. StarCraft 2 설치
2. python-sc2 설치
3. sc2minigame 플랫폼 설치

**StarCraft 2 설치**

Windows 환경에서는 일반적인 방식으로 Battle.net을 통해 StarCraft 2를 설치한다.
python-sc2에서 StarCraft 2 기본 설치 경로(C:\Program Files (x86)\StarCraft II)에서
실행파일을 찾기 때문에, 가급적이면 설치 경로를 바꾸지 않는 것이 좋다. 설치 경로를 바꿔야 할 필요가 있다면,
환경변수 SC2PATH를 변경해야 한다 (python-sc2 문서를 참조).

Linux 환경에서는 Linux용 바이너리를 다운받아서 ~/StarCraftII에 압축을 해제하는
것으로 간단하게 설치 가능하다. 경진대회는 Windows용 바이너리를 기준으로 진행되지만,
Linux용 바이너리는 화면을 렌더링하지 않아서(Headless), 기계학습에 더 용이하다.
Linux용은 Windows용 바이너리보다 약간 버전이 낮은 경향이 있다.

- 공식 저장소: https://github.com/Blizzard/s2client-proto#downloads

.. code-block:: bash

   $ # StarCraft 2 설치, 19-04-19 최신 버전인 4.7.1 기준
   $ cd ~
   $ wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.7.1.zip  # 다운
   $ unzip ~/SC2.4.7.1.zip -d StarCraftII  # 압축해제 암호: iagreetotheeula
   $ chmod +x ~/StarCraftII/bin/Versions/Base*/SC2_x64  # 실행권한 부여

**python-sc2 설치**

여기서는 python anaconda 배포판을 사용하는 것을 기준으로 설명한다.
Windows와 Linux 모두 동일한 절차를 진행한다.

1. anaconda를 다운받아서 설치(python 3.7과 2.7 모두 사용 가능)

  https://www.anaconda.com/distribution/

2. python-sc2를 설치

*StarCraft II 4.8.5 이후 버전*

.. code-block:: bash

   $ # 가상환경 생성
   $ conda create -n sc2 python=3.6 -y  # python 3.6 환경 생성
   $ # 가상환경 활성화
   $ conda activate sc2
   $ # 주요 모듈 설치
   $ conda install -y ipython numpy=1.15.4 scipy scikit-image matplotlib psutil tqdm tensorflow
   $ conda install -y pytorch-cpu torchvision-cpu -c pytorch  # pytorch cpu 버전
   $ conda install -y tensorboardx -c conda-forge
   $ pip install visdom
   $ # python-sc2 개발판 설치
   $ # - Windows용 StarCraft II가 4.8.5이후 버전에서 python-sc2 정식 배포판이 작동하지 않음
   $ # - 정식 배포판 대신 개발용 버전을 설치해야 정상적으로 작동함
   $ # - 참고: https://github.com/Dentosal/python-sc2/issues/266
   $ pip install pipenv
   $ pip install --upgrade --force-reinstall https://github.com/Dentosal/python-sc2/archive/develop.zip

*StarCraft II 4.8.5 이전버전*

.. code-block:: bash

   $ # 가상환경 생성
   $ conda create -n sc2 python=3.6 -y  # python 3.6 환경 생성
   $ # 가상환경 활성화
   $ conda activate sc2
   $ # 주요 모듈 설치
   $ conda install -y ipython numpy=1.15.4 scipy scikit-image matplotlib psutil tqdm tensorflow
   $ conda install -y pytorch-cpu torchvision-cpu -c pytorch  # pytorch cpu 버전
   $ conda install -y tensorboardx -c conda-forge
   $ pip install visdom
   $ # python-sc2 설치
   $ pip install sc2

3. sc2minigame 설치

설치를 원하는 경로에 sc2minigame 압축해제한다.

게임 실행
---------

**예제 AI vs. StarCraft 기본 AI**

구현한 AI와 기본 컴퓨터 AI끼리 플레이를 할 때는 다음 명령을 입력한다.

.. code-block:: bash

   (sc2) ~/sc2minigame $ python run_sc2minigame.py \
                         --bot1=bots.nc_example_v6.drop_bot.DropBot \
                         --realtime=True \
                         --debug=True

--bot1 옵션은 1번 플레이어 클래스를 지정하는 옵션이고
--bot2에 기본 플레이어 옵션으로 기본 AI가 지정되어 있다.

bots.nc_example_v6.drop_bot.DropBot AI는
./bots/nc_example_v6/drop_bot.py 파일에 있다.

--realtime 옵션이 True 일때는 게임이 실시간으로 실행되고
False 일때는 최대한 빠르게 가속되어 실행된다.

--debug 옵션은 AI를 초기화할 때 사용되며 DropBot은 debug 옵션이 True일때,
게임 화면에 디버그 정보를 출력한다.

게임이 성공적으로 실행되면, 플랫폼 설치가 완료된 것이다.

**예제 AI vs. 예제 AI**

다른 두 예제 AI끼리 게임을 하려면 다음 처럼 --bot1과 --bot2 옵션으로
게임을 하려는 AI를 지정하면 된다.

python-sc2를 이용해 구현한 AI는 게임 에서는 인간 플레이어로 취급되므로,
기본 AI로 플레이 할때와 달리 게임이 두 개가 실행된다.
게임 하나는 서버가 되고, 하나는 클라이언트가 되어 멀티 플레이로 게임이 실행된다.

.. code-block:: bash

   (sc2) ~/sc2minigame $ python run_sc2minigame.py \
                         --bot1=bots.nc_example_v6.DropBot \
                         --bot2=bots.nc_example_v5.ReaperBot \
                         --realtime=False

**인간 vs. 예제AI**

python-sc2로 구현한 AI는 게임 중에 사람의 입력을 그대로 받을 수 있다.
따라서, run_sc2minigame.py에서는 아무 행동도 하지 않는 AI인 DummyBot을 실행해서
AI와 게임을 플레이 할 수 있도록 했다.

.. code-block:: bash

   (sc2) ~/sc2minigame $ python run_sc2minigame.py \
                         --bot1=bots.nc_example_v0.DummyBot \
                         --bot2=bots.nc_example_v6.DropBot \
                         --realtime=True

python-sc2에는 인간 플레이어를 직접 지정하는 할 수 있으므로 그 기능을 사용해도 무방하다.

.. [#] https://github.com/Blizzard/s2client-api
.. [#] https://github.com/Dentosal/python-sc2
