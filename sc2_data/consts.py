
# 학습 성능과 관련이 적은 파라미터 저장(실험할 때마다 수정할 필요가 없음)
# 실험할 때마다 수정할 필요가 많은 파라미터는 initializer.py에서 관리


class SC2:
    fast_reload = True
    ws_timeout = 2 * 60  # ws 최대 대기 시간 (default: 120)
    step_time_limit = 2 * 60  # 최대 step 시간 2분
    game_time_limit = 20 * 60  # 최대 게임시간 15분
    game_step_limit = 5000  # 최대 게임 스텝, 초과하면 무승부로 게임 종료
    # max_game_limit = 10  # 하나의 서버로 재시작 않하고 하는 최대 게임 수

    feature_layer_width = 24  # 뭔지 알 수 없군;; pysc2에서 복사함
    feature_layer_resolution = (84, 84)  # feature layer 해상도
    feature_layer_minimap_resolution = (128, 128)  # minimap 해상도


class Learner:
    server_ip = "172.20.86.170"
    server_port = 8000
    frontend_port = 5559
    backend_port = 5560


class Actor:
    # 원격 actor 한 대에서 실행 가능한 actor의 최대 개수, eg. 99, 999, 9999, ...
    max_remote_actors_per_host = 999 
    


class Msg:
    PING = 0
    GAME_LOG = 1
    ERROR_LOG = 2