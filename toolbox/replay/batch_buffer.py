
import numpy as np


class batch_buffer(object):

    def __init__(self, batch_size, n_steps):
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.max_buffer_size = 2 * batch_size
        self.buffer = list()

    def __repr__(self):
        buff = list()
        buff += [f'META']
        buff += [f'- batch_size: {self.batch_size}, max buffer size {self.max_buffer_size}']
        buff += [f'- n_steps: {self.n_steps}']
        buff += [f'DATA START']
        for idx, (eid, episode) in enumerate(self.buffer):
            check_ready = lambda : len(episode) >= self.n_steps
            check_done = lambda : episode[-1]["done"] if len(episode) > 0 else 0
            buff += [f'- idx: {idx}, eid: {eid}, len: {len(episode)} ready: {check_ready()}, done: {check_done()}']
        buff += [f'DATA END']
        return '\n'.join(buff)


class StaticBatchBuffer(batch_buffer):

    def __init__(self, batch_size, n_steps):
        super().__init__(batch_size, n_steps)
        self.buffer = [[-1, []] for _ in range(batch_size)]

    def put(self, id_episode):
        min_idx = np.argmin([len(self.buffer[idx][1]) for idx in range(self.batch_size)])
        self.buffer[min_idx][0] = id_episode[0]
        self.buffer[min_idx][1] += id_episode[1][:]

    def ready(self):
        if min([len(self.buffer[idx][1]) for idx in range(self.batch_size)]) >= self.n_steps:
            return True
        else:
            return False

    def get(self):
        batch = list()
        if self.ready():
            for idx, (_, _) in enumerate(self.buffer):
                rollout =  self.buffer[idx][1][:self.n_steps]
                self.buffer[idx][1] = self.buffer[idx][1][self.n_steps:]
                batch.append(rollout)
        return batch


class DynamicBatchBuffer(batch_buffer):
    
    def __init__(self, batch_size, n_steps):
        super().__init__(batch_size, n_steps)

    def put(self, id_episode):
        inserted = False

        # eid가 같은 episode가 있으면 새로운 episode를 병합
        for idx, (eid, _) in enumerate(self.buffer):
            if eid == id_episode[0]:
                self.buffer[idx][1] += id_episode[1][:]
                inserted = True
                break

        # buffer에 저장된 episode개수가 배치 크기보다 작을 경우 무조건 추가
        if not inserted and len(self.buffer) < self.batch_size:
            self.buffer.append([id_episode[0], id_episode[1][:]])
            inserted = True

        # eid가 같은 episode가 없을 경우 가장 짧은 종료된 에피소드 뒤에 추가
        if not inserted:
            # 최대 에피소드 길이, 
            # dynamic batch buffer를 사용하는 이유는 policy lag을 가능한 줄이려는 것이기 때문에,
            # buffer 크기가 매우 작아야 하므로, 100000 보다는 작아야 함
            max_len = 100000

            ep_len = list()
            for eid, episode in self.buffer:
                # 종료된 에피소드가 있는 버퍼는 길이를 기록하고,
                # 종료되지 않은 에피소드가 있는 버퍼는 최대 값을 기록
                if episode[-1]['done']:
                    ep_len.append(len(episode))
                else:
                    ep_len.append(max_len)

            # 가장 짧은 에피소드 길이는 최대 에피소드 길이보다 짧아야 함
            if np.min(ep_len) < max_len:
                # 종료된 에피소드 중 가장 짧은 에피소드 뒤에 새로운 에피소드 추가
                min_idx = np.argmin(ep_len)
                self.buffer[min_idx][0] = id_episode[0]
                self.buffer[min_idx][1] += id_episode[1][:]
                inserted = True

        # 모두 실패 했으면, 새로 episode를 추가함; 버퍼의 너비가 배치보다 커짐
        if not inserted:
            self.buffer.append([id_episode[0], id_episode[1][:]])

        # 너무 오래된 episode 제거: 워커 사망?
        self.buffer = self.buffer[-self.max_buffer_size:]

    def ready(self):
        count = sum([1 for eid, episode in self.buffer if len(episode) >= self.n_steps])
        if count >= self.batch_size:
            return True
        else:
            return False

    def get(self):
        batch = list()
        count = sum([1 for eid, episode in self.buffer if len(episode) >= self.n_steps])

        if count >= self.batch_size:
            for idx, (_, episode) in enumerate(self.buffer):
                if len(episode) >= self.n_steps:
                    rollout =  self.buffer[idx][1][:self.n_steps]
                    self.buffer[idx][1] = self.buffer[idx][1][self.n_steps:]
                    batch.append(rollout)
                    if len(batch) >= self.batch_size:
                        break
        
        # 비어있는 episode 버퍼 제거
        self.buffer = [[eid, episode] for eid, episode in self.buffer if len(episode) > 0]
        
        return batch


if __name__ == '__main__':
    
    from IPython import embed

    batch_size = 4
    rollout_length = 4

    buff = DynamicBatchBuffer(batch_size, rollout_length)
    print(buff)

    episode0 = [0, [dict(done=False), dict(done=False), dict(done=False), dict(done=False)]]
    episode1 = [1, [dict(done=False), dict(done=False), dict(done=False), dict(done=False)]]
    episode2 = [2, [dict(done=False), dict(done=False), dict(done=False), dict(done=False)]]
    episode3 = [3, [dict(done=False), dict(done=False), dict(done=False), dict(done=False)]]
    episode4 = [4, [dict(done=False), dict(done=False), dict(done=False), dict(done=False)]]
    episode5 = [5, [dict(done=False), dict(done=False), dict(done=False), dict(done=True)]]
    episode6 = [6, [dict(done=False), dict(done=False), dict(done=False), dict(done=True)]]

    buff.put(episode0)
    buff.put(episode1)
    buff.put(episode2)
    buff.put(episode3)
    buff.put(episode4)
    buff.put(episode5)
    buff.put(episode6)
    buff.put([0,[dict(done=False), dict(done=False), dict(done=False), dict(done=True)]])
    embed()

    buff = StaticBatchBuffer(batch_size, rollout_length)
    buff.put(episode0)
    buff.put(episode1)
    buff.put(episode2)
    buff.put(episode3)
    buff.put(episode4)
    buff.put(episode5)
    buff.put(episode6)
    buff.put([0,[dict(done=False), dict(done=False), dict(done=False), dict(done=True)]])
    embed()