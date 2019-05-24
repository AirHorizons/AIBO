
import random
import shelve
import tempfile
from collections import deque, namedtuple
from enum import Enum
from pathlib import Path

import numpy as np

from ..utils import singleton


class ReplayManager(metaclass=singleton.Singleton):
    """
    - Replay buffer를 생성하고 관리하기 위한 용도로 사용하는 싱글톤 객체
    - 여러 Trainer가 동일한 replay buffer에 접근하기 위해 사용함
    """

    def __init__(self, path=None):
        self._memory = dict()
        self.path = path

    def deque(self, tag, capacity):
        if tag not in self._memory:
            self._memory[tag] = DequeReplayBuffer(capacity)
        return self._memory[tag]

    def array(self, tag, capacity):
        if tag not in self._memory:
            memory = ArrayReplayBuffer(capacity)
            if self.path:
                memory.path = Path(self.path) / Path(tag + '.mem')
            self._memory[tag] = memory
        return self._memory[tag]

    def __getattr__(self, name):
        return self.get(name)

    def get(self, name):
        assert name in self._memory
        return self._memory.get(name, None)


class DequeReplayBuffer(deque):
    """
    간단하게 실험해 볼 때 사용하는 덱기반 replay buffer
    """
    def __init__(self, capacity):
        super().__init__(maxlen=capacity)

    @property
    def size(self):
        return len(self)

    @property
    def capacity(self):
        return self.maxlen

    def put(self, data):
        self.append(data)

    def sample(self, n_samples):
        return random.sample(self, n_samples)


class ArrayReplayBuffer(object):
    def __init__(self, capacity):
        self.path = None
        self._memory = None
        self.capacity = capacity
        self._indexes = list()

        self._current_idx = 0
        self.size = 0
        self._memory_assigned = False

    def _assign_memory(self, data):
        self.dtypes = [d.dtype for d in data]
        self.shapes = [d.shape for d in data]
        self.n_dims = sum([np.product(s) for s in self.shapes])  # 전체 float 개수

        # tuple의 개별 요소들을 인코딩하고 디코딩 하기 쉽게 하기 위해 
        # 개별 요소들이 시작되는 부분과 끝 부분의 인덱스를 저장
        self._indexes = list()
        start_idx = 0
        for shape in self.shapes:
            end_idx = start_idx + np.product(shape)
            self._indexes.append((start_idx, end_idx))
            start_idx = end_idx

        if self.path:
            # memory file 생성
            mode = 'r+' if Path(self.path).exists() else 'w+'
            self._memory = np.memmap(self.path, dtype=np.float32, mode=mode, 
                shape=(self.capacity, self.n_dims))
            # meta data 저장소 생성
            with shelve.open(str(self.path) + '.db') as db:
                self._current_idx = db.get('_current_idx', 0)
                self.size = db.get('size', 0)
        else:
            self._memory = np.zeros(dtype=np.float32, shape=(self.capacity, self.n_dims))

        self._memory_assigned = True

    def _encode(self, _data):
        encoded_data = [d.reshape(-1) for d in _data]
        return np.concatenate(encoded_data)

    def _decode(self, _data):
        buff = list()
        for (start_idx, end_idx), dtype, shape in zip(self._indexes, self.dtypes, self.shapes):
            decoded = _data[start_idx: end_idx]
            decoded = decoded.astype(dtype)
            decoded = decoded.reshape(shape)
            decoded = np.array(decoded)
            buff.append(decoded)
        return buff

    def put(self, data):
        # 데이터 샘플 하나씩 저장
        if not self._memory_assigned:
            self._assign_memory(data)
            
        self._memory[self._current_idx] = self._encode(data)  # 현재 위치에 데이터 기록
        self._current_idx = (self._current_idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def batch_put(self, data_list):
        # 여러 데이터 샘플을 한번에 저장
        for data in data_list:
            self.put(data)

        # 파일에 meta data를 업데이트
        if self.path:
            with shelve.open(self.path + '-db') as db:
                db['_current_idx'] = self._current_idx
                db['size'] = self.size

    def get(self, idx):
        # index로 하나씩 샘플에 접근
        return self._decode(self._memory[idx])

    def sample(self, n_samples):
        # 여러 데이터 샘플을 샘플링
        assert n_samples < self.size
        choices = np.random.randint(0, self.size - 1, n_samples)
        buff = list()
        for choice in choices:
            buff.append(self.get(choice))
        
        # TODO: 같은 종류의 요소끼리 묶여서(axis=0로) 반환되도록 구현 필요
        return buff

    def __str__(self):
        buff = [
            f'path: {self.path}',
            f'size: {self.size} / {self.capacity}',
            self._memory.__str__()
        ]
        return '\n'.join(buff)


if __name__ == '__main__':
    
    import tempfile
    from IPython import embed

    zero_memory = ReplayManager().deque('zero', 100)
    nonzero_memory = ReplayManager().deque('nonzero', 100)
    ReplayManager().path = '.'
    mem_memory = ReplayManager().array('mem_test', 100)

    for i in range(10):
        data = [
            np.random.random((36, 36)).astype(np.float32), 
            np.random.random(10).astype(np.float32), 
            np.random.randint(0, 10, 1)]
        mem_memory.put(data)

        out = mem_memory.get(i)
        for i in range(3):
            assert (out[i] == data[i]).all()

    embed()
