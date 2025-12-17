import random
import torch


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, s_next, done):
        data = (s, a, r, s_next, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.cat(s),
            torch.cat(a),
            torch.cat(r),
            torch.cat(s_next),
            torch.cat(done),
        )

    def __len__(self):
        return len(self.buffer)
