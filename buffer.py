import numpy as np
import torch
import warnings


class ReplayBuffer:
    def __init__(self, state_shape, action_shape, max_num_seq=int(2**12), seq_len=30,
                 batch_size=32, device='cpu', obs_uint8=False):
        super(ReplayBuffer, self).__init__()

        self.seq_len = seq_len
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.batch_size = batch_size
        self.obs_uint8 = obs_uint8
        
        self.device = device

        self.state_dtype = torch.float32 if not obs_uint8 else torch.uint8
        
        self.length = torch.zeros((max_num_seq,), dtype=torch.int, device=self.device)
        self.masks = torch.zeros((max_num_seq, seq_len,), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((max_num_seq, seq_len, *action_shape), dtype=torch.float32, device=self.device)
        self.states = torch.zeros((max_num_seq, seq_len + 1, *state_shape), dtype=self.state_dtype, device=self.device)
        self.rewards = torch.zeros((max_num_seq, seq_len,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((max_num_seq, seq_len), dtype=torch.float32, device=self.device)
        self.tail, self.size = 0, 0
        self.max_num_seq = max_num_seq

        self.sum_steps = 0
        self.min_length = 0
        self.max_length = 0

        # Allocate GPU space for sampling batch data
        self.length_b = torch.zeros((batch_size,), dtype=torch.int, device=self.device)
        self.masks_b = torch.zeros((batch_size, seq_len,), dtype=bool, device=self.device)
        self.actions_b = torch.zeros((batch_size, seq_len, *self.action_shape), dtype=torch.float32, device=self.device)
        self.states_b = torch.zeros((batch_size, seq_len + 1, *self.state_shape), dtype=torch.float32, device=self.device)
        self.rewards_b = torch.zeros((batch_size, seq_len,), dtype=torch.float32, device=self.device)
        self.dones_b = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=self.device)

    def append_episode(self, states, actions, r, done, length):

        if length < 1:
            warnings.warn("Episode length < 1, not recorded!")
            
        # shape: trajectory_length x data shape
        
        if self.obs_uint8 and (states.dtype != np.uint8):
            states = np.clip(states, 0, 1)
            states = (255 * states).astype(np.uint8)

        if self.length[self.tail] != 0:
            self.sum_steps -= self.length[self.tail]
        self.length[self.tail] = length
        self.sum_steps += length
        self.min_length = min(self.min_length, length)
        self.max_length = max(self.max_length, length)

        self.length[self.tail] = length
        self.masks[self.tail][:length] = 1
        self.masks[self.tail][length:] = 0
        self.dones[self.tail][:length] = torch.from_numpy(done[:length]).to(device=self.device)
        self.dones[self.tail][length:] = 1
        self.states[self.tail][:length + 1] = torch.from_numpy(states[:length + 1]).to(device=self.device)
        self.states[self.tail][length + 1:] = 0
        self.actions[self.tail][:length] = torch.from_numpy(actions[:length]).to(device=self.device)
        self.actions[self.tail][length:] = 0
        self.rewards[self.tail][:length] = torch.from_numpy(r[:length]).to(device=self.device)
        self.rewards[self.tail][length:] = 0

        self.tail = (self.tail + 1) % self.max_num_seq
        self.size = min(self.size + 1, self.max_num_seq)

    def sample_batch(self):
        sampled_episodes = torch.from_numpy(np.random.choice(self.size, [self.batch_size])).to(torch.int64)

        self.masks_b.fill_(0)
        self.actions_b.fill_(0)
        self.states_b.fill_(0)
        self.rewards_b.fill_(0)
        self.dones_b.fill_(0)

        self.length_b[:] = self.length[sampled_episodes]
        self.actions_b[:] = self.actions[sampled_episodes]
        self.rewards_b[:] = self.rewards[sampled_episodes]
        self.dones_b[:] = self.dones[sampled_episodes]
        self.masks_b[:] = self.masks[sampled_episodes]

        if self.obs_uint8:
            self.states_b[:] = (self.states[sampled_episodes].to(torch.float32)) / 255
        else:
            self.states_b[:] = self.states[sampled_episodes]

        return self.states_b, self.actions_b, self.rewards_b, self.dones_b, self.masks_b, self.length_b
    
    def sample_latest_experience(self):
        sampled_episodes = np.arange(self.tail - self.batch_size, self.tail)

        self.masks_b.fill_(0)
        self.actions_b.fill_(0)
        self.states_b.fill_(0)
        self.rewards_b.fill_(0)
        self.dones_b.fill_(0)

        self.length_b[:] = self.length[sampled_episodes]
        self.actions_b[:] = self.actions[sampled_episodes]
        self.rewards_b[:] = self.rewards[sampled_episodes]
        self.dones_b[:] = self.dones[sampled_episodes]
        self.masks_b[:] = self.masks[sampled_episodes]

        if self.obs_uint8:
            self.states_b[:] = (self.states[sampled_episodes].to(torch.float32)) / 255
        else:
            self.states_b[:] = self.states[sampled_episodes]

        return self.states_b, self.actions_b, self.rewards_b, self.dones_b, self.masks_b, self.length_b