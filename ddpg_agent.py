import numpy as np
import random
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.optim as optim
import copy
import torch.nn.functional as F

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, random_seed,
                 buffer_size=int(1e6), batch_size=128, gamma=0.99, tau=1e-3,
                 lr_actor=1e-4, lr_critic=3e-4, critic_weight_decay=0.0001,
                 update_every=14, update_num_repeats=10,
                 noise_decay=0.99):
        """Base class for agents

        Args:
            state_size (flat): the number of states
            action_size (float): the number of actions
            random_seed (int): Description
            buffer_size (int, optional): the maximum number of elements in the replay buffer
            batch_size (int, optional): mini batch size
            gamma (float, optional): reward discount rate
            tau (int, optional): target network update rate. Use 1 if the target network is updated entirely from the local network at once
            lr_actor (float, optional): actor network learning rate on the Adam optimizer
            lr_critic (float, optional): critic network learning rate on the Adam optimizer
            critic_weight_decay (float, optional): critic network L2 weight decay
            update_every (int, optional): how often we learn the learning step (i.e. 4 means the learning step is executed ever 4 action taken)
            update_num_repeats (int, optional): how many times to sample the replay buffer and update networks each learning step
            noise_decay (float, optional): how much we decay the OU noise each step
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=critic_weight_decay)

        self.soft_update(self.actor_local, self.actor_target, tau=1.0)
        self.soft_update(self.critic_local, self.critic_target, tau=1.0)

        # Noise process
        self.noise = OUNoise(1, action_size, random_seed, decay=noise_decay)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, random_seed)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.update_num_repeats = update_num_repeats

        self.update_t = 0

    def reset(self):
        self.noise.reset()
        self.update_t = 0

    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy.

        Params
        ======
        Args:
            state (array): current state
            add_noise (bool, optional): True if we like to add OU noise for exploration

        Returns:
            TYPE: Description
        """

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        """Add an episode to the memory and run a learning step after sampling from the replay buffer

        Args:
            state (list): current state
            action (int): the action taken according to the policy given the state
            reward (float): the reward from the environment
            next_state (list): the next state returned from the environment given the action
            done (float): flag returned from the environment to indicate if the current episode is done
            beta (float): beta value for important sampling weight
        """

        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        self.update_t = (self.update_t + 1) % self.update_every
        if self.update_t == 0:
            for repeat in range(self.update_num_repeats):
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update Critic
        next_actions = self.actor_target(next_states)
        next_states_target_value = self.critic_target(next_states, next_actions)
        target_value = rewards + self.gamma * (next_states_target_value * (1. - dones))
        estimated_value = self.critic_local(states, actions)

        critic_loss = F.mse_loss(target_value, estimated_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Update actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, num_agents, size, seed, mu=0., theta=0.15, sigma=0.2, decay=0.99):
        """Initialize parameters and noise process."""
        self.num_agents = num_agents
        self.size = size
        self.mu = mu * np.ones((num_agents, size))
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.decay = decay
        self.epsilon = 1.

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.epsilon *= self.decay

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.uniform(-1, 1, (self.num_agents, self.size))
        self.state = x + dx
        return self.state * self.epsilon


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
