from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
from collections import deque
import matplotlib.pyplot as plt
import sys
import torch


def Train(env, agent, num_agents, num_episodes=200):
    """Summary

    Args:
        env (Unity ML environment): environment
        agent (DQN agent): the learning agent
        num_episodes (int, optional): the number of episodes to run
   """
    score_window_len = 100
    scores = []
    scores_window = deque(maxlen=score_window_len)
    avg_score_to_solve = 0.5

    print("Total number of episodes {}".format(num_episodes))
    for episode in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        states = env_info.vector_observations                 # get the current state
        score_array = np.zeros(num_agents)                    # initialize the scre
        agent.reset()
        num_steps = 0
        while True:
            actions = agent.act(states, True)
            env_info = env.step(actions)[brain_name]          # get the transition from the environment with the chosen action
            next_states = env_info.vector_observations       # get the next state
            rewards = env_info.rewards                       # get the reward
            score_array += rewards
            dones = env_info.local_done                      # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            num_steps += 1
            if dones[0]:                                     # exit loop if episode finished
                break
        score = np.amax(score_array)
        scores_window.append(score)
        scores.append(score)
        if episode >= score_window_len:
            average_over_solving_window = np.mean(scores_window)
            print("Score: {} at episode {} and average score over last 100 episodes is {}".format(score, episode, average_over_solving_window))
            if average_over_solving_window > avg_score_to_solve:
                print("Solved at episode {}".format(episode))
                break
        else:
            print("Score: {} at episode {} total steps {}".format(score, episode, num_steps))

    env.close()
    plot(scores, "rewards.png")
    torch.save(agent.actor_local.state_dict(), 'model_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'model_critic.pth')


def Test(env, agent, num_agents, num_episodes=200):
    """Summary

    Args:
        env (Unity ML environment): environment
        agent (DQN agent): the learning agent
        num_episodes (int, optional): the number of episodes to run
   """
    score_window_len = 100
    scores = []
    scores_window = deque(maxlen=score_window_len)

    agent.actor_local.load_state_dict(torch.load('model_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('model_critic.pth'))

    print("Total number of episodes {}".format(num_episodes))
    for episode in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
        states = env_info.vector_observations                 # get the current state
        score_array = np.zeros(num_agents)                    # initialize the scre
        agent.reset()
        num_steps = 0
        while True:
            actions = agent.act(states, False)
            env_info = env.step(actions)[brain_name]          # get the transition from the environment with the chosen action
            next_states = env_info.vector_observations       # get the next state
            rewards = env_info.rewards                       # get the reward
            score_array += rewards
            dones = env_info.local_done                      # see if episode has finished
            states = next_states
            num_steps += 1
            if dones[0]:                                     # exit loop if episode finished
                break
        score = np.amax(score_array)
        scores_window.append(score)
        scores.append(score)
        if episode >= score_window_len:
            average_over_solving_window = np.mean(scores_window)
            print("Score: {} at episode {} and average score over last 100 episodes is {}".format(score, episode, average_over_solving_window))
            if average_over_solving_window > avg_score_to_solve:
                print("Solved at episode {}".format(episode))
                break
        else:
            print("Score: {} at episode {} total steps {}".format(score, episode, num_steps))

    env.close()
    plot(scores, "rewards_test.png")


def plot(scores, file_name):
    xlable = np.arange(len(scores), dtype=int)
    plt.plot(xlable, scores)
    plt.ylabel('total rewards')
    plt.xlabel('episode')
    plt.savefig(file_name)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        env = None
        try:
            env = UnityEnvironment(file_name=sys.argv[2])
            isForTrain = sys.argv[1] is "train"

            if isForTrain is True:
                print("The mode is training...")
            else:
                print("The mode is testing...")

            brain_name = env.brain_names[0]
            brain = env.brains[brain_name]
            action_size = brain.vector_action_space_size
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            state_size = len(state)
            num_agents = len(env_info.vector_observations)

            print("num agents {} state size {} action size {}".format(num_agents, state_size, action_size))

            agent = Agent(state_size, action_size, random_seed=0,
                          buffer_size=int(1e5),
                          batch_size=512,
                          tau=1e-3,
                          lr_actor=3e-4, lr_critic=3e-4,
                          critic_weight_decay=0.0,
                          update_every=4, update_num_repeats=2,
                          noise_decay=0.999)

            if isForTrain is True:
                print("start training....")
                Train(env, agent, num_agents, num_episodes=3000)
            else:
                Test(env, agent, num_agents, num_episodes=200)

        except KeyboardInterrupt:
            print("Keyboard interrupted")
            env.close()
    else:
        print("usage - python training.py [train|test] path to the agent executable")
