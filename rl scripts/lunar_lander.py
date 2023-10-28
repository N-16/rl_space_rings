import gym, time
import numpy as np
import matplotlib.pyplot as plt
from dueling_dqn import Agent

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig =plt.figure()
    ax =fig.add_subplot(111, label="1")
    ax2 =fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_games = 1000
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, alpha=5e-4,
                  input_dims=[8], n_actions=4, mem_size=100000, eps_min=0.01,
                  batch_size=64, eps_dec=1e-3, replace=100)

    if load_checkpoint:
        agent.load_models()

    filename = 'LunarLander-Dueling-128-128-Adam-lr0005-replace100.png'
    scores = []
    eps_history = []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation, _ = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncuated, info = env.step(action)
            done = terminated or truncuated
            n_steps += 1
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_


        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score %.1f ' % score,
             ' average score %.1f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
        #if i > 0 and i % 10 == 0:
        #    agent.save_models()

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, filename)

