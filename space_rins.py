from dueling_dqn import Agent
from dueling_dqn import DuelingLinearDeepQNetwork
import matplotlib.pyplot as plt
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.base_env import ActionTuple
import numpy as np
import traceback
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import datetime
import torch as T


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

'''if __name__ == "__main__":
    channel = EngineConfigurationChannel()
    env = UE(file_name='space_rings_env', seed=1, worker_id=3, side_channels=[channel], no_graphics=False)
    channel.set_configuration_parameters(time_scale = 12.0, quality_level=0)

    env.reset()
    try:
        num_games = 5000
        load_checkpoint = True
        agent = Agent(gamma=0.5, epsilon=0.05, alpha=1e-5,
                    input_dims=[11], n_actions=9, mem_size=100000, eps_min=0.01,
                    batch_size=128, eps_dec=1e-6, replace=100, chkpt_dir='models2')
        if load_checkpoint:
            agent.load_models()
        filename = 'SpaceRings-Dueling-NewShapedReward' + str(datetime.datetime.now()) 
        scores = []
        eps_history = []
        n_steps = 0
        avg_score = 0

    
        behavior_name = list(env.behavior_specs.keys())[0]
        spec = env.behavior_specs[behavior_name]

        print("Number of observations: ", len(spec.observation_specs[0].shape))
        action_spec = spec.action_spec

        #if spec.action_spec.is_discrete():
        #    print("Actions are discrete")
        #decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        for episode in range(num_games):
            env.reset()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            tracked_agent = -1 # -1 indicates not yet tracking
            done = False # For the tracked_agent
            episode_rewards = 0 # For the tracked_agent
            while not done:
                reward = 0
                # Track the first agent we see if not tracking
                # Note : len(decision_steps) = [number of agents that requested a decision]
                if tracked_agent == -1 and len(decision_steps) >= 1:
                    tracked_agent = decision_steps.agent_id[0]
                # Generate an action for all agents
                observation_before = decision_steps[tracked_agent].obs
                observation_next = []
                action = agent.choose_action(observation_before[0])
                # Set the actions
                env.set_actions(behavior_name, ActionTuple(discrete=np.array([[action]], dtype=np.int32)))
                # Move the simulation forward
                env.step()
                # Get the new simulation results
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                if tracked_agent in decision_steps.agent_id: # The agent requested a decision
                    reward = decision_steps[tracked_agent].reward
                    observation_next = decision_steps[tracked_agent].obs
                    #print("reward added ", decision_steps[tracked_agent].reward)
                if tracked_agent in terminal_steps.agent_id: # The agent terminated its episode
                    reward = terminal_steps[tracked_agent].reward 
                    observation_next = terminal_steps[tracked_agent].obs
                    done = True
                agent.store_transition(observation_before[0], action,
                                    reward, observation_next[0], int(done))
                agent.learn()
                episode_rewards += reward
            #print(f"Total rewards for episode {episode} is {episode_rewards}")
            scores.append(episode_rewards)
            avg_score = np.mean(scores[max(0, episode-100):(episode+1)])
            print('episode: ', episode,'score %.1f ' % episode_rewards,
                ' average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
            if episode > 0 and episode % 1000 == 0:
                agent.save_models()

            #if episode % 1000 == 0:
                #agent.reset_epsilon()

            eps_history.append(agent.epsilon)
        
        x = [i+1 for i in range(num_games)]
        plotLearning(x, scores, eps_history, filename + "avg" + str(avg_score) + '.png')
        agent.save_onnx()
        env.close() 
        
    except Exception as e:
        try:
            traceback.print_exc()
            x = [i+1 for i in range(len(scores))]
            plotLearning(x, scores, eps_history, filename + "avg" + str(avg_score) + '.png')
            env.close()
        except:
            traceback.print_exc()     
            env.close() '''
    

def testAgent(model,env, num_games = 100):
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    scores = []
    behavior_name = list(env.behavior_specs.keys())[0]
    for episode in range(num_games):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        tracked_agent = -1 # -1 indicates not yet tracking
        done = False # For the tracked_agent
        episode_rewards = 0 # For the tracked_agent
        while not done:
            reward = 0
            # Track the first agent we see if not tracking
            # Note : len(decision_steps) = [number of agents that requested a decision]
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]
            # Generate an action for all agents
            observation = decision_steps[tracked_agent].obs
            observation = observation[0]
            observation = observation[np.newaxis,:]
            state = T.tensor(observation).to(device)
            _, advantage = model.forward(state)
            action = T.argmax(advantage).item()
            # Set the actions
            env.set_actions(behavior_name, ActionTuple(discrete=np.array([[action]], dtype=np.int32)))
            # Move the simulation forward
            env.step()
            # Get the new simulation results
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if tracked_agent in decision_steps.agent_id: # The agent requested a decision
                reward = decision_steps[tracked_agent].reward
                #print("reward added ", decision_steps[tracked_agent].reward)
            if tracked_agent in terminal_steps.agent_id: # The agent terminated its episode
                reward = terminal_steps[tracked_agent].reward 
                done = True
            episode_rewards += reward
        #print(f"Total rewards for episode {episode} is {episode_rewards}")
        scores.append(episode_rewards)
        avg_score = np.mean(scores[0:(episode+1)])
        print('episode: ', episode,'score %.1f ' % episode_rewards,
            ' average score %.1f' % avg_score)
try:        
    channel = EngineConfigurationChannel()
    env = UE(file_name='space_rings_env', seed=1, worker_id=3, side_channels=[channel], no_graphics=False)
    channel.set_configuration_parameters(time_scale = 12.0, quality_level=0)

    env.reset()
    num_games = 100
    model = DuelingLinearDeepQNetwork(0, 9, input_dims=[11],
                                    name='q_eval', chkpt_dir='models2/backup/test_test')
    model.load_checkpoint()
    testAgent(model, env)
    env.close()
except Exception as e:
    traceback.print_exc()  
    env.close()
    
