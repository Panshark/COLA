#!/usr/bin/env python
# n-step Asynchronous Advantage Actor-Critic Agent (A3C) | Praveen Palanisamy
# Chapter 8, Hands-on Intelligent Agents with OpenAI Gym, 2018

from argparse import ArgumentParser
from datetime import datetime
import time
from collections import namedtuple
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp
import torch.nn.functional as F
#from environment.utils import make_env
import gym
try:
    import roboschool
except ImportError:
    pass
from tensorboardX import SummaryWriter
from utils.params_manager import ParamsManager
from function_approximator.shallow import Actor as ShallowActor
from function_approximator.shallow import DiscreteActor as ShallowDiscreteActor
from function_approximator.shallow import Critic as ShallowCritic
from function_approximator.deep import Actor as DeepActor
from function_approximator.deep import DiscreteActor as DeepDiscreteActor
from function_approximator.deep import Critic as DeepCritic
from function_approximator.deep import PredictActor,ResNet9
from environment import carla_gym
import environment.atari as Atari

parser = ArgumentParser("deep_ac_agent")
parser.add_argument("--env", help="Name of the Gym environment", default="Pendulum-v0", metavar="ENV_ID")
parser.add_argument("--params-file", help="Path to the parameters file. Default= ./COLA_rl_parameters.json",
                    default="COLA_rl_parameters.json", metavar="async_a2c_parameters.json")
parser.add_argument("--model-dir", default="COLA_rl/trained_models/", metavar="MODEL_DIR",
                    help="Directory to save/load trained model. Default= ./COLA_rl/trained_models/")
parser.add_argument("--render", action='store_true', default=False,
                    help="Whether to render the environment to the display. Default=False")
parser.add_argument("--test", help="Tests a saved Agent model to see the performance. Disables learning",
                    action='store_true', default=False)
parser.add_argument("--gpu-id", help="GPU device ID to use. Default:0", type=int, default=0, metavar="GPU_ID")
args = parser.parse_args()

global_step_num = 0
params_manager= ParamsManager(args.params_file)
summary_file_path_prefix = params_manager.get_agent_params()['summary_file_path_prefix']
summary_file_path= summary_file_path_prefix + args.env + "_" + datetime.now().strftime("%y-%m-%d-%H-%M")
writer = SummaryWriter(summary_file_path)
# Export the parameters as json files to the log directory to keep track of the parameters used in each experiment
params_manager.export_env_params(summary_file_path + "/" + "env_params.json")
params_manager.export_agent_params(summary_file_path + "/" + "agent_params.json")
use_cuda = params_manager.get_agent_params()['use_cuda']
# Introduced in PyTorch 0.4
device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")

seed = params_manager.get_agent_params()['seed']  # With the intent to make the results reproducible
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)

Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])


class DeepActorCriticAgent(mp.Process):
    def __init__(self, id, env_name, agent_params, shared_state, env_params):
        """
        An Asynchronous implementation of an Advantage Actor-Critic Agent that uses a Deep Neural Network to represent it's Policy and the Value function
        :param state_shape:
        :param action_shape:
        """
        super(DeepActorCriticAgent, self).__init__()
        self.id = id
        if id == 0:
            self.actor_name = "global"
        else:
            self.actor_name = "actor" + str(self.id)
        self.shared_state = shared_state
        self.env_name = env_name
        self.params = agent_params
        self.env_conf = env_params
        self.policy = self.multi_variate_gaussian_policy
        self.gamma = self.params['gamma']
        self.trajectory = []  # Contains the trajectory of the agent as a sequence of Transitions
        self.rewards = []  # Contains the rewards obtained from the env at every step
        self.global_step_num = 0
        self.best_mean_reward = - float("inf") # Agent's personal best mean episode reward
        self.best_reward = - float("inf")
        self.saved_params = False  # Whether or not the params have been saved along with the model to model_dir
        self.continuous_action_space = True  # Assumption by default unless env.action_space is Discrete
        self.policies = [] # tool models
        self.accracy = 0
        self.step_num = 0
        self.accracy_sum = 0
        self.BayesProb = [0.5,0.5]
        self.Bayes = self.params["bayes"]

    def multi_variate_gaussian_policy(self, obs):
        """
        Calculates a multi-variate gaussian distribution over actions given observations
        :param obs: Agent's observation
        :return: policy, a distribution over actions for the given observation
        """
        mu, sigma = self.actor(obs)
        value = self.critic(obs)
        [ mu[:, i].clamp_(float(self.env.action_space.low[i]), float(self.env.action_space.high[i]))
         for i in range(self.action_shape)]  # Clamp each dim of mu based on the (low,high) limits of that action dim
        sigma = torch.nn.Softplus()(sigma).squeeze() + 1e-7  # Let sigma be (smoothly) +ve
        self.mu = mu.to(torch.device("cpu"))
        self.sigma = sigma.to(torch.device("cpu"))
        self.value = value.to(torch.device("cpu"))
        if len(self.mu.shape) == 0: # See if mu is a scalar
            # self.mu = self.mu.unsqueeze(0)  # This prevents MultivariateNormal from crashing with SIGFPE
            self.mu.unsqueeze_(0)
        self.action_distribution = MultivariateNormal(self.mu, torch.eye(self.action_shape) * self.sigma, validate_args=True)
        return self.action_distribution

    def discrete_policy(self, obs, predict = False):
        """
        Calculates a discrete/categorical distribution over actions given observations
        :param obs: Agent's observation
        :return: policy, a distribution over actions for the given observation
        """
        if predict:
            logits = self.actor(obs)
            self.logits = logits.to(torch.device("cpu"))
            return torch.nn.functional.softmax(self.logits, dim=1)
        else:
            logits = self.policies[self.selected](obs)
            self.logits = logits.to(torch.device("cpu"))
            self.action_distribution = Categorical(logits=self.logits)
            return self.action_distribution

    def preproc_obs(self, obs):
        obs = np.array(obs)  # Obs could be lazy frames. So, force fetch before moving forward
        if len(obs.shape) == 3:
            # Reshape obs from (H x W x C) order to this order: C x W x H and resize to (C x 84 x 84)
            obs = np.reshape(obs, (obs.shape[2], obs.shape[1], obs.shape[0]))
            obs = np.resize(obs, (obs.shape[0], 84, 84))
        #  Convert to torch Tensor, add a batch dimension, convert to float repr
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        return obs

    def process_action(self, action):
        if self.continuous_action_space:
            [action[:, i].clamp_(float(self.env.action_space.low[i]), float(self.env.action_space.high[i]))
             for i in range(self.action_shape)]  # Limit the action to lie between the (low, high) limits of the env
        action = action.to(torch.device("cpu"))
        return action.numpy().squeeze(0)  # Convert to numpy ndarray, squeeze and remove the batch dimension
    
    def get_action(self, obs):
        obs = self.preproc_obs(obs)
        action_distribution = self.policy(obs)  # Call to self.policy(obs) also populates self.value with V(obs)
        action = action_distribution.sample()
        action = self.process_action(action)
        # Store the n-step trajectory for learning. Skip storing the trajectories in test only mode
        return action
    
    def get_predict_action(self, obs):
        obs = self.preproc_obs(obs)
        action_distribution = self.policy(obs, True)  # Call to self.policy(obs) also populates self.value with V(obs)
        if self.Bayes:
            p1 = action_distribution[0][0].item()
            p2 = action_distribution[0][1].item()
            Denominator = self.BayesProb[0]*p1 + self.BayesProb[1]*p2
            P1 = self.BayesProb[0]*p1/Denominator
            P2 = self.BayesProb[1]*p2/Denominator
            self.BayesProb = [np.clip(P1,0.01,0.99), np.clip(P2,0.01,0.99)]
            self.selected = np.argmax(self.BayesProb)
        else:
            self.selected = action_distribution.argmax().item()
        # Store the n-step trajectory for learning. Skip storing the trajectories in test only mode
        return action_distribution, self.logits

    def calculate_n_step_return(self, n_step_rewards, final_state, done, gamma):
        """
        Calculates the n-step return for each state in the input-trajectory/n_step_transitions
        :param n_step_rewards: List of rewards for each step
        :param final_state: Final state in this n_step_transition/trajectory
        :param done: True rf the final state is a terminal state if not, False
        :return: The n-step return for each state in the n_step_transitions
        """
        g_t_n_s = list()
        with torch.no_grad():
            g_t_n = torch.tensor([[0]]).float() if done else self.critic(self.preproc_obs(final_state)).cpu()
            for r_t in n_step_rewards[::-1]:  # Reverse order; From r_tpn to r_t
                g_t_n = torch.tensor(r_t).float() + self.gamma * g_t_n
                g_t_n_s.insert(0, g_t_n)  # n-step returns inserted to the left to maintain correct index order
            return g_t_n_s

    def calculate_loss(self):
        """
        Calculates the critic and actor losses using the td_targets and self.trajectory
        :param td_targets:
        :return:
        """
        # actor_loss = torch.norm(self.Predited_prob-self.true_weather)
        actor_loss = torch.nn.functional.cross_entropy(self.Predited_logits,self.true_weather)
        if self.actor_name == "global":
            writer.add_scalar(self.actor_name + "/actor_loss", actor_loss, self.global_step_num)

        return actor_loss

    def pull_params_from_global_agent(self):
        # If this is the very beginning of the procs, the global agent may not have started yet.
        # Wait for the global agent proc to start and make the shared state dict available
        while "actor_state_dict" not in self.shared_state:
            time.sleep(2)
        self.actor.load_state_dict(self.shared_state["actor_state_dict"])
        self.actor.to(device)

    def push_params_to_global_agent(self):
        self.shared_state["actor_state_dict"] = self.actor.cpu().state_dict()
        # To make sure that the actor models are on the desired device
        self.actor.to(device)

    def learn(self, grad_clip=False):
        actor_loss = self.calculate_loss()
        
        self.actor_optimizer.zero_grad()
        if grad_clip: 
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), grad_clip)
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        # print("lr: ", self.actor_optimizer.param_groups[0]['lr'])

        self.rewards.clear()

    def save(self):
        model_file_name = self.params["model_dir"] + "Async-A2C_" + self.env_name + ".ptm"
        agent_state = {"Actor": self.actor.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward}
        torch.save(agent_state, model_file_name)
        print("Agent's state is saved to", model_file_name)
        # Export the params used if not exported already
        if not self.saved_params:
            params_manager.export_agent_params(model_file_name + ".agent_params")
            print("The parameters have been saved to", model_file_name + ".agent_params")
            self.saved_params = True

    def load(self):
        model_file_name = self.params["model_dir"] + "Async-A2C_" + self.env_name + ".ptm"
        agent_state = torch.load(model_file_name, map_location= lambda storage, loc: storage)
        self.actor.load_state_dict(agent_state["Actor"])
        self.actor.to(device)
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print("Loaded Advantage Actor model state from", model_file_name,
              " which fetched a best mean reward of:", self.best_mean_reward,
              " and an all time best reward of:", self.best_reward)
    def get_lr(self, optimizer): 
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    def run(self):
        # If a custom useful_region configuration for this environment ID is available, use it if not use the Default.
        # Currently this is utilized for only the Atari env. Follows the same procedure as in Chapter 6
        custom_region_available = False
        for key, value in self.env_conf['useful_region'].items():
            if key in args.env:
                self.env_conf['useful_region'] = value
                custom_region_available = True
                break
        if custom_region_available is not True:
            self.env_conf['useful_region'] = self.env_conf['useful_region']['Default']
        atari_env = False
        for game in Atari.get_games_list():
            if game.replace("_", "") in args.env.lower():
                atari_env = True
        if atari_env:  # Use the Atari wrappers (like we did in Chapter 6) if it's an Atari env
            self.env = Atari.make_env(self.env_name, self.env_conf)
        else:
            #print("Given environment name is not an Atari Env. Creating a Gym env")
            self.env = gym.make(self.env_name)

        self.state_shape = self.env.observation_space.shape
        # print(self.state_shape)
        if isinstance(self.env.action_space.sample(), int):  # Discrete action space
            self.action_shape = self.env.action_space.n
            self.policy = self.discrete_policy
            self.continuous_action_space = False

        else:  # Continuous action space
            self.action_shape = self.env.action_space.shape[0]
            self.policy = self.multi_variate_gaussian_policy
            
        self.actor_shape = 2
        if len(self.state_shape) == 3:  # Screen image is the input to the agent
            if self.continuous_action_space:
                self.actor= DeepActor(self.state_shape, self.action_shape, device).to(device)
            else:  # Discrete action space
                self.actor = ResNet9(self.state_shape, self.actor_shape, device).to(device)
                self.policy1 = DeepDiscreteActor(self.state_shape, self.action_shape, device).to(device)
                self.policies.append(self.policy1)
                self.policy2 = DeepDiscreteActor(self.state_shape, self.action_shape, device).to(device)
                self.policies.append(self.policy2)
        else:  # Input is a (single dimensional) vector
            if self.continuous_action_space:
                self.actor = ShallowActor(self.state_shape, self.action_shape, device).to(device)
            else:  # Discrete action space
                self.actor = ShallowDiscreteActor(self.state_shape, self.action_shape, device).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.params["learning_rate"], weight_decay=0.001)
        self.actor_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.actor_optimizer, max_lr=self.params["learning_rate"], total_steps=self.params["max_num_episodes"])
        
        # layer_num = 1
        # for param1, param2 in zip(self.policy1.parameters(), self.policy2.parameters()):
        #     print("P1 layer",layer_num,": ",torch.max(torch.abs(param1)).item(),end = "")
        #     print(" P2 layer",layer_num,": ",torch.max(torch.abs(param2)).item(),end = "")
        #     print(" layer",layer_num,": ",torch.max(torch.abs(param1-param2)).item())
        #     layer_num += 1

        if self.actor_name == "global":

            # Handle loading and saving of trained Agent's model
            episode_rewards = list()
            prev_checkpoint_mean_ep_rew = self.best_mean_reward
            num_improved_episodes_before_checkpoint = 0  # To keep track of the num of ep with higher perf to save model
            #print("Using agent_params:", self.params)
            if self.params['load_trained_model']:
                try:
                    self.load()
                    prev_checkpoint_mean_ep_rew = self.best_mean_reward
                    
                    model_file_name = "tool_models/model1/Async-A2C_Carla-v0.ptm"
                    agent_state = torch.load(model_file_name, map_location= lambda storage, loc: storage)
                    self.policy1.load_state_dict(agent_state["Actor"])
                    self.policy1.to(device)
                    
                    model_file_name = "tool_models/model2/Async-A2C_Carla-v0.ptm"
                    agent_state = torch.load(model_file_name, map_location= lambda storage, loc: storage)
                    self.policy2.load_state_dict(agent_state["Actor"])
                    self.policy2.to(device)
                    
                except FileNotFoundError:
                    if args.test:  # Test a saved model
                        print("FATAL: No saved model found. Cannot test. Press any key to train from scratch")
                        input()
                    else:
                        print("WARNING: No trained model found for this environment. Training from scratch.")
            self.actor.share_memory()
            # Initialize the global shared actor parameters
            self.shared_state["actor_state_dict"] = self.actor.cpu().state_dict()

        for episode in range(self.params["max_num_episodes"]):
            if self.actor_name == "global" and self.global_step_num != 0:
               writer.add_scalar(self.actor_name + "/accuracy", self.accracy/(self.step_num+1), self.global_step_num)
               self.actor_scheduler.step()
               writer.add_scalar(self.actor_name + "/lr", self.get_lr(self.actor_optimizer), self.global_step_num)
               
            if args.test and episode != 0:
                self.accracy_sum += self.accracy/(self.step_num+1)
                writer.add_scalar(self.actor_name + "/mean_accuracy", self.accracy_sum/episode, self.global_step_num)
            self.accracy = 0
            obs = self.env.reset()
            done = False
            ep_reward = 0.0
            step_num = 0
            self.step_num = 0
            self.BayesProb = [0.5,0.5]
            cloudy = 0

            self.pull_params_from_global_agent()  # Synchronize local-agent specific parameters from the global
            while not done:
                self.Predited_prob, self.Predited_logits = self.get_predict_action(obs)
                action = self.get_action(obs)
                next_obs, reward, done, _, weather_num = self.env.step(action)
                # print("weather:", weather_num, type(weather_num))
                if isinstance(weather_num,int):
                    if weather_num == 0:
                        self.true_weather = torch.tensor([0])
                        cloudy += 1
                    else:
                        self.true_weather = torch.tensor([1])
                        
                    if self.selected == self.true_weather.item():
                       self.accracy += 1
                else:    
                    self.true_weather = torch.tensor([np.argmax(weather_num)])
                    if self.selected == np.argmax(weather_num):
                        self.accracy += 1
                    if np.argmax(weather_num) == 0:
                        cloudy += 1
            
                print("Prediction: {}/{} , "
                      "True Weather {},{}, Selected Action: {}".format(
                          [float('{:.4f}'.format(i)) for i in self.Predited_prob.tolist()[0]],
                          self.BayesProb,
                          weather_num, self.true_weather,
                          self.selected,
                      ))
                self.rewards.append(reward)
                ep_reward += reward
                step_num += 1
                self.step_num += 1
                if not args.test and (step_num >= self.params["learning_step_thresh"] or done):
                    self.learn(0.01)
                    step_num = 0
                    # Async send updates to the global shared parameters
                    self.push_params_to_global_agent()

                    # Monitor performance and save Agent's state when perf improves
                    if done and self.actor_name == "global":
                        episode_rewards.append(ep_reward)
                        if ep_reward > self.best_reward:
                            self.best_reward = ep_reward
                        if np.mean(episode_rewards) > prev_checkpoint_mean_ep_rew:
                            num_improved_episodes_before_checkpoint += 1
                        if num_improved_episodes_before_checkpoint >= self.params["save_freq_when_perf_improves"]:
                            prev_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                            self.best_mean_reward = np.mean(episode_rewards)
                            self.save()
                            num_improved_episodes_before_checkpoint = 0
                if done and self.actor_name == "global" and args.test:
                        episode_rewards.append(ep_reward)
                        writer.add_scalar(self.actor_name + "/cloudy_proportion", cloudy/self.step_num, self.global_step_num)
                        # print("cloudy:", cloudy/self.global_step_num)
                obs = next_obs
                self.global_step_num += 1
                if self.actor_name == "global":
                    if args.render:
                        self.env.render()
                    #print(self.actor_name + ":Episode#:", episode, "step#:", step_num, "\t rew=", reward, end="\r")
                    writer.add_scalar(self.actor_name + "/reward", reward, self.global_step_num)
            # Print stats at the end of episodes
                
            if self.actor_name == "global":
                print("{}:Episode#:{} \t ep_reward:{} \t mean_ep_rew:{}\t best_ep_reward:{}".format(
                            self.actor_name, episode, ep_reward, np.mean(episode_rewards), self.best_reward))
                writer.add_scalar(self.actor_name + "/ep_reward", ep_reward, self.global_step_num)
                if args.test:
                    writer.add_scalar(self.actor_name + "/mean_ep_reward", np.mean(episode_rewards), self.global_step_num)


if __name__ == "__main__":
    agent_params = params_manager.get_agent_params()
    agent_params["model_dir"] = args.model_dir
    agent_params["test"] = args.test
    env_params = params_manager.get_env_params()  # Used with Atari environments
    env_params["env_name"] = args.env
    mp.set_start_method('spawn')  # Prevents RuntimeError during cuda init

    manager = mp.Manager()
    shared_state = manager.dict()
    if not args.test:
        agent_procs =[DeepActorCriticAgent(id, args.env, agent_params, shared_state, env_params)
                      for id in range(agent_params["num_agents"])]
        [p.start() for p in agent_procs]
        [p.join() for p in agent_procs]
    else:
        test_agent_proc = DeepActorCriticAgent(0, args.env, agent_params, shared_state, env_params)
        test_agent_proc.start()
        test_agent_proc.join()

