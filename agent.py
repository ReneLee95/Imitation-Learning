import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
import csv
from collections import deque
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','done'))


def get_demo_traj():
    demo_memory = ReplayMemory(1000)
    data = np.load("demo_traj_2.npy", allow_pickle=True)

    for i in range(len(data)):
        for j in range(len(data[i])):
            state = data[i][j][0]
            action = data[i][j][1]
            reward = data[i][j][2]
            new_state = data[i][j][3]
            done = ~data[i][j][4]

            demo_memory.write(Transition(state, action, new_state, reward,done))
    
    return demo_memory                       

class Tree(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.lazy = np.zeros(capacity, dtype=object)
        self.position = 0

    def propagation(self,idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self.propagation(parent, change)
               
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self.propagation(idx, change)
    
    def add(self, p, data):
        tree_idx = self.position + self.capacity - 1
        self.lazy[self.position] = data
        
        self.position += 1
        if self.position >= self.capacity:
            self.position = 0
        
        self.update(tree_idx, p)
    
    def get_value(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.lazy[data_idx]
    
    def total(self):
        return self.tree[0]  
    
    

##########################################################################
############                                                  ############
############                  DQfDNetwork 구현                 ############
############                                                  ############
##########################################################################
class PER_ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = capacity       
        self.e = 0.001
        self.a = 0.4
        self.beta = 0.3

        self.per_buffer = Tree(capacity * 2)
        self.lengthCount = 0
            
    def get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a    
            
    def write_PER(self,error, transition):
        data = self.get_priority(error)
        self.per_buffer.add(data, transition)          
        
        if self.per_buffer.capacity > self.lengthCount:
            self.lengthCount += 1
           
    def sample_PER(self, batch_size):
        batch, idx = [], []
        seg = self.per_buffer.total() / batch_size
        priority = []

        for i in range(batch_size):
            s = random.uniform(seg * i , seg * (i + 1))
            index, p, data = self.per_buffer.get_value(s)           
            
            priority.append(p)
            batch.append(data)
            idx.append(index)

        sampling_prob = priority / self.per_buffer.total()
        weight = np.power(self.lengthCount * sampling_prob, -self.beta)
        weight = weight/weight.max()

        return batch, idx, weight
    
    def __len__(self):
        return self.lengthCount


class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.currentPosition = 0
        self.buffer = deque(maxlen=self.capacity) 

    
    def write(self, transition):
        self.buffer.append(transition)
        self.currentPosition = self.currentPosition + 1           
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
        


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class DQfDNetwork(nn.Module):
    def __init__(self, input_shape=4, num_actions=1):
        super(DQfDNetwork, self).__init__()

        self.lin1 = nn.Linear(input_shape, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128,128)
        self.lin4 = nn.Linear(128, num_actions)
        
        init_weights(self.lin1)
        init_weights(self.lin2)
        init_weights(self.lin3)
        init_weights(self.lin4)

    def forward(self, states):
        x = self.lin1(states)
        x = F.leaky_relu(self.lin2(x))
        x = F.leaky_relu(self.lin3(x))
        return self.lin4(x)
##########################################################################
############                                                  ############
############                  DQfDagent 구현                   ############
############                                                  ############
##########################################################################

def update_lr(optimizer,reward):
    if reward > 500:
        optimizer.param_groups[0]['lr'] = 0.00006
        return optimizer
    elif reward > 400:
        optimizer.param_groups[0]['lr'] = 0.00006
        return optimizer
    elif reward > 300:
        optimizer.param_groups[0]['lr'] = 0.00006
        return optimizer
    elif reward > 200:
        optimizer.param_groups[0]['lr'] = 0.00007  
        return optimizer
    elif reward > 100:
        optimizer.param_groups[0]['lr'] = 0.00008
        return optimizer

class DQfDAgent(object):
    def __init__(self, env, use_per, n_episode):
        super(DQfDAgent, self).__init__()
        
        self.n_EPISODES = n_episode

        self.env = env       

        self.margin = 8.0
        self.opt_step = 0
        self.eps_steps_done = 0
        
        self.use_per = use_per  
        
        if self.use_per == False:
            self.epsilon = 0.3           
            self.lr = 0.00005
            self.decay = 0.99
            self.epsilon_end = 0.1
            self.epsilon_decay = 0.999  
            self.steps_done = 0
            self.gamma = 1.0
            self.decay = 0.99

            self.memory = ReplayMemory(100000)
            self.alpha = 0.1
        else:
            self.epsilon = 0.3
            self.epsilon_end = 0.1
            self.epsilon_decay = 0.999  
            self.eps_steps_done = 0
            self.steps_done = 0
            self.gamma = 1.0
            self.decay = 0.99
            self.alpha = 0.1
            self.lr = 0.00008
            self.memory = PER_ReplayMemory(100000)

        self.main_network = DQfDNetwork(self.env.observation_space.shape[0],self.env.action_space.n)
        self.target_network = DQfDNetwork(self.env.observation_space.shape[0],self.env.action_space.n)

        self.optimizer = optim.Adam(self.main_network.parameters(), self.lr, weight_decay=1e-5)
        self.demo_memory = get_demo_traj()

        self.update_network_parameters()
        
    def epsilon_update(self,epsilon):
        epsilon_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(-1. * self.eps_steps_done / self.epsilon_decay)
        self.eps_steps_done += 1
        
        return epsilon_threshold
        
    def get_action(self, state):
        sample = random.random()
        self.steps_done += 1
        if sample > self.epsilon:
            q_vals = self.main_network(torch.Tensor(state)).data
            return q_vals.argmax().numpy()
        else:
            return self.env.action_space.sample()
        
    def sample_opt(self, batch_size):
        demo_trans = []
        proportion = 0.3
        demo_samples = int(batch_size * proportion)
        if demo_samples > 0:
            demo_trans = self.demo_memory.sample(demo_samples)

        if demo_samples != batch_size:
            if self.use_per:
                agent_trans, idx, _ = self.memory.sample_PER(batch_size - demo_samples)
            else:
                agent_trans = self.memory.sample(batch_size - demo_samples)
            transitions = demo_trans + agent_trans
        else:
            transitions = demo_trans
        batch = Transition(*zip(*transitions))

        next_state_batch = torch.Tensor(batch.next_state)
        state_batch = torch.Tensor(batch.state)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        n_reward_batch = torch.Tensor(batch.reward)
        done_batch = torch.Tensor(batch.done)

        q_value = self.main_network(state_batch)
        state_action_value = q_value.gather(1, action_batch).squeeze()    
        
        expected = torch.argmax(self.main_network(next_state_batch).detach().squeeze(), dim=1).view(-1, 1)
        one_step = 1.0 + (done_batch) * self.gamma * self.target_network(next_state_batch).squeeze().gather(1, expected)

        J_DQ_loss = F.mse_loss(one_step,self.main_network(state_batch).squeeze().gather(1,action_batch))

        J_n_loss = F.mse_loss(state_action_value, n_reward_batch, reduction="mean")

        num_actions = q_value.size(1)
        margins = (torch.ones(num_actions, num_actions) - torch.eye(num_actions)) * self.margin
        batch_margin = margins[action_batch.data.squeeze()]
        q_value = q_value + torch.Tensor(batch_margin).type(torch.FloatTensor)

        J_E_loss = (q_value.max(1)[0] - state_action_value).pow(2)[:demo_samples].sum()

        J_L2_loss = torch.tensor(0.)
        for param in self.main_network.parameters():
            J_L2_loss += torch.norm(param)
        
        l1, l2, l3 = 1.0, 1.0, 1e-5

        loss = J_DQ_loss + l1 * J_n_loss + l2 * J_E_loss + l3 * J_L2_loss

        
        self.main_network.train()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.main_network.parameters(), 100)
        self.optimizer.step()

        if self.opt_step % 1000 == 0:
            self.epsilon = max(self.epsilon_update(self.epsilon), 0.01)
        if self.opt_step % 10000 == 0:
            self.update_network_parameters()

        self.opt_step += 1
        
    def pretrain(self):
        ## Do pretrain for 1000 steps
        batch_size = 64      
        
        for i in range(1000):           
            demo_trans = []
            demo_samples = int(batch_size)
            if demo_samples > 0:
                demo_trans = self.demo_memory.sample(demo_samples)

            transitions = demo_trans
            batch = Transition(*zip(*transitions))

            next_state_batch = torch.Tensor(batch.next_state)
            state_batch = torch.Tensor(batch.state)
            action_batch = torch.LongTensor(batch.action).unsqueeze(1)
            reward_batch = torch.Tensor(batch.reward)
            done_batch = torch.Tensor(batch.done)

            q_value = self.main_network(state_batch)
            state_action_value = q_value.gather(1, action_batch).squeeze()

            expected = torch.argmax(self.main_network(next_state_batch).detach().squeeze(), dim=1).view(-1, 1)
            one_step = 1.0 + (done_batch) * self.gamma * self.target_network(next_state_batch).squeeze().gather(1, expected)

            J_DQ_loss = F.mse_loss(one_step,self.main_network(state_batch).squeeze().gather(1,action_batch))

            J_n_loss = F.mse_loss(state_action_value, reward_batch, reduction="mean")

            num_actions = q_value.size(1)
            margins = (torch.ones(num_actions, num_actions) - torch.eye(num_actions)) * self.margin
            batch_margin = margins[action_batch.data.squeeze()]
            q_value = q_value + torch.Tensor(batch_margin).type(torch.FloatTensor)

            J_E_loss = (q_value.max(1)[0] - state_action_value).pow(2)[:demo_samples].sum()

            J_L2_loss = torch.tensor(0.)
            for param in self.main_network.parameters():
                J_L2_loss += torch.norm(param)
            
            l1, l2, l3 = 1.0, 1.0, 1e-5
            loss = J_DQ_loss + l1 * J_n_loss + l2 * J_E_loss + l3 * J_L2_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.main_network.parameters(), 50)
            self.optimizer.step()

            self.opt_step += 1

    def train(self):
        ###### 1. DO NOT MODIFY FOR TESTING ######
        test_mean_episode_reward = deque(maxlen=20)
        test_over_reward = False
        test_min_episode = np.inf
        ###### 1. DO NOT MODIFY FOR TESTING ######
        
        # Do pretrain
        self.pretrain()            

        for e in range(self.n_EPISODES):
            ########### 2. DO NOT MODIFY FOR TESTING ###########
            test_episode_reward = 0
            ########### 2. DO NOT MODIFY FOR TESTING  ###########          
            
            done = False
            state = self.env.reset()
            transitions = []
            action_list, next_state_list, reward_list = [], [], []
            
            while not done:

                action = self.get_action(state)

                next_state, reward, done, _ = self.env.step(action)
                
                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward      
                ########### 3. DO NOT MODIFY FOR TESTING  ###########

                reward = torch.Tensor([reward])
                
                action_list.append(action)
                next_state_list.append(next_state)
                reward_list.append(reward)
                
                for idx in range(len(action_list)):
                    new_reward = reward_list[idx] + self.gamma * reward
                    transitions.append(Transition(state,action,next_state,new_reward,torch.zeros(1)))
                    self.gamma = self.gamma * self.decay

                if done:
                    for idx in range(len(transitions)):
                        if self.use_per:
                            old_value = self.main_network(torch.Tensor(transitions[idx].state)).detach().numpy()[transitions[idx].action]
                            if idx == 0:
                                new_value = transitions[idx].reward.numpy()[0]
                            else:
                                new_value = self.main_network(torch.Tensor(transitions[idx].next_state)).detach().numpy()[transitions[idx].action]
                                old_value = (new_value * self.decay) + transitions[idx].reward.numpy()[0]
                            old_value = (1 - self.alpha) * old_value + (self.alpha * new_value)
                            error = abs(new_value - old_value)
                            error = np.clip(error, 0, 1)  

                            self.memory.write_PER(error, transitions[idx])
                        else:
                            self.memory.write(transitions[idx])
                            
                    update_lr(self.optimizer,test_episode_reward)
                    
                else:
                    q_value = self.main_network(torch.Tensor(next_state)).data
                    if len(transitions) >= 10:
                        trans_pop = transitions.pop()
                        trans_pop = trans_pop._replace(reward=trans_pop.reward + self.gamma * q_value.max())
                        if self.use_per:
                            error = self.get_td_error(trans_pop)
                            self.memory.write_PER(error, trans_pop)
                        else:
                            self.memory.write(trans_pop)
                                                
                state = next_state

                if len(self.memory) >= 1000:
                    self.sample_opt(64)
                    

                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward)==20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                
            ########### 5. DO NOT MODIFY FOR TESTING  ###########
            if test_over_reward:
                print("END train function")
                break
            ########### 5. DO NOT MODIFY FOR TESTING  ###########

        ########### 6. DO NOT MODIFY FOR TESTING  ###########
        return test_min_episode, np.mean(test_mean_episode_reward)
        ########### 6. DO NOT MODIFY FOR TESTING  ###########
    
    def get_td_error(self,trans):
        old_value = self.main_network(torch.Tensor(trans.state)).detach().numpy()[trans.action]
        new_value = self.main_network(torch.Tensor(trans.next_state)).detach().numpy()[trans.action]
        new_value = (new_value * self.decay) + trans.reward.numpy()[0]
        old_value = (1 - self.alpha) * old_value + (self.alpha * new_value)
        error = abs(new_value - old_value)
        error = np.clip(error, 0, 1)  
        
        return error
        
    def update_network_parameters(self):        
        for main_param, target_param in zip(self.main_network.parameters(), self.target_network.parameters()):
            target_param._grad = main_param.grad
        
        self.optimizer.step()
        
        self.target_network.load_state_dict(self.main_network.state_dict())
        