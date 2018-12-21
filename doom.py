from vizdoom import *
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random 
import time
import copy
import argparse
import cv2
import warnings
warnings.filterwarnings('ignore')

from model import DDDQNNet , init_weights
from utils import SumTree , Memory , predict_action ,create_enviornment , preprocess_frame

parser = argparse.ArgumentParser(description="Agent DOoM")
parser.add_argument('--learning_rate', type=float,help='Learning rate for DQN',default=0.025)
parser.add_argument('--episodes', type=int,help='Number of episodes for training',default=5000)
parser.add_argument('--steps', type=int,help='Maximum steps per episode',default=5000)
parser.add_argument('--tau', type=int,help='Time to refil target network weights',default=3000)
parser.add_argument('--explore_stop', type=float,help='Stopping exploration probability',default=0.01)
parser.add_argument('--decay_rate', type=float,help='Decay rate for exploration probability',default=0.00005)
parser.add_argument('--discount_factor', type=float,help='Discounting factor for returns',default=0.95)
parser.add_argument('--pretrain_length', type=int,help='number of iterations to fill replay memory',default=10000)
parser.add_argument('--mem_size', type=int,help='Size of the replay memory',default=10000)
parser.add_argument('--train', type=bool,help='want to train or not',default=False)
parser.add_argument('--batch_size', type=int,help='Batch size to sample from replay memory',default=64)
parser.add_argument('--gpu', type=bool,help='Use GPU',default=False)
parser.add_argument('--weights', type=str,help='load pretrained weights',default=None)
parser.add_argument('--save_weights', type=bool,help='save the DQN weights',default=False)
parser.add_argument('--graph', type=bool,help='save the DQN weights',default=False)
parser.add_argument('--test', type=bool,help='Generate testing video',default=False)
args = parser.parse_args()
game,possible_actions = create_enviornment()

stack_size=4
stacked_frames = deque([np.zeros((108,124),dtype=np.int) for i in range(stack_size)],maxlen=4)

def stack_frames(stacked_frames,state,is_new_episode):
	frame = preprocess_frame(state)
	if is_new_episode:
		stacked_frames = deque([frame for i in range(stack_size)],maxlen=4)
	else:
		stacked_frames.append(frame)
	stacked_state = torch.stack(list(stacked_frames),1)
	return stacked_state, stacked_frames

state_size = [4,108,124]
action_size = game.get_available_buttons_size()
explore_start = 1.0
episode_render = False

if args.weights is None:
	DQNetwork = DDDQNNet(state_size,action_size,name="DQNetwork")
	init_weights(DQNetwork)
else:
	DQNetwork = torch.load(args.weights)
	print("Weights loaded")

TargetNetwork = DDDQNNet(state_size,action_size,name="TargetNetwork")
if args.gpu:
	DQNetwork , TargetNetwork = DQNetwork.cuda() , TargetNetwork.cuda()
optimizerDQ = optim.RMSprop(DQNetwork.parameters(),lr=args.learning_rate)

memory = Memory(args.mem_size)
game.new_episode()

if args.train:
	print("Training started")
	for i in range(args.pretrain_length):
		if i==0:
			state = game.get_state().screen_buffer
			state,stacked_frames = stack_frames(stacked_frames,state,True)

		action=random.choice(possible_actions)
		reward = game.make_action(action)
		done = game.is_episode_finished()

		if done:
			next_state = np.zeros(state.shape,dtype=np.float32)
			experience = state,action,reward,next_state,done
			memory.store(experience)
			game.new_episode()
			state = game.get_state().screen_buffer
			state,stacked_frames = stack_frames(stacked_frames,state,True)
		else:
			next_state = game.get_state().screen_buffer
			next_state,stacked_frames = stack_frames(stacked_frames,next_state,False)
			experience = state,action,reward,next_state,done
			memory.store(experience)
			state= next_state

	Summary = { "Loss":[] , "Rewards":[] }

	decay_step=0
	tau=0
	game.init()
	TargetNetwork = copy.deepcopy(DQNetwork)
	for episode in range(args.episodes):
		loss=0
		step=0
		episode_rewards=[]
		game.new_episode()
		state = game.get_state().screen_buffer
		state,stacked_frames = stack_frames(stacked_frames,state,True)

		while step<args.steps:
			step+=1
			tau+=1
			decay_step+=1
			action,explore_probability,is_greedy= predict_action(explore_start,args.explore_stop,args.decay_rate,decay_step,state,possible_actions,DQNetwork,args.gpu)
			reward = game.make_action(action)
			done = game.is_episode_finished()
			episode_rewards.append(reward)
			optimizerDQ.zero_grad()

			if done:
				next_state=np.zeros(state.shape,dtype=np.float32)
				step=args.steps
				total_reward = np.sum(episode_rewards)

				Summary["Rewards"].append(total_reward)
				Summary["Loss"].append(loss)

				print('Episode: {}'.format(episode),'Total reward: {}'.format(total_reward),
					'Training loss: {:.4f}'.format(loss),
					'Explore P: {:.4f}'.format(explore_probability),'Is Greedy: {}'.format(is_greedy))

				experience= state,action,reward,next_state,done
				memory.store(experience)

			else:
				next_state = game.get_state().screen_buffer
				next_state , stacked_frames = stack_frames(stacked_frames,next_state,False)
				experience= state,action,reward,next_state,done
				memory.store(experience)
				state = next_state

			ISWeights_mb,tree_idx,batch = memory.sample(args.batch_size)
			ISWeights_mb = torch.tensor(ISWeights_mb,requires_grad=True)
			if args.gpu:
				ISWeights_mb=ISWeights_mb.cuda()
			ISWeights_mb = ISWeights_mb.squeeze(1)
			states_mb = torch.cat([Variable(each[0][0],requires_grad=True) for each in batch],0)
			actions_mb = torch.cat([torch.tensor(each[0][1],requires_grad=True) for each in batch],0)
			rewards_mb = torch.tensor([each[0][2] for each in batch],requires_grad=True) 
			next_states_mb = torch.cat([torch.tensor(each[0][3],requires_grad=True) for each in batch], 0)
			dones_mb = [each[0][4] for each in batch]

			target_Qs_batch =[]
			predicted_Qs_batch=[]

			if args.gpu:
				states_mb,next_states_mb = states_mb.cuda() , next_states_mb.cuda()

			q_state = DQNetwork(states_mb)
			q_next_state = DQNetwork(next_states_mb)
			q_target_next_state = TargetNetwork(next_states_mb)

			for i in range(len(batch)):
				terminal = dones_mb[i]
				_,action = torch.max(q_next_state[i],0)
				_,ac = torch.max(actions_mb[i],0)
				predicted_Qs_batch.append(q_state[i][ac])
				if terminal:
					target_Qs_batch.append(rewards_mb[i])
				else:
					target = rewards_mb[i].cuda() + args.discount_factor*q_target_next_state[i][action]
					target_Qs_batch.append(target)

			targets_mb = torch.tensor([each for each in target_Qs_batch],requires_grad=True)
			pre_targets_mb = torch.tensor([each for each in predicted_Qs_batch],requires_grad=True)
			if args.gpu:
				targets_mb , pre_targets_mb = targets_mb.cuda() , pre_targets_mb.cuda()
			absolute_errors = torch.abs(targets_mb-pre_targets_mb)
			loss = (targets_mb-pre_targets_mb)*(targets_mb-pre_targets_mb)
			loss = torch.mean(ISWeights_mb*loss)
			loss = Variable(loss,requires_grad=True)
			
			loss.backward()
			optimizerDQ.step()
			if args.gpu:
				memory.batch_update(tree_idx,absolute_errors.cpu().detach().numpy())
			else:
				memory.batch_update(tree_idx,absolute_errors.detach().numpy())

			if tau>args.tau:
				TargetNetwork = copy.deepcopy(DQNetwork)
				tau=0
				print("Model updated")

		if episode%5==0 and args.save_weights:
			torch.save(DQNetwork,'./dqnet.pt')
			print("Model Saved")



if args.graph:
	x = [i+1 for i in range(len(Summary["Loss"]))]
	y1 = Summary["Loss"]
	y2 = Summary["Rewards"]
	plt.subplot(2,1,1)
	plt.plot(x,y1)
	plt.title("Summary Graphs")
	plt.ylabel("Loss")

	plt.subplot(2,1,2)
	plt.plot(x,y2)
	plt.xlabel("episodes")
	plt.ylabel("Rewads")
	plt.savefig('./graph.png')


if args.test:
	print("Testing")
	for i in range(10):
		fourcc = cv2.VideoWriter_fourcc(*'DIVX')
		video = cv2.VideoWriter('./video{}.avi'.format(i),fourcc,10,(320,240))
		game.new_episode()
		state = game.get_state().screen_buffer
		st = state.transpose(1,2,0)
		video.write(st)
		state,stacked_frames = stack_frames(stacked_frames,state,True)
		s=0
		while not game.is_episode_finished():
			s+=1
			action,explore_probability,is_greedy = predict_action(0.01,0,0,0,state,possible_actions,DQNetwork,args.gpu)
			game.make_action(action)
			done = game.is_episode_finished()

			if done:
				break
			else:
				next_state = game.get_state().screen_buffer
				st = next_state.transpose(1,2,0)
				video.write(st)
				next_state,stacked_frames = stack_frames(stacked_frames,next_state,False)
				state = next_state

		score = game.get_total_reward()
		print("Score : ",score,"Steps :",s)
		video.release()
	game.close()
