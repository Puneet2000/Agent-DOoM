from vizdoom import *
import torch
import torchvision.transforms as transforms
import numpy as np
from collections import deque
import random 
import warnings
warnings.filterwarnings('ignore')

class SumTree(object):


	def __init__(self,capacity):
		self.capacity = capacity
		self.tree = np.zeros(2*capacity-1)
		self.data = np.zeros(capacity,dtype=object)
		self.data_pointer=0

	def add(self,priority,data):
		tree_index = self.data_pointer + self.capacity -1
		self.data[self.data_pointer]=data
		self.update(tree_index,priority)
		self.data_pointer+=1

		if self.data_pointer>=self.capacity:
			self.data_pointer=0

	def update(self,tree_index,priority):
		change = priority-self.tree[tree_index]
		self.tree[tree_index] = priority

		while tree_index!=0:
			tree_index=(tree_index-1)//2
			self.tree[tree_index]+=change

	def get_leaf(self,v):
		parent_index=0
		while True:
			left_child_index = 2*parent_index+1
			right_child_index = left_child_index+1
			if left_child_index>=len(self.tree):
				leaf_index = parent_index
				break
			else:
				if v<=self.tree[left_child_index]:
					parent_index=left_child_index
				else:
					v-=self.tree[left_child_index]
					parent_index=right_child_index
		data_index = leaf_index-self.capacity+1
		return leaf_index,self.tree[leaf_index],self.data[data_index]

	@property
	def total_priority(self):
		return self.tree[0]

class Memory(object):
	PER_e=0.01
	PER_a = 0.6
	PER_b = 0.4
	PER_b_increment =0.001
	absolute_upper_error = 1

	def __init__(self,capacity):
		self.tree= SumTree(capacity)

	def store(self,experience):
		max_priority = np.max(self.tree.tree[-self.tree.capacity:])
		if max_priority==0:
			max_priority=self.absolute_upper_error
		self.tree.add(max_priority,experience)

	def sample(self,n):
		memory_b=[]
		b_idx , b_ISWeights = np.empty((n,),dtype=np.int32),np.empty((n,1),dtype=np.float32)
		priority_segment = self.tree.total_priority/n
		self.PER_b = min(1.0,self.PER_b+self.PER_b_increment)

		p_min = np.min(self.tree.tree[-self.tree.capacity:])/self.tree.total_priority
		max_weight = (p_min*n)**(-self.PER_b)

		for i in range(n):
			a,b = priority_segment*i , priority_segment*(i+1)
			value = np.random.uniform(a,b)
			index, priority,data = self.tree.get_leaf(value)
			sampling_probs = priority/self.tree.total_priority
			b_ISWeights[i,0]= np.power(n*sampling_probs,-self.PER_b)/max_weight
			b_idx[i]=index
			experience=[data]
			memory_b.append(experience)
		return b_ISWeights,b_idx,memory_b

	def batch_update(self,tree_idx,abs_errors):
		abs_errors+=self.PER_e
		clipped_errors = np.minimum(abs_errors,self.absolute_upper_error)
		ps = np.power(clipped_errors,self.PER_a)
		for ti,p in zip(tree_idx,ps):
			self.tree.update(ti,p)

def create_enviornment():
	game = DoomGame()
	game.load_config("deadly_corridor.cfg")
	game.set_doom_scenario_path("deadly_corridor.wad")
	game.set_window_visible(False)
	game.init()
	possible_actions = np.identity(7,dtype=np.float).tolist()
	return game,possible_actions

def preprocess_frame(frame):
	cropped_frame = frame[:,15:-5,20:-20]
	normalized_frame = transforms.ToTensor()(cropped_frame)
	normalized_frame = transforms.ToPILImage()(normalized_frame)
	normalized_frame = transforms.Grayscale()(normalized_frame)
	preprocessed_frame = transforms.Resize((108,124))(normalized_frame)
	preprocessed_frame = transforms.ToTensor()(preprocessed_frame)
	return preprocessed_frame

def predict_action(explore_start,explore_stop,decay_rate,decay_step,state,actions,DQNetwork,gpu):
	explore_probability = explore_stop +(explore_start-explore_stop)*np.exp(-decay_rate*decay_step)
	A = np.ones(len(actions),dtype=np.float)*explore_probability/len(actions)
	if gpu:
		state = state.cuda()
	Qs = DQNetwork(state)
	_,choice = torch.max(Qs,1)
	A[int(choice)] += 1-explore_probability
	ch = np.random.choice(np.arange(len(actions)),p=A)
	action = actions[int(ch)]
	if int(ch)==int(choice):
		return action,explore_probability,True
	else:
		return action,explore_probability,False