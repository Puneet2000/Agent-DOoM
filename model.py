import torch
import torch.nn as nn
import torch.nn.functional as F

class DDDQNNet(nn.Module):
	def __init__(self,state_size,action_size,name):
		super(DDDQNNet,self).__init__()
		self.state_size=state_size
		self.action_size=action_size
		self.name=name

		self.conv1 = nn.Sequential( nn.Conv2d(4,32,8,4,0),
									nn.ELU() )
		self.conv2 = nn.Sequential( nn.Conv2d(32,64,4,2,0),
									nn.ELU())
		self.conv3 = nn.Sequential( nn.Conv2d(64,128,4,2,0),
									nn.ELU())
		self.value_fc = nn.Sequential( nn.Linear(5*6*128,512),
										nn.ELU(),
										nn.Linear(512,1))
		self.advantage_fc = nn.Sequential( nn.Linear(5*6*128,512),
											nn.ELU(),
											nn.Linear(512,self.action_size))

	def forward(self,x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		flatten = conv3.view(-1,5*6*128)
		value = self.value_fc(flatten)
		advantage = self.advantage_fc(flatten)
		q_value = value + (advantage - torch.mean(advantage))
		return q_value

def init_weights(m):
	if type(m)==nn.Linear or type(m) ==nn.Conv2d :
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.01)