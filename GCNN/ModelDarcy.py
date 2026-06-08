import torch
import torch.nn.functional as F
import pdb
from torch_geometric.nn import GCNConv, ChebConv, GATConv, TransformerConv, TAGConv, ARMAConv, SGConv,MFConv, RGCNConv
from torch_geometric.data import InMemoryDataset
import numpy as np

def T_connectivity(e2vcg,type='iso'):
	NnG=np.max(e2vcg)+1
	NnE=e2vcg.shape[1]
	if type=='ele':
		connectivity=[]
		for i in range(NnG):
			positions=np.argwhere(e2vcg==i)[:,0]
			#pdb.set_trace()
			for j in positions:
				for k in range(NnE):
					if e2vcg[j,k]!=i:
						connectivity.append(np.asarray([i,e2vcg[j,k]]))
		return torch.tensor(torch.from_numpy(np.asarray(connectivity).T).to('cuda'),dtype=torch.long)
	elif type=='iso':
		connectivity=[[i for i in range(NnG)],[i for i in range(NnG)]]
		return torch.tensor(torch.from_numpy(np.asarray(connectivity)).to('cuda'),dtype=torch.long)
	elif type=='eletruncate':
		connectivity=[]
		for i in range(NnG):
			positions=np.argwhere(e2vcg==i)[:,0]
			for j in positions:
				for k in range(NnE):
					if e2vcg[j,k]!=i:
						connectivity.append(np.asarray([i,e2vcg[j,k]]))
		return torch.tensor(torch.from_numpy(np.asarray(connectivity).T).to('cuda'),dtype=torch.long)

class Darcy_Chebnet(torch.nn.Module):
	def __init__(self,split):
		super(Darcy_Chebnet, self).__init__()
		nci=2;nco=1
		# nci = 3;nco = 1#验证后nci是（x,y）,并非（D,u,p）
		kk=10
		self.split=split
		# 共八层卷积
		#Darcy
		self.conv1 = ChebConv(nci, 32,K=kk)
		#in_channels 输入样本的通道数 out_channels 输出样本通道数 kk 切比雪夫普图卷积多项式阶数
		self.conv2 = ChebConv(32, 64,K=kk)
		self.conv3 = ChebConv(64, 128,K=kk)
		self.conv4 = ChebConv(128, 256,K=kk)
		self.conv5 = ChebConv(256, 128,K=kk)
		self.conv6 = ChebConv(128, 64,K=kk)
		self.conv7 = ChebConv(64, 32,K=kk)
		self.conv8 = ChebConv(32, nco,K=kk)

		try:
			self.conv1=last_chance0(self.conv1)
			self.conv2=last_chance0(self.conv2)
			self.conv3=last_chance0(self.conv3)
			self.conv4=last_chance0(self.conv4)
			self.conv5=last_chance0(self.conv5)
			self.conv6=last_chance0(self.conv6)
			self.conv7=last_chance0(self.conv7)
			self.conv8=last_chance1(self.conv8)


		except:
			torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv2.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv3.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv4.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv5.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv6.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv7.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv8.weight)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		n1=self.split[0]#Darcy域节点数
		n2=self.split[1]#Darcy域单元数
		idx1=[i for i in range(n1)]
		x1=x[idx1,:]
		edge_index1=edge_index[:,0:n2]
		
		x1 = self.conv1(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv2(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv3(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv4(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv5(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv6(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv7(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv8(x1, edge_index1)
		return torch.cat([x1],axis=0)#F.log_softmax(x, dim=1)

# 主要目的是对神经网络层的权重进行正交初始化
def last_chance0(maru):
	f91=torch.cat([dark.weight.T.unsqueeze(0) for dark in maru.lins],dim=0)
	f91 = torch.nn.init.orthogonal_(f91, torch.nn.init.calculate_gain('relu'))
	for i in range(len(maru.lins)):
		macsed= torch.nn.Parameter(f91[i,:,:].T)
		maru.lins[i].weight=macsed
	return maru

def last_chance1(maru):
	weights=torch.cat([dark.weight.T.unsqueeze(0) for dark in maru.lins],dim=0)
	weights = torch.nn.init.orthogonal_(weights)
	for i in range(len(maru.lins)):
		w_= torch.nn.Parameter(weights[i,:,:].T)
		maru.lins[i].weight=w_
	return maru