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

class NS_Chebnet(torch.nn.Module):
	def __init__(self,split):
		super(NS_Chebnet, self).__init__()
		nci=2;nco=1
		kk=10
		self.split=split
		# 共八层卷积
		#NS
		self.conv11 = ChebConv(nci, 32, K=kk)
		self.conv22 = ChebConv(32, 64, K=kk)
		self.conv33 = ChebConv(64, 128, K=kk)
		self.conv44 = ChebConv(128, 256, K=kk)
		self.conv55 = ChebConv(256, 128, K=kk)
		self.conv66 = ChebConv(128, 64, K=kk)
		self.conv77 = ChebConv(64, 32, K=kk)
		self.conv88 = ChebConv(32, nco, K=kk)

		self.conv111 = ChebConv(nci, 32, K=kk)
		self.conv222 = ChebConv(32, 64, K=kk)
		self.conv333 = ChebConv(64, 128, K=kk)
		self.conv444 = ChebConv(128, 256, K=kk)
		self.conv555 = ChebConv(256, 128, K=kk)
		self.conv666 = ChebConv(128, 64, K=kk)
		self.conv777 = ChebConv(64, 32, K=kk)
		self.conv888 = ChebConv(32, nco, K=kk)

		self.conv1111 = ChebConv(nci, 32, K=kk)
		self.conv2222 = ChebConv(32, 64, K=kk)
		self.conv3333 = ChebConv(64, 128, K=kk)
		self.conv4444 = ChebConv(128, 256, K=kk)
		self.conv5555 = ChebConv(256, 128, K=kk)
		self.conv6666 = ChebConv(128, 64, K=kk)
		self.conv7777 = ChebConv(64, 32, K=kk)
		self.conv8888 = ChebConv(32, nco, K=kk)


		try:
			self.conv11 = last_chance0(self.conv11)
			self.conv22 = last_chance0(self.conv22)
			self.conv33 = last_chance0(self.conv33)
			self.conv44 = last_chance0(self.conv44)
			self.conv55 = last_chance0(self.conv55)
			self.conv66 = last_chance0(self.conv66)
			self.conv77 = last_chance0(self.conv77)
			self.conv88 = last_chance1(self.conv88)

			self.conv111 = last_chance0(self.conv111)
			self.conv222 = last_chance0(self.conv222)
			self.conv333 = last_chance0(self.conv333)
			self.conv444 = last_chance0(self.conv444)
			self.conv555 = last_chance0(self.conv555)
			self.conv666 = last_chance0(self.conv666)
			self.conv777 = last_chance0(self.conv777)
			self.conv888 = last_chance1(self.conv888)

			self.conv1111 = last_chance0(self.conv1111)
			self.conv2222 = last_chance0(self.conv2222)
			self.conv3333 = last_chance0(self.conv3333)
			self.conv4444 = last_chance0(self.conv4444)
			self.conv5555 = last_chance0(self.conv5555)
			self.conv6666 = last_chance0(self.conv6666)
			self.conv7777 = last_chance0(self.conv7777)
			self.conv8888 = last_chance1(self.conv8888)
		except:

			torch.nn.init.orthogonal_(self.conv11.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv22.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv33.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv44.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv55.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv66.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv77.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv88.weight)

			torch.nn.init.orthogonal_(self.conv111.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv222.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv333.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv444.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv555.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv666.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv777.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv888.weight)

			torch.nn.init.orthogonal_(self.conv1111.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv2222.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv3333.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv4444.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv5555.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv6666.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv7777.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv8888.weight)
	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		n1=self.split[0]#NS_U 节点数
		n2=self.split[1]#NS_p节点数
		n3=self.split[2]#NS单元数

		idx1 = [2 * i for i in range(n1)]
		idx2 = [2 * i + 1 for i in range(n1)]
		idx3 = [i + n1 * 2 for i in range(n2)]
		x2 = x[idx1, :]
		x3 = x[idx2, :]
		x4 = x[idx3, :]
		edge_index2 = edge_index[:, 0:n3]
		edge_index3 = edge_index[:, n3:2 * n3]
		edge_index4 = edge_index[:, 2 * n3:]

		x2 = self.conv11(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv22(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv33(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv44(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv55(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv66(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv77(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv88(x2, edge_index2)

		x3 = self.conv111(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv222(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv333(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv444(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv555(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv666(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv777(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv888(x3, edge_index3)

		x4 = self.conv1111(x4, edge_index4)
		x4 = F.relu(x4)
		x4 = self.conv2222(x4, edge_index4)
		x4 = F.relu(x4)
		x4 = self.conv3333(x4, edge_index4)
		x4 = F.relu(x4)
		x4 = self.conv4444(x4, edge_index4)
		x4 = F.relu(x4)
		x4 = self.conv5555(x4, edge_index4)
		x4 = F.relu(x4)
		x4 = self.conv6666(x4, edge_index4)
		x4 = F.relu(x4)
		x4 = self.conv7777(x4, edge_index4)
		x4 = F.relu(x4)
		x4 = self.conv8888(x4, edge_index4)

		uv = torch.cat([x2, x3], axis=0)
		# for i in range(n3):
		# 	# uv.append(torch.cat([x3[i:i+1,0:],x3[i:i+1,0:]],axis=0))
		# 	uv.append(torch.cat([x2[i,:], x3[i,:], axis=0))
		# uv_=torch.cat(uv,axis=0)
		return torch.cat([uv, x4], axis=0)

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