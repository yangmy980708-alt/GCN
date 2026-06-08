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
	



class Ns_Darcy_Chebnet(torch.nn.Module):
	def __init__(self, split, hidden_channels=32, num_layers=6, heads=1):
		"""
        多领域GAT网络
        参数:
            split: 各领域节点划分
            hidden_channels: 隐藏层通道数
            num_layers: 每领域网络层数
            heads: GAT注意力头数
        """
		super(Ns_Darcy_Chebnet, self).__init__()
		nci, nco = 2, 1
		# nci, nco = 3, 1#输入通道数为3报错
		self.split = split

		# 创建四个领域网络
		self.darcy_net = self._build_network(nci, nco, hidden_channels, num_layers, heads)
		self.ns_u_net = self._build_network(nci, nco, hidden_channels, num_layers, heads)
		self.ns_v_net = self._build_network(nci, nco, hidden_channels, num_layers, heads)
		self.ns_p_net = self._build_network(nci, nco, hidden_channels, num_layers, heads)

		# 初始化权重
		self._initialize_weights()

	def _build_network(self, in_channels, out_channels, hidden_channels, num_layers, heads):
		"""构建GAT网络块"""
		layers = torch.nn.ModuleList()
		# 编码部分
		for i in range(num_layers // 2):
			in_dim = in_channels if i == 0 else hidden_channels * (2 ** (i - 1))
			out_dim = hidden_channels * (2 ** i)
			layers.append(GATConv(in_dim, out_dim, heads=heads))

		# 解码部分
		for i in range(num_layers // 2, num_layers - 1):
			in_dim = hidden_channels * (2 ** (num_layers - i - 1))
			out_dim = hidden_channels * (2 ** (num_layers - i - 2))
			layers.append(GATConv(in_dim, out_dim, heads=heads))

		# 最后一层
		layers.append(GATConv(hidden_channels, out_channels, heads=1))
		return layers

	def _initialize_weights(self):
		"""初始化网络权重"""
		for net in [self.darcy_net, self.ns_u_net, self.ns_v_net, self.ns_p_net]:
			for i, layer in enumerate(net):
				if i < len(net) - 1:
					torch.nn.init.orthogonal_(layer.lin_src.weight, gain=torch.nn.init.calculate_gain('relu'))
					torch.nn.init.orthogonal_(layer.lin_dst.weight, gain=torch.nn.init.calculate_gain('relu'))
				else:
					torch.nn.init.orthogonal_(layer.lin_src.weight)
					torch.nn.init.orthogonal_(layer.lin_dst.weight)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		n1, n2, n3, n4, n5 = self.split

		# 分割输入数据
		x1 = x[:n1, :]  # Darcy
		x2 = x[n1:n1 + n3, :]  # NS U
		x3 = x[n1 + n3:n1 + 2 * n3, :]  # NS V
		x4 = x[n1 + 2 * n3:n1 + 2 * n3 + n4, :]  # NS P

		edge_index1 = edge_index[:, :n2]  # Darcy
		edge_index2 = edge_index[:, n2:n2 + n5]  # NS U
		edge_index3 = edge_index[:, n2 + n5:n2 + 2 * n5]  # NS V
		edge_index4 = edge_index[:, n2 + 2 * n5:n2 + 3 * n5]  # NS P

		# 处理各领域
		x1 = self._process_domain(x1, edge_index1, self.darcy_net)
		x2 = self._process_domain(x2, edge_index2, self.ns_u_net)
		x3 = self._process_domain(x3, edge_index3, self.ns_v_net)
		x4 = self._process_domain(x4, edge_index4, self.ns_p_net)

		# 合并结果
		uv = torch.cat([x2, x3], dim=0)
		return torch.cat([x1, uv, x4], dim=0)

	def _process_domain(self, x, edge_index, layers):
		"""处理单个领域的前向传播"""
		for i, layer in enumerate(layers[:-1]):
			x = layer(x, edge_index)
			x = F.relu(x)
		x = layers[-1](x, edge_index)
		return x