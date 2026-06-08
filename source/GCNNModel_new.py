import torch
import torch.nn.functional as F
import pdb
from torch_geometric.nn import GCNConv, ChebConv, GATConv, TransformerConv, TAGConv, ARMAConv, SGConv,MFConv, RGCNConv
from torch_geometric.data import InMemoryDataset
import numpy as np

def e2vcg2connectivity(e2vcg,type='iso'):
	"""
	e2vcg should be in np.array
	"""
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
	


'''
1. 改变卷积层数
2. 改变GCN结构 例如带有残差连接(residual connections)
3. 改变样本 输入/出 通道数
4. 改变激活函数
'''
class Ns_Chebnet_new(torch.nn.Module):
	def __init__(self,split):
		super(Ns_Chebnet_new, self).__init__()
		nci=2;nco=1
		kk=10
		self.split=split
		# 共八层卷积
		self.conv1 = ChebConv(nci, 8,K=kk)
		#in_channels 输入样本的通道数 out_channels 输出样本通道数 kk 切比雪夫普图卷积多项式阶数
		self.conv2 = ChebConv(8, 16,K=kk)
		self.conv3 = ChebConv(16, 32,K=kk)
		self.conv4 = ChebConv(32, 16,K=kk)
		#16，8
		self.conv5_FC = ChebConv(16,nco,K=kk)
		# self.conv5 = ChebConv(256, 128,K=kk)
		# self.conv6 = ChebConv(128, 64,K=kk)
		# self.conv7 = ChebConv(64, 32,K=kk)
		# self.conv8 = ChebConv(32, nco,K=kk)

		self.conv11 = ChebConv(nci, 8,K=kk)
		self.conv22 = ChebConv(8, 16,K=kk)
		self.conv33 = ChebConv(16, 32,K=kk)
		self.conv44 = ChebConv(32, 16,K=kk)
		self.conv55_FC = ChebConv(16,nco,K=kk)
		# self.conv55 = ChebConv(256, 128,K=kk)
		# self.conv66 = ChebConv(128, 64,K=kk)
		# self.conv77 = ChebConv(64, 32,K=kk)
		# self.conv88 = ChebConv(32, nco,K=kk)

		self.conv111 = ChebConv(nci, 8, K=kk)
		self.conv222 = ChebConv(8, 16, K=kk)
		self.conv333 = ChebConv(16, 32, K=kk)
		self.conv444 = ChebConv(32, 16, K=kk)
		self.conv555_FC = ChebConv(16, nco, K=kk)


		try:
			self.conv1=last_chance0(self.conv1)
			self.conv2=last_chance0(self.conv2)
			self.conv3=last_chance0(self.conv3)
			self.conv4=last_chance0(self.conv4)
			self.conv5_FC = last_chance1(self.conv5_FC)

			self.conv11=last_chance0(self.conv11)
			self.conv22=last_chance0(self.conv22)
			self.conv33=last_chance0(self.conv33)
			self.conv44=last_chance0(self.conv44)
			self.conv55_FC = last_chance1(self.conv55_FC)

			self.conv111=last_chance0(self.conv111)
			self.conv222=last_chance0(self.conv222)
			self.conv333=last_chance0(self.conv333)
			self.conv444=last_chance0(self.conv444)
			self.conv555_FC = last_chance1(self.conv555_FC)
		except:
			torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv2.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv3.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv4.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv5_FC.weight)

			torch.nn.init.orthogonal_(self.conv11.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv22.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv33.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv44.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv55_FC.weight)

			torch.nn.init.orthogonal_(self.conv111.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv222.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv333.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv444.weight, torch.nn.init.calculate_gain('relu'))
			torch.nn.init.orthogonal_(self.conv555_FC.weight)
		#torch.nn.init.orthogonal_(self.conv4.weight)
		#torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
		#torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
	"""
	try except 用于代码的异常处理
	try:
    <代码块1>
    except:
    <代码块2>
    <代码块1>中一般是一些容易“报错”的代码，如果<代码块1>中的代码能够正常运行，那么程序就会逃过<代码块2>去执行后续的其他代码；
    而<代码块2>中的代码一般是<代码块1>报错时的应对措施。
	"""

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		'''split = [xcg.shape[1], msh_.xcg.shape[1], connectivity_uv.shape[1]]'''
		n1=self.split[0] # xcg.shape[1] 速度场节点个数
		n2=self.split[1] # msh_.xcg.shape[1] 压力场节点个数
		n3=self.split[2] # connectivity_uv.shape[1] = 2* (2*n1+n2) 例如：2剖分 n3 = 108
		idx1=[2*i for i in range(n1)]
		idx2=[2*i+1 for i in range(n1)]
		idx3=[i+n1*2 for i in range(n2)]
		x1=x[idx1,:]
		x2=x[idx2,:]
		x3=x[idx3,:]
		edge_index1=edge_index[:,0:n3]
		edge_index2=edge_index[:,n3:2*n3]
		edge_index3=edge_index[:,2*n3:]
		
		x11 = self.conv1(x1, edge_index1)
		x11 = F.relu(x11)
		x12 = self.conv2(x11, edge_index1)
		x12 = F.relu(x12)
		x13 = self.conv3(x12, edge_index1)
		x13 = F.relu(x13)
		x14 = self.conv4(x13, edge_index1)
		x14 = F.relu(x14)
		# x15 = self.conv5(x14, edge_index1)
		# x15 = F.relu(x15)
		# x16 = self.conv6(x15, edge_index1)
		# x16 = F.relu(x16)
		# x17 = self.conv7(x16, edge_index1)
		# x17 = F.relu(x17)
		# x18 = self.conv8(x17, edge_index1)
		xx1 = x1 + x11 + x12 + x13 + x14 # 会出现的问题 这里的维度不同
		x1 = self.conv5_FC(xx1, edge_index1) # 疑惑 这里返回的是x1吗？


		x21 = self.conv11(x2, edge_index2)
		x21 = F.relu(x21)
		x22 = self.conv22(x21, edge_index2)
		x22 = F.relu(x22)
		x23 = self.conv33(x22, edge_index2)
		x23 = F.relu(x23)
		x24 = self.conv44(x23, edge_index2)
		x24 = F.relu(x24)
		# x25 = self.conv55(x24, edge_index2)
		# x25 = F.relu(x25)
		# x26 = self.conv66(x25, edge_index2)
		# x26 = F.relu(x26)
		# x27 = self.conv77(x26, edge_index2)
		# x27 = F.relu(x27)
		# x28 = self.conv88(x27, edge_index2)
		xx2 = x2 + x21 + x22 + x23 + x24
		x2 = self.conv55_FC(xx2, edge_index2)

		x31 = self.conv111(x3, edge_index3)
		x31 = F.relu(x31)
		x32 = self.conv222(x31, edge_index3)
		x32 = F.relu(x32)
		x33 = self.conv333(x32, edge_index3)
		x33 = F.relu(x33)
		x34 = self.conv444(x33, edge_index3)
		x34 = F.relu(x34)
		# x35 = self.conv555(x34, edge_index3)
		# x35 = F.relu(x35)
		# x36 = self.conv666(x35, edge_index3)
		# x36 = F.relu(x36)
		# x37 = self.conv777(x36, edge_index3)
		# x37 = F.relu(x37)
		# x38 = self.conv888(x37, edge_index3)
		xx3 = x3+ x31 + x32 + x33 +x34
		x3 = self.conv555_FC(xx3, edge_index3)

		uv=[]
		for i in range(n1):
			uv.append(torch.cat([x1[i:i+1,0:],x2[i:i+1,0:]],axis=0))
		uv_=torch.cat(uv,axis=0)
		return torch.cat([uv_,x3],axis=0)#F.log_softmax(x, dim=1)


def last_chance0(maru): #这里的 maru 对应卷积层conv
	f91=torch.cat([dark.weight.T.unsqueeze(0) for dark in maru.lins],dim=0)
	"""
	这一行使用列表推导式从 maru.lins 中提取所有线性层 (dark)
	对每个线性层的权重进行转置 (dark.weight.T) 并增加一个维度 (unsqueeze(0))
	然后，使用 torch.cat 在第一个维度上将所有这些权重张量连接起来，形成一个新的张量 f91
	"""
	f91 = torch.nn.init.orthogonal_(f91, torch.nn.init.calculate_gain('relu'))
	"""
	这行代码对 f91 进行正交初始化
	torch.nn.init.calculate_gain('relu') 计算用于 ReLU 激活函数的增益。
	正交初始化可以帮助模型在训练初期保持良好的信号传递，从而促进更快的收敛。
	"""
	for i in range(len(maru.lins)):
		macsed= torch.nn.Parameter(f91[i,:,:].T)
		"""
		这个循环遍历 maru.lins 中的每一个线性层
		取出 f91 张量的第 i 个切片，转置后将其转换为 torch.nn.Parameter
		然后将其重新赋值给对应线性层的权重( 对应下一行maru.lins[i].weight=macsed)
		这样实现了对每个线性层权重的设置。
		"""
		"""
		首先可以把这个函数理解为类型转换函数
		将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
		(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)
		所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
		使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
		"""
		maru.lins[i].weight=macsed
	return maru

def last_chance1(maru): # 与last_chance0 用法相同
	weights=torch.cat([dark.weight.T.unsqueeze(0) for dark in maru.lins],dim=0)
	weights = torch.nn.init.orthogonal_(weights)
	for i in range(len(maru.lins)):
		w_= torch.nn.Parameter(weights[i,:,:].T)
		maru.lins[i].weight=w_
	return maru