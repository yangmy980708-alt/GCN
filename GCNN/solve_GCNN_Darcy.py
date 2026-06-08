import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import sparse
import torch
import time
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../GCNN')
from GCNN.basis_Ab import Double,ReshapeFix
def solve_GCNN_Darcy(number_of_unknowns_Darcy, number_of_FE_nodes_u, number_of_FE_nodes_p,DataLoader,LossF,model,tol=1e-3,maxit=2000,qoiidx=None,softidx=None,penaltyConstant=None):
	"""Wrapper"""
	startime=time.time()
	model,Loss,Er_D,Erlist_D,erlist_D= \
        solve_SGD(number_of_unknowns_Darcy, number_of_FE_nodes_u, number_of_FE_nodes_p,
                  DataLoader, LossF, model, tol, maxit, qoiidx, softidx, penaltyConstant, plotFlag=True)
	print('wallclock time of all epochs = ',time.time()-startime)
    #Er_D等是储存了平均相对误差的列表
    #Elist_u等是储存了所有的相对误差的列表
	return model,Loss,Er_D,Erlist_D,erlist_D


def solve_SGD(number_of_unknowns_Darcy, number_of_FE_nodes_u, number_of_FE_nodes_p,
              DataLoader, LossF, model, tol, maxit, qoiidx, softidx, penaltyConstant, plotFlag=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # 初始化存储变量
    Er_D = []; Loss = []; Erlist_D = []

    for epoch in range(maxit):
        print('epoch = ', epoch)
        startime = time.time()

        # 训练模型
        erD_0, loss, model, erlist_D = \
            trainmodel(number_of_unknowns_Darcy, number_of_FE_nodes_u, number_of_FE_nodes_p,
                      DataLoader, LossF, model, optimizer, criterion, qoiidx, softidx, penaltyConstant)
        print('Darcy Error = ', erD_0)
        print('Loss = ', loss)
        print('Wallclock time of this epoch = ', time.time() - startime)

        # 记录当前结果

        Er_D.append(erD_0); Loss.append(loss)
        Erlist_D.append(erlist_D)

        # 仅当 loss < tol 时停止训练
        if loss < tol:
            print(f'Training stopped: loss < {tol} reached!')
            break  # 直接退出训练循环

        # 可选：定期保存模型（例如每 100 个 epoch）
        if epoch % 100 == 0:
            torch.save(model, f'./Checkpoint_Epoch_{epoch}.pth')

    # 训练结束后保存最终模型和结果
    torch.save(model, './FinalModel.pth')
    np.savetxt('./FinalLoss.txt', np.asarray(Loss))
    # if plotFlag:
    #         fig = plt.figure()
    #         ax = plt.subplot(1, 1, 1)
    #         ax.plot(Er_D, label='Relative Error Darcy')
    #         ax.plot(Er_u, label='Relative Error U')  # u相对误差
    #         ax.plot(Er_p, label='Relative Error P')
    #         ax.plot(Loss, label='Loss')  # 损失
    #         ax.legend()
    #         ax.set_xlabel('Epoch')
    #         ax.set_yscale('log')
    #         fig.savefig('./LossResidual_total.png', bbox_inches='tight')  # 绘制折线图
    #         plt.show()
    #
    # if plotFlag:
    #         fig = plt.figure()
    #         ax = plt.subplot(1, 1, 1)
    #         ax.plot(Er_D, label='Relative Error Darcy')
    #         ax.plot(Er_u1, label='Relative Error U1')  # u1相对误差
    #         ax.plot(Er_u2, label='Relative Error U2')  # u2相对误差
    #         ax.plot(Er_p, label='Relative Error P')
    #         ax.plot(Loss, label='Loss')  # 损失
    #         ax.legend()
    #         ax.set_xlabel('Epoch')
    #         ax.set_yscale('log')
    #         fig.savefig('./LossResidual.png', bbox_inches='tight')  # 绘制折线图
    #         plt.show()
    return model,Loss,Er_D,Erlist_D,erlist_D




def trainmodel(number_of_unknowns_Darcy, number_of_FE_nodes_u, number_of_FE_nodes_p,
               DataLoader, LossF, model, optimizer, criterion, qoiidx, softidx, penaltyConstant):
    # 设置模型为训练模式
    model.train()
    erD_0 = 0;
    erlist_D = []  # 新增 Darcy 系统的误差记录
    ReList = []

    # 清空梯度
    optimizer.zero_grad()

    # 遍历数据加载器中的每个 batch
    for data in DataLoader:
        # 将输入数据移动到 GPU
        input = data[0].to('cuda')

        # 提取真实值和损失函数 ID
        fcn_id = data[0].y[0, 0]
        truth = data[0].y[1:, 0:]

        # 选择损失函数
        fcn = LossF[int(fcn_id)]
        assert (int(fcn_id) - fcn_id) ** 2 < 1e-12, 'The loss function is selected right!'

        # 前向传播
        tic = time.time()
        output = model(input)
        Re, boundary_nodes_D, r_D_boundary, r_Stokes_boundary= fcn(output)
        print('wallclock time of evl Res= ', time.time() - tic)

        # 存储残差
        ReList.append(torch.abs(Re))

        # 处理解
        solution = ReshapeFix(torch.clone(output), [len(output.flatten()), 1], 'C')
        #边界处理还需要
        solution[boundary_nodes_D[1, :].to(torch.long),:] = r_D_boundary
        # solution[boundary_nodes_S[2, :].to(torch.long),:] = r_Stokes_boundary

        # 划分解为 Darcy 部分、NS 的 u1 和 u2 部分、压力 p 部分
        solution_D = solution[:number_of_unknowns_Darcy]  # Darcy 部分

        # 提取真实值的对应部分
        truth_D = truth[:number_of_unknowns_Darcy]
        # 计算 Darcy 部分的相对误差
        erD_0 += torch.sqrt(criterion(solution_D, truth_D) / criterion(truth_D, truth_D * 0)).item()
        erlist_D.append(torch.sqrt(criterion(solution_D, truth_D) / criterion(truth_D, truth_D * 0)).item())

    # 计算总损失
    loss = ReList[0] * 0
    for i in range(len(ReList)):
        loss = loss + ReList[i]
    loss = torch.norm(loss)

    # 处理软约束
    if softidx is not None and penaltyConstant is not None:
        print('DataLoss = ', criterion(solution[softidx], truth[softidx]) * penaltyConstant)
        loss = criterion(solution[softidx], truth[softidx]) * penaltyConstant + loss

    # 处理量化指标
    if qoiidx is not None:
        QOI_ER = torch.sqrt(
            criterion(solution[qoiidx], truth[qoiidx]) / criterion(truth[qoiidx], truth[qoiidx] * 0)).item()
        print('QOI Error=', QOI_ER)
        os.system("touch QOIError.txt")
        os.system("touch QOIValue.txt")
        file1 = open("QOIError.txt", "a")
        file1.writelines(str(QOI_ER) + "\n")
        file2 = open("QOIValue.txt", "a")
        file2.writelines(str(solution[qoiidx].detach().cpu().numpy().reshape([1, -1])[:]) + "\n")
        file1.close()
        file2.close()

    # 反向传播
    # tic = time.time()
    # loss.backward()
    # print('wallclock time of this BP= ', time.time() - tic)

    # 参数更新
    optimizer.step()

    print('>>>>>>>max error D<<<<<<< ====================================', max(erlist_D))  # 新增 Darcy 系统的误差打印
    try:
        print('>>>>>>>model source<<<<<<< =======================', model.source)
        os.system("touch ModelSource.txt")
        file3 = open("ModelSource.txt", "a")
        object2write = model.source.detach().cpu().numpy().reshape([1, -1])
        for ifer in range(2):
            try:
                file3.writelines(str(object2write[0, ifer]) + "\n")
            except:
                pass
        file3.close()
    except:
        pass

    # 返回结果
    #eru_0 / len(DataLoader)表示输出平均相对误差
    #loss.norm().item() / len(DataLoader)每次训练的平均损失
    return erD_0 / len(DataLoader), loss.norm().item() / len(DataLoader),model, erlist_D