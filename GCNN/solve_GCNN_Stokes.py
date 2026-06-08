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

#solve_GCNN(number_of_unknowns_Darcy,number_of_FE_nodes_u,
                                             # number_of_FE_nodes_p,TrainDataloader,
                                             # LossF, model, tol, maxit)
def solve_GCNN_Stokes(number_of_unknowns_Darcy, number_of_FE_nodes_u, number_of_FE_nodes_p,DataLoader,LossF,model,tol=1e-3,maxit=2000,qoiidx=None,softidx=None,penaltyConstant=None):
	"""Wrapper"""
	startime=time.time()
	model,Loss,Er_u,Er_u1,Er_u2,Er_p,Erlist_u,Erlist_u1,Erlist_u2,Erlist_p,erlist_u, erlist_u1, erlist_u2, erlist_p, = \
        solve_SGD(number_of_unknowns_Darcy, number_of_FE_nodes_u, number_of_FE_nodes_p,
                  DataLoader, LossF, model, tol, maxit, qoiidx, softidx, penaltyConstant, plotFlag=True)
	print('wallclock time of all epochs = ',time.time()-startime)
    #Er_D等是储存了平均相对误差的列表
    #Elist_u等是储存了所有的相对误差的列表
	return model,Loss,Er_u,Er_u1,Er_u2,Er_p,Erlist_u,Erlist_u1,Erlist_u2,Erlist_p,erlist_u, erlist_u1, erlist_u2, erlist_p


def solve_SGD(number_of_unknowns_Darcy, number_of_FE_nodes_u, number_of_FE_nodes_p,
              DataLoader, LossF, model, tol, maxit, qoiidx, softidx, penaltyConstant, plotFlag=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # 初始化存储变量
    Er_u = []; Er_u1 = []; Er_u2 = []; Er_p = []; Loss = []
    Erlist_u = []; Erlist_u1 = []; Erlist_u2 = []; Erlist_p = [];

    for epoch in range(maxit):
        print('epoch = ', epoch)
        startime = time.time()

        # 训练模型
        eru_0, eru1_0, eru2_0, erp_0, loss, model, erlist_u, erlist_u1, erlist_u2, erlist_p = \
            trainmodel(number_of_unknowns_Darcy, number_of_FE_nodes_u, number_of_FE_nodes_p,
                      DataLoader, LossF, model, optimizer, criterion, qoiidx, softidx, penaltyConstant)
        print('NS Velocity Error = ', eru_0)
        print('NS u1 Velocity Error = ', eru1_0)
        print('NS u2 Velocity Error = ', eru2_0)
        print('NS Pressure Error = ', erp_0)
        print('Loss = ', loss)
        print('Wallclock time of this epoch = ', time.time() - startime)

        # 记录当前结果
        Er_u.append(eru_0); Er_u1.append(eru1_0); Er_u2.append(eru2_0)
        Er_p.append(erp_0); Loss.append(loss)
        Erlist_u.append(erlist_u); Erlist_u1.append(erlist_u1); Erlist_u2.append(erlist_u2)
        Erlist_p.append(erlist_p);

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
    np.savetxt('./FinalEr_u.txt', np.asarray(Er_u))
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
    return model,Loss,Er_u,Er_u1,Er_u2,Er_p,Erlist_u,Erlist_u1,Erlist_u2,Erlist_p,erlist_u, erlist_u1, erlist_u2, erlist_p




def trainmodel(number_of_unknowns_Darcy, number_of_FE_nodes_u, number_of_FE_nodes_p,
               DataLoader, LossF, model, optimizer, criterion, qoiidx, softidx, penaltyConstant):
    # 设置模型为训练模式
    model.train()
    eru_0 = 0;
    eru1_0 = 0;
    eru2_0 = 0;
    erp_0 = 0;
    loss_0 = 0
    erlist_u = [];
    erlist_u1 = [];
    erlist_u2 = [];
    erlist_p = [];
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
        Re, boundary_nodes_S,r_D_boundary, r_Stokes_boundary= fcn(output)
        print('wallclock time of evl Res= ', time.time() - tic)

        # 存储残差
        ReList.append(torch.abs(Re))

        # 处理解
        solution = ReshapeFix(torch.clone(output), [len(output.flatten()), 1], 'C')
        #边界处理还需要
        solution[boundary_nodes_S[2, :].to(torch.long),:] = r_Stokes_boundary

        # 划分解为NS 的 u1 和 u2 部分、压力 p 部分
        solution_u1 = solution[:number_of_FE_nodes_u]  # NS 的 u1 部分
        solution_u2 = solution[number_of_FE_nodes_u: 2 * number_of_FE_nodes_u]  # NS 的 u2 部分
        solution_p = solution[2 * number_of_FE_nodes_u: 2 * number_of_FE_nodes_u + number_of_FE_nodes_p]  # 压力 p 部分
        solution_u = torch.sqrt(solution_u1**2 + solution_u2**2)

        # 提取真实值的对应部分
        truth_u1 = truth[:number_of_FE_nodes_u]
        truth_u2 = truth[number_of_FE_nodes_u:2 * number_of_FE_nodes_u]
        truth_p = truth[2 * number_of_FE_nodes_u: 2 * number_of_FE_nodes_u + number_of_FE_nodes_p]
        truth_u = torch.sqrt(truth_u1**2 + truth_u2**2)

        # 计算 NS 的 u1 和 u2 部分的相对误差
        eru1_0 += torch.sqrt(criterion(solution_u1, truth_u1) / criterion(truth_u1, truth_u1 * 0)).item()
        erlist_u1.append(torch.sqrt(criterion(solution_u1, truth_u1) / criterion(truth_u1, truth_u1 * 0)).item())
        eru2_0 += torch.sqrt(criterion(solution_u2, truth_u2) / criterion(truth_u2, truth_u2 * 0)).item()
        erlist_u2.append(torch.sqrt(criterion(solution_u2, truth_u2) / criterion(truth_u2, truth_u2 * 0)).item())

        eru_0 += torch.sqrt(criterion(solution_u, truth_u) / criterion(truth_u, truth_u * 0)).item()
        erlist_u.append(torch.sqrt(criterion(solution_u, truth_u) / criterion(truth_u, truth_u * 0)).item())

        # 计算压力 p 部分的相对误差
        erp_0 += torch.sqrt(criterion(solution_p, truth_p) / criterion(truth_p, truth_p * 0)).item()
        erlist_p.append(torch.sqrt(criterion(solution_p, truth_p) / criterion(truth_p, truth_p * 0)).item())

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
    tic = time.time()
    loss.backward()
    print('wallclock time of this BP= ', time.time() - tic)

    # 参数更新
    optimizer.step()

    # 打印最大误差
    print('>>>>>>>max error u<<<<<<< ====================================', max(erlist_u))
    print('>>>>>>>max error u1<<<<<<< ====================================', max(erlist_u1))
    print('>>>>>>>max error u2<<<<<<< ====================================', max(erlist_u2))
    print('>>>>>>>max error p<<<<<<< ====================================', max(erlist_p))
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
    return eru_0 / len(DataLoader),eru1_0 / len(DataLoader),eru2_0 / len(DataLoader), erp_0 / len(DataLoader), loss.norm().item() / len(DataLoader),model, erlist_u, erlist_u1,erlist_u2,erlist_p