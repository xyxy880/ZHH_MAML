from torch import optim
from my_loss import *
from model.MTL_x4 import Mtl4
from copy import deepcopy

class Meta(nn.Module):

    def __init__(self,args):

        super(Meta, self).__init__()

        self.task_lr = args.task_lr

        self.spt = args.spt
        self.qry = args.qry

        self.update_step = args.update_step

        #模型
        self.net = Mtl4()
        # 加载预训练模型参数
        # self.net.load_state_dict(torch.load('./pth/171001.pth'))

        #损失函数
        self.loss_fn=My_loss()
        #元优化器
        self.meta_optim = optim.Adam(self.net.parameters(),lr=args.meta_lr)

        self.fine_net=None
        #15x3x40x30
    def forward(self, x_spt, y_spt, x_qry, y_qry,epoch):

        #拿来保存损失，第一个损失是元模型的，其他是元模型更新i次的loss
        loss_q = [0 for _ in range(self.update_step+1) ]

        pred_img=[]

        #模型参数更新一次
        pred = self.net(x_spt, vars=None)
        loss = self.loss_fn(pred, y_spt)
        grad = torch.autograd.grad(loss, self.net.parameters(),create_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad, self.net.parameters())))

        #拿元模型计算损失
        with torch.no_grad():
            pred_0 = self.net(x_qry, self.net.parameters())
            pred_img.append(pred_0)
            loss = self.loss_fn(pred_0, y_qry)
            loss_q[0]=loss
        #一次更新的loss
        pred_1 = self.net(x_qry, fast_weights)
        pred_img.append(pred_1)
        loss = self.loss_fn(pred_1, y_qry)
        loss_q[1] = loss

        # 2-5次更新
        for i in range(2,self.update_step+1):
            pred = self.net(x_spt, fast_weights)
            loss = self.loss_fn(pred, y_spt)
            grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad, fast_weights)))

            pred = self.net(x_qry, fast_weights)
            pred_img.append(pred)
            loss = self.loss_fn(pred, y_qry)
            loss_q[i] = loss

        #多步损失优化
        update_loss=0
        for idx,loss in enumerate(loss_q):
            update_loss+=loss*pow(0.1,idx)

        #梯度清零
        self.meta_optim.zero_grad()
        #loss回传，计算梯度
        update_loss.backward()
        #元模型更新一次
        self.meta_optim.step()

        #返回损失
        Loss = [loss.item() for loss in loss_q]
        #保存模型
        if epoch%3000==0:
            torch.save(self.net.state_dict(),'./pth/{}.pth'.format(epoch))
        return  Loss,pred_img


    def finetune(self, x_spt, y_spt, x_qry, y_qry):
        #模型
        # net=EDSR4()
        # net.load_state_dict(torch.load('../pth/0.pth'))
        net=deepcopy(self.net)
        # 拿来保存损失，第一个损失是元模型的，其他是元模型更新i次的loss
        #优化器，使用任务级别的学习率
        optimizer = optim.Adam(net.parameters(), lr=self.task_lr)
        #保存预测图
        for _ in range(1):
            pred=net(x_spt)
            loss=self.loss_fn(pred,y_spt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred=net(x_qry)
        loss=self.loss_fn(pred,y_qry)

        return loss.item(),pred



if __name__ == '__main__':
    import argparse
    import numpy as np

    torch.manual_seed(222)
    #为CPU设置种子用于生成随机数，以使得结果是确定的。
    torch.cuda.manual_seed_all(222)
    #为GPU设置种子用于生成随机数，以使得结果是确定的。
    np.random.seed(222)
    #用于生成指定的随机数

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dir1', type=str, help='root', default='../../data/tian_x4/fine')
    argparser.add_argument('--dir2s', type=list, default=['HR', 'LR'])
    argparser.add_argument('--dir3s', type=list, default=['GFS025', 'GFS050', 'era5_2020', 'era5_2000'])
    argparser.add_argument('--dir4s', type=list, default=['u10', 'v10', 'd2m', 'sp', 't2m', 'tp'])

    argparser.add_argument('--train', type=int, help='train number', default=12)
    argparser.add_argument('--test', type=int, help='test number', default=8)
    argparser.add_argument('--spt', type=int, help='k shot for support set', default=7)
    argparser.add_argument('--qry', type=int, help='k shot for query set', default=5)

    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-5)
    argparser.add_argument('--task_lr', type=float, help='task-level inner update learning rate', default=1e-4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)

    argparser.add_argument('--env', type=str, help='visdom windows title', default='ee')
    args = argparser.parse_args()

    #support set
    x_spt=torch.rand(10,3,20,20)
    y_spt=torch.rand(10,3,80,80)

    #query set
    x_qry=torch.rand(20,3,20,20)
    y_qry=torch.rand(20,3,80,80)
    maml=Meta(args)
    loss,pred_img=maml(x_spt,y_spt,x_qry,y_qry,0)
    print(loss)
    #support set
    x_spt=torch.rand(10,3,30,30)
    y_spt=torch.rand(10,3,120,120)

    #query set
    x_qry=torch.rand(20,3,30,30)
    y_qry=torch.rand(20,3,120,120)
    loss,pred = maml.finetune(x_spt, y_spt, x_qry, y_qry)
    print(loss)

#[1.2051810026168823, 1.2040518522262573]
# tensor(0.4061, grad_fn=<MeanBackward0>)