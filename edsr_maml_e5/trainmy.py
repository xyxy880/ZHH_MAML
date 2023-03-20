import argparse
import numpy as np
import  torch
import visdom


from loadermy import Mydataloader
from meta_meta import Meta
from show_img import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
def main(args):

    torch.manual_seed(222)
    #为CPU设置种子用于生成随机数，以使得结果是确定的。
    torch.cuda.manual_seed_all(222)
    #为GPU设置种子用于生成随机数，以使得结果是确定的。
    np.random.seed(222)
    #用于生成指定的随机数

    device = torch.device(args.cuda)

    #定义元学习
    maml = Meta(args).to(device)

    #filter()函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
    #接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回True
    #和False，最后将返回True的元素放到新列表中。
    #.requires_grad  表明当前变量是否需要在计算中保留对应的梯度信息。
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    #map函数的意思就是将function（第一个参数）应用于iterable（第二个参数）的每一个元素，结果以列表的形式返回。
    #x.shape就是变量x的大小（比如3x3的矩阵）
    #np.prod() 计算所有元素的乘积
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    #打印总共需要训练多少个模型参数
    print('Total trainable tensors:', num)

    viz = visdom.Visdom(env=args.env)

    x=[]
    y=[]
    x_test=[]
    y_test=[]

    perm = np.random.permutation(args.train + args.test)

    #定义训练加载器
    db = {}
    for dir2 in args.dir2s:
        db[dir2] = {}
        for dir3 in args.dir3s:
            db[dir2][dir3] = {}
            for dir4 in args.dir4s:
                db[dir2][dir3][dir4] = Mydataloader('{}/{}/{}/{}'.format(args.dir1,dir2,dir3,dir4), perm, 'train',args)
    # 定义测试加载器
    db1 = {}
    for dir2 in args.dir2s:
        db1[dir2] = {}
        for dir3 in args.dir3s:
            db1[dir2][dir3] = {}
            for dir4 in args.dir4s:
                db1[dir2][dir3][dir4] = Mydataloader('{}/{}/{}/{}'.format(args.dir1,dir2,dir3,dir4), perm, 'test',args)

    t_num=0
    t_test = 0
    for epoch in range(args.epoches):
        for i in np.random.permutation(24):
            if i<6:
                x_spt,x_qry=db['LR']['GFS025'][args.dir4s[i]].next()
                y_spt,y_qry=db['HR']['GFS025'][args.dir4s[i]].next()
            elif i<12:
                x_spt, x_qry = db['LR']['GFS050'][args.dir4s[i-6]].next()
                y_spt, y_qry = db['HR']['GFS050'][args.dir4s[i-6]].next()
            elif i<18:
                x_spt, x_qry = db['LR']['era5_2020'][args.dir4s[i - 12]].next()
                y_spt, y_qry = db['HR']['era5_2020'][args.dir4s[i - 12]].next()
            else:
                x_spt, x_qry = db['LR']['era5_2000'][args.dir4s[i - 18]].next()
                y_spt, y_qry = db['HR']['era5_2000'][args.dir4s[i - 18]].next()

            #把数据转为tensor类型放到GPU上训练
            x_spt,y_spt,x_qry,y_qry=torch.tensor(x_spt).to(device),torch.tensor(y_spt).to(device),torch.tensor(x_qry).to(device),torch.tensor(y_qry).to(device)
            loss_train,pred_img = maml(x_spt, y_spt, x_qry, y_qry,epoch)
            print('epoch:{} ,task:{}, training loss:{}'.format(epoch,i,loss_train))
            x.append(t_num)
            y.append(loss_train[1])
            t_num+=1
            if t_num==1:
                loss_window = viz.line(
                    X=np.expand_dims(x, axis=1)[0],
                    Y=np.expand_dims(y, axis=1)[0],
                    opts={'xlabel': 'update_meta', 'ylabel': 'train_loss', 'title': 'edsr'}
                )
            else:
                viz.line(
                    X=np.expand_dims(x, axis=1)[t_num-1],
                    Y=np.expand_dims(y, axis=1)[t_num-1],
                    win=loss_window,
                    update='append'
                )
        if (epoch+1)%1==0:
            for i in range(24):
                if i<6:
                    x_spt,x_qry=db1['LR']['GFS025'][args.dir4s[i]].next()
                    y_spt,y_qry=db1['HR']['GFS025'][args.dir4s[i]].next()
                elif i<12:
                    x_spt, x_qry = db1['LR']['GFS050'][args.dir4s[i-6]].next()
                    y_spt, y_qry = db1['HR']['GFS050'][args.dir4s[i-6]].next()
                elif i<18:
                    x_spt, x_qry = db1['LR']['era5_2020'][args.dir4s[i - 12]].next()
                    y_spt, y_qry = db1['HR']['era5_2020'][args.dir4s[i - 12]].next()
                else:
                    x_spt, x_qry = db1['LR']['era5_2000'][args.dir4s[i - 18]].next()
                    y_spt, y_qry = db1['HR']['era5_2000'][args.dir4s[i - 18]].next()
                x_spt, y_spt, x_qry, y_qry = torch.tensor(x_spt).to(device), torch.tensor(y_spt).to(device), torch.tensor(
                    x_qry).to(device), torch.tensor(y_qry).to(device)
                loss_test, pred_img = maml.finetune(x_spt, y_spt, x_qry, y_qry)
                x_test.append(t_test)
                y_test.append(loss_test)
                t_test += 1
                if t_test == 1:
                    test_window = viz.line(
                        X=np.expand_dims(x_test, axis=1)[0],
                        Y=np.expand_dims(y_test, axis=1)[0],
                        opts={'xlabel': 'update_num', 'ylabel': 'test_loss', 'title': 'test'}
                    )
                else:
                    viz.line(
                        X=np.expand_dims(x_test, axis=1)[t_test - 1],
                        Y=np.expand_dims(y_test, axis=1)[t_test - 1],
                        win=test_window,
                        update='append'
                    )
                # x.data.cpu().numpy()

                show_pred=[]
                show_gt=[]
                for xx in pred_img:
                    xx = xx.data.cpu().numpy()  # 用了gpu就加这行
                    xx = np.asarray(xx)
                    xx = np.transpose(xx, (1, 2, 0))  # 转成 chw 才能用cvt函数
                    xx = cv2.cvtColor(xx, cv2.COLOR_BGR2RGB)#必须转，不然颜色偏黄
                    xx = np.transpose(xx, (2, 0, 1))
                    show_pred.append(xx)
                for xx in y_qry:
                    xx = xx.data.cpu().numpy()  # 用了gpu就加这行
                    xx = np.asarray(xx)
                    xx = np.transpose(xx, (1, 2, 0))  # 转成 chw 才能用cvt函数
                    xx = cv2.cvtColor(xx, cv2.COLOR_BGR2RGB)#必须转，不然颜色偏黄
                    xx = np.transpose(xx, (2, 0, 1))
                    show_gt.append(xx)

                pp = np.mean([psnr(p,y,data_range=255) for p,y in zip(show_pred,show_gt)])
                ss = np.mean([ssim(np.transpose(p, (1, 2, 0)), np.transpose(y, (1, 2, 0)),multichannel=True) for p,y in zip(show_pred,show_gt)])
                mm = np.mean([np.sqrt(mse(p,y)) for p,y in zip(show_pred,show_gt)])
                viz.images(show_pred, nrow=args.qry, win='pred:{}'.format(i), opts=dict(title='{%.2f}/{%.2f}/{%.2f}/{%d}'%(pp,ss,mm,i)))
                viz.images(show_gt, nrow=args.qry, win='true:{}'.format(i), opts=dict(title='true value'))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoches', type=int, help='epoch number', default=2000000000)
    argparser.add_argument('--cuda', type=str, help='device', default='cuda:0')

    argparser.add_argument('--dir1', type=str, help='root', default='/hdd/zhanghonghu/D')
    # argparser.add_argument('--dir1', type=str, help='root', default='../../data/tian_x4/test')
    argparser.add_argument('--dir2s', type=list, default=['HR', 'LR'])
    argparser.add_argument('--dir3s', type=list, default=['GFS025', 'GFS050', 'era5_2020', 'era5_2000'])
    argparser.add_argument('--dir4s', type=list, default=['u10', 'v10', 'd2m', 'sp', 't2m', 'tp'])

    argparser.add_argument('--train', type=int, help='train number', default=2400)
    argparser.add_argument('--test', type=int, help='test number', default=700)
    argparser.add_argument('--spt', type=int, help='k shot for support set', default=20)
    argparser.add_argument('--qry', type=int, help='k shot for query set', default=5)

    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-5)
    argparser.add_argument('--task_lr', type=float, help='task-level inner update learning rate', default=1e-4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)

    #下面两个相加必须为已有因子的总数5r

    argparser.add_argument('--env', type=str, help='visdom windows title', default='lr_e5')
    args = argparser.parse_args()
    main(args)