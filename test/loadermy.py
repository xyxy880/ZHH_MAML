from os_filename import get_filename
import cv2
import numpy as np

class Mydataloader:

    def __init__(self, dir,perm,flag,args):
        #dir是图片路径的目录路径，perm，是打乱的图片序列，flag是表明是train模式还是测试模式
        self.spt,self.qry = args.spt,args.qry

        self.idx=0#当前取到哪里
        self.data=[]#保存当前加载器的所有图片路径

        paths=get_filename(dir,args.train,args.test,perm,flag)
        for path in paths:
            img_path='{}/{}'.format(dir,path)
            self.data.append(img_path)

    def getImg(self,x):
        x=cv2.imread(x)
        x=np.transpose(x,[2,0,1])
        return x.astype(np.float32)

    def next(self):
        #最简单的处理方法，溢出就重来
        if (self.idx+self.spt+self.qry)>=len(self.data):
            self.idx=0
        cur_paths=[self.data[i+self.idx] for i in range(self.spt+self.qry)]
        self.idx+=self.spt+self.qry

        imgs=[]
        for path in cur_paths:
            imgs.append(self.getImg(path))

        return np.array(imgs[:self.spt]),np.array(imgs[self.spt:self.spt+self.qry])

if __name__ == '__main__':

    import  time
    import  visdom
    from show_img import *
    import argparse
    import torch

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

    argparser.add_argument('--env', type=str, help='visdom windows title', default='ee')
    args = argparser.parse_args()


    viz = visdom.Visdom(env=args.env)
    #train loader
    perm = np.random.permutation(args.train + args.test)
    db={}
    for dir2 in args.dir2s:
        db[dir2]={}
        for dir3 in args.dir3s:
            db[dir2][dir3]={}
            for dir4 in args.dir4s:
                db[dir2][dir3][dir4]=Mydataloader('{}/{}/{}/{}'.format(args.dir1,dir2,dir3,dir4),perm,'train',args)
    db1={}
    #test loader
    for dir2 in args.dir2s:
        db1[dir2]={}
        for dir3 in args.dir3s:
            db1[dir2][dir3]={}
            for dir4 in args.dir4s:
                db1[dir2][dir3][dir4]=Mydataloader('{}/{}/{}/{}'.format(args.dir1,dir2,dir3,dir4),perm,'test',args)

    img_spt,img_qry=db['HR']['GFS050']['tp'].next()

    img=img_show(img_spt)
    img1=img_show(img_qry)

    viz.images(img, nrow=7, win='spt',env=args.env,opts=dict(title='哈ss哈'))
    viz.images(img1, nrow=5, win='qry', env=args.env,opts=dict(title='哈test哈'))

    time.sleep(10)

