import  torch
from    torch import nn
from    torch.nn import functional as F


class Mtl4(nn.Module):

    def __init__(self):

        super(Mtl4, self).__init__()

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()

        w1=nn.Parameter(torch.ones(128,3,3,3))
        w1=torch.nn.init.kaiming_normal_(w1)
        b1=nn.Parameter(torch.zeros(128))
        self.vars.append(w1)
        self.vars.append(b1)

        w2=nn.Parameter(torch.ones(128,128,3,3))
        w2=torch.nn.init.kaiming_normal_(w2)
        b2=nn.Parameter(torch.zeros(128))
        self.vars.append(w2)
        self.vars.append(b2)

        w3 = nn.Parameter(torch.ones(48, 128, 3, 3))
        w3 = torch.nn.init.kaiming_normal_(w3)
        b3 = nn.Parameter(torch.zeros(48))
        self.vars.append(w3)
        self.vars.append(b3)

        w4 = nn.Parameter(torch.ones(128, 3, 3, 3))
        w4 = torch.nn.init.kaiming_normal_(w4)
        b4 = nn.Parameter(torch.zeros(128))
        self.vars.append(w4)
        self.vars.append(b4)

        w5 = nn.Parameter(torch.ones(48, 128, 3, 3))
        w5 = torch.nn.init.kaiming_normal_(w5)
        b5 = nn.Parameter(torch.zeros(48))
        self.vars.append(w5)
        self.vars.append(b5)

        w6 = nn.Parameter(torch.ones(128, 3, 3, 3))
        w6 = torch.nn.init.kaiming_normal_(w6)
        b6 = nn.Parameter(torch.zeros(128))
        self.vars.append(w6)
        self.vars.append(b6)

        w7 = nn.Parameter(torch.ones(48, 128, 3, 3))
        w7 = torch.nn.init.kaiming_normal_(w7)
        b7 = nn.Parameter(torch.zeros(48))
        self.vars.append(w7)
        self.vars.append(b7)

        w8 = nn.Parameter(torch.ones(128, 3, 3, 3))
        w8 = torch.nn.init.kaiming_normal_(w8)
        b8 = nn.Parameter(torch.zeros(128))
        self.vars.append(w8)
        self.vars.append(b8)

        w9 = nn.Parameter(torch.ones(48, 128, 3, 3))
        w9 = torch.nn.init.kaiming_normal_(w9)
        b9 = nn.Parameter(torch.zeros(48))
        self.vars.append(w9)
        self.vars.append(b9)

        w10 = nn.Parameter(torch.ones(3, 3, 3, 3))
        w10 = torch.nn.init.kaiming_normal_(w10)
        b10 = nn.Parameter(torch.zeros(3))
        self.vars.append(w10)
        self.vars.append(b10)




#对每层的网络进行参数初始化（kaiming初始化），并把模型参数保存到一起。



    def forward(self, x,vars=None):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        w, b = vars[idx], vars[idx + 1]
        x1=F.conv2d(x,w,b,stride=1,padding=(w.shape[-1]-1)//2)
        x1=F.relu(x1)

        idx+=2
        w, b = vars[idx], vars[idx + 1]
        x2=F.conv2d(x1,w,b,stride=1,padding=(w.shape[-1]-1)//2)
        x2=F.relu(x2)
        tt=idx
        xx=0
        for i in range(2):
            idx=tt
            idx+=2
            w, b = vars[idx], vars[idx + 1]
            x3=F.conv2d(x2,w,b,stride=1,padding=(w.shape[-1]-1)//2)
            x3=F.relu(x3)
            x3=torch.pixel_shuffle(x3, 4)

            idx+=2
            w, b = vars[idx], vars[idx + 1]
            x4=F.conv2d(x3,w,b,stride=4,padding=(w.shape[-1]-1)//2)
            x4=F.relu(x4)
            x4=torch.subtract(x4,x2)

            idx+=2
            w, b = vars[idx], vars[idx + 1]
            x5=F.conv2d(x4,w,b,stride=1,padding=(w.shape[-1]-1)//2)
            x5=F.relu(x5)
            x5 = torch.pixel_shuffle(x5, 4)
            x5=torch.add(x3,x5)

            idx+=2
            w, b = vars[idx], vars[idx + 1]
            x6=F.conv2d(x5,w,b,stride=4,padding=(w.shape[-1]-1)//2)
            x6=F.relu(x6)

            idx+=2
            w, b = vars[idx], vars[idx + 1]
            x7=F.conv2d(x6,w,b,stride=1,padding=(w.shape[-1]-1)//2)
            x7=F.relu(x7)
            x7=torch.pixel_shuffle(x7, 4)
            x7=torch.subtract(x5,x7)#     ??????????????

            idx+=2
            w, b = vars[idx], vars[idx + 1]
            x8=F.conv2d(x7,w,b,stride=4,padding=(w.shape[-1]-1)//2)
            x8=F.relu(x8)
            x8=torch.add(x8,x6)
            xx+=x8*pow(0.1,i-1)
            x2=x8

        idx+=2
        w, b = vars[idx], vars[idx + 1]
        x9=F.conv2d(xx,w,b,stride=1,padding=(w.shape[-1]-1)//2)
        x9=F.relu(x9)
        x9=torch.pixel_shuffle(x9, 4)

        idx+=2
        w, b = vars[idx], vars[idx + 1]
        x10=F.conv2d(x9,w,b,stride=1,padding=(w.shape[-1]-1)//2)
        x10=F.relu(x10)

        return x10

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

if __name__=='__main__':
    x=torch.ones(1,3,28,28)
    net=Mtl4()
    y=net(x,None)
    print(y.shape)
