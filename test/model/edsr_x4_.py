import  torch
from    torch import nn
from    torch.nn import functional as F


class EDSR4(nn.Module):

    def __init__(self):

        super(EDSR4, self).__init__()

        self.B=16 #残差块的数目
        self.scale=4#放大倍数

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()

        #0-63
        for i in range(0,self.B*2):
            w=nn.Parameter(torch.ones(64,64,3,3))
            w=torch.nn.init.kaiming_normal_(w)
            b=nn.Parameter(torch.zeros(64))
            self.vars.append(w)
            self.vars.append(b)
        #64，65
        w64=nn.Parameter(torch.ones(64,3,3,3))
        w64=torch.nn.init.kaiming_normal_(w64)
        b64=nn.Parameter(torch.zeros(64))
        self.vars.append(w64)
        self.vars.append(b64)

        #66.67
        w65=nn.Parameter(torch.ones(64,64,3,3))
        w65=torch.nn.init.kaiming_normal_(w65)
        b65=nn.Parameter(torch.zeros(64))
        self.vars.append(w65)
        self.vars.append(b65)

        #68，69
        w66=nn.Parameter(torch.ones(48,64,3,3))
        w66=torch.nn.init.kaiming_normal_(w66)
        b66=nn.Parameter(torch.zeros(48))
        self.vars.append(w66)
        self.vars.append(b66)




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
        #64-65
        idx = 64
        w, b = vars[idx], vars[idx + 1]
        x=F.conv2d(x,w,b,stride=1,padding=(w.shape[-1]-1)//2)
        x=F.relu(x)
        out1=x
        for i in range(0,self.B*4,4):
            idx = i
            t=x#保留x
            w, b = vars[idx], vars[idx + 1]
            x = F.conv2d(x, w, b, stride=1, padding=(w.shape[-1] - 1) // 2)
            x = F.relu(x)

            w, b = vars[idx+2], vars[idx + 3]
            x = F.conv2d(x, w, b, stride=1, padding=(w.shape[-1] - 1) // 2)

            x=t+x*0.1
        #66-67
        idx = 66
        w, b = vars[idx], vars[idx + 1]
        x = F.conv2d(x, w, b, stride=1, padding=(w.shape[-1] - 1) // 2)
        x = F.relu(x)
        x=x+out1

        #上采样
        #68-69
        idx=68
        w, b = vars[idx], vars[idx + 1]
        x=F.conv2d(x,w,b,stride=1,padding=(w.shape[-1]-1)//2)
        x=torch.pixel_shuffle(x, 4)
        x=self.max_min_normalize(x)
        return x

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

    def max_min_normalize(self,matrix):
        # get the min and max values of the original matrix
        min = torch.min(matrix)
        max = torch.max(matrix)

        # calculate the normalized matrix using the formula
        normalized_matrix = (matrix - min) / (max - min) * 255

        # return the normalized matrix
        return normalized_matrix

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

if __name__=='__main__':
    x=torch.ones(10,3,40,12)
    net=EDSR4()
    y=net(x,None)
    print(y.shape)
