#测试相同的像素数 分十次卷积和一次卷积速度比较
import logging
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from datetime import datetime 
# from einops import rearrange, reduce, repeat
# 建立一个简单的indexconv用于测试
class IndexedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,output_len,bias=False):
        super(IndexedConv, self).__init__()
        
        self.logger = logging.getLogger(__name__ + '.IndexedConv')

        groups = 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_width = output_len

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        nbatch = input.shape[0]
        # input is of shape (N, C_in, K, L)
        input = input.reshape(nbatch, -1, self.output_width)
        out = torch.bmm(self.weight.view(self.out_channels, -1).expand(nbatch, -1, -1), input)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)

        return out

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

# def one_conv(x,model,times):
#     # time begin
#     for _ in range(times):
#         out= model(x)
#         print(out.shape)
#     #time end
#     #avg time = time /times
# def ten_conv(x,model,times):
#     # y = rearrange(x, 'b p h w c ->  p b h w c')
#
#     # time begin
#     for time in range(times) :
#         for i in range(10):
#             out= model(x[i])
#             # print(i,out.shape)
#     #time end
#
#     #avg time = time /times
def main(B,H,W,C,k,x_ten_conv,x_one_conv):

    # x = torch.randn(10,B,C,H,W ) 

    # model_ten_conv =  nn.Conv2d(in_channels=C,out_channels=C,kernel_size=3,stride=1,padding=1) 
    # model_one_conv =  nn.Conv2d(in_channels=C,out_channels=C,kernel_size=(1,k),stride=1,padding=0) 
    k_one = k*k
    L = 10 * H * W
    model_ten_conv =  nn.Conv2d(in_channels=C,out_channels=C,kernel_size=k,stride=1,padding=1)



    times =100


    model_one_conv =IndexedConv(in_channels=C, out_channels=C, kernel_size=k_one,output_len=L)
    use_cuda=True
    if use_cuda:
        model_ten_conv=model_ten_conv.cuda()
        model_one_conv=model_one_conv.cuda()
        x_ten_conv=x_ten_conv.cuda()
        x_one_conv=x_one_conv.cuda()
##跑十次
    print(f'x_ten_conv:{x_ten_conv.shape}')
    print(f'out_ten_conv:{model_ten_conv(x_ten_conv[0]).shape}')
    print(f'x_one_conv:{x_one_conv.shape}')
    print(f'out_one_conv:{model_one_conv(x_one_conv).shape}')
    c=datetime.now()
    print(c )
    for time in range(times) :
        for i in range(10):
            out= model_ten_conv(x_ten_conv[i])
            # print(i,out.shape)
    d=datetime.now()
    print(d)
    print((d-c) /(times*1.0))
    deltaten =(d-c) /(times*1.0)

##一次跑完

    a=datetime.now()
    print(a )
    for _ in range(times):
      out =model_one_conv(x_one_conv )
    b=datetime.now()
    print(b)
    deltaone =(b-a) /(times*1.0)
    print((b-a) /(times*1.0))
    print(deltaten/deltaone)


if __name__ == '__main__':
    B=8
    H=64
    W=64
    C=128
    k=3
    L = 10 * H * W
    # for k in range(3,8):
    # for B in [4,8,16,32]:
    #     print(f'B:{B}')
    for C in [256]:
        k_one = k * k

        x_one_conv = torch.randn(B, C, L, k_one)
        x_ten_conv = torch.randn(10, B, C, H, W)

        main(B, H, W, C, k,x_ten_conv,x_one_conv)