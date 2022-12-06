# 测试相同的像素数 分十次卷积和一次卷积速度比较
import logging
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from datetime import datetime
from einops import rearrange, reduce, repeat


# 建立一个简单的indexconv用于测试
class IndexedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(IndexedConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels*kernel_size))
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

    def forward(self, input, indices):
        # input=> N C_in L_in  indices=> L_out k
        nbatch = input.shape[0]
        col = input[..., indices]
        # print(f'col:{col.shape}')
        # col is of shape (N, C_in, L_out, k)
        # col = col.reshape(nbatch, -1, out_len)
        # view expand都不重新赋值内存 expand会改变 连续性
        col = rearrange(col, 'N C L K  ->  N (C K) L ')
        out = torch.bmm(self.weight.expand(nbatch, -1, -1), col)
        # print(f'col:{col.shape},out:{out.shape},wei:{self.weight.shape}')

        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)
        return out


class spherePHDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(spherePHDConv, self).__init__()
        self.model = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=1, padding=0)

    def forward(self, input, indices):
        # input=> N C_in L_in  indices=> L_out C_in
        # nbatch = input.shape[0]
        col = input[..., indices]
        # print(f'phd input{col.shape}')
        # out_len = indices.shape[0]
        # col is of shape (N, C_in, L_out, k)
        # col is of shape (N, C_in, L_out, k)
        out = self.model(col).squeeze(-1)
        return out


class NormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(NormConv, self).__init__()
        self.model = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2)
        self.topo = 'D20'  # 菱形20面体

    def forward(self, input):

        # input=>  N C_in L_in  indices=> L_out C_in
        N, C, L = input.shape
        # print(f'in_shape:{input.shape},{input}')
        if self.topo == 'D20':
            P = 10
        w = h = int(math.sqrt(L // 10))
        # zero padding
        input = rearrange(input, 'N C (p w h) -> p N C w h ', w=w, h=h)

        # print(f'in_shape:{input.shape},{input}')
        outs = []
        for i in range(P):
            out = self.model(input[i])
            outs.append(out)
        outs = torch.stack(outs)
        outs = rearrange(outs, 'P N C  w h  ->  N C (P w h) ')
        return outs


def test_normConv(input, C, k):
    # input = torch.randn(B, C, L)
    # print(input)
    model = NormConv(in_channels=C, out_channels=C, kernel_size=k)
    print(model(input).shape)


def test_PHDConv(input, indices, C, k, use_cuda):
    # input = torch.randn(B, C, L)
    # indices = torch.randint(0, L,size=[L,k*k])

    # print(input)
    model = spherePHDConv(in_channels=C, out_channels=C, kernel_size=k * k)
    print('PHDConv', model(input, indices).shape)


def test_IndexConv(input, indices, C, k, use_cuda):
    # input = torch.randn(B, C, L)
    # indices = torch.randint(0, L,size=[L,k*k])

    # print(input)
    model = IndexedConv(in_channels=C, out_channels=C, kernel_size=k * k)
    print('IndexConv', model(input, indices).shape)


def test_model(model, input, indices=None):
    if indices is None:
        out = model(input)
    else:
        out = model(input, indices)

    return out
def main():
    # x = torch.randn(10,B,C,H,W )

    # model_ten_conv =  nn.Conv2d(in_channels=C,out_channels=C,kernel_size=3,stride=1,padding=1) 
    # model_one_conv =  nn.Conv2d(in_channels=C,out_channels=C,kernel_size=(1,k),stride=1,padding=0)
    k = 3
    B = 32
    H = 64
    W = 64
    c = 64
    c_out = 128
    k_squ = k * k
    L = 10 * H * W


    input = torch.randn(B, c, L)

    indices = torch.randint(0, L, size=[L, k * k])


    print(f'input:{input.shape}')
    print(f'indices:{indices.shape}')
    n_conv = NormConv(in_channels=c, out_channels=c_out, kernel_size=k)
    s_conv = spherePHDConv(in_channels=c, out_channels=c_out, kernel_size=k_squ)
    i_conv = IndexedConv(in_channels=c, out_channels=c_out, kernel_size=k_squ)
    # out = i_conv(input, indices)
    #计算参数量
    # total = sum([param.nelement() for param in n_conv.parameters()]) #计算总参数量
    # print("Number of n_conv parameter: %.6f" % (total)) #输出

    # total = sum([param.nelement() for param in s_conv.parameters()]) #计算总参数量
    # print("Number of s_conv parameter: %.6f" % (total)) #输出

    # total = sum([param.nelement() for param in i_conv.parameters()]) #计算总参数量
    # print("Number of i_conv parameter: %.6f" % (total)) #输出
    use_cuda = True
    if use_cuda:
        n_conv = n_conv.cuda()
        s_conv = s_conv.cuda()
        i_conv = i_conv.cuda()
        input = input.cuda()
        indices = indices.cuda()

    ##跑十次

    # print(f'x_one_conv:{x_one_conv.shape}')
    # print(f'out_one_conv:{model_one_conv(x_one_conv).shape}')
    # for _ in range(20):
    #     test_model(n_conv, input)
    times = 100

    for i in range(1):
        print(f'{i} time ,times:{times}')

        print('n_conv:', end=' ')
        time0 = datetime.now()
        for time in range(times):
            out_n = test_model(n_conv, input)
        time1 = datetime.now()
        print(time0, end=' ')
        print(time1)
        n_time = (time1 - time0) / (times * 1.0)
        print(f'n_time:{n_time}')
        # 0:00:00.016201

        print('s_conv:', end=' ')
        time0 = datetime.now()
        for time in range(times):
           out_s = test_model(s_conv, input, indices)
        time1 = datetime.now()
        print(time0, end=' ')
        print(time1)
        s_time = (time1 - time0) / (times * 1.0)
        print(f's_time:{s_time}')
        # #0:00:00.018110

        print('i_conv', end=' ')
        time0 = datetime.now()
        for time in range(times):
            out_i = test_model(i_conv, input , indices)
        time1 = datetime.now()
        print(time0, end=' ')
        print(time1)
        i_time = (time1 - time0) / (times * 1.0)
        print(f'i_time:{i_time}')
        #i_time:0:00:00.015453

        # print(f'n/i:{n_time/i_time}')
        # print(f's/i:{s_time/i_time}')
        # print(f'n/s:{n_time/s_time}')

    print(f'out_n:{out_n.shape},outs:{out_s.shape},out_i:{out_i.shape}')
        # print(f'{ out_s[0,0,0:10]}, {out_i[0,0,0:10]} ')
        #
        # print(f'{torch.allclose(out_s[0,0], out_i[0,0])}')
if __name__ == '__main__':
    # test_normConv()
    # test_PHDConv()
    # test_IndexConv()
    # B=8
    # for _ in range (1):
        main()

    # H=64
    # W=64
    # C=128
    # k=3
    # L = 10 * H * W
    # # for k in range(3,8):
    # # for B in [4,8,16,32]:
    # #     print(f'B:{B}')
    # for C in [256]:
    #     k_one = k * k

    #     x_one_conv = torch.randn(B, C, L, k_one)
    #     x_ten_conv = torch.randn(10, B, C, H, W)

    #     main(B, H, W, C, k,x_ten_conv,x_one_conv)
