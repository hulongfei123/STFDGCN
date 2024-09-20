import math
from math import sqrt

import torch.nn as nn
import torch
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # 检查 A 是否包含批次维度
        if A.dim() == 3:  # A 是 (B, N, N)
            x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        else:  # A 是 (N, N)
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout, support_len, order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class TemporalAttentionModel(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(TemporalAttentionModel, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.fc12 = torch.nn.Linear(model_dim, model_dim)
        self.fc13 = torch.nn.Linear(model_dim, model_dim)
        self.fc14 = torch.nn.Linear(model_dim, model_dim)
        self.fc15 = torch.nn.Linear(model_dim, 4*model_dim)
        self.fc16 = torch.nn.Linear(4*model_dim, model_dim)


        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X, Mask=True):
        #X(B,H,N,T)
        X = X.transpose(1, 3) #(B,T,N,H)
        B, T, N, _ = X.shape

        query = F.gelu(self.fc12(X))#(B,T,N,H)
        key = F.gelu(self.fc13(X))
        value = F.gelu(self.fc14(X))

        query = query.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        key = key.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        value = value.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)

        key = key.transpose(-1, -2)
        attention = (query @ key) / self.head_dim ** 0.5  # self.head_dim = 38

        if Mask == True:
            mask = torch.ones(T, T).to(X.device) # [T,T]
            mask = torch.tril(mask) # [T,T]下三角为1其余为0
            zero_vec = torch.tensor(-9e15).to(X.device)
            mask = mask.to(torch.bool)  #里面元素全是负无穷大
            attention = torch.where(mask, attention, zero_vec)

        attention = self.softmax(attention)
        X = torch.matmul(attention, value) #(B,N,h,T,h_d)
        X = X.transpose(2, 3).reshape(B, N, T, self.model_dim)  # (B, N, 12, M)
        X = self.dropout(self.fc16(F.gelu(self.fc15(X)))).permute(0, 3, 1, 2)

        return X  #X(B,H,N,T)

class DynamicGraphGenerator(nn.Module):
    def __init__(self, in_steps, hidden_dim, embed_dim, nodes, c):
        super(DynamicGraphGenerator, self).__init__()
        self.in_steps = in_steps
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.topk = int(c * math.log(nodes, 2))
        fft_dim = (in_steps // 2) + 1

        self.Wx = nn.Parameter(torch.randn(fft_dim, embed_dim), requires_grad=True)
        self.Wd = nn.Parameter(torch.randn(nodes, 4*embed_dim, embed_dim), requires_grad=True)
        self.Wxabs = nn.Parameter(torch.randn(embed_dim, embed_dim), requires_grad=True)

        self.layersnorm = torch.nn.LayerNorm(normalized_shape=[nodes, embed_dim], eps=1e-08,elementwise_affine=False)
        self.drop = nn.Dropout(p=0.1)

        nn.init.xavier_uniform_(self.Wx)
        nn.init.xavier_uniform_(self.Wd)
        nn.init.xavier_uniform_(self.Wxabs)

    def cat(self,x1, x2):
        M = []
        for i in range(x1.size(0)):
            x = x1[i]
            new_x = torch.cat([x, x2], dim=1)
            M.append(new_x)
        result = torch.stack(M, dim=0)
        return result

    def dy_mask_graph(self, adj, k):
        M = []
        for i in range(adj.size(0)):
            adp = adj[i]#nn
            mask = torch.zeros(adj.size(1), adj.size(2)).to(adj.device)
            mask = mask.fill_(float("0"))
            s1, t1 = (adp + torch.rand_like(adp) * 0.01).topk(k, 1)
            mask = mask.scatter_(1, t1, s1.fill_(1))
            M.append(mask)
        mask = torch.stack(M, dim=0)
        adj = adj * mask #使得邻接矩阵只保留前 k 个最大值的连接。
        return adj

    def forward(self, x, T_D, D_W, E):

        xn1 = torch.fft.rfft(x, dim=1) #(B,7,N)
        xn1 = torch.abs(xn1)

        xn1 = torch.nn.functional.normalize(xn1, p=2.0, dim=1, eps=1e-12, out=None)
        xn1 = torch.nn.functional.normalize(xn1, p=2.0, dim=2, eps=1e-12, out=None)
        #x 经过FC
        x = torch.einsum("btn,td->bnd", (xn1, self.Wx))#bnd
        #x||E
        xe = self.cat(x, E) #B N 2*d
        # x||E||TD||TW
        xedw = torch.cat([xe, T_D, D_W], dim=2)#B N 2d+2*12

        #conv1d
        x1 = torch.bmm(xedw.permute(1, 0, 2), self.Wd).permute(1, 0, 2)#bnd

        x1 = torch.relu(x1)#N B d
        x1k = self.layersnorm(x1)
        DE = self.drop(x1k)
        #conv1d
        DEw = torch.einsum("bne,ek->bnk", (DE, self.Wxabs)).contiguous()#b n 30

        #第一次卷积反转和第二次卷积结果相乘
        adj = torch.bmm(DEw, x1.permute(0, 2, 1)) #b n n

        adp = torch.relu(adj) #b n n
        #topk
        if self.topk>0:
            adp = self.dy_mask_graph(adp, self.topk)

        AD = F.softmax(adp, dim=2)#b n n

        return AD

class STFDGCN(nn.Module):
    def __init__(
        self,
        supports,
        num_nodes,
        in_steps=12,
        out_steps=12,
        input_dim=3,
        output_dim=1,
        embed_dim=10,
        hidden_dim=64,
        cheb_k=2,
        num_heads=8,
        num_layers=1,
        dropout=0.1,
        c=5,
    ):
        super().__init__()
        self.supports = supports
        self.num_nodes = num_nodes#N
        self.in_steps = in_steps#12
        self.out_steps = out_steps#12
        self.input_dim = input_dim#1
        self.output_dim = output_dim#1
        self.embed_dim = embed_dim#
        self.hidden_dim = hidden_dim
        self.cheb_k = cheb_k
        self.num_heads = num_heads#4
        self.num_layers = num_layers#1
        self.dropout = dropout
        self.c = c

        residual_channels = self.hidden_dim
        dilation_channels = self.hidden_dim
        skip_channels = 256
        end_channels = 512
        self.blocks=4
        self.layers=2
        self.supports_len = 4



        #time embedding
        self.T_i_D_emb  = nn.Parameter(torch.empty(288, self.embed_dim))#(288,12)
        self.D_i_W_emb  = nn.Parameter(torch.empty(7,  self.embed_dim))#(7,12)
        #node embeddings Eu(N,10) Ed(N,10)
        self.E1 = nn.Parameter(torch.empty(self.num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.empty(self.num_nodes, embed_dim))
        self.E3 = nn.Parameter(torch.empty(self.num_nodes, embed_dim))


        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                     out_channels=self.hidden_dim,
                                     kernel_size=(1, 1))

        self.graph_generator = DynamicGraphGenerator(in_steps,self.hidden_dim,self.embed_dim,self.num_nodes, self.c)

        self.temporalattention = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.bn = nn.ModuleList()


        kernel_size = 2
        receptive_field = 1
        for _ in range(self.blocks):
            # block=4,kernel_size=2,layers=2
            additional_scope = kernel_size - 1
            new_dilation = 1
            # new_dilation: 1->2
            # receptive_field: 1->13
            for _ in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

                self.gconv.append(gcn(dilation_channels, residual_channels, dropout, self.supports_len))
                self.temporalattention.append(TemporalAttentionModel(residual_channels, self.num_heads))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_steps,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)
        nn.init.xavier_uniform_(self.E3)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def forward(self, x):
        # x: (B, 12, N, 3)
        # x1 = x[:, :, :, 0:1]  # (B, 12, N, 1)
        # y_day = x[:, :, :, 1:2]  # (B, 12, N, 1)
        # y_week = x[:, :, :, 2:3]  # (B, 12, N, 1)
        T_D = self.T_i_D_emb[(x[:, :, :, 1] * 288).type(torch.LongTensor)][:, -1, :, :]
        D_W = self.D_i_W_emb[(x[:, :, :, 1 + 1]).type(torch.LongTensor)][:, -1, :, :]

        Adap = F.softmax(F.relu(torch.mm(self.E1, self.E2.T)), dim=1)  # (N,20)*(20,h)*(h,d)=(N,d)
        supports = self.supports + [Adap]
        AD = self.graph_generator(x[:, :, :, 0], T_D, D_W, self.E3)
        new_supports = supports + [AD]

        x = x.transpose(1, 3).contiguous()#x(B,3,N,12)


        in_len = x.size(3)
        # self.receptive_field: 13
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = x

        x = self.start_conv(x) #x(B,32,N,13)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x

            # TA
            x = self.temporalattention[i](residual)
            x = x + residual

            # dilated convolution
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)

            x = filter * gate

            #skip
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            #d-gcn
            x = self.gconv[i](x, new_supports) #(B,F,N,T)

            #Add
            x = x + residual[:, :, :, -x.size(3):]

            #bn
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


