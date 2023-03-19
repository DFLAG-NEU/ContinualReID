# Main part of our model
# This code is modified from https://github.com/Diego999/pyGAT 
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from IPython import embed

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features # F
        self.out_features = out_features  # F' 
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) 
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) 
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) 
        h_prime = torch.matmul(attention, Wh) 

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout # 0.6

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


def Truncated_initializer(m):
    # sample u1:
    size = m.size()
    u1 = torch.rand(size)*(1-np.exp(-2)) + np.exp(-2) # torch.rand(3,3)
    # torch.rand(size)=torch.rand(torch.size([1,2048]))
    # sample u2:
    u2 = torch.rand(size)
    # sample the truncated gaussian ~TN(0,1,[-2,2]):
    z = torch.sqrt(-2*torch.log(u1)) * torch.cos(2*np.pi*u2)
    m.data = z

class MetaGraph_fd(nn.Module):
    def __init__(self, hidden_dim, input_dim, sigma=2.0, proto_graph_vertex_num=16, meta_graph_vertex_num=128):
        super(MetaGraph_fd, self).__init__()
        self.hidden_dim, self.input_dim, self.sigma = hidden_dim, input_dim, sigma
        adj_mlp = nn.Linear(hidden_dim, 1) 
        # nn.Linear() set fully connected layer
        Truncated_initializer(adj_mlp.weight)
        nn.init.constant_(adj_mlp.bias, 0.1)

        gate_mlp = nn.Linear(hidden_dim, 1) # Linear(in_features=hidden_dim, out_features=1, bias=True)
        Truncated_initializer(gate_mlp.weight)
        nn.init.constant_(gate_mlp.bias, 0.1)

        self.softmax = nn.Softmax(dim=0)
        self.meta_graph_vertex_num = meta_graph_vertex_num # The number of VA
        self.proto_graph_vertex_num = proto_graph_vertex_num #  VT
        self.meta_graph_vertex = nn.Parameter(torch.rand(meta_graph_vertex_num, input_dim)) 
        self.distance = nn.Sequential(adj_mlp, nn.Sigmoid()) 
        self.gate = nn.Sequential(gate_mlp, nn.Sigmoid()) 
        self.device = torch.device('cuda')
        self.att = GAT(nfeat=self.hidden_dim,nclass=self.input_dim, nhid=8,dropout=0.6, nheads=8,alpha=0.2).to(self.device)
        

        self.MSE = nn.MSELoss(reduce='mean')
        self.register_buffer('meta_graph_vertex_buffer', torch.rand(self.meta_graph_vertex.size(), requires_grad=False))

    def StabilityLoss(self, old_vertex, new_vertex):
        old_vertex = F.normalize(old_vertex) 
        new_vertex = F.normalize(new_vertex)

        return torch.mean(torch.sum((old_vertex-new_vertex).pow(2), dim=1, keepdim=False)) 
    def forward(self, inputs): # inputs = features 

       

        correlation_meta = self._correlation(self.meta_graph_vertex_buffer, self.meta_graph_vertex.detach())
        
        self.meta_graph_vertex_buffer = self.meta_graph_vertex.detach() 


        batch_size = inputs.size(0)
        protos = inputs 
 
        
        # Accumulated knowledge graph
        meta_graph = self._construct_graph_samegraph(self.meta_graph_vertex, self.meta_graph_vertex).to(self.device)

        # Temporary knowledge graph
        proto_graph = self._construct_graph_samegraph(protos, protos).to(self.device)


        m, n = protos.size(0), self.meta_graph_vertex.size(0) 



        cross_graph = self._construct_graph_crossgraph(protos, self.meta_graph_vertex).to(self.device)
        
        # super_graph based accumulated knowledge graph and temporary knowledge graph
        meta_graph = self._construct_graph_sa
        super_garph = torch.cat((torch.cat((proto_graph, cross_graph),  dim=1), torch.cat((cross_graph.t(), meta_graph), dim=1)), dim=0)

     

        feature = torch.cat((protos, self.meta_graph_vertex), dim=0).to(self.device)     

        # Propagation of knowledge with GAT
        representation = self.att(feature, super_garph)
        
        correlation_transfer_meta = self._correlation(representation[batch_size:].detach(), self.meta_graph_vertex.detach())
        
        correlation_protos = self._correlation(representation[0:batch_size].detach(), protos.detach())

        return representation[0:batch_size].to(self.device), representation[-batch_size:].to(self.device), [correlation_meta,correlation_transfer_meta, correlation_protos] 

    def _construct_graph(self, A, B):
        m = A.size(0) # 32
        n = B.size(0) # 32
        I = torch.eye(n, requires_grad=False).to(self.device) 
       
        index_aabb = torch.arange(0, m, requires_grad=False).repeat_interleave(n, dim=0).long()
        index_abab = torch.arange(0, n, requires_grad=False).repeat(m).long()
        diff = A[index_aabb] - B[index_abab]
        graph = self.distance(diff).view(m, n) 
        graph = graph.to(self.device) * (1 - I) + I
        return graph 

    def _construct_graph_samegraph(self, A, B):
        
        m = A.size(0) # 32
        n = B.size(0) # 32
        I = torch.eye(n, requires_grad=False).to(self.device)
        graph=torch.tensor(np.ones((m,n))).to(self.device)-I 

        return graph
        

    def _construct_graph_crossgraph(self, A, B):

        m = A.size(0) # 32
        n = B.size(0) # 32
        graph=torch.tensor(np.ones((m,n))).to(self.device)
        # graph = self.distance(graph)
        return graph


    def _correlation(self, A, B):
        similarity = F.cosine_similarity(A,B)
        similarity = torch.mean(similarity) # 取平均 means(tensor([1,2,3,4])) = 2.5
        return similarity



