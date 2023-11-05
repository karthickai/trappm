# MIT License
# 
# Copyright (c) [2023] [Karthick PANNER SELVAM] [karthick.pannerselvam@uni.lu]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch_geometric.nn as tgnn

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.bn1 = nn.BatchNorm1d(2 * out_channels)
        self.conv2 = SAGEConv(2 * out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = SAGEConv(out_channels, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.conv4 = SAGEConv(out_channels, out_channels)
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.gnn_drop_1 = nn.Dropout(p=0.5)
        self.gnn_drop_2 = nn.Dropout(p=0.5)
        self.gnn_drop_3 = nn.Dropout(p=0.5)
        self.gnn_relu1 = nn.ReLU()
        self.gnn_relu2 = nn.ReLU()
        self.gnn_relu3 = nn.ReLU()
        self.gnn_relu4 = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.gnn_relu1(x)
        x = self.gnn_drop_1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.gnn_relu2(x)
        x = self.gnn_drop_2(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.gnn_relu3(x)
        x = self.gnn_drop_3(x)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = self.gnn_relu4(x)
        return x
    

class DownstreamModel(torch.nn.Module):
    def __init__(
        self,
        num_node_features=113,
        gnn_layer="SAGEConv",
        gnn_hidden=512,
        fc_hidden=512,
        reduce_func="sum",
    ):
        super(DownstreamModel, self).__init__()

        self.reduce_func = reduce_func
        self.num_node_features = num_node_features
        self.gnn_layer_func = getattr(tgnn, gnn_layer)

        self.graph_conv_1 = self.gnn_layer_func(num_node_features, gnn_hidden, normalize=True)
        self.graph_conv_2 = self.gnn_layer_func(gnn_hidden, gnn_hidden, normalize=True)
        self.gnn_drop_1 = nn.Dropout(p=0.05)
        self.gnn_drop_2 = nn.Dropout(p=0.05)
        self.gnn_relu1 = nn.ReLU()
        self.gnn_relu2 = nn.ReLU()

        sf_hidden = 6
        self.fc_1 = nn.Linear(gnn_hidden + sf_hidden + gnn_hidden, fc_hidden)
        self.fc_2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_drop_1 = nn.Dropout(p=0.05)
        self.fc_drop_2 = nn.Dropout(p=0.05)
        self.fc_relu1 = nn.ReLU()
        self.fc_relu2 = nn.ReLU()
        self.predictor = nn.Linear(fc_hidden, 1)

    def forward(self, data, z):
        x, A = data.x, data.edge_index
        x = x.to(dtype=torch.float)
        x = self.graph_conv_1(x, A)
        x = self.gnn_relu1(x)
        x = self.gnn_drop_1(x)
        x = self.graph_conv_2(x, A)
        x = self.gnn_relu2(x)
        x = self.gnn_drop_2(x)
        gnn_feat = scatter(x, data.batch, dim=0, reduce=self.reduce_func)
        x = torch.cat([gnn_feat, z ], dim=1)
        x = self.fc_1(x)
        x = self.fc_relu1(x)
        x = self.fc_drop_1(x)
        x = self.fc_2(x)
        x = self.fc_relu2(x)
        feat = self.fc_drop_2(x)
        x = self.predictor(feat)
        x = -F.logsigmoid(x) 
        return x
   