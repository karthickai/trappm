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

import onnx
import numpy as np
import networkx as nx
from . import onnx_tool
from .features import nodefeature
from torch_geometric.utils.convert import from_networkx

def fill_onnx_weight(G):
    for init in G.graph.initializer:
        fill_a_tensor(init)
    for node in G.graph.node:
        # conv weight may be constant node
        if node.op_type == "Constant":
           for attr in node.attribute:
               if attr.type == onnx.AttributeProto.TENSOR:
                   fill_a_tensor(attr.t)


def fill_a_tensor(T, min_len=10):

    def stat_size(T):
        L = 0
        L = max(L, len(T.float_data))
        L = max(L, len(T.double_data))
        L = max(L, len(T.int32_data))
        L = max(L, len(T.int64_data))
        L = max(L, len(T.uint64_data))
        L = max(L, len(T.raw_data) // 4)
        return L

    # in case of overlapping the shape constant & initializer
    length = np.prod(T.dims)
    # real size == 0 need to fill too
    if stat_size(T) > 0:
        return

    if T.data_type == onnx.TensorProto.FLOAT:
        T.float_data[:] = np.random.rand(length) - 0.5

    elif T.data_type == onnx.TensorProto.DOUBLE:
        T.double_data[:] = np.random.rand(length) - 0.5

    elif T.data_type == onnx.TensorProto.COMPLEX64:
        T.float_data[:] = np.random.rand(length * 2) - 0.5

    elif T.data_type == onnx.TensorProto.COMPLEX128:
        T.double_data[:] = np.random.rand(length * 2) - 0.5

    elif T.data_type == onnx.TensorProto.INT64:
        T.int64_data[:] = (np.random.rand(length) * 127).astype(np.int64) - 64

    elif T.data_type == onnx.TensorProto.UINT32 or \
        T.data_type == onnx.TensorProto.UINT64:
        T.uint64_data[:] = (np.random.rand(length) * 127).astype(np.uint64)

    elif T.data_type == onnx.TensorProto.UINT8 or \
        T.data_type == onnx.TensorProto.INT8 or \
        T.data_type == onnx.TensorProto.UINT16 or \
        T.data_type == onnx.TensorProto.INT16 or \
        T.data_type == onnx.TensorProto.INT32 or \
        T.data_type == onnx.TensorProto.BOOL or \
        T.data_type == onnx.TensorProto.FLOAT16 or \
        T.data_type == onnx.TensorProto.BFLOAT16:
        T.int32_data[:] = (np.random.rand(length) * 127).astype(np.int32) - 64

    else:
        # onnx.TensorProto.UNDEFINED,
        #onnx.TensorProto.STRING,
        pass


def pass_remove_ceil_mode_for_MaxPool(G):
    for node in G.graph.node:
        if node.op_type.endswith("Pool"):
            for attr in node.attribute:
                if attr.name == "ceil_mode":
                    # print("remove attirbute ceil_mode for {}: {}".format(node.op_type, node.name))
                    node.attribute.remove(attr)

def onnx_to_nx(onnx_path):
    G = onnx.load(onnx_path)
    fill_onnx_weight(G)
    pass_remove_ceil_mode_for_MaxPool(G)
    batch_size = None
    g = onnx_tool.model_profile(G)
    nx_graph = nx.DiGraph()
    if(g):
        for node in g.nodemap.values():
            if not batch_size:
                batch_size = node.inshape[0]
            node_feature = {
                "name": node.name,
                "op_type": node.op_type,
                "inshape": node.inshape,
                "outshape": node.outshape,
                "macs": node.macs,
                "params": node.params,
                "memory": node.memory,
            }
            for i in node.nextnodes:
                nx_graph.add_edge(node.name, i.name)
            for i in node.prevnodes:
                nx_graph.add_edge(i.name, node.name)
            nx_graph.add_node(node.name, node_feature=node_feature)
        graph_feature = {
            "file_name": "test.gexf",
            "total_nodes": len(nx_graph.nodes),
            "total_edges": len(nx_graph.edges),
            "total_macs": g.macs,
            "total_params": g.params,
            "total_memory": g.memory,
        }
        nx_graph.graph["name"] = graph_feature
    
        return nx_graph, batch_size
    else:
        raise Exception("onnx model profile error")


def to_pt(onnx_file):
    G, batch_size = onnx_to_nx(onnx_file)
    G = nodefeature(G, batch_size)
    data = from_networkx(G, all)
    mean = data.static.mean()
    std = data.static.std()
    data.static = (data.static - mean) / std 
    return data
    