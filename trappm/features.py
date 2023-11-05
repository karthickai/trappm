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

import numpy as np

ops = ['Sub', 'Add', 'Min', 'Max', 'Neg', 'Div', 'Mul', 'Abs', 'Ceil', 
       'Exp', 'Softmax', 'Log', 'ImageScaler', 'InstanceNormalization', 
       'Sqrt', 'Pow', 'Sin', 'Cos', 'Range', 'Sigmoid', 'Tanh', 'HardSigmoid', 
       'Relu', 'PRelu', 'LeakyRelu', 'Sum', 'NonMaxSuppression', 'LRN', 'Less', 
       'LessOrEqual', 'Not', 'And', 'Where', 'Transpose', 'Gemm', 'MatMul', 'Tile', 
       'Gather', 'Clip', 'Reciprocal', 'Relu6', 'Constant', 'Concat', 'OneHot', 
       'Einsum', 'Unsqueeze', 'Squeeze', 'Shape', 'Resize', 'Upsample', 'PoolBase', 
       'AveragePool', 'MaxPool', 'Dropout', 'GlobalAveragePool', 'Expand', 'Pad', 'Identity', 
       'Erf', 'BatchNormalization', 'Flatten', 'ArgMax', 'ArrayFeatureExtractor', 'ZipMap', 
       'Slice', 'ReduceMean', 'ReduceProd', 'ReduceSum', 'ReduceMin', 'ReduceMax', 'TopK',
       'Scan', 'Compress', 'Hardmax', 'CategoryMapper', 'LSTM', 'Conv', 'ReduceL2', 'CumSum',
        'NonZero', 'Equal', 'Floor', 'RoiAlign', 'ScatterElements', 'ScatterND', 
        'Greater', 'DequantizeLinear', 'QuantizeLinear', 'MatMulInteger', 'QLinearMatMul', 
        'QLinearConv', 'ConvTranspose', 'Reshape', 'GRU', 'ConstantOfShape', 'Cast', 'Split', 'None']

dtypes = ["float32", "float16", "mixed"]

opts = ["adam", "sgd", "adamw"]

def opt2vec(opt):
    vec = np.zeros(len(opts))
    vec[opts.index(opt)] = 1
    return vec.tolist()

def dtype2vec(dtype):
    vec = {}
    vec[dtypes.index(dtype)] = 1
    return vec.tolist()

def onehotencode(op):
    if op not in ops:
        op = "None"
    op_vec = np.zeros(len(ops))
    op_vec[ops.index(op)] = 1
    return op_vec

            
def input_size2vec(input_size):
    input_size = input_size.split("_")
    input_size = [int(x) for x in input_size]
    input_size = input_size + [0] * (4 - len(input_size)) # supported for 3D CNN input
    return input_size


def nodefeature(G, batch):
    try:
        static = G.graph["name"]
        G.graph['static'] = np.array([static['total_nodes'], static['total_edges'], static['total_macs'],  static['total_params'], static['total_memory']])
        del G.graph["name"]
        G.graph['static'] = np.concatenate(([int(batch)], G.graph['static']), axis=None)
        for node in G.nodes(data=True):
            feature = node[1]['node_feature']  
            feature['op_type'] = onehotencode(feature['op_type'])
            feature['inshape'] = np.array(feature['inshape']).flatten().tolist()
            feature['inshape'] = feature['inshape'] + [0] * (6 - len(feature['inshape']))
            feature['outshape'] = np.array(feature['outshape']).flatten().tolist()
            feature['outshape'] = feature['outshape'] + [0] * (6 - len(feature['outshape']))
            feature = feature['op_type'].tolist() + feature['inshape'] + feature['outshape'] + [feature['macs'], feature['params'], feature['memory']]
            del G.nodes[node[0]]["node_feature"]
            G.nodes[node[0]]["x"] = np.array(feature)
        return G
    except:
        raise Exception("node feature error")


