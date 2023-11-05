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

import argparse
from timm.models import create_model
import torch
from timm.models import create_model
from timm.data import resolve_data_config


def get_model(model_name, model_dtype, device, drop_rate):
    model = create_model(
        model_name, pretrained=False, in_chans=3, exportable=True, drop_rate=drop_rate, scriptable=False)
    model.to(device=device, dtype=model_dtype)
    data_config = resolve_data_config({}, model=model)
    input_size = data_config['input_size']
    num_classes = model.num_classes
    return model, input_size, num_classes

def get_input(batch_size, input_size, data_dtype, device):
    dummy_inputs = torch.randn(
        (batch_size,) + input_size, device=device, dtype=data_dtype)
    return dummy_inputs

def to_onnx(model, dummy_inputs, onnx_file):
    with torch.no_grad():
        model.eval()
        torch.onnx.export(model, dummy_inputs, onnx_file, verbose=True, export_params=False)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TraPPM Performance Prediction Model')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--batch_size', type=int, help='batch size')

    args = parser.parse_args()
    device = torch.device('cuda')
    model_dtype = torch.float32
    model, input_size, num_classes = get_model(args.model, model_dtype, device, 0.0)
    dummy_inputs = get_input(args.batch_size, input_size, model_dtype, device)
    onnx_file = f'{args.model}_{args.batch_size}.onnx'
    to_onnx(model, dummy_inputs, onnx_file)

