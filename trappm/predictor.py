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

import os
import torch
from torch_scatter import scatter
from torch_geometric.nn import GAE

from . import model
from .utils import set_seed
from .converter import to_pt


SEED = 1337
set_seed(SEED)

current_file_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_file_dir, "models")
GAE_MODEL_PATH = f"{MODEL_DIR}/autoenc_343.pth"    

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

gen_model = GAE(model.GCNEncoder(in_channels=113, out_channels=512)).to(device)
gen_model.load_state_dict(torch.load(GAE_MODEL_PATH))
gen_model.eval()

def evaluate(downstream_model, device, data):
    downstream_model.eval()
    data = data.to(device)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
    embeddings = gen_model.encode((data.x).to(dtype=torch.float), data.edge_index)
    embeddings = scatter(embeddings, data.batch, dim=0, reduce="sum")
    static = (data.static.to(device, dtype=torch.float)).view(-1, 6)
    z = (torch.cat([embeddings, static], dim=1)).to(device, dtype=torch.float)
    out = downstream_model(data, z)
    out = round(out.item(), 2)
    return out


def get_rich_table(results, save_file):
    from rich.console import Console
    from rich.table import Table
    from rich.terminal_theme import MONOKAI
    colums = ['GPU Metrics', 'A100 GPU Prediction']
    rich_table = Table(show_header=True, show_lines=True)
    for column in colums:
        rich_table.add_column(column, justify="right", header_style="bold")
    for k, v in results.items():
        rich_table.add_row(k, v)
    console = Console(record=True)
    console.print("TraPPM Performance Prediction Report", style="bold red", justify="center")
    console.print(rich_table, justify="center")
    console.save_svg(save_file, theme=MONOKAI, title="")


def predict(onnx_file):
    data = to_pt(onnx_file)
    
    labels = ['Train_Gpu_Memory_Mb', 'Train_Gpu_Power_W', 'Train_Step_Time','Inference_Step_Time']
    print_labels = ['Train Memory', 'Train Power', 'Train Step Time','Inference Step Time']
    unit = ['Mb', 'W', 'ms', 'ms']

    results = {}

    for k, pk, u in zip(labels, print_labels, unit):
        downstream_model = model.DownstreamModel().to(device)
        downstream_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"fold_1_{k}.pth")))
        out = evaluate(downstream_model, device, data)
        results[pk] = str(out) + " " + u

    try:
        import rich
        save_file = onnx_file.split("/")[-1].split(".")[0] + ".svg"
        get_rich_table(results, save_file)
    except:
        for k, v in results.items():
            print(f"{k}: {v}")
    
    return results
    



            

