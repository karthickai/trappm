import argparse
import pandas as pd
import torch
import yaml
from train import Trainer

from utils import logger, setup_logger, CSV_RW
config ={}

parser = argparse.ArgumentParser(description='TraPPM vision dataset generator')
parser.add_argument('--config', type=str, default="config.yml", help='config file')
parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--hpo', type=str, default="exp1", help='hpo name')

args = parser.parse_args()

if args.config:
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        setup_logger(config['log_file'])
        logger.info("Config file loaded successfully")

if args.model not in config['models']:
    logger.error("Model not found in config file")
    # exit(1)

if args.hpo not in config:
    logger.error("HPO not found in config file")
    # exit(1)

hpo = config[str(args.hpo)]['hyperparameters']
bench = config[str(args.hpo)]['benchmark']

csv = CSV_RW(bench["csv_file"])
flag = True
unsup = False
if unsup:
    for model_name in config["models"]:
        for batch in hpo['batch_size']:
            logger.info(model_name, batch)
            torch.cuda.empty_cache()
            train = Trainer(model_name=model_name, batch_size=batch, opt="adam", 
                            lr=0.1, weight_decay=0.0, drop_rate=0.0, 
                            model_dtype=torch.float32, device=torch.device('cuda'), 
                            num_warm_iter=1, num_bench_iter=1)
                                
            if not train.export_all_models(onnx_path=bench["onnx_path"]):
                break
else:
    for model_name in config["models"]:
        for batch in hpo['batch_size']:
            for drop_rate in hpo['drop_rate']:
                for optimizer in hpo["optimizer"]:
                    for lr in hpo["learning_rate"]:
                        for weight_decay in hpo["weight_decay"]:
                            for model_dtype in hpo["model_dtype"]:
                                if model_dtype == "float32":
                                    model_dtype = torch.float32
                                elif model_dtype == "float16":
                                    model_dtype = torch.float16
                                else:
                                    model_dtype = torch.float32
                                new_row = {'model_name': model_name, 'model_dtype': str(model_dtype).split('.')[-1], 
                                        'batch_size': batch, 'drop_rate': drop_rate, 'opt': optimizer, 
                                        'lr': lr, 'weight_decay': weight_decay}
                                if not csv.check_row_exists(new_row):
                                    torch.cuda.empty_cache()
                                    train = Trainer(model_name=model_name, batch_size=batch, opt=optimizer, 
                                                    lr=lr, weight_decay=weight_decay, drop_rate=drop_rate, 
                                                    model_dtype=torch.float32, device=torch.device('cuda'), 
                                                    num_warm_iter=bench["warmp_iter"], num_bench_iter=bench["bench_iter"])
                                    
                                    if train.run(csv_file=bench["csv_file"], onnx_path=bench["onnx_path"]):
                                    # if train.export_all_models(onnx_path=bench["onnx_path"]):
                                        flag = True
                                    else: 
                                        flag = False
                                        break
                                if not flag: break
                            if not flag: break
                        if not flag: break
                    if not flag: break
                if not flag: break
            if not flag: break



