import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import GAE

from models import DownstreamModel, GCNEncoder
from utils import get_logger, set_seed
from ppm_dataset import get_dataset, labels

SEED = 1337
set_seed(SEED)
logger = get_logger("trappm.log")
NUM_EPOCHS = 100

RESULTS_SAVE_DIR = "SOURCE_CODE_DIR"
DATASET_ROOT_DIR = "DATASET_DIR/A100/"
GAE_MODEL_PATH = "SOURCE_CODE_DIR/models/autoenc.pth"       

k_folds = get_dataset(DATASET_ROOT_DIR, batch_size=1, base_seed=1337, ratio=0.7, num_folds=5)

device = torch.device('cuda')
gen_model = GAE(GCNEncoder(in_channels=113, out_channels=512)).to(device)
gen_model.load_state_dict(torch.load(GAE_MODEL_PATH))


def train(model, optimizer, criterion, dataloader, device, y_idx):
    model.train()
    gen_model.eval()
    train_loss = 0
    y_true_list = []
    y_pred_list = []
    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device) 
        embeddings = gen_model.encode((data.x).to(dtype=torch.float), data.edge_index)
        embeddings = scatter(embeddings, data.batch, dim=0, reduce="sum")
        static = (data.static.to(device, dtype=torch.float)).view(-1, 6)
        z = (torch.cat([embeddings, static], dim=1)).to(device, dtype=torch.float)
        y = ((data.y)[y_idx]).view(-1, 1).to(device, dtype=torch.float)
        out = downstream_model(data, z)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        y_true_list.append(y.detach().cpu().numpy())
        y_pred_list.append(out.detach().cpu().numpy())
    train_loss = train_loss / len(dataloader)
    y_true = np.concatenate(y_true_list).flatten()
    y_pred = np.concatenate(y_pred_list).flatten()
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return train_loss, mape, rmse

def evaluate(downstream_model, gen_model, criterion, dataloader, device, y_idx=0):
    downstream_model.eval()
    gen_model.eval()
    val_loss = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device) 
            embeddings = gen_model.encode((data.x).to(dtype=torch.float), data.edge_index)
            embeddings = scatter(embeddings, data.batch, dim=0, reduce="sum")
            static = (data.static.to(device, dtype=torch.float)).view(-1, 6)
            z = (torch.cat([embeddings, static], dim=1)).to(device, dtype=torch.float)
            y = ((data.y)[y_idx]).view(-1, 1).to(device, dtype=torch.float)
            out = downstream_model(data, z)
            loss = criterion(out, y)
            val_loss += loss.item()
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(out.cpu().numpy())
        val_loss = val_loss / len(dataloader)
        y_true = np.concatenate(y_true_list).flatten()
        y_pred = np.concatenate(y_pred_list).flatten()
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return val_loss, mape, rmse           

for k, v in labels.items():
    train_results = []
    test_results = []
    for i, fold in enumerate(k_folds):
        train_loader = fold["train"]
        test_loader = fold["test"]
        downstream_model = DownstreamModel().to(device)
        optimizer = torch.optim.Adam(downstream_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        criterion = nn.MSELoss()
        best_loss = float('inf')
        best_epoch = 0
        for epoch in range(NUM_EPOCHS):
            train_loss, mape, rmse = train(downstream_model, optimizer, criterion, train_loader, device, v)
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                torch.save(downstream_model.state_dict(), os.path.join(RESULTS_SAVE_DIR, f"models/fold_{i+1}_{k}.pth"))
            logger.info(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.4f} - {k}')
            result = {
                "fold": i+1,
                "epoch": epoch+1,
                "train_loss": train_loss,
                "mape": mape,
                "rmse": rmse,
                "metric": k
            }
            train_results.append(result)
        logger.info(f'Best Epoch: {best_epoch+1:02}, Best Train Loss: {best_loss:.4f} - {k}')
        logger.info("======= Test dataset evaluation =======")
        downstream_model = DownstreamModel().to(device)
        downstream_model.load_state_dict(torch.load(os.path.join(RESULTS_SAVE_DIR, f"models/fold_{i+1}_{k}.pth")))
        for data in test_loader:
            dataset = data["dataset"]
            category = data["category"]
            test_loss, mape, rmse = evaluate(downstream_model, gen_model, criterion, dataset, device, v)
            logger.info(f'{category} - {k} Test Loss: {test_loss:.4f}, Test MAPE: {mape:.4f}, Test RMSE: {rmse:.4f}')
            result = {
                "category": category,
                "fold": i+1,
                "loss": test_loss,
                "mape": mape,
                "rmse": rmse,
                "metric": k
            }
            test_results.append(result)
    train_df = pd.DataFrame(train_results)
    test_df = pd.DataFrame(test_results)
    train_df.to_csv(os.path.join(RESULTS_SAVE_DIR, f"results/train_{k}.csv"), index=False)
    test_df.to_csv(os.path.join(RESULTS_SAVE_DIR, f"results/test_{k}.csv"), index=False)