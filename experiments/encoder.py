from torch_geometric.data import DataLoader
import torch
from torch_geometric.nn import GAE
from ppm_dataset import PPMDataset
from utils import get_logger, set_seed
from model import GCNEncoder
logger = get_logger("autoencoder.log")
set_seed(1337)

device = torch.device('cuda')
dataset = DataLoader(PPMDataset(root="UNSUP_DATASET_DIR", unsup=True), batch_size=1, shuffle=False)


in_channels, out_channels = 113, 512

model = GAE(GCNEncoder(in_channels, out_channels))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

def train():
    model.train()
    total_loss = 0
    for data in dataset:
        data = data.to(device)
        optimizer.zero_grad()
        z = model.encode((data.x).to(device=device, dtype=torch.float), data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)
    
best_loss = float('inf')
for epoch in range(400):
    loss = train()
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), f'models/autoenc.pth')
    logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    

