import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from power_dataset import StreamingPowerDataset
from gnn_model import MultiObjectiveGNN

def train_model():
    print("[INFO] Initializing Training Pipeline...")
    
    # 1. Setup Device & Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StreamingPowerDataset(csv_file="data/labels.csv", tensor_dir="data/tensors")
    
    # Note: PyG DataLoader handles batching graph components automatically
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize Model & Optimizer
    model = MultiObjectiveGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 50
    best_loss = float('inf')
    
    # 3. Main Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (graph_data, recipe_str, targets) in enumerate(train_loader):
            graph_data = graph_data.to(device)
            targets = targets.to(device)
            
            # Dummy encoding for text recipe strings (simplification for this loop)
            # In production, map 'recipe_str' to integers using a vocabulary dictionary
            recipe_seq = torch.randint(0, 7, (targets.size(0), 20)).to(device) 
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(graph_data.x, graph_data.edge_index, graph_data.batch, recipe_seq)
            
            # Weighted Loss: Power(1.0) + Area(0.1) + Delay(0.1)
            loss_power = criterion(predictions[:, 0], targets[:, 0])
            loss_area = criterion(predictions[:, 1], targets[:, 1])
            loss_delay = criterion(predictions[:, 2], targets[:, 2])
            loss = loss_power + (0.1 * loss_area) + (0.1 * loss_delay)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
        
        # Save best model checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(" -> Checkpoint saved.")

if __name__ == "__main__":
    train_model()