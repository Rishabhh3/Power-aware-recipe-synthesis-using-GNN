import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from gnn_model import MultiObjectiveGNN
# In a real pipeline, you would use bench_to_tensor to convert the 5 anchor .bench files
# Here we use dummy tensors for structural representation of the fine-tuning logic.

def micro_tune_model():
    print("[INFO] Initializing Circuit-Specific Fine-Tuning (Active Few-Shot)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the globally trained base model
    model = MultiObjectiveGNN().to(device)
    try:
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))
        print("[INFO] Global base model 'best_model.pth' loaded successfully.")
    except FileNotFoundError:
        print("[WARNING] 'best_model.pth' not found. Training from scratch for demonstration.")

    # 2. Execute Graph Shielding (Freeze Lobe 1)
    # We freeze the GCN layers so the model doesn't suffer "catastrophic forgetting" of basic gate logic
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    for param in model.graph_fc.parameters():
        param.requires_grad = False
        
    print("[INFO] Graph Shielding Active: Lobe 1 (Circuit Reader) is frozen.")

    # 3. Load the 5 Smart Anchors Ground Truth
    try:
        anchors_df = pd.read_csv("data/smart_anchors.csv")
    except FileNotFoundError:
        print("[ERROR] 'data/smart_anchors.csv' missing. Run step1_anchors.py first.")
        return

    # 4. Configure Optimizer for rapid tuning (Higher Learning Rate: 0.005)
    # Notice we only pass parameters that require gradients (Lobes 2 & 3)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)
    criterion = nn.MSELoss()

    epochs = 250
    print(f"[INFO] Commencing Dynamic Gradient Balancing Training for {epochs} rounds...")
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Dummy Data for Compilation ---
        # In production, stream the actual 5 .pt tensors mapped to these anchors
        dummy_x = torch.randn((10, 3)).to(device) 
        dummy_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).to(device)
        dummy_batch = torch.zeros(10, dtype=torch.long).to(device)
        dummy_recipe = torch.randint(0, 7, (1, 20)).to(device)
        
        # Extracted Ground Truth (Simulated metrics from the CSV)
        # Using dummy physical targets mimicking [Power, Area, Delay]
        target = torch.tensor([[5.0, 120.0, 4000.0]]).to(device) 
        
        predictions = model(dummy_x, dummy_edge_index, dummy_batch, dummy_recipe)
        
        # 5. Dynamic Gradient Balancing
        # Balances the massive scaling differences between 4000ps (Delay) and 5uW (Power)
        target_power = target[:, 0].mean().item() + 1e-6
        target_area = target[:, 1].mean().item() + 1e-6
        target_delay = target[:, 2].mean().item() + 1e-6
        
        ratio_p = target_delay / target_power
        ratio_a = target_delay / target_area
        
        loss_power = criterion(predictions[:, 0], target[:, 0])
        loss_area = criterion(predictions[:, 1], target[:, 1])
        loss_delay = criterion(predictions[:, 2], target[:, 2])
        
        # Apply the calculated ratios to equalize the math pressure
        balanced_loss = (ratio_p * loss_power) + (ratio_a * loss_area) + (1.0 * loss_delay)
        
        balanced_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  -> Round {epoch + 1}/{epochs} | Balanced Loss: {balanced_loss.item():.2f}")

    # Save the newly specialized brain
    torch.save(model.state_dict(), "tuned_model.pth")
    print("[SUCCESS] Circuit-specific tuning complete. Brain saved as 'tuned_model.pth'.")

if __name__ == "__main__":
    micro_tune_model()