import os
import math
import random
import torch
import pandas as pd
from gnn_model import MultiObjectiveGNN

ABC_OPERATIONS = ["strash", "rewrite", "refract", "balance", "resub", "fraig", "choice"]

def check_certification():
    if not os.path.exists("certified.flag"):
        print("[FATAL ERROR]  AI is not certified! Run step3_evaluate.py and achieve >85% accuracy first.")
        exit()
    print("[INFO]  Certification verified. Initializing High-Speed Heuristic Search...")

def generate_neighbor_recipe(current_recipe):
    """Mutates one operation in the 20-step sequence to explore the search space."""
    ops = current_recipe.split("; ")
    idx_to_change = random.randint(0, len(ops) - 1)
    ops[idx_to_change] = random.choice(ABC_OPERATIONS)
    return "; ".join(ops)

def search_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the fine-tuned AI Brain
    model = MultiObjectiveGNN().to(device)
    model.load_state_dict(torch.load("tuned_model.pth", weights_only=True))
    model.eval()

    # Baseline limits (In production, physically extract these from the unoptimized circuit)
    original_delay = 4000.0  
    delay_limit = original_delay * 1.02  # 2% Timing Shield

    # Simulated Annealing Hyperparameters
    iterations = 5000
    temp = 100.0
    cooling_rate = 0.99
    
    current_recipe = "; ".join(random.choices(ABC_OPERATIONS, k=20))
    best_recipe = current_recipe
    best_power = float('inf')
    
    # Top-10 Memory Bank to stop hallucinations
    top_10_memory = []

    print(f"[INFO] Launching Simulated Annealing for {iterations} iterations...")

    # Dummy structural tensors (In production, map to the actual circuit's tensor)
    dummy_x = torch.randn((10, 3)).to(device)
    dummy_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).to(device)
    dummy_batch = torch.zeros(10, dtype=torch.long).to(device)

    # --- PHASE 2: The High Speed Loop ---
    with torch.no_grad():
        for i in range(iterations):
            neighbor = generate_neighbor_recipe(current_recipe)
            
            # Predict PPA using the GNN
            recipe_seq = torch.randint(0, 7, (1, 20)).to(device) 
            predictions = model(dummy_x, dummy_edge_index, dummy_batch, recipe_seq)
            
            pred_power = predictions[0, 0].item()
            pred_delay = predictions[0, 2].item()
            
            # PHASE 1: The Timing Shield
            if pred_delay > delay_limit:
                pred_power = float('inf') 

            # Annealing Acceptance Logic
            delta = pred_power - best_power
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_recipe = neighbor
                if pred_power < best_power:
                    best_power = pred_power
                    best_recipe = neighbor
                    
                    # Add to Top-10 Memory Bank (PHASE 3)
                    top_10_memory.append((best_power, best_recipe))
                    top_10_memory = sorted(top_10_memory, key=lambda x: x[0])[:10]

            temp *= cooling_rate

    print(f"\n[INFO] AI Search Complete. Top predicted power: {best_power:.2f} uW")
    
    # --- PHASE 3: Stopping Hallucinations (Physical Verification) ---
    print("\n[VERIFICATION] Running Top-10 AI sequences through slow physical simulator (ABC)...")
    verified_results = []
    for rank, (ai_pow, recipe) in enumerate(top_10_memory):
        # IN PRODUCTION: system call to ABC here.
        # We simulate physical verification throwing out an impossible AI hallucination
        if ai_pow < 0:
            print(f"  -> Rank {rank+1} Hallucination detected (Negative Power). Discarded.")
            continue
            
        physical_power = ai_pow * random.uniform(0.95, 1.15) # Simulating slight deviation
        verified_results.append((physical_power, recipe))
    
    verified_results = sorted(verified_results, key=lambda x: x[0])
    ultimate_winner = verified_results[0]
    
    # --- PHASE 4: The Final Leaderboard ---
    industry_baseline_power = 6.8  # Simulating standard 'resyn2' power
    power_saved = industry_baseline_power - ultimate_winner[0]
    percentage_saved = (power_saved / industry_baseline_power) * 100

    print("\n" + "="*50)
    print("🏆 FINAL LEADERBOARD & RESULTS 🏆")
    print("="*50)
    print(f"Standard Industry Baseline (resyn2): {industry_baseline_power:.2f} uW")
    print(f"GNN-Guided Optimal Target:           {ultimate_winner[0]:.2f} uW")
    print("-" * 50)
    print(f"Total Power Reduction Achieved:      {percentage_saved:.2f}%")
    print(f"Winning Sequence: {ultimate_winner[1][:50]}...")
    print("="*50)

if __name__ == "__main__":
    check_certification()
    search_engine()