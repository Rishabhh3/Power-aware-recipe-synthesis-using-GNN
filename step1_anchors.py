import os
import random
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# The 7 core optimization operations
ABC_OPERATIONS = ["strash", "rewrite", "refract", "balance", "resub", "fraig", "choice"]

def generate_random_recipes(num_recipes=1500, length=20):
    """Generates a massive pool of random hardware optimization sequences."""
    recipes = []
    for _ in range(num_recipes):
        # Pick 20 random operations and join them with semicolons
        recipe = "; ".join(random.choices(ABC_OPERATIONS, k=length))
        recipes.append(recipe)
    return recipes

def find_smart_anchors():
    print("[INFO] Generating 1500 random synthesis recipes...")
    recipes = generate_random_recipes(1500, 20)

    print("[INFO] Vectorizing recipes to map structural similarities...")
    # TF-IDF converts the sequence of text commands into numerical matrices
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(recipes)

    print("[INFO] Executing K-Means Clustering (K=5)...")
    # Group the 1500 recipes into 5 distinct structural impact zones
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X)

    # Locate the single recipe that is mathematically closest to each cluster's center
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    anchors = [recipes[i] for i in closest_indices]
    
    print("\n[SUCCESS] 5 Smart Anchors Discovered:")
    for i, anchor in enumerate(anchors):
        # Print a snippet of the recipe for visual confirmation
        print(f"  Anchor {i+1}: {anchor[:60]}...")

    # Save these specific 5 targets to a file
    os.makedirs("data", exist_ok=True)
    output_path = "data/smart_anchors.csv"
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["anchor_id", "recipe_string", "power", "area", "delay"])
        for i, anchor in enumerate(anchors):
            # We leave physical metrics blank for now; they must be physically simulated next
            writer.writerow([i, anchor, "", "", ""])
            
    print(f"\n[INFO] Anchors safely stored at: {output_path}")
    print("[ALERT] The physical metrics (power, area, delay) are blank. You must simulate these 5 recipes in ABC to gather ground truth.")

if __name__ == "__main__":
    find_smart_anchors()