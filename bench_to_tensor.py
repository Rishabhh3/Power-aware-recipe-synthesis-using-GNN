import os
import torch
from torch_geometric.data import Data
import re

def parse_bench_to_pyg(bench_filepath):
    """
    Parses a .bench file and converts it into a PyTorch Geometric Data object.
    """
    node_to_id = {}
    node_features = []
    edges_src = []
    edges_dst = []
    edge_attrs = []
    
    current_node_id = 0
    
    # Simple One-Hot Encoding Map for Node Types
    # [PI, PO, GATE]
    encode_map = {
        'INPUT': [1.0, 0.0, 0.0],
        'OUTPUT': [0.0, 1.0, 0.0],
        'GATE': [0.0, 0.0, 1.0]
    }

    with open(bench_filepath, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        # Parse Primary Inputs
        if line.startswith("INPUT"):
            match = re.search(r"INPUT\((.*?)\)", line)
            if match:
                node_name = match.group(1).strip()
                if node_name not in node_to_id:
                    node_to_id[node_name] = current_node_id
                    node_features.append(encode_map['INPUT'])
                    current_node_id += 1

        # Parse Primary Outputs (Outputs are just declarations, they usually link from a gate)
        elif line.startswith("OUTPUT"):
            match = re.search(r"OUTPUT\((.*?)\)", line)
            if match:
                node_name = match.group(1).strip()
                if node_name not in node_to_id:
                    node_to_id[node_name] = current_node_id
                    node_features.append(encode_map['OUTPUT'])
                    current_node_id += 1
                    
        # Parse Logic Gates and Wires (e.g., n4 = AND(n1, n2) or n5 = NOT(n3))
        elif "=" in line:
            parts = line.split("=")
            dest_node = parts[0].strip()
            logic_expr = parts[1].strip()
            
            # Register the destination gate if not seen
            if dest_node not in node_to_id:
                node_to_id[dest_node] = current_node_id
                node_features.append(encode_map['GATE'])
                current_node_id += 1
                
            dest_id = node_to_id[dest_node]
            
            # Extract source nodes from the logic expression
            match = re.search(r"\((.*?)\)", logic_expr)
            if match:
                sources = [s.strip() for s in match.group(1).split(",")]
                is_inversion = 1.0 if "NOT" in logic_expr else 0.0
                
                for src in sources:
                    if src not in node_to_id:
                        # Fallback for unregistered internal nodes
                        node_to_id[src] = current_node_id
                        node_features.append(encode_map['GATE'])
                        current_node_id += 1
                    
                    src_id = node_to_id[src]
                    
                    # Append directional edge (Source -> Destination)
                    edges_src.append(src_id)
                    edges_dst.append(dest_id)
                    
                    # Append edge weight tracking logic inversions
                    edge_attrs.append([is_inversion])

    # Convert Python lists to PyTorch Tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # Construct the final PyG Graph Data object
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph_data

def process_directory(input_dir, output_dir):
    print(f"[INFO] Initializing Tensor Conversion on directory: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".bench"):
            bench_path = os.path.join(input_dir, filename)
            tensor_filename = filename.replace(".bench", ".pt")
            tensor_path = os.path.join(output_dir, tensor_filename)
            
            try:
                # Convert the graph
                graph_data = parse_bench_to_pyg(bench_path)
                
                # Save the compressed PyTorch geometric object
                torch.save(graph_data, tensor_path)
                processed_count += 1
                
                # Dynamic Memory Management: Delete the bulky raw text file once compressed
                os.remove(bench_path)
                
            except Exception as e:
                print(f"[ERROR] Corrupted file skipped: {filename}. Reason: {e}")
                
    print(f"[SUCCESS] Converted {processed_count} graphs into PyG tensors at: {output_dir}")
    print("[INFO] Cleaned up bulky raw `.bench` files to conserve disk space.")

if __name__ == "__main__":
    SOURCE_BENCH_DIR = "data/generated_bench"
    TARGET_TENSOR_DIR = "data/tensors"
    
    if os.path.exists(SOURCE_BENCH_DIR):
        process_directory(SOURCE_BENCH_DIR, TARGET_TENSOR_DIR)
    else:
        print(f"[ERROR] Directory '{SOURCE_BENCH_DIR}' not found.")