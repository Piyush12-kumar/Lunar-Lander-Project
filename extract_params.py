import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python extract_params.py <checkpoint_file>")
    sys.exit(1)

checkpoint_file = sys.argv[1]
output_file = "extracted_params.npy"

# Load the checkpoint with allow_pickle=True
try:
    checkpoint_data = np.load(checkpoint_file, allow_pickle=True).item()
    best_params = checkpoint_data['best_params']
    best_avg = checkpoint_data['best_avg']
    
    # Save just the parameters (no pickle needed)
    np.save(output_file, best_params)
    
    print(f"Successfully extracted parameters with fitness: {best_avg:.2f}")
    print(f"Saved to {output_file}")
except Exception as e:
    print(f"Error: {e}")