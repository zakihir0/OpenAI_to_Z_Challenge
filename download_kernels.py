#!/usr/bin/env python3

from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()
api.authenticate()

# List of interesting kernels found
kernels = [
    "rackovic1994/peru-settlements-as-a-dynamic-graph",
    "paultimothymooney/how-to-use-openai-models-on-kaggle", 
    "zdanovic/hybrid-cv-llm-for-lost-city-discovery"
]

# Create directory for kernels
os.makedirs("kernels", exist_ok=True)

for kernel_ref in kernels:
    try:
        print(f"Downloading kernel: {kernel_ref}")
        api.kernels_pull(kernel_ref, path=f"kernels/{kernel_ref.replace('/', '_')}")
        print(f"  ✓ Downloaded to kernels/{kernel_ref.replace('/', '_')}")
    except Exception as e:
        print(f"  ✗ Error downloading {kernel_ref}: {e}")

print("\nListing downloaded kernels:")
for root, dirs, files in os.walk("kernels"):
    level = root.replace("kernels", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")