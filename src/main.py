import numpy as np
from datasets import generate_cell_data
from models import PseudoCellGenerator
from alignment import TemporalAligner

# -----------------------
# Config
# -----------------------
NUM_TIMES = 3
NUM_CELLS = 300
NUM_GENES = 50
NUM_PSEUDOCELLS = 10
NUM_ALIGN_CLUSTERS = 8

# -----------------------
# Step 1: Generate data
# -----------------------
expr_by_time = [
    generate_cell_data(NUM_CELLS, NUM_GENES, t)
    for t in range(NUM_TIMES)
]

# -----------------------
# Step 2: Pseudocell construction
# -----------------------
pseudocells_by_time = []
labels_by_time = []

generator = PseudoCellGenerator(NUM_PSEUDOCELLS)

for t, expr in enumerate(expr_by_time):
    pcs, labels = generator.fit(expr)
    pseudocells_by_time.append(pcs)
    labels_by_time.append(labels)

    print(f"[Time {t}] Pseudocells shape: {pcs.shape}")

# -----------------------
# Step 3: Temporal alignment
# -----------------------
aligner = TemporalAligner(NUM_ALIGN_CLUSTERS)
aligned_labels, centers = aligner.align(pseudocells_by_time)

# -----------------------
# Results
# -----------------------
print("\n=== Alignment Results ===")
for t, labels in aligned_labels.items():
    print(f"Time {t} aligned clusters:", labels)

print("\nGlobal aligned centers shape:", centers.shape)