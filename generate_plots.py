import matplotlib.pyplot as plt
import numpy as np
import os

# Create assets folder if not exists
if not os.path.exists("assets"):
    os.makedirs("assets")

# --- DATA FROM YOUR JSON RESULTS ---
# Text Retrieval (R@10)
tasks_text = ['Short Text (SciFact)', 'Long Context (LoCo/Qasper)']
baseline_text = [0.220, 0.950]  # SBERT
dragon_text = [0.185, 0.920]    # Dragon (Compressed)

# Vision Retrieval (R@1)
tasks_vis = ['Logic Test (Captions)', 'Real World (E2E Agent)']
baseline_vis = [0.680, 0.717]   # CLIP
dragon_vis = [0.662, 0.460]     # Dragon

# --- PLOT 1: TEXT RETRIEVAL ---
x = np.arange(len(tasks_text))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, baseline_text, width, label='Baseline (SBERT)', color='#bdc3c7')
rects2 = ax.bar(x + width/2, dragon_text, width, label='Dragon (Compressed)', color='#e74c3c')

ax.set_ylabel('Recall @ 10')
ax.set_title('Text Retrieval: Compression Efficiency')
ax.set_xticks(x)
ax.set_xticklabels(tasks_text)
ax.legend()
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')
plt.tight_layout()
plt.savefig('assets/benchmark_text.png')
print("Generated assets/benchmark_text.png")

# --- PLOT 2: VISION RETRIEVAL ---
x = np.arange(len(tasks_vis))

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, baseline_vis, width, label='Baseline (CLIP)', color='#bdc3c7')
rects2 = ax.bar(x + width/2, dragon_vis, width, label='Dragon (Agent)', color='#8e44ad')

ax.set_ylabel('Recall @ 1')
ax.set_title('Vision Retrieval: Logic vs. Reality')
ax.set_xticks(x)
ax.set_xticklabels(tasks_vis)
ax.legend()
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')
plt.tight_layout()
plt.savefig('assets/benchmark_vision.png')
print("Generated assets/benchmark_vision.png")