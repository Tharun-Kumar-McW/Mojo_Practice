import matplotlib.pyplot as plt

# Data
data = {
    '32 x 32': {'Serial': 0.108, 'Loop_Reordered': 0.102, 'Loop_Tiling': 0.065},
    '64 x 64': {'Serial': 0.576, 'Loop_Reordered': 0.407, 'Loop_Tiling': 0.275},
    '128 x 128': {'Serial': 3.214, 'Loop_Reordered': 2.779, 'Loop_Tiling': 2.100},
    '256 x 256': {'Serial': 32.200, 'Loop_Reordered': 22.773, 'Loop_Tiling': 15.343},
    '512 x 512': {'Serial': 298.448, 'Loop_Reordered': 191.247, 'Loop_Tiling': 128.307},
    '1024 x 1024': {'Serial': 4415.529, 'Loop_Reordered': 1555.253, 'Loop_Tiling': 1026.054},
}

# Colors for each method
colors = {
    'Serial': 'skyblue',
    'Loop_Reordered': 'orange',
    'Loop_Tiling': 'green'
}

# Plot
for size, values in data.items():
    methods = list(values.keys())
    times = list(values.values())

    plt.figure()

    bars = plt.bar(methods, times, color=[colors[m] for m in methods])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.xlabel("Method")
    plt.ylabel("Time (ms)")
    plt.title(f"Scalar Dot Product Attention Comparison (Input Size {size})")

    plt.ylim(0, max(times) + (max(times) * 0.2))  # spacing on top

    plt.show()