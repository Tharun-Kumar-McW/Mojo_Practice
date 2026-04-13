import matplotlib.pyplot as plt

# Data
data = {
    '100': {'Serial': 0.016, 'Parallel': 0.264, 'Parallel-SIMD': 0.269},
    '10000': {'Serial': 0.031, 'Parallel': 0.284, 'Parallel-SIMD': 0.252},
    '1000000': {'Serial': 0.683, 'Parallel': 0.415, 'Parallel-SIMD': 0.328},
    '100000000': {'Serial': 49.365, 'Parallel': 33.316, 'Parallel-SIMD': 32.155}
}

# Colors for each method
colors = {
    'Serial': 'skyblue',
    'Parallel': 'orange',
    'Parallel-SIMD': 'green'
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
    plt.title(f"Dot Product Comparison (Input Size {size})")

    plt.ylim(0, max(times) + (max(times) * 0.2))  # spacing on top

    plt.show()