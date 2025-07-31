import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

output_path = "outputs/conference/market_price_vs_soc4.png"

sns.set(style="whitegrid")

# California
vector_hourly = [40.12, 42.28, 43.46, 46.59, 60.65, 65.67, 70.41, 85.57, 142.20, 248.21, 116.28, 82.23]
vector_15min = [
                35.0,
                33.923394187871416,
                32.84678837574283,
                31.770182563614252,
                30.693576751485672,
                29.616970939357092,
                28.540365127228505,
                27.46375931509993,
                26.38715350297135,
                25.31054769084277,
                24.23394187871419,
                23.15733606658561,
                22.08073025445703,
                21.00517728460388,
                19.929624314750736,
                19.845274165604746,
                19.760924016458755,
                21.196424016458757,
                22.631924016458758,
                24.06742401645876,
                25.50292401645876,
                26.93842401645876,
                28.373924016458762,
                29.809424016458763,
                31.244924016458764,
                31.498561092558354,
                32.93406109255835,
                34.36956109255835,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173,
                34.97890689655173
            ]

# Create x-axis values:
# For the hourly vector: assume 12 hours starting at 0
x_hourly = np.arange(10, 10+len(vector_hourly))  # 0, 1, ..., 11
# For the 15-min vector: 12 hours with 15-min intervals: 0, 0.25, ..., 12.0 (49 points)
x_15min = np.arange(10, 22 + 0.25, 0.25)

# Convert vectors to numpy arrays
vector_hourly = np.array(vector_hourly)
vector_15min = np.array(vector_15min)

# Create the plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot the hourly series on the left y-axis (blue)
l1, = ax1.plot(x_hourly, vector_hourly, linestyle='-', color='tab:blue', label='Market prices')
ax1.set_xlabel("Time (h)", labelpad=20, fontsize=22)
ax1.set_ylabel("Energy Price ($/MWh)", color='blue', labelpad=20, fontsize=22)
ax1.tick_params(axis='both', which='major', labelsize=22)
# Set left y-axis tick labels to blue
ax1.tick_params(axis='y', which='both', colors='blue')
ax1.set_xticks(np.arange(10, 23, 1)) 

# Create a second y-axis for the 15-min series (orange)
ax2 = ax1.twinx()
l2, = ax2.plot(x_15min, vector_15min, linestyle='-', color='tab:orange', label='SoC of EV 4')
ax2.set_ylabel("State of Charge (kWh)", color='orange', labelpad=15, fontsize=22)
# Set right y-axis tick labels to orange
ax2.tick_params(axis='y', which='both', colors='orange', labelsize=22)

# # Create a combined legend
lines = [l1, l2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, fontsize=22, loc='upper left')

# Add a grid
ax1.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(output_path)
plt.close()
logging.info(f"Plot 'time_vectors' saved to {output_path}")
