import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class PlotHandler:
    @staticmethod
    def plot_objective_vs_cost(results, time_axis, output_path):
        sns.set(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(12, 6))

        operator_objective = results["operator_objective_vector"]
        energy_costs = results["energy_cost_vector"]

        # Operator Objective
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Operator Objective", color="tab:blue")
        ax1.plot(time_axis, operator_objective, label="Operator Objective", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Energy Costs
        ax2 = ax1.twinx()
        ax2.set_ylabel("Energy Costs", color="tab:orange")
        ax2.plot(time_axis, energy_costs, label="Energy Costs", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # Finalize plot
        fig.suptitle("Operator Objective vs. Energy Costs")
        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")

    @staticmethod
    def plot_soc_evolution(soc_over_time, time_axis, output_path, ev_config):
        sns.set(style="whitegrid")
        num_evs = len(soc_over_time)

        # Extract start and end times from the configuration
        start_time, end_time = ev_config["time_range"]
        num_time_steps = len(time_axis)
        delta_t = (end_time - start_time) / num_time_steps  # Time step size

        # Create time edges including the end_time
        time_edges = np.linspace(start_time, end_time, num_time_steps + 1)
        time_centers = (time_edges[:-1] + time_edges[1:]) / 2  # Midpoints for interpolation

        # Initialize the SoC matrix
        soc_matrix = np.zeros((num_evs, num_time_steps))
        ev_ids = list(soc_over_time.keys())

        for i, ev_id in enumerate(ev_ids):
            # Get the EV-specific configuration
            ev = next(ev for ev in ev_config["evs"] if ev["id"] == ev_id)
            battery_capacity = ev["battery_capacity"]
            min_soc = ev["min_soc"]

            # Interpolate the SoC values
            raw_soc = np.interp(time_centers, time_axis, soc_over_time[ev_id])

            # Normalize SoC using battery capacity and min_soc for this EV
            normalized_soc = (raw_soc - min_soc) / (battery_capacity - min_soc) if battery_capacity != min_soc else 0
            soc_matrix[i, :] = normalized_soc

        # Insert white space between rows by adding NaN rows
        num_rows = num_evs * 2 - 1
        expanded_soc_matrix = np.full((num_rows, num_time_steps), np.nan)

        for i in range(num_evs):
            expanded_soc_matrix[i * 2, :] = soc_matrix[i, :]

        # Flip the matrix vertically so that the first EV is at the top
        expanded_soc_matrix_flipped = np.flipud(expanded_soc_matrix)

        # Create the plot
        plt.figure(figsize=(12, num_evs))  # Adjust height dynamically based on number of EVs

        # Set extent to align the image pixels with the time edges
        extent = [start_time, end_time, 0, num_rows]

        # Plot the heatmap with fixed color range [0, 1]
        plt.imshow(expanded_soc_matrix_flipped, aspect='auto', cmap='RdYlGn', interpolation='none', extent=extent, 
                vmin=0, vmax=1)

        # Set x-ticks at the time edges, including the last time stamp
        plt.xticks(time_edges, labels=[f"{t:.1f}" for t in time_edges])
        plt.xlabel("Time")

        # Set y-ticks
        y_ticks = np.arange(0, num_rows, 2) + 0.5  # Adjust y-ticks to center labels
        y_tick_labels = np.flip(ev_ids)
        plt.yticks(y_ticks, y_tick_labels)

        # Set limits to match the extent
        plt.xlim(start_time, end_time)
        plt.ylim(0, num_rows)

        # Add colorbar with fixed range [0, 1]
        cbar = plt.colorbar()
        cbar.set_label('Normalized State of Charge (SoC)')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['0 %', '100 %'])

        # Set title
        plt.title("State of Charge (SoC) Evolution Over Time (Normalized)")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")
