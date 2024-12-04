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
        # Remove the last value of time axis to make the vectors the same length
        time_axis = time_axis[:-1]

        # Plot both metrics on the same axis
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Objective/Cost [$]")
        ax1.plot(time_axis, operator_objective, label="Operator Objective", color="tab:blue")
        ax1.plot(time_axis, energy_costs, label="Energy Costs", color="tab:orange")
        ax1.tick_params(axis="y")

        # Add a legend to distinguish the lines
        ax1.legend(loc="best")

        # Finalize plot
        fig.suptitle("Operator Objective vs. Energy Costs")
        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")


    @staticmethod
    def plot_soc_evolution(schedule_results, time_axis, output_path, ev_config):
        sns.set(style="whitegrid")
        ev_ids = list(schedule_results["soc_over_time"].keys())
        num_evs = len(ev_ids)

        # Create the figure and subplots for each EV
        fig, axes = plt.subplots(
            num_evs, 1, figsize=(14, 1.5 * num_evs), sharex=True, gridspec_kw={'hspace': 0.7}
        )
        if num_evs == 1:
            axes = [axes]  # Ensure axes is iterable when there is only one EV

        for ax, ev_id in zip(axes, ev_ids):
            # Get the EV-specific configuration
            ev = next(ev for ev in ev_config["evs"] if ev["id"] == ev_id)
            battery_capacity = ev["battery_capacity"]
            min_soc = ev["min_soc"]

            # Normalize SoC using battery capacity and min_soc
            raw_soc = np.array(schedule_results["soc_over_time"][ev_id])
            if battery_capacity != min_soc:
                normalized_soc = (raw_soc - min_soc) / (battery_capacity - min_soc)
            else:
                normalized_soc = np.zeros_like(raw_soc)

            # Plot using imshow with interpolation
            ax.imshow(
                [normalized_soc],  # Data is a 2D array with one row
                aspect='auto',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                extent=[time_axis[0], time_axis[-1], 0, 1],
                interpolation='bilinear',
                origin='lower'
            )

            # Add ticks for desired and actual disconnect times
            desired_time = schedule_results["desired_disconnect_time"][ev_id]
            actual_time = schedule_results["actual_disconnect_time"][ev_id]

            ax.axvline(desired_time, color='magenta', linewidth=4, linestyle='-', label='Desired Disconnect')
            ax.axvline(actual_time, color='darkblue', linewidth=4, linestyle='-', label='Actual Disconnect')

            # Customize the subplot
            ax.set_yticks([])
            ax.set_ylabel(ev_id, rotation=0, labelpad=15, fontsize=10, ha='center', va='center')
            ax.set_xlim(time_axis[0], time_axis[-1])

        # Shared x-axis labels
        plt.xlabel("Time", fontsize=12)
        plt.xticks(time_axis, labels=[f"{t:.1f}" for t in time_axis])

        # Add a single colorbar
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1)),
            ax=axes,
            orientation='vertical',
            fraction=0.03,
            pad=0.1  # Enlarge the space between the color legend and the plots
        )
        cbar.set_label('Normalized State of Charge (SoC)', fontsize=12)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['0 %', '100 %'])

        # Add a vertical "EVs" label on the left
        fig.text(0.04, 0.5, 'EVs', va='center', rotation='vertical', fontsize=12)

        # Add a title
        fig.suptitle("State of Charge (SoC) Evolution Over Time (Normalized)", fontsize=14, y=1.02)

        # Add a legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', fontsize=10)

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {output_path}")


    @staticmethod
    def plot_market_prices(market_prices, time_range, output_path):
        """
        Plot the day-ahead market prices in $/MWh for each hour of the specified time range.
        
        Args:
            market_prices (list): List of market prices corresponding to each hour in the time range.
            time_range (list): List with start and end hours [start, end].
            output_path (str): Path to save the output plot.
        """
        sns.set(style="whitegrid")
        hours = list(range(time_range[0], time_range[1]))  # Generate the time axis based on the time range

        # Ensure the lengths of market_prices and hours match
        if len(market_prices) != len(hours):
            raise ValueError("Length of market_prices does not match the length of the time range.")

        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(hours, market_prices, marker='o', linestyle='-', color='tab:blue')

        # Customize the plot
        plt.title("Day-Ahead Market Prices", fontsize=14)
        plt.xlabel("Hour of Day", fontsize=12)
        plt.ylabel("Price ($/kWh)", fontsize=12)
        plt.xticks(hours)
        plt.grid(True, linestyle='--', linewidth=0.5)

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Market prices plot saved to {output_path}")


    @staticmethod
    def plot_summary_bars(results_by_experiment, output_path):
        """
        Plot a bar chart summarizing the operator objective and energy costs
        for each experiment type with 95% confidence intervals.

        Args:
            results_by_experiment (dict): Dictionary where keys are experiment types,
                                        and values are lists of results for all runs.
            output_path (str): Path to save the output plot.
        """
        sns.set(style="whitegrid")

        experiment_types = list(results_by_experiment.keys())
        objectives = []
        costs = []
        ci_objectives = []
        ci_costs = []

        for xp_type in experiment_types:
            xp_results = results_by_experiment[xp_type]
            operator_objectives = [res["sum_operator_objective"] for res in xp_results]
            energy_costs = [res["sum_energy_costs"] for res in xp_results]

            # Calculate means and 95% CI
            objectives.append(np.mean(operator_objectives))
            costs.append(np.mean(energy_costs))
            ci_objectives.append(1.96 * np.std(operator_objectives) / np.sqrt(len(operator_objectives)))
            ci_costs.append(1.96 * np.std(energy_costs) / np.sqrt(len(energy_costs)))

        # Bar positions
        x = np.arange(len(experiment_types))
        width = 0.35  # Bar width

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, objectives, width, label="Operator Objective", color="blue", yerr=ci_objectives, capsize=5)
        bars2 = ax.bar(x + width / 2, costs, width, label="Energy Costs", color="orange", yerr=ci_costs, capsize=5)

        # Add labels, title, and legend
        ax.set_xlabel("Experiment Type", fontsize=12)
        ax.set_ylabel("Value ($)", fontsize=12)
        ax.set_title("Summary of Operator Objective and Energy Costs", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(experiment_types, fontsize=10)
        ax.legend(fontsize=10)

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Summary bar plot saved to {output_path}")
