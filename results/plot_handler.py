import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle


class PlotHandler:
    @staticmethod
    def plot_total_cost_vs_energy_cost(results, time_axis, output_path):
        sns.set(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(12, 6))

        operator_cost = results["operator_cost_over_time"]
        energy_costs = results["energy_cost_over_time"]
        # Remove the last value of time axis to make the vectors the same length
        time_axis = time_axis[:-1]

        # Plot both metrics on the same axis
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Total Cost/Energy Cost [$]")
        ax1.plot(time_axis, operator_cost, label="Total Costs", color="tab:blue")
        ax1.plot(time_axis, energy_costs, label="Energy Costs", color="tab:orange")
        ax1.tick_params(axis="y")

        # Add a legend to distinguish the lines
        ax1.legend(loc="best")

        # Finalize plot
        fig.suptitle("Total Costs vs. Energy Costs")
        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Plot 'total_cost_vs_energy_cost' saved to {output_path}")

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

            # Normalize SoC
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

            # Add lines for desired and actual disconnect times
            desired_time = schedule_results["desired_disconnection_time"][ev_id]
            actual_time = schedule_results["actual_disconnection_time"][ev_id]

            ax.axvline(desired_time, color='magenta', linewidth=4, linestyle='-', label='Desired Disconnect')
            ax.axvline(actual_time, color='darkblue', linewidth=4, linestyle='-', label='Actual Disconnect')

            # Customize the subplot
            ax.set_yticks([])
            ax.set_ylabel(str(ev_id), rotation=0, labelpad=15, fontsize=10, ha='center', va='center')
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
            pad=0.1
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
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot 'soc_evolution' saved to {output_path}")

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
        logging.info(f"Plot 'market_prices' saved to {output_path}")

    @staticmethod
    def plot_summary_bars(results_by_experiment, output_path):
        """
        Plot a bar chart summarizing the operator costs and energy costs
        for each experiment type with 95% confidence intervals.

        Args:
            results_by_experiment (dict): Dictionary where keys are experiment types,
                                          and values are lists of results for all runs.
            output_path (str): Path to save the output plot.
        """
        sns.set(style="whitegrid")

        experiment_types = list(results_by_experiment.keys())
        operator_costs = []
        energy_costs = []
        ci_operator_costs = []
        ci_energy_costs = []

        for xp_type in experiment_types:
            xp_results = results_by_experiment[xp_type]
            operator_cost = [res["sum_operator_costs"] for res in xp_results]
            energy_cost = [res["sum_energy_costs"] for res in xp_results]

            # Calculate means and 95% CI
            operator_costs.append(np.mean(operator_cost))
            energy_costs.append(np.mean(energy_cost))
            ci_operator_costs.append(1.96 * np.std(operator_cost) / np.sqrt(len(operator_cost)))
            ci_energy_costs.append(1.96 * np.std(energy_cost) / np.sqrt(len(energy_cost)))

        # Bar positions
        x = np.arange(len(experiment_types))
        width = 0.35  # Bar width

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, operator_costs, width, label="Operator Costs",
               color="blue", yerr=ci_operator_costs, capsize=5)
        ax.bar(x + width / 2, energy_costs, width, label="Energy Costs",
               color="orange", yerr=ci_energy_costs, capsize=5)

        # Add labels, title, and legend
        ax.set_xlabel("Experiment Type", fontsize=12)
        ax.set_ylabel("Value ($)", fontsize=12)
        ax.set_title("Summary of Operator Costs and Energy Costs", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(experiment_types, fontsize=10)
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Plot 'summary_bars' saved to {output_path}")

    @staticmethod
    def plot_cost_benchmarking_bars(results_by_experiment, output_path):
        """
        Plot a bar chart showing the percentage change in total_cost
        for each method compared to the uncoordinated method.

        Args:
            results_by_experiment (dict): Dictionary where keys are experiment types,
                                          and values are lists of results for all runs.
            output_path (str): Path to save the output plot.
        """
        sns.set(style="whitegrid")

        # Ensure 'uncoordinated' is present
        if "uncoordinated" not in results_by_experiment:
            raise ValueError("'uncoordinated' experiment type is required for percentage change plot.")

        # Extract uncoordinated results
        uncoord_results = results_by_experiment["uncoordinated"]
        uncoord_costs = [res["sum_operator_cost"] for res in uncoord_results]
        mean_uncoord = np.mean(uncoord_costs)

        if mean_uncoord == 0:
            raise ValueError("The mean operator cost for 'uncoordinated' is zero, cannot compute percentage changes.")

        # Prepare data for other experiment types
        experiment_types = [xp for xp in results_by_experiment.keys()]
        percentage_changes = []
        ci_percentage_changes = []
        colors = []

        for xp_type in experiment_types:
            xp_results = results_by_experiment[xp_type]
            xp_costs = [res["sum_operator_cost"] for res in xp_results]

            if not xp_costs:
                logging.warning(f"No results for experiment type '{xp_type}'. Skipping.")
                continue

            mean_xp = np.mean(xp_costs)
            pct_change = ((mean_xp - mean_uncoord) / mean_uncoord) * 100
            percentage_changes.append(pct_change)

            # Compute CI only if more than one run
            if len(xp_costs) > 1:
                ci = (1.96 * (np.std(xp_costs, ddof=1) / np.sqrt(len(xp_costs)))) * 100 / mean_uncoord
            else:
                ci = 0.0

            ci_percentage_changes.append(ci)
            colors.append("red" if pct_change >= 0 else "green")

        if not percentage_changes:
            logging.warning("No data available to plot percentage changes.")
            return

        # Dynamically set figure width based on the number of experiment types
        num_experiments = len(experiment_types)

        if num_experiments == 1:
            figsize = (8, 6)
            bar_width = 0.4
            x = np.array([0])
        else:
            width_per_bar = 0.8
            total_width = max(6, num_experiments * width_per_bar)
            figsize = (total_width, 6)
            bar_width = 0.6 if num_experiments > 1 else 0.4
            x = np.arange(len(experiment_types))

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x, percentage_changes, bar_width, yerr=ci_percentage_changes,
               capsize=5, color=colors, alpha=0.7)

        # Center single bar if there's only one experiment
        if num_experiments == 1:
            ax.set_xticks([0])
        else:
            ax.set_xticks(x)

        ax.set_xlabel("Experiment Type", fontsize=12)
        ax.set_ylabel("Percentage Change in Total Costs (%)", fontsize=12)
        ax.set_title("Benchmarking of Methods", fontsize=14, pad=15)
        ax.set_xticklabels(experiment_types, fontsize=10, rotation=45, ha='right')
        ax.axhline(0, color='black', linewidth=0.8)

        if num_experiments == 1:
            ax.set_xlim(-0.5, 0.5)
        else:
            ax.set_xlim(-0.5, num_experiments - 0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot 'cost_benchmarking_bars' saved to {output_path}")

    @staticmethod
    def plot_v2g_fraction_bars(results_by_experiment, output_path):
        """
        Plot a bar chart showing the fraction of energy transferred to V2G
        as a percentage of the energy needed to charge connected EVs for each experiment type.

        Args:
            results_by_experiment (dict): Dictionary where keys are experiment types,
                                          and values are lists of results for all runs.
            output_path (str): Path to save the output plot.
        """
        sns.set(style="whitegrid")

        experiment_types = list(results_by_experiment.keys())
        fractions = []
        ci_fractions = []
        colors = []

        for xp_type in experiment_types:
            xp_results = results_by_experiment[xp_type]
            fractions_list = [res["v2g_fraction"] for res in xp_results]

            if not fractions_list:
                logging.warning(f"No results for experiment type '{xp_type}'. Skipping.")
                continue

            mean_fraction = np.mean(fractions_list)
            fractions.append(mean_fraction)

            # Calculate 95% CI
            if len(fractions_list) > 1:
                ci = 1.96 * (np.std(fractions_list, ddof=1) / np.sqrt(len(fractions_list)))
            else:
                ci = 0.0

            ci_fractions.append(ci)
            colors.append("blue")  # or any color logic you prefer

        if not fractions:
            logging.warning("No data available to plot V2G fraction bars.")
            return

        num_experiments = len(experiment_types)
        if num_experiments == 1:
            figsize = (8, 6)
            x = np.array([0])
        else:
            width_per_bar = 0.8
            total_width = max(6, num_experiments * width_per_bar)
            figsize = (total_width, 6)
            x = np.arange(len(experiment_types))

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(x, fractions, 0.6, yerr=ci_fractions,
                      capsize=5, color=colors, alpha=0.7)

        ax.set_xlabel("Experiment Type", fontsize=12)
        ax.set_ylabel("Energy Transferred to V2G (%)", fontsize=12)
        ax.set_title("Energy Transferred to V2G as a Fraction of Energy Needed to Charge EVs", fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(experiment_types, fontsize=10, rotation=45, ha='right')
        ax.axhline(0, color='black', linewidth=0.8)

        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot 'v2g_fraction_bars' saved to {output_path}")
    
    @staticmethod
    def plot_payment_comparison(results, time_axis, output_path):
        """
        Plots a bar chart comparing the congestion_cost and vcg_tax for each EV.
        The x-axis corresponds to the EV IDs, and for each EV two bars are drawn:
        one for congestion_cost and one for vcg_tax.
        This plot is only generated if both 'congestion_cost' and 'vcg_tax' are available in results.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style="whitegrid")
        
        # Check if both congestion_cost and vcg_tax are available in results
        if "congestion_cost" not in results or "vcg_tax" not in results:
            plt.figure()
            plt.text(0.5, 0.5, "No payment data available", ha='center', va='center')
            plt.savefig(output_path)
            plt.close()
            import logging
            logging.warning("Payment comparison plot skipped because 'congestion_cost' or 'vcg_tax' not found in results.")
            return

        # Both keys exist: extract the dictionaries.
        congestion_cost = results["congestion_cost"]
        vcg_tax = results["vcg_tax"]
        
        # Determine the EV IDs that appear in both dictionaries.
        ev_ids = sorted(set(congestion_cost.keys()) & set(vcg_tax.keys()))
        if not ev_ids:
            import logging
            logging.warning("No matching EV ids found in congestion_cost and vcg_tax; skipping payment_comparison plot.")
            return

        # Prepare the data for plotting.
        congestion_values = [congestion_cost[ev] for ev in ev_ids]
        vcg_values = [vcg_tax[ev] for ev in ev_ids]

        x = np.arange(len(ev_ids))  # positions for EV groups
        width = 0.35  # width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, congestion_values, width, label='Congestion Cost', color='tab:blue')
        rects2 = ax.bar(x + width/2, vcg_values, width, label='VCG Tax', color='tab:orange')

        # Labeling
        ax.set_xlabel("EV id")
        ax.set_ylabel("Payment")
        ax.set_title("Payment Comparison: Congestion Cost vs. VCG Tax per EV")
        ax.set_xticks(x)
        ax.set_xticklabels(ev_ids)
        ax.legend()

        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()
        import logging
        logging.info(f"Plot 'payment_comparison' saved to {output_path}")

    @staticmethod
    def plot_nash_grid(tau_values, alpha_values, cost_grid, output_path, true_tau=None, true_alpha=None):
        """
        Plot a heatmap (grid) of the target EV's utility as a function of its bid parameters.
        - tau_values (x-axis): candidate disconnection_time bids.
        - alpha_values (y-axis): candidate disconnection_time_preference_coefficient bids.
        - cost_grid: a 2D array (shape: len(alpha_values) x len(tau_values))
          where each cell is the EV's utility computed as energy_cost + congestion_cost + adaptability_cost.
        - If true_tau and true_alpha are provided, the cell matching these values is highlighted with a thick yellow border.
        The colormap runs from green (low cost) to red (high cost).
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        # Use the reversed RdYlGn colormap so that low values are green and high values are red.
        cmap = "RdYlGn_r"
        ax = sns.heatmap(cost_grid, xticklabels=tau_values, yticklabels=alpha_values, cmap=cmap, annot=True, fmt=".2f")
        plt.xlabel("Bid: disconnection time")
        plt.ylabel("Bid: adaptability coefficient)")
        plt.title("Cost of Target EV as a Function of its Bid")
        
        # If true_tau and true_alpha are provided, highlight that cell.
        if true_tau is not None and true_alpha is not None:
            try:
                col_index = tau_values.index(true_tau)
                row_index = alpha_values.index(true_alpha)
                # The heatmap's coordinate system: each cell is 1 unit. Add a Rectangle patch.
                rect = Rectangle((col_index, row_index), 1, 1, fill=False, edgecolor='blue', lw=3)
                ax.add_patch(rect)
            except ValueError:
                logging.warning("True bid values not found in the candidate lists; skipping cell highlight.")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Plot 'nash_grid' saved to {output_path}")
