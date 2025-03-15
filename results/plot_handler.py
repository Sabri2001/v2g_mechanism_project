import logging
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd


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

        fig, axes = plt.subplots(
            num_evs, 1, figsize=(14, 1.5 * num_evs), sharex=True,
            gridspec_kw={'hspace': 0.7}
        )
        if num_evs == 1:
            axes = [axes]

        for ax, ev_id in zip(axes, ev_ids):
            # Fetch EV config and normalize SoC
            ev = next(ev for ev in ev_config["evs"] if ev["id"] == ev_id)
            battery_capacity = ev["battery_capacity"]
            min_soc = 0
            raw_soc = np.array(schedule_results["soc_over_time"][ev_id])
            normalized_soc = (
                (raw_soc - min_soc) / (battery_capacity - min_soc)
                if battery_capacity != min_soc else np.zeros_like(raw_soc)
            )

            # Plot SoC as an image
            ax.imshow(
                [normalized_soc],
                aspect='auto',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                extent=[time_axis[0], time_axis[-1], 0, 1],
                interpolation='bilinear',
                origin='lower'
            )

            # Desired / actual disconnection lines
            desired_time = schedule_results["desired_disconnection_time"][ev_id]
            actual_time = schedule_results["actual_disconnection_time"][ev_id]
            ax.axvline(desired_time, color='magenta', linewidth=4, linestyle='-',
                    label='Desired Disconnection')
            ax.axvline(actual_time, color='darkblue', linewidth=4, linestyle='-',
                    label='Actual Disconnection')

            # Hide y‐axis ticks and label each row with EV ID
            ax.set_yticks([])
            ax.set_ylabel(str(ev_id), rotation=0, labelpad=15, fontsize=12,
                        ha='center', va='center')
            ax.set_xlim(time_axis[0], time_axis[-1])

        # Label the shared x axis
        plt.xlabel("Time", fontsize=14, labelpad=25)
        for ax in axes:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x)}:00")
            )

        # Adjust layout to leave room on the right for the colorbar,
        # and some space at the bottom for the legend
        fig.subplots_adjust(left=0.12,
                            right=0.84,
                            bottom=0.15,
                            top=0.95)

        # Colorbar on the right
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1)),
            ax=axes,
            orientation='vertical',
            fraction=0.03,
            pad=0.07,
            shrink=0.8
        )
        cbar.set_label('State of Charge (%)', fontsize=10)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['0 %', '100 %'])

        # EVs label slightly further from plots (tweak x position)
        fig.text(0.05, 0.55, 'EVs', va='center', rotation='vertical', fontsize=14)

        # Retrieve the line handles for desired/actual and place legend below
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.45, -0.05),
            ncol=2,
            fontsize=10
        )

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
    def plot_total_cost_and_energy(results_by_experiment, output_path):
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
    def plot_total_cost(results_by_experiment, output_path):
        """
        Plot a bar chart summarizing the total operator costs
        for each experiment type with 95% confidence intervals.

        Args:
            results_by_experiment (dict): Dictionary where keys are experiment types,
                                        and values are lists of results for all runs.
            output_path (str): Path to save the output plot.
        """
        sns.set(style="whitegrid")

        experiment_types = list(results_by_experiment.keys())
        operator_costs = []
        ci_operator_costs = []

        for xp_type in experiment_types:
            xp_results = results_by_experiment[xp_type]
            operator_cost = [res["sum_operator_costs"] for res in xp_results]

            # Calculate mean and 95% CI for operator costs
            operator_costs.append(np.mean(operator_cost))
            ci_operator_costs.append(1.96 * np.std(operator_cost) / np.sqrt(len(operator_cost)))

        # Bar positions
        x = np.arange(len(experiment_types))
        width = 0.5  # Adjusted bar width since only one bar is plotted per experiment type

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, operator_costs, width, label="Total Costs",
            color="blue", yerr=ci_operator_costs, capsize=5)

        # Add labels, title, and legend
        ax.set_xlabel("Experiment Type", fontsize=12)
        ax.set_ylabel("Value ($)", fontsize=12)
        # ax.set_title("Summary of Operator Costs", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['ADMM', 'Optimal'], fontsize=10)
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Plot 'summary_bars' saved to {output_path}")

    @staticmethod
    def plot_gap_violin(results_by_experiment, output_path):
        """
        Compute the percentage gap between ADMM and Optimal solutions,
        and display it as a violin plot across all experiment runs.

        Args:
            results_by_experiment (dict): Dict of lists of run results.
            output_path (str): Path to save the output plot.
        """
        sns.set(style="whitegrid")

        admm_costs = [run["sum_operator_costs"] for run in results_by_experiment["coordinated"]]
        opt_costs = [run["sum_operator_costs"] for run in results_by_experiment["centralized"]]

        gaps = []
        for admm, opt in zip(admm_costs, opt_costs):
            gap = 0.0 if opt == 0 else 100.0 * (admm - opt) / opt
            gaps.append(gap)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(x=gaps, ax=ax, orient='h', cut=0)

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)

        # Adjust padding and remove title
        ax.set_xlabel("Gap (%)", fontsize=12, labelpad=15)
        ax.set_ylabel("", labelpad=15)  # No y-label, but increased padding if needed

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Violin plot of gaps saved to {output_path}")

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
        - alpha_values (y-axis): candidate disconnection_time_flexibility bids.
        - cost_grid: a 2D array (shape: len(alpha_values) x len(tau_values))
        where each cell is the EV's utility computed as energy_cost + congestion_cost + adaptability_cost.
        - If true_tau and true_alpha are provided, the cell matching these values is highlighted with a thick blue border.
        The colormap runs from green (low cost) to red (high cost).
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        # Use the reversed RdYlGn colormap so that low values are green and high values are red.
        cmap = "RdYlGn_r"
        ax = sns.heatmap(cost_grid, xticklabels=tau_values, yticklabels=alpha_values, cmap=cmap, annot=True, fmt=".2f")
        plt.xlabel("Bid: disconnection time", labelpad=15)  # Increased labelpad for x-axis
        plt.ylabel("Bid: adaptability coefficient", labelpad=15)  # Increased labelpad for y-axis
        # plt.title("Cost of Target EV as a Function of its Bid")
        
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

    @staticmethod
    def plot_cost_savings_vs_alpha(plot_data, output_path):
        """
        Expects plot_data as a dictionary where keys are battery_factor values and values are dictionaries with:
        - "alpha": list of alpha_factor values (x-axis)
        - "mean": list of mean cost savings (%) for that battery_factor
        - "std": list of standard deviations of cost savings
        Each battery_factor will be represented as a separate line on the plot.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for battery_factor, data in plot_data.items():
            plt.errorbar(
                data["alpha"], data["mean"], yerr=data["std"],
                label=f'Battery Factor {battery_factor}', marker='o', capsize=5
            )
        plt.xlabel("Alpha Factor")
        plt.ylabel("Cost Savings (%)")
        # plt.title("Cost Savings vs. Alpha Factor for different Battery Factors")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        import logging
        logging.info(f"Plot 'cost_savings_vs_alpha' saved to {output_path}")

    @staticmethod
    def plot_cost_savings_bars(chart_data, output_path):
        """
        chart_data is a dict of dicts, e.g.:
        {
            xp_label: {
            bf_value: [list of runs' total_costs],
            ...
            },
            ...
        }
        We'll produce a grouped bar chart:
        x-axis: xp_label (in alphabetical or custom order)
        for each xp_label group, 2 bars for battery_factor=1.0, 0.25 (colors).
        Error bars = std of each [list of runs].
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        sns.set(style="whitegrid")

        xp_labels = list(chart_data.keys())
        # We'll define battery_factors in a consistent order. 
        battery_factors = [1.0, 0.25]

        x = np.arange(len(xp_labels))  # positions for xp groups
        width = 0.3  # width of each bar

        fig, ax = plt.subplots(figsize=(10, 6))

        # For each battery_factor, we offset the bar positions slightly
        offsets = {
            1.0: -width/2,
            0.25: width/2
        }
        colors = {
            1.0: 'blue',
            0.25: 'orange'
        }

        # We'll loop over battery_factors, then plot bars for each xp_label
        for bf in battery_factors:
            means = []
            stds = []
            for xp_label in xp_labels:
                runs_list = chart_data[xp_label].get(bf, [])
                if runs_list:
                    mean_val = np.mean(runs_list)
                    std_val = np.std(runs_list)
                else:
                    mean_val = 0
                    std_val = 0
                means.append(mean_val)
                stds.append(std_val)

            # compute the bar positions for this battery_factor
            bar_positions = x + offsets[bf]

            # plot
            ax.bar(
                bar_positions,
                means,
                width,
                label=f"BF={bf}",
                yerr=stds,
                capsize=5,
                color=colors[bf],
                alpha=0.7
            )

        ax.set_xticks(x)
        ax.set_xticklabels(xp_labels, fontsize=10)

        # Increase labelpad for more spacing to the axes
        ax.set_xlabel("Experiment Setup", fontsize=12, labelpad=15)
        ax.set_ylabel("Total Cost ($)", fontsize=12, labelpad=15)

        ax.legend(title="Battery Factor")

        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()

        import logging
        logging.info(f"Plot 'cost_savings_bars' saved to {output_path}")

    @staticmethod
    def plot_fake_alpha_tau_bars(chart_data, output_path):
        """
        Bar chart with 3 bars (one for each scenario),
        plus std error bars from the run-to-run variation.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        sns.set(style="whitegrid")

        xp_labels = list(chart_data.keys()) 
        means = []
        stds = []
        for xp_label in xp_labels:
            runs_list = chart_data[xp_label]
            mean_val = np.mean(runs_list)
            std_val  = np.std(runs_list)
            means.append(mean_val)
            stds.append(std_val)

        x_positions = np.arange(len(xp_labels))
        width = 0.5  # single bar width

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            x_positions,
            means,
            width,
            yerr=stds,
            capsize=5,
            color=["blue", "orange", "green"][:len(xp_labels)],
            alpha=0.7
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xp_labels, fontsize=10)
        ax.set_xlabel("Scenario", fontsize=12, labelpad=10)
        ax.set_ylabel("Total Cost ($)", fontsize=12, labelpad=10)

        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()

        import logging
        logging.info(f"Plot 'fake_alpha_tau_bars' saved to {output_path}")

    @staticmethod
    def plot_admm_iterations_violin(results_by_experiment, output_path):
        """
        Produces a horizontal violin plot of the ADMM iteration counts
        for the 'coordinated' experiment, separated by nu_multiplier (1.0 vs 1.1).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import logging

        coordinated_runs = results_by_experiment.get("coordinated", [])

        data_rows = []
        for run_results in coordinated_runs:
            iters = run_results.get("admm_iterations", None)
            nu_mult = run_results.get("nu_multiplier", None)

            if iters is not None and nu_mult is not None and nu_mult in [1.0, 1.1]:
                data_rows.append({
                    "nu_multiplier": nu_mult,
                    "admm_iterations": iters
                })

        if not data_rows:
            logging.warning("No ADMM iteration data (with nu_multiplier=1.0 or 1.1) found. Skipping plot.")
            return

        df = pd.DataFrame(data_rows)

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.violinplot(
            data=df,
            y="nu_multiplier",
            x="admm_iterations",
            orient="h",
            cut=0,
            ax=ax
        )

        # Increased padding around axis labels
        ax.set_xlabel("ADMM Iterations", fontsize=12, labelpad=15)
        ax.set_ylabel("nu_multiplier", fontsize=12, labelpad=15)
        # ax.set_title(...) removed as requested

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Plot 'admm_iterations_violin' saved to {output_path}")

    @staticmethod
    def plot_vcg_tax_violin(results_by_experiment, output_path):
        """
        Creates a violin + swarm plot on a symlog x-axis for total VCG taxes
        (as a percentage of total energy costs) across all runs/experiment types.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        import logging

        # Collect VCG tax percentages
        vcg_tax_percentages = []
        for xp_type, runs_list in results_by_experiment.items():
            for run_results in runs_list:
                if "vcg_tax" in run_results:
                    total_vcg_tax = sum(run_results["vcg_tax"].values())
                    total_energy_cost = run_results.get("sum_energy_costs", 0.0)
                    if total_energy_cost != 0.0:
                        pct = (total_vcg_tax / total_energy_cost) * 100
                        vcg_tax_percentages.append(pct)
                    else:
                        logging.warning("Total energy cost is zero; skipping run.")

        if not vcg_tax_percentages:
            logging.warning("No VCG tax data found. Skipping 'plot_vcg_tax_violin'.")
            return

        # Prepare data for Seaborn
        df = pd.DataFrame({
            "value": vcg_tax_percentages,
            "category": ["VCG Tax Distribution"] * len(vcg_tax_percentages)
        })

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))

        # Violin plot (horizontal) for distribution shape
        sns.violinplot(
            x="value",
            y="category",
            data=df,
            orient="h",
            ax=ax,
            inner=None,  # no internal lines
            color="lightblue",
            cut=0         # limit the kernel extension
        )

        # Swarm plot to show individual points
        sns.swarmplot(
            x="value",
            y="category",
            data=df,
            orient="h",
            ax=ax,
            color="black",
            size=4
        )

        # Symmetrical log scale to handle negative & large positive values
        ax.set_xscale("symlog", linthresh=1)

        # Reference lines
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
        mean_val = np.mean(vcg_tax_percentages)
        ax.axvline(mean_val, color="blue", linestyle="-", linewidth=1,
                label=f"Mean = {mean_val:.2f}")
        ax.legend()

        # Hide the single category label
        ax.set_ylabel("")      # remove y-axis label
        ax.set_yticks([])      # remove the category text on the y-axis

        # Increase space between x-axis label and the plot
        ax.set_xlabel("VCG Tax (% of Energy Costs)", fontsize=12, labelpad=20)

        # Final layout and save
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Plot 'vcg_tax_violin' saved to {output_path}")

    @staticmethod
    def plot_warmstart(times_dict, output_path):
        """
        Creates a boxplot of partial scheduling times
        for ADMM vs. Gurobi, comparing No Warmstart vs. Warmstart.

        times_dict keys:
        - "coord_no_ws": ADMM partial no-warmstart times (list of floats)
        - "coord_ws": ADMM partial warmstart times (list of floats)
        - "cent_no_ws": Gurobi partial no-warmstart times (list of floats)
        - "cent_ws": Gurobi partial warmstart times (list of floats)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        # DataFrame construction
        df_plot = pd.concat([
            pd.DataFrame({"time": times_dict["coord_no_ws"], "method": "ADMM", "warmstart": "No Warmstart"}),
            pd.DataFrame({"time": times_dict["coord_ws"], "method": "ADMM", "warmstart": "Warmstart"}),
            pd.DataFrame({"time": times_dict["cent_no_ws"], "method": "Gurobi", "warmstart": "No Warmstart"}),
            pd.DataFrame({"time": times_dict["cent_ws"], "method": "Gurobi", "warmstart": "Warmstart"}),
        ], ignore_index=True)

        # Plot
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(
            x="method",
            y="time",
            hue="warmstart",
            data=df_plot,
            palette="Set2",
            ax=ax
        )

        # Labels and padding
        ax.set_xlabel("Method", fontsize=12, labelpad=15)
        ax.set_ylabel("Computation Time (s)", fontsize=12, labelpad=15)

        # Logarithmic y-axis
        ax.set_yscale("log")

        ax.legend(title="")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # @staticmethod
    # def plot_warmstart(times_dict, output_path):
    #     """
    #     Creates a grouped bar chart (mean ± std) of partial scheduling times
    #     for ADMM vs. Gurobi, comparing No Warmstart vs. Warmstart.
        
    #     times_dict keys:
    #     - "coord_no_ws": ADMM partial no-warmstart times
    #     - "coord_ws": ADMM partial warmstart times
    #     - "cent_no_ws": Gurobi partial no-warmstart times
    #     - "cent_ws": Gurobi partial warmstart times
    #     """
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     import seaborn as sns

    #     sns.set(style="whitegrid")

    #     # Means and stds
    #     coord_no_ws_mean = np.mean(times_dict["coord_no_ws"])
    #     coord_no_ws_std  = np.std(times_dict["coord_no_ws"])
    #     coord_ws_mean    = np.mean(times_dict["coord_ws"])
    #     coord_ws_std     = np.std(times_dict["coord_ws"])

    #     cent_no_ws_mean = np.mean(times_dict["cent_no_ws"])
    #     cent_no_ws_std  = np.std(times_dict["cent_no_ws"])
    #     cent_ws_mean    = np.mean(times_dict["cent_ws"])
    #     cent_ws_std     = np.std(times_dict["cent_ws"])

    #     # Two categories on the x-axis: 0 -> ADMM, 1 -> Gurobi
    #     x = np.arange(2)
    #     width = 0.35

    #     # We define arrays for No Warmstart and Warmstart
    #     no_ws_means = [coord_no_ws_mean, cent_no_ws_mean]
    #     no_ws_stds  = [coord_no_ws_std,  cent_no_ws_std]
    #     ws_means    = [coord_ws_mean,    cent_ws_mean]
    #     ws_stds     = [coord_ws_std,     cent_ws_std]

    #     fig, ax = plt.subplots(figsize=(8, 5))

    #     # Plot No Warmstart bars (shift left)
    #     ax.bar(
    #         x - width/2, no_ws_means, width,
    #         yerr=no_ws_stds, capsize=5,
    #         alpha=0.8, label="No Warmstart"
    #     )

    #     # Plot Warmstart bars (shift right)
    #     ax.bar(
    #         x + width/2, ws_means, width,
    #         yerr=ws_stds, capsize=5,
    #         alpha=0.8, label="Warmstart"
    #     )

    #     # X-axis labels
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(["ADMM", "Gurobi"], fontsize=11)

    #     ax.set_ylabel("Computation Time (s)", fontsize=12)
    #     ax.set_title("Partial Scheduling: No Warmstart vs. Warmstart", fontsize=13)

    #     ax.legend(fontsize=10)

    #     plt.tight_layout()
    #     plt.savefig(output_path)
    #     plt.close()

    # @staticmethod
    # def plot_vcg_tax_violin(results_by_experiment, output_path):
        # """
        # Creates a violin plot of total VCG taxes as a percentage of total energy costs,
        # for all runs (across all experiment types) in a single distribution.
        
        # The percentage is computed as:
        #     100 * ( sum of all EVs' vcg_tax ) / ( sum_energy_costs ).
        # """
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # import numpy as np
        # import logging

        # # Gather VCG tax % data from all experiment types (if present).
        # vcg_tax_percentages = []
        # for xp_type, runs_list in results_by_experiment.items():
        #     for run_results in runs_list:
        #         # Only consider runs that actually have vcg_tax computed
        #         if "vcg_tax" in run_results:
        #             total_vcg_tax = sum(run_results["vcg_tax"].values())
        #             print(f"Total VCG tax for run: {total_vcg_tax}")
        #             total_energy_cost = run_results.get("sum_energy_costs", 0.0)
        #             print(f"Total energy cost for run: {total_energy_cost}")

        #             if total_energy_cost != 0.0:
        #                 vcg_tax_pct = (total_vcg_tax / total_energy_cost) * 100
        #                 vcg_tax_percentages.append(vcg_tax_pct)
        #             else:
        #                 print("Warning: Total energy cost is zero, skipping this run.")
        #                 pass

        # if not vcg_tax_percentages:
        #     logging.warning("No VCG tax data found to plot. Skipping 'plot_vcg_tax_violin'.")
        #     return

        # # Create a single violin plot, similarly to plot_gap_violin
        # sns.set(style="whitegrid")
        # fig, ax = plt.subplots(figsize=(8, 5))

        # # Plot horizontally with cut=0 to avoid extending tails
        # sns.violinplot(x=vcg_tax_percentages, orient='h', cut=0, ax=ax)
        
        # # Optional vertical reference line at 0
        # ax.axvline(x=0, color='red', linestyle='--', linewidth=1)

        # ax.set_xlabel("VCG Tax (% of Energy Costs)", fontsize=12)
        # # ax.set_title("Violin Plot: Distribution of VCG Taxes as % of Energy Costs", fontsize=14)

        # plt.tight_layout()
        # plt.savefig(output_path)
        # plt.close()
        # logging.info(f"Plot 'vcg_tax_violin' saved to {output_path}")