import logging
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd


class PlotHandler:
    @staticmethod
    def plot_soc_evolution(schedule_results, time_axis, output_path, ev_config):
        import matplotlib.ticker as ticker
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
            offset = 0.05  # adjust as needed based on time scale resolution
            ax.axvline(desired_time - offset, color='magenta', linewidth=4, linestyle='-',
                    label='Desired Disconnection')
            ax.axvline(actual_time + offset, color='darkblue', linewidth=4, linestyle='-',
                    label='Actual Disconnection')

            # Hide y-axis ticks and label each row with EV ID
            ax.set_yticks([])
            ax.set_ylabel(str(ev_id + 1), rotation=0, labelpad=15, fontsize=30,
                        ha='center', va='center')
            ax.set_xlim(time_axis[0], time_axis[-1])

        # Label the shared x-axis with "Time (h)" and increased font size
        plt.xlabel("Time (h)", fontsize=30, labelpad=30)
        for ax in axes:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            # Format hours as two-digit numbers (e.g., "10") without ":00"
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x):02d}")
            )
            ax.tick_params(axis='x', labelsize=30)

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
        cbar.set_label('State of Charge (%)', fontsize=30)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['0 %', '100 %'])
        cbar.ax.tick_params(labelsize=30)

        # EVs label slightly further from plots
        fig.text(0.05, 0.55, 'EVs', va='center', rotation='vertical', fontsize=30)

        # Retrieve the line handles for desired/actual and place legend further below the plot
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.45, -0.20),
            ncol=2,
            fontsize=30
        )

        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot 'soc_evolution' saved to {output_path}")
