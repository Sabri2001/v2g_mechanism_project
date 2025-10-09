import logging
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd


class PlotHandler:
    # @staticmethod
    # def plot_soc_evolution(schedule_results, time_axis, output_path, ev_config):
    #     import matplotlib.ticker as ticker
    #     sns.set(style="whitegrid")
    #     ev_ids = list(schedule_results["soc_over_time"].keys())
    #     num_evs = len(ev_ids)

    #     fig, axes = plt.subplots(
    #         num_evs, 1, figsize=(14, 1.5 * num_evs), sharex=True,
    #         gridspec_kw={'hspace': 0.7}
    #     )
    #     if num_evs == 1:
    #         axes = [axes]

    #     for ax, ev_id in zip(axes, ev_ids):
    #         # Fetch EV config and normalize SoC
    #         ev = next(ev for ev in ev_config["evs"] if ev["id"] == ev_id)
    #         battery_capacity = ev["battery_capacity"]
    #         min_soc = 0
    #         raw_soc = np.array(schedule_results["soc_over_time"][ev_id])
    #         normalized_soc = (
    #             (raw_soc - min_soc) / (battery_capacity - min_soc)
    #             if battery_capacity != min_soc else np.zeros_like(raw_soc)
    #         )

    #         # Plot SoC as an image
    #         ax.imshow(
    #             [normalized_soc],
    #             aspect='auto',
    #             cmap='RdYlGn',
    #             vmin=0,
    #             vmax=1,
    #             extent=[time_axis[0], time_axis[-1], 0, 1],
    #             interpolation='bilinear',
    #             origin='lower'
    #         )

    #         # Desired / actual disconnection lines
    #         desired_time = schedule_results["desired_disconnection_time"][ev_id]
    #         actual_time = schedule_results["actual_disconnection_time"][ev_id]
    #         offset = 0.05  # adjust as needed based on time scale resolution
    #         ax.axvline(desired_time - offset, color='magenta', linewidth=4, linestyle='-',
    #                 label='Desired Disconnection')
    #         ax.axvline(actual_time + offset, color='darkblue', linewidth=4, linestyle='-',
    #                 label='Actual Disconnection')

    #         # Hide y-axis ticks and label each row with EV ID
    #         ax.set_yticks([])
    #         ax.set_ylabel(str(ev_id + 1), rotation=0, labelpad=15, fontsize=30,
    #                     ha='center', va='center')
    #         ax.set_xlim(time_axis[0], time_axis[-1])

    #     # Label the shared x-axis with "Time (h)" and increased font size
    #     plt.xlabel("Time (h)", fontsize=30, labelpad=30)
    #     for ax in axes:
    #         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #         # Format hours as two-digit numbers (e.g., "10") without ":00"
    #         ax.xaxis.set_major_formatter(
    #             ticker.FuncFormatter(lambda x, pos: f"{int(x):02d}")
    #         )
    #         ax.tick_params(axis='x', labelsize=30)

    #     fig.subplots_adjust(left=0.12,
    #                         right=0.84,
    #                         bottom=0.15,
    #                         top=0.95)

    #     # Colorbar on the right
    #     cbar = fig.colorbar(
    #         plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1)),
    #         ax=axes,
    #         orientation='vertical',
    #         fraction=0.03,
    #         pad=0.07,
    #         shrink=0.8
    #     )
    #     cbar.set_label('State of Charge (%)', fontsize=30)
    #     cbar.set_ticks([0, 1])
    #     cbar.set_ticklabels(['0 %', '100 %'])
    #     cbar.ax.tick_params(labelsize=30)

    #     # EVs label slightly further from plots
    #     fig.text(0.05, 0.55, 'EVs', va='center', rotation='vertical', fontsize=30)

    #     # Retrieve the line handles for desired/actual and place legend further below the plot
    #     handles, labels = axes[0].get_legend_handles_labels()
    #     fig.legend(
    #         handles, labels,
    #         loc='lower center',
    #         bbox_to_anchor=(0.45, -0.20),
    #         ncol=2,
    #         fontsize=30
    #     )

    #     plt.savefig(output_path, bbox_inches='tight')
    #     plt.close()
    #     logging.info(f"Plot 'soc_evolution' saved to {output_path}")

    @staticmethod
    def plot_soc_evolution(schedule_results, time_axis, output_path, ev_config):
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        sns.set(style="whitegrid")
        ev_ids = list(schedule_results["soc_over_time"].keys())
        num_evs = len(ev_ids)
        
        # Get market prices from results
        market_prices = schedule_results.get("market_prices", [])
        
        # Create subplots: num_evs for SoC + 1 for market prices
        fig, axes = plt.subplots(
            num_evs + 1, 1, figsize=(14, 1.5 * num_evs + 1.5), sharex=True,
            gridspec_kw={'hspace': 0.7}
        )
        if num_evs == 0:
            axes = [axes]
        
        # Separate EV axes from market price axis
        ev_axes = axes[:-1] if num_evs > 0 else []
        market_ax = axes[-1]
        
        for ax, ev_id in zip(ev_axes, ev_ids):
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
            offset = 0.05
            ax.axvline(desired_time - offset, color='magenta', linewidth=4, linestyle='-',
                    label='Desired Disconnection')
            ax.axvline(actual_time + offset, color='darkblue', linewidth=4, linestyle='-',
                    label='Actual Disconnection')
            
            # Hide y-axis ticks and label each row with EV ID
            ax.set_yticks([])
            ax.set_ylabel(str(ev_id + 1), rotation=0, labelpad=15, fontsize=30,
                        ha='center', va='center')
            ax.set_xlim(time_axis[0], time_axis[-1])
        
        # ✅ Plot market prices in the additional bottom bar (same length as SoC bars)
        if len(market_prices) > 0:
            min_price = np.min(market_prices)
            max_price = np.max(market_prices)
            normalized_prices = (np.array(market_prices) - min_price) / \
                                (max_price - min_price) \
                                if max_price != min_price \
                                else np.zeros_like(market_prices)
            
            # Force same visual dimensions as SoC bars
            market_ax.imshow(
                [normalized_prices],
                aspect='auto',
                cmap='Blues',
                vmin=0,
                vmax=1,
                extent=[time_axis[0], time_axis[-1], 0, 1],
                interpolation='bilinear',
                origin='lower'
            )
        
        market_ax.set_yticks([])
        market_ax.set_ylabel('', rotation=0, labelpad=15, fontsize=30,
                            ha='center', va='center')
        market_ax.set_xlim(time_axis[0], time_axis[-1])
        
        plt.xlabel("Time (h)", fontsize=30, labelpad=30)
        for ax in axes:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x):02d}")
            )
            ax.tick_params(axis='x', labelsize=30)
        
        fig.subplots_adjust(left=0.12,
                            right=0.70,
                            bottom=0.15,
                            top=0.95)
        
        # SoC Colorbar (unchanged)
        cbar_soc = fig.colorbar(
            plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1)),
            ax=ev_axes,
            orientation='vertical',
            fraction=0.03,
            pad=0.07,
            shrink=0.8
        )
        cbar_soc.set_label('State of Charge (%)', fontsize=30)
        cbar_soc.set_ticks([0, 1])
        cbar_soc.set_ticklabels(['0 %', '100 %'])
        cbar_soc.ax.tick_params(labelsize=30)
        
        # Market Price Colorbar (independent positioning to match SoC colorbar width)
        if len(market_prices) > 0:
            # Get the SoC colorbar position and an EV axis position to match dimensions
            soc_cbar_pos = cbar_soc.ax.get_position()
            ev_ax_pos = ev_axes[0].get_position() if len(ev_axes) > 0 else None
            
            # Match the market_ax position to EV axes dimensions
            if ev_ax_pos:
                market_ax_pos = market_ax.get_position()
                market_ax.set_position([
                    ev_ax_pos.x0,
                    market_ax_pos.y0,
                    ev_ax_pos.width,
                    market_ax_pos.height
                ])
                market_ax_pos = market_ax.get_position()
            else:
                market_ax_pos = market_ax.get_position()
            
            # Create colorbar axis manually with 1.5x height
            cbar_height = market_ax_pos.height * 1.5
            cbar_width = soc_cbar_pos.width
            
            cax_price = fig.add_axes([
                soc_cbar_pos.x0,
                market_ax_pos.y0 + (market_ax_pos.height - cbar_height) / 2,
                cbar_width,
                cbar_height
            ])
            
            cbar_price = fig.colorbar(
                plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 1)),
                cax=cax_price,
                orientation='vertical'
            )
            fig.text(0.79, 0.20, 'Price', va='center', rotation='vertical', fontsize=30)
            fig.text(0.82, 0.20, '($/kWh)', va='center', rotation='vertical', fontsize=25)
            cbar_price.set_ticks([0, 1])
            cbar_price.set_ticklabels([f'{min_price:.3f}', f'{max_price:.3f}'])
            cbar_price.ax.tick_params(labelsize=30)
        
        fig.text(0.05, 0.55, 'EVs', va='center', rotation='vertical', fontsize=30)
        
        handles, labels = ev_axes[0].get_legend_handles_labels()
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

    @staticmethod
    def plot_nash_grid(tau_values, alpha_values, cost_grid, output_path, true_tau=None, true_alpha=None):
        """
        Plot a heatmap (grid) of the target EV's utility as a function of its bid parameters.
        - tau_values (x-axis): candidate disconnection_time bids.
        - alpha_values (y-axis): candidate disconnection_time_flexibility bids.
        - cost_grid: a 2D array (shape: len(alpha_values) x len(tau_values))
        where each cell is the EV's utility computed as energy_cost + congestion_cost + adaptability_cost.
        - If true_tau and true_alpha are provided, the cell matching these values is highlighted 
        with a thick blue border. The colormap runs from green (low cost) to red (high cost).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Rectangle
        import logging

        # Set the overall style for the plot.
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))

        # Use the reversed RdYlGn colormap so that low values are green and high values are red.
        cmap = "RdYlGn_r"
        ax = sns.heatmap(
            cost_grid,
            xticklabels=tau_values,
            yticklabels=alpha_values,
            cmap=cmap,
            annot=True,
            fmt=".2f"
        )

        # Increase font size for the axis labels.
        plt.xlabel("Bid: disconnection time (h)", labelpad=15, fontsize=15)
        plt.ylabel("Bid: flexibility coefficient ($/h²)", labelpad=15, fontsize=15)
        
        # Increase font size for tick labels.
        ax.tick_params(axis='both', which='major', labelsize=15)

        # Increase font size for the colorbar ticks.
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        
        # Add a legend for the color bar with increased label distance.
        cbar.set_label("Utility ($)", fontsize=15, labelpad=15)

        # If true_tau and true_alpha are provided, highlight that cell.
        if true_tau is not None and true_alpha is not None:
            try:
                col_index = tau_values.index(true_tau)
                row_index = alpha_values.index(true_alpha)
                rect = Rectangle((col_index, row_index), 1, 1, fill=False, edgecolor='blue', lw=3)
                ax.add_patch(rect)
            except ValueError:
                logging.warning("True bid values not found in the candidate lists; skipping cell highlight.")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Plot 'nash_grid' saved to {output_path}")