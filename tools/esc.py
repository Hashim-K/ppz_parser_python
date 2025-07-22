import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("ESC Telemetry")
class ESCTool(BaseTool):
    """
    Plots detailed telemetry (RPM, Volts, Amps, Power, Temp) for specified ESCs.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["ESC"]

    def parse(self, motor_ids: List[int] = None):
        """
        Parses telemetry data for the specified motor_ids.
        """
        esc_df = self.all_data["ESC"]

        # If no motor IDs are specified, the tool is not valid.
        if not motor_ids:
            self.is_valid = False
            return

        # Filter the dataframe to only include rows with the specified motor IDs.
        filtered_df = esc_df[esc_df["motor_id"].isin(motor_ids)]

        if filtered_df.empty:
            # If no data is found for the requested motors, the tool is not valid.
            self.is_valid = False
            return

        self.parsed_data = filtered_df
        # Create a dynamic name for the plot window.
        self.figure_name = f"ESC Telemetry (Motors: {motor_ids})"
        # Store motor IDs for plotting.
        self.motor_ids = sorted(filtered_df["motor_id"].unique())

    def plot(self):
        """
        Generates a multi-subplot figure for ESC telemetry.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            5, 1, figsize=(15, 18), sharex=True, num=self.figure_name
        )
        fig.suptitle(f"ESC Telemetry for Motors {self.motor_ids}", fontsize=16)

        plot_metrics = {
            "RPM": ("rpm", "RPM", axs[0]),
            "Voltage": ("motor_volts", "Volts (V)", axs[1]),
            "Current": ("amps", "Amps (A)", axs[2]),
            "Power": ("power", "Watts (W)", axs[3]),
            "Temperature": ("temperature", "Celsius", axs[4]),
        }

        # Iterate through each metric and plot data for each motor.
        for title, (col, ylabel, ax) in plot_metrics.items():
            for motor_id in self.motor_ids:
                motor_df = df[df["motor_id"] == motor_id]
                ax.plot(
                    motor_df["timestamp"], motor_df[col], label=f"Motor {int(motor_id)}"
                )

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
