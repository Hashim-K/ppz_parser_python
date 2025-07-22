import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Energy")
class EnergyTool(BaseTool):
    """
    Plots energy data from the ENERGY message.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["ENERGY"]

    def parse(self, *args, **kwargs):
        """
        Parses the energy data.
        """
        self.parsed_data = self.all_data["ENERGY"]

    def plot(self):
        """
        Generates a plot with voltage, current, and power consumption.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True, num="Energy")
        fig.suptitle("Energy Consumption", fontsize=16)

        # Plot Voltage
        axs[0].plot(df["timestamp"], df["voltage"], label="Voltage")
        axs[0].set_title("Battery Voltage")
        axs[0].set_ylabel("Volts (V)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot Current
        axs[1].plot(df["timestamp"], df["current"], label="Current", color="orange")
        axs[1].set_title("Current Draw")
        axs[1].set_ylabel("Amps (A)")
        axs[1].legend()
        axs[1].grid(True)

        # Calculate and Plot Power
        power = df["voltage"] * df["current"]
        axs[2].plot(df["timestamp"], power, label="Power", color="red")
        axs[2].set_title("Power Consumption")
        axs[2].set_ylabel("Watts (W)")
        axs[2].legend()
        axs[2].grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
