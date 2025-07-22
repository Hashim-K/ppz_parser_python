import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Hybrid Guidance")
class HybridGuidanceTool(BaseTool):
    """
    Plots data from the GUIDANCE message for the hybrid guidance controller.
    """

    @property
    def required_messages(self) -> List[str]:
        # This tool specifically requires the standard GUIDANCE message.
        return ["GUIDANCE"]

    def parse(self, *args, **kwargs):
        """
        Parses the hybrid guidance data.
        """
        self.parsed_data = self.all_data["GUIDANCE"]

    def plot(self):
        """
        Generates plots for guidance position, velocity, and feedforward terms.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            3, 1, figsize=(15, 12), sharex=True, num="Hybrid Guidance"
        )
        fig.suptitle("Hybrid Guidance Controller", fontsize=16)

        # Plot 1: Position
        # This plots the x, y, z setpoints from the guidance message.
        axs[0].plot(df["timestamp"], df["x"], label="x (sp)")
        axs[0].plot(df["timestamp"], df["y"], label="y (sp)")
        axs[0].plot(df["timestamp"], df["z"], label="z (sp)")
        axs[0].set_title("Position Setpoint")
        axs[0].set_ylabel("Position (m)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Velocity
        # This plots the velocity setpoints.
        axs[1].plot(df["timestamp"], df["xd"], label="xd (sp)")
        axs[1].plot(df["timestamp"], df["yd"], label="yd (sp)")
        axs[1].plot(df["timestamp"], df["zd"], label="zd (sp)")
        axs[1].set_title("Velocity Setpoint")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Feedforward Acceleration
        # This plots the feedforward acceleration terms.
        axs[2].plot(df["timestamp"], df["xdd"], label="xdd (ff)")
        axs[2].plot(df["timestamp"], df["ydd"], label="ydd (ff)")
        axs[2].set_title("Feedforward Acceleration")
        axs[2].set_ylabel("m/s^2")
        axs[2].legend()
        axs[2].grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
