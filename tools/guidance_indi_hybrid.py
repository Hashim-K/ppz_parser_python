import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Hybrid Guidance")
class GuidanceIndiHybridTool(BaseTool):
    """
    Plots data from the GUIDANCE_INDI_HYBRID message.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["GUIDANCE_INDI_HYBRID"]

    def parse(self, *args, **kwargs):
        """
        Parses the hybrid guidance data.
        """
        self.parsed_data = self.all_data["GUIDANCE_INDI_HYBRID"]

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
        axs[0].plot(df["timestamp"], df["px"], label="px (est)")
        axs[0].plot(df["timestamp"], df["py"], label="py (est)")
        axs[0].plot(df["timestamp"], df["pz"], label="pz (est)")
        axs[0].plot(df["timestamp"], df["px_ref"], label="px (ref)", linestyle="--")
        axs[0].plot(df["timestamp"], df["py_ref"], label="py (ref)", linestyle="--")
        axs[0].plot(df["timestamp"], df["pz_ref"], label="pz (ref)", linestyle="--")
        axs[0].set_title("Position: Estimated vs. Reference")
        axs[0].set_ylabel("Position (m)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Velocity
        axs[1].plot(df["timestamp"], df["vx"], label="vx (est)")
        axs[1].plot(df["timestamp"], df["vy"], label="vy (est)")
        axs[1].plot(df["timestamp"], df["vz"], label="vz (est)")
        axs[1].plot(df["timestamp"], df["vx_ref"], label="vx (ref)", linestyle="--")
        axs[1].plot(df["timestamp"], df["vy_ref"], label="vy (ref)", linestyle="--")
        axs[1].plot(df["timestamp"], df["vz_ref"], label="vz (ref)", linestyle="--")
        axs[1].set_title("Velocity: Estimated vs. Reference")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Feedforward Acceleration
        axs[2].plot(df["timestamp"], df["ax_ff"], label="ax_ff")
        axs[2].plot(df["timestamp"], df["ay_ff"], label="ay_ff")
        axs[2].plot(df["timestamp"], df["az_ff"], label="az_ff")
        axs[2].set_title("Feedforward Acceleration")
        axs[2].set_ylabel("m/s^2")
        axs[2].legend()
        axs[2].grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
