import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Hover Loop")
class HoverLoopTool(BaseTool):
    """
    Plots data from the rotorcraft flight plan (ROTORCRAFT_FP) message,
    focusing on hover control loop performance.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["ROTORCRAFT_FP"]

    def parse(self, *args, **kwargs):
        """
        Parses the flight plan data.
        """
        self.parsed_data = self.all_data["ROTORCRAFT_FP"]

    def plot(self):
        """
        Generates plots for position, velocity, and acceleration setpoints vs. actuals.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True, num="Hover Loop")
        fig.suptitle("Hover Control Loop", fontsize=16)

        # Plot 1: Position (East, North, Up)
        axs[0].plot(df["timestamp"], df["east"], label="East (est)")
        axs[0].plot(df["timestamp"], df["north"], label="North (est)")
        axs[0].plot(df["timestamp"], df["up"], label="Up (est)")
        axs[0].plot(df["timestamp"], df["pe"], label="East (sp)", linestyle="--")
        axs[0].plot(df["timestamp"], df["pn"], label="North (sp)", linestyle="--")
        axs[0].plot(df["timestamp"], df["pup"], label="Up (sp)", linestyle="--")
        axs[0].set_title("Position: Estimated vs. Setpoint")
        axs[0].set_ylabel("Position (m)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Velocity (East, North, Up)
        axs[1].plot(df["timestamp"], df["veast"], label="VEast (est)")
        axs[1].plot(df["timestamp"], df["vnorth"], label="VNorth (est)")
        axs[1].plot(df["timestamp"], df["vup"], label="VUp (est)")
        axs[1].plot(df["timestamp"], df["pde"], label="VEast (sp)", linestyle="--")
        axs[1].plot(df["timestamp"], df["pdn"], label="VNorth (sp)", linestyle="--")
        axs[1].plot(df["timestamp"], df["pdup"], label="VUp (sp)", linestyle="--")
        axs[1].set_title("Velocity: Estimated vs. Setpoint")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Acceleration (East, North, Up)
        axs[2].plot(df["timestamp"], df["ae"], label="Acc East (sp)")
        axs[2].plot(df["timestamp"], df["an"], label="Acc North (sp)")
        axs[2].plot(df["timestamp"], df["aup"], label="Acc Up (sp)")
        axs[2].set_title("Acceleration Setpoints")
        axs[2].set_ylabel("m/s^2")
        axs[2].legend()
        axs[2].grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
