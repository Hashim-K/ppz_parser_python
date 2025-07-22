import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Rotorcraft FP")
class RotorcraftFPTool(BaseTool):
    """
    Plots data from the rotorcraft flight plan (ROTORCRAFT_FP) message.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["ROTORCRAFT_FP"]

    def parse(self, *args, **kwargs):
        """
        Parses the rotorcraft flight plan data.
        """
        self.parsed_data = self.all_data["ROTORCRAFT_FP"]

    def plot(self):
        """
        Generates plots for position, velocity, and attitude from ROTORCRAFT_FP.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            3, 1, figsize=(15, 12), sharex=True, num="Rotorcraft Flight Plan"
        )
        fig.suptitle("Rotorcraft Flight Plan", fontsize=16)

        # Plot 1: Position (East, North, Up)
        axs[0].plot(df["timestamp"], df["east"], label="East (est)")
        axs[0].plot(df["timestamp"], df["north"], label="North (est)")
        axs[0].plot(df["timestamp"], df["up"], label="Up (est)")
        axs[0].plot(
            df["timestamp"], df["carrot_east"], label="East (sp)", linestyle="--"
        )
        axs[0].plot(
            df["timestamp"], df["carrot_north"], label="North (sp)", linestyle="--"
        )
        axs[0].plot(df["timestamp"], df["carrot_up"], label="Up (sp)", linestyle="--")
        axs[0].set_title("Position: Estimated vs. Setpoint")
        axs[0].set_ylabel("Position (m)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Velocity (East, North, Up)
        axs[1].plot(df["timestamp"], df["veast"], label="VEast")
        axs[1].plot(df["timestamp"], df["vnorth"], label="VNorth")
        axs[1].plot(df["timestamp"], df["vup"], label="VUp")
        axs[1].set_title("Velocity")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Attitude (Roll, Pitch, Yaw)
        axs[2].plot(df["timestamp"], df["phi"], label="Roll (phi)")
        axs[2].plot(df["timestamp"], df["theta"], label="Pitch (theta)")
        axs[2].plot(df["timestamp"], df["psi"], label="Yaw (psi)")
        axs[2].plot(
            df["timestamp"], df["carrot_psi"], label="Yaw setpoint", linestyle="--"
        )
        axs[2].set_title("Attitude Angles")
        axs[2].set_ylabel("Angle (rad)")
        axs[2].legend()
        axs[2].grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
