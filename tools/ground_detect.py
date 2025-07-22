import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Ground Detection")
class GroundDetectTool(BaseTool):
    """
    Plots data from the ground detection system.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["GROUND_DETECT"]

    def parse(self, *args, **kwargs):
        """
        Parses the ground detection data.
        """
        self.parsed_data = self.all_data["GROUND_DETECT"]

    def plot(self):
        """
        Generates a plot showing ground proximity and throttle status.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            2, 1, figsize=(15, 10), sharex=True, num="Ground Detection"
        )
        fig.suptitle("Ground Detection Status", fontsize=16)

        # Plot 1: Ground Proximity
        axs[0].plot(df["timestamp"], df["ground_proximity"], label="Ground Proximity")
        axs[0].set_title("Ground Proximity Sensor")
        axs[0].set_ylabel("Sensor Value")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Throttle is Low Flag
        axs[1].plot(
            df["timestamp"],
            df["throttle_low"],
            label="Throttle Low Flag",
            drawstyle="steps-post",
        )
        axs[1].set_title("Throttle Status")
        axs[1].set_ylabel("Boolean Flag")
        axs[1].set_ylim(-0.1, 1.1)  # Set Y-axis limits for boolean data
        axs[1].legend()
        axs[1].grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
