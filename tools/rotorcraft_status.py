import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Rotorcraft Status")
class RotorcraftStatusTool(BaseTool):
    """
    Plots data from the ROTORCRAFT_STATUS message.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["ROTORCRAFT_STATUS"]

    def parse(self, *args, **kwargs):
        """
        Parses the rotorcraft status data.
        """
        self.parsed_data = self.all_data["ROTORCRAFT_STATUS"]

    def plot(self):
        """
        Generates plots for failsafe, vehicle, and flight modes.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            3, 1, figsize=(15, 12), sharex=True, num="Rotorcraft Status"
        )
        fig.suptitle("Rotorcraft Status", fontsize=16)

        # Plot 1: Autopilot Mode
        axs[0].plot(
            df["timestamp"],
            df["ap_mode"],
            label="AP Mode",
            drawstyle="steps-post",
        )
        axs[0].set_title("Autopilot Mode")
        axs[0].set_ylabel("Mode ID")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Horizontal Mode
        axs[1].plot(
            df["timestamp"],
            df["ap_h_mode"],
            label="Horizontal Mode",
            drawstyle="steps-post",
        )
        axs[1].set_title("Horizontal Mode")
        axs[1].set_ylabel("Mode ID")
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Vertical Mode
        axs[2].plot(
            df["timestamp"],
            df["ap_v_mode"],
            label="Vertical Mode",
            drawstyle="steps-post",
        )
        axs[2].set_title("Vertical Mode")
        axs[2].set_ylabel("Mode ID")
        axs[2].legend()
        axs[2].grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
