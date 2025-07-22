import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Rotorcraft CMD")
class RotorcraftCmdTool(BaseTool):
    """
    Plots the commands from the ROTORCRAFT_CMD message.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["ROTORCRAFT_CMD"]

    def parse(self, *args, **kwargs):
        """
        Parses the rotorcraft command data.
        """
        self.parsed_data = self.all_data["ROTORCRAFT_CMD"]

    def plot(self):
        """
        Generates a plot showing the roll, pitch, and yaw commands.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(15, 8), num="Rotorcraft Commands")

        plt.plot(df["timestamp"], df["roll"], label="Roll CMD")
        plt.plot(df["timestamp"], df["pitch"], label="Pitch CMD")
        plt.plot(df["timestamp"], df["yaw"], label="Yaw CMD")

        plt.title("Rotorcraft Commands")
        plt.xlabel("Time (s)")
        plt.ylabel("Command Value")
        plt.legend()
        plt.grid(True)
