import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Device Powers")
class PowersTool(BaseTool):
    """
    Plots power consumption for various devices from the POWER message.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["POWER"]

    def parse(self, *args, **kwargs):
        """
        Parses the power data.
        """
        self.parsed_data = self.all_data["POWER"]

    def plot(self):
        """
        Generates a plot showing the power values for each device over time.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(15, 8), num="Device Power Consumption")

        # Find all columns that represent power values (e.g., p1, p2, ...).
        power_cols = sorted(
            [col for col in df.columns if col.startswith("p") and col[1:].isdigit()]
        )

        if not power_cols:
            print("Skipping Device Powers plot: No power columns (p1, p2, ...) found.")
            return

        # Plot each power value over time.
        for col in power_cols:
            plt.plot(df["timestamp"], df[col], label=f"Device {col}")

        plt.title("Device Power Consumption")
        plt.xlabel("Time (s)")
        plt.ylabel("Power (implementation-specific units)")
        plt.legend()
        plt.grid(True)
