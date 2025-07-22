import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Airspeed")
class AirspeedTool(BaseTool):
    """
    Plots airspeed data from the AIRSPEED message.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["AIRSPEED"]

    def parse(self, *args, **kwargs):
        """
        Parses the airspeed data.
        """
        self.parsed_data = self.all_data["AIRSPEED"]

    def plot(self):
        """
        Generates a plot with airspeed, scaled pressure, and differential pressure.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True, num="Airspeed")

        # Plot Airspeed
        axs[0].plot(df["timestamp"], df["airspeed"], label="Airspeed")
        axs[0].set_title("Airspeed")
        axs[0].set_ylabel("m/s")
        axs[0].legend()
        axs[0].grid(True)

        # Plot Scaled Pressure
        axs[1].plot(
            df["timestamp"],
            df["scaled_pressure"],
            label="Scaled Pressure",
            color="green",
        )
        axs[1].set_title("Scaled Pressure")
        axs[1].set_ylabel("Pascals (Pa)")
        axs[1].legend()
        axs[1].grid(True)

        # Plot Differential Pressure
        axs[2].plot(
            df["timestamp"],
            df["differential_pressure"],
            label="Differential Pressure",
            color="red",
        )
        axs[2].set_title("Differential Pressure")
        axs[2].set_ylabel("Pascals (Pa)")
        axs[2].legend()
        axs[2].grid(True)

        plt.xlabel("Time (s)")
        plt.suptitle("Airspeed Sensor Data", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
