import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("GPS Data")
class GPSTool(BaseTool):
    """
    Plots detailed GPS data including position, velocity, and status.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["GPS"]

    def parse(self, *args, **kwargs):
        """
        Parses the GPS data from the main data dictionary.
        """
        self.parsed_data = self.all_data["GPS"]

    def plot(self):
        """
        Generates a comprehensive multi-subplot figure for all key GPS metrics.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        # Create a figure with 4 rows and 2 columns of subplots.
        fig, axs = plt.subplots(4, 2, figsize=(16, 18), num="GPS Data")
        fig.suptitle("GPS Data", fontsize=16)

        # Plot 1: Mode & Number of Satellites
        ax = axs[0, 0]
        ax.plot(df["timestamp"], df["mode"], label="Mode")
        ax.plot(df["timestamp"], df["nb_sats"], label="Num Sats")
        ax.set_title("Mode & Satellites")
        ax.legend()
        ax.grid(True)

        # Plot 2: UTM Position
        ax = axs[0, 1]
        ax.plot(df["timestamp"], df["utm_east"], label="UTM East")
        ax.plot(df["timestamp"], df["utm_north"], label="UTM North")
        ax.set_title("UTM Position")
        ax.set_ylabel("m")
        ax.legend()
        ax.grid(True)

        # Plot 3: Altitude
        ax = axs[1, 0]
        ax.plot(df["timestamp"], df["alt"], label="Altitude")
        ax.set_title("Altitude")
        ax.set_ylabel("m")
        ax.legend()
        ax.grid(True)

        # Plot 4: Climb Rate
        ax = axs[1, 1]
        ax.plot(df["timestamp"], df["climb"], label="Climb")
        ax.set_title("Climb Rate")
        ax.set_ylabel("m/s")
        ax.legend()
        ax.grid(True)

        # Plot 5: Speed
        ax = axs[2, 0]
        ax.plot(df["timestamp"], df["speed"], label="Speed (3D)")
        ax.plot(df["timestamp"], df["gspeed"], label="Ground Speed (2D)")
        ax.set_title("Speed")
        ax.set_ylabel("m/s")
        ax.legend()
        ax.grid(True)

        # Plot 6: Course
        ax = axs[2, 1]
        ax.plot(df["timestamp"], df["course"], label="Course")
        ax.set_title("Course")
        ax.set_ylabel("degrees")
        ax.legend()
        ax.grid(True)

        # Plot 7: Accuracy
        ax = axs[3, 0]
        ax.plot(df["timestamp"], df["hmsl_acc"], label="HMSL Accuracy")
        ax.plot(df["timestamp"], df["vel_acc"], label="Velocity Accuracy")
        ax.set_title("Accuracy")
        ax.set_ylabel("m or m/s")
        ax.legend()
        ax.grid(True)

        # Plot 8: Dilution of Precision (DOP)
        ax = axs[3, 1]
        ax.plot(df["timestamp"], df["pdop"], label="pDOP")
        ax.plot(df["timestamp"], df["hdop"], label="hDOP")
        ax.plot(df["timestamp"], df["vdop"], label="vDOP")
        ax.set_title("Dilution of Precision")
        ax.legend()
        ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
