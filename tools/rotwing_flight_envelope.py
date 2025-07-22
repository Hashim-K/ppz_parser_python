import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Rotwing Flight Envelope")
class RotwingFlightEnvelopeTool(BaseTool):
    """
    Plots the flight envelope of the rotating wing aircraft.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["ROTWING_STATE"]

    def parse(self, *args, **kwargs):
        """
        Parses the rotating wing state data.
        """
        self.parsed_data = self.all_data["ROTWING_STATE"]

    def plot(self):
        """
        Generates a plot of airspeed vs. dynamic pressure ratio.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 8), num="Rotwing Flight Envelope")

        # Create a scatter plot of airspeed vs. q_slash_q_max
        plt.scatter(df["q_slash_q_max"], df["airspeed"], s=10)

        plt.title("Rotating Wing Flight Envelope")
        plt.xlabel("q / q_max (Dynamic Pressure Ratio)")
        plt.ylabel("Airspeed (m/s)")
        plt.grid(True)
