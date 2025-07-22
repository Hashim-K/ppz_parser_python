import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Rotwing State")
class RotwingStateTool(BaseTool):
    """
    Plots detailed state information for a rotating wing aircraft.
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
        Generates plots for RPM, tilt angle, thrust, and other state variables.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(3, 2, figsize=(16, 15), num="Rotwing State")
        fig.suptitle("Rotating Wing State", fontsize=16)

        # Plot 1: Motor RPM
        ax = axs[0, 0]
        ax.plot(df["timestamp"], df["rpm_ref"], label="RPM (ref)", linestyle="--")
        ax.plot(df["timestamp"], df["rpm_mes"], label="RPM (mes)")
        ax.set_title("Motor RPM")
        ax.set_ylabel("RPM")
        ax.legend()
        ax.grid(True)

        # Plot 2: Tilt Angle
        ax = axs[0, 1]
        ax.plot(
            df["timestamp"], df["tilt_angle_ref"], label="Tilt (ref)", linestyle="--"
        )
        ax.plot(df["timestamp"], df["tilt_angle_mes"], label="Tilt (mes)")
        ax.set_title("Tilt Angle")
        ax.set_ylabel("Angle (rad)")
        ax.legend()
        ax.grid(True)

        # Plot 3: Thrust
        ax = axs[1, 0]
        ax.plot(df["timestamp"], df["thrust_ref"], label="Thrust (ref)", linestyle="--")
        ax.plot(df["timestamp"], df["thrust_mes"], label="Thrust (mes)")
        ax.set_title("Thrust")
        ax.set_ylabel("Normalized")
        ax.legend()
        ax.grid(True)

        # Plot 4: Dynamic Pressure Ratio
        ax = axs[1, 1]
        ax.plot(df["timestamp"], df["q_slash_q_max"], label="q / q_max")
        ax.set_title("Dynamic Pressure Ratio")
        ax.set_ylabel("Ratio")
        ax.legend()
        ax.grid(True)

        # Plot 5: Airspeed
        ax = axs[2, 0]
        ax.plot(df["timestamp"], df["airspeed"], label="Airspeed")
        ax.set_title("Airspeed")
        ax.set_ylabel("m/s")
        ax.legend()
        ax.grid(True)

        # Plot 6: Pitch (Theta)
        ax = axs[2, 1]
        ax.plot(df["timestamp"], df["theta_mes"], label="Theta (mes)")
        ax.set_title("Pitch Angle")
        ax.set_ylabel("Angle (rad)")
        ax.legend()
        ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
