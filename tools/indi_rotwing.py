import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("INDI Rotwing")
class IndiRotwingTool(BaseTool):
    """
    Plots data from the INDI_ROTWING message for the rotating wing controller.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["INDI_ROTWING"]

    def parse(self, *args, **kwargs):
        """
        Parses the INDI rotating wing controller data.
        """
        self.parsed_data = self.all_data["INDI_ROTWING"]

    def plot(self):
        """
        Generates plots for angular accelerations, control inputs, and other INDI variables.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True, num="INDI Rotwing")
        fig.suptitle("INDI Rotating Wing Controller", fontsize=16)

        # Plot 1: Angular Accelerations (p, q, r)
        axs[0].plot(
            df["timestamp"], df["p_dot_ref"], label="p_dot (ref)", linestyle="--"
        )
        axs[0].plot(
            df["timestamp"], df["q_dot_ref"], label="q_dot (ref)", linestyle="--"
        )
        axs[0].plot(
            df["timestamp"], df["r_dot_ref"], label="r_dot (ref)", linestyle="--"
        )
        axs[0].plot(df["timestamp"], df["p_dot_mes"], label="p_dot (mes)")
        axs[0].plot(df["timestamp"], df["q_dot_mes"], label="q_dot (mes)")
        axs[0].plot(df["timestamp"], df["r_dot_mes"], label="r_dot (mes)")
        axs[0].set_title("Angular Accelerations: Reference vs. Measured")
        axs[0].set_ylabel("rad/s^2")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Control Input (u)
        axs[1].plot(df["timestamp"], df["u_l"], label="u_l")
        axs[1].plot(df["timestamp"], df["u_m"], label="u_m")
        axs[1].plot(df["timestamp"], df["u_n"], label="u_n")
        axs[1].set_title("Control Input (u)")
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Incremental Control Input (delta_u)
        axs[2].plot(df["timestamp"], df["delta_u_l"], label="delta_u_l")
        axs[2].plot(df["timestamp"], df["delta_u_m"], label="delta_u_m")
        axs[2].plot(df["timestamp"], df["delta_u_n"], label="delta_u_n")
        axs[2].set_title("Incremental Control Input (delta_u)")
        axs[2].legend()
        axs[2].grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
