import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Stabilization Attitude")
class StabAttitudeTool(BaseTool):
    """
    Plots the estimated vs. reference attitude from the STAB_ATTITUDE message.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["STAB_ATTITUDE"]

    def parse(self, *args, **kwargs):
        """
        Parses the stabilization attitude data.
        """
        self.parsed_data = self.all_data["STAB_ATTITUDE"]

    def plot(self):
        """
        Generates plots for roll, pitch, and yaw, comparing estimated vs. reference values.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            3, 1, figsize=(15, 12), sharex=True, num="Stabilization Attitude"
        )
        fig.suptitle("Stabilization Attitude: Estimated vs. Reference", fontsize=16)

        # Define plot details for each axis.
        plot_map = {
            "Roll": ("roll", "roll_ref", axs[0]),
            "Pitch": ("pitch", "pitch_ref", axs[1]),
            "Yaw": ("yaw", "yaw_ref", axs[2]),
        }

        # Create each subplot.
        for title, (est_col, ref_col, ax) in plot_map.items():
            if est_col in df.columns:
                ax.plot(df["timestamp"], df[est_col], label=f"Est. {title}")
            if ref_col in df.columns:
                ax.plot(
                    df["timestamp"], df[ref_col], label=f"Ref. {title}", linestyle="--"
                )

            ax.set_title(f"{title} Angle")
            ax.set_ylabel("Angle (rad)")
            ax.legend()
            ax.grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
