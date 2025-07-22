import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Quaternions")
class QuaternionTool(BaseTool):
    """
    Plots the estimated vs. reference quaternions from AHRS_REF_QUAT.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["AHRS_REF_QUAT"]

    def parse(self, *args, **kwargs):
        """
        Parses the quaternion data.
        """
        self.parsed_data = self.all_data["AHRS_REF_QUAT"]

    def plot(self):
        """
        Generates a plot comparing the estimated and reference quaternions.
        """
        df = self.parsed_data
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            4,
            1,
            figsize=(15, 14),
            sharex=True,
            num="Quaternions (Estimated vs. Reference)",
        )
        fig.suptitle("Quaternions: Estimated vs. Reference", fontsize=16)

        # Define the plot details for each quaternion component.
        quat_map = {
            "q0 (w)": ("body_qi", "ref_qi"),
            "q1 (x)": ("body_qx", "ref_qx"),
            "q2 (y)": ("body_qy", "ref_qy"),
            "q3 (z)": ("body_qz", "ref_qz"),
        }

        # Plot each component on its own subplot.
        for i, (title, (est_col, ref_col)) in enumerate(quat_map.items()):
            ax = axs[i]
            if est_col in df.columns:
                ax.plot(df["timestamp"], df[est_col], label="Estimated", color="b")
            if ref_col in df.columns:
                ax.plot(
                    df["timestamp"],
                    df[ref_col],
                    label="Reference",
                    color="c",
                    linestyle="--",
                )

            ax.set_title(title)
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
