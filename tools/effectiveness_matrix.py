import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import numpy as np

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Effectiveness Matrix")
class EffectivenessMatrixTool(BaseTool):
    """
    Plots the effectiveness matrix from the EFF_MAT message.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["EFF_MAT"]

    def parse(self, *args, **kwargs):
        """
        Parses the effectiveness matrix data.
        """
        self.parsed_data = self.all_data["EFF_MAT"]

    def plot(self):
        """
        Generates a plot with a separate subplot for each element of the 3xN matrix.
        """
        df = self.parsed_data

        # Identify all columns that are part of the effectiveness matrix
        eff_mat_cols = sorted([col for col in df.columns if col.startswith("eff_mat_")])

        if not eff_mat_cols:
            print("Skipping Effectiveness Matrix plot: No 'eff_mat_' columns found.")
            return

        # Determine the dimensions of the matrix from the column names
        indices = [
            list(map(int, col.replace("eff_mat_", "").split("_")))
            for col in eff_mat_cols
        ]
        rows = max(idx[0] for idx in indices) + 1
        cols = max(idx[1] for idx in indices) + 1

        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            rows,
            cols,
            figsize=(5 * cols, 4 * rows),
            sharex=True,
            num="Effectiveness Matrix",
        )

        fig.suptitle("Effectiveness Matrix", fontsize=16)

        for i in range(rows):
            for j in range(cols):
                col_name = f"eff_mat_{i}_{j}"
                ax = (
                    axs[i, j]
                    if rows > 1 and cols > 1
                    else (axs[j] if cols > 1 else axs)
                )

                if col_name in df.columns:
                    ax.plot(df["timestamp"], df[col_name])
                    ax.set_title(f"Element ({i}, {j})")
                    ax.grid(True)
                else:
                    ax.set_title(f"Element ({i}, {j}) - No Data")
                    ax.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
