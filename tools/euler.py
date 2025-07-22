import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool
from utils.transforms import eulers_of_quat


@register_tool("Euler Angles")
class EulerTool(BaseTool):
    """
    Plots estimated vs. reference Euler angles (roll, pitch, yaw).

    This tool is now more robust. It will prioritize the 'AHRS_REF_QUAT' message,
    which contains both estimated and reference quaternions. As a fallback,
    it will use the 'ATTITUDE' message for estimated angles only.
    """

    @property
    def required_messages(self) -> List[str]:
        return []  # Logic is handled in check_required_messages

    def check_required_messages(self) -> bool:
        """
        Checks if either AHRS_REF_QUAT or ATTITUDE messages are present.
        """
        self.source_msg = None
        # Prioritize the message that contains both estimated and reference values.
        if "AHRS_REF_QUAT" in self.all_data:
            self.source_msg = "AHRS_REF_QUAT"
            return True
        # Fallback to the message with only estimated values.
        if "ATTITUDE" in self.all_data:
            self.source_msg = "ATTITUDE"
            return True
        return False

    def parse(self, order: str = "ZYX"):
        """
        Parses Euler angles from the identified source message.
        """
        self.parsed_data = {}
        if self.source_msg == "AHRS_REF_QUAT":
            print("Info: EulerTool using AHRS_REF_QUAT for estimated vs. reference.")
            quat_df = self.all_data["AHRS_REF_QUAT"]

            # Check for estimated body quaternion columns.
            body_quat_cols = ["body_qi", "body_qx", "body_qy", "body_qz"]
            if all(c in quat_df.columns for c in body_quat_cols):
                # Convert estimated quaternion to eulers.
                self.parsed_data["est"] = eulers_of_quat(quat_df, *body_quat_cols)

            # Check for reference quaternion columns.
            ref_quat_cols = ["ref_qi", "ref_qx", "ref_qy", "ref_qz"]
            if all(c in quat_df.columns for c in ref_quat_cols):
                # Convert reference quaternion to eulers.
                self.parsed_data["ref"] = eulers_of_quat(quat_df, *ref_quat_cols)

        elif self.source_msg == "ATTITUDE":
            print("Info: EulerTool using ATTITUDE message (estimated angles only).")
            # If we only have ATTITUDE, we only have estimated angles.
            self.parsed_data["est"] = self.all_data["ATTITUDE"]

    def plot(self):
        """
        Generates the plot comparing estimated and reference Euler angles.
        """
        if not self.parsed_data:
            return

        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            3,
            1,
            figsize=(15, 12),
            sharex=True,
            num="Euler Angles (Estimated vs. Reference)",
        )
        fig.suptitle("Euler Angles: Estimated vs. Reference", fontsize=16)

        # Plot estimated data if available.
        if "est" in self.parsed_data:
            est_df = self.parsed_data["est"]
            axs[0].plot(
                est_df["timestamp"], est_df["phi"], label="Est. Roll (φ)", color="b"
            )
            axs[1].plot(
                est_df["timestamp"], est_df["theta"], label="Est. Pitch (θ)", color="b"
            )
            axs[2].plot(
                est_df["timestamp"], est_df["psi"], label="Est. Yaw (ψ)", color="b"
            )

        # Plot reference data if available, using dashed lines.
        if "ref" in self.parsed_data:
            ref_df = self.parsed_data["ref"]
            axs[0].plot(
                ref_df["timestamp"],
                ref_df["phi"],
                label="Ref. Roll (φ)",
                color="c",
                linestyle="--",
            )
            axs[1].plot(
                ref_df["timestamp"],
                ref_df["theta"],
                label="Ref. Pitch (θ)",
                color="c",
                linestyle="--",
            )
            axs[2].plot(
                ref_df["timestamp"],
                ref_df["psi"],
                label="Ref. Yaw (ψ)",
                color="c",
                linestyle="--",
            )

        # Set titles and labels for each subplot.
        axs[0].set_title("Roll Angle")
        axs[0].set_ylabel("Angle (rad)")
        axs[1].set_title("Pitch Angle")
        axs[1].set_ylabel("Angle (rad)")
        axs[2].set_title("Yaw Angle")
        axs[2].set_ylabel("Angle (rad)")

        for ax in axs:
            ax.legend()
            ax.grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
