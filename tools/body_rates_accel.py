import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Body Rates and Accelerations")
class BodyRatesAccelTool(BaseTool):
    """
    Plots body rates from IMU_GYRO and accelerations from IMU_ACCEL.
    """

    @property
    def required_messages(self) -> List[str]:
        # This tool can work if either accel or gyro is present.
        return []

    def check_required_messages(self) -> bool:
        """
        Check if at least one of the required IMU messages is available.
        """
        # Note: This checks for the raw IMU messages, as per the MATLAB script.
        self.has_accel = "IMU_ACCEL" in self.all_data
        self.has_gyro = "IMU_GYRO" in self.all_data
        return self.has_accel or self.has_gyro

    def parse(self, *args, **kwargs):
        """
        Data is accessed directly in the plot method since it comes from two sources.
        """
        pass

    def plot(self):
        """
        Generates plots for body rates (gyro) and accelerations.
        """
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            2, 1, figsize=(15, 10), sharex=True, num="Body Rates and Accelerations"
        )

        # Plot gyroscope data if available.
        if self.has_gyro:
            df_gyro = self.all_data["IMU_GYRO"]
            axs[0].plot(df_gyro["timestamp"], df_gyro["p"], label="p (roll rate)")
            axs[0].plot(df_gyro["timestamp"], df_gyro["q"], label="q (pitch rate)")
            axs[0].plot(df_gyro["timestamp"], df_gyro["r"], label="r (yaw rate)")
            axs[0].set_title("Body Rates (Gyroscope)")
            axs[0].set_ylabel("Angular Rate (rad/s)")
            axs[0].legend()
            axs[0].grid(True)
        else:
            # If no data, display an empty plot with a clear title.
            axs[0].set_title("Body Rates (Gyroscope) - No Data")
            axs[0].grid(True)

        # Plot accelerometer data if available.
        if self.has_accel:
            df_accel = self.all_data["IMU_ACCEL"]
            axs[1].plot(df_accel["timestamp"], df_accel["ax"], label="ax")
            axs[1].plot(df_accel["timestamp"], df_accel["ay"], label="ay")
            axs[1].plot(df_accel["timestamp"], df_accel["az"], label="az")
            axs[1].set_title("Body Accelerations")
            axs[1].set_ylabel("Acceleration (m/s^2)")
            axs[1].legend()
            axs[1].grid(True)
        else:
            # If no data, display an empty plot with a clear title.
            axs[1].set_title("Body Accelerations - No Data")
            axs[1].grid(True)

        plt.xlabel("Time (s)")
        plt.suptitle("IMU Body Rates and Accelerations", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
