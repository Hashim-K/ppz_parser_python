import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("IMU Scaled")
class IMUScaledTool(BaseTool):
    """
    Plots scaled data from IMU_ACCEL_SCALED and IMU_GYRO_SCALED.
    """

    @property
    def required_messages(self) -> List[str]:
        # This tool can work if either accel or gyro is present.
        return []

    def check_required_messages(self) -> bool:
        """
        Check if at least one of the scaled IMU messages is available.
        """
        self.has_accel = "IMU_ACCEL_SCALED" in self.all_data
        self.has_gyro = "IMU_GYRO_SCALED" in self.all_data
        return self.has_accel or self.has_gyro

    def parse(self, *args, **kwargs):
        """
        Data is accessed directly in the plot method since it comes from two sources.
        """
        pass

    def plot(self):
        """
        Generates plots for scaled body rates (gyro) and accelerations.
        """
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            2, 1, figsize=(15, 10), sharex=True, num="IMU Scaled Data"
        )
        fig.suptitle("IMU Scaled Sensor Data", fontsize=16)

        # Plot gyroscope data if available.
        if self.has_gyro:
            df_gyro = self.all_data["IMU_GYRO_SCALED"]
            axs[0].plot(df_gyro["timestamp"], df_gyro["p"], label="p (roll rate)")
            axs[0].plot(df_gyro["timestamp"], df_gyro["q"], label="q (pitch rate)")
            axs[0].plot(df_gyro["timestamp"], df_gyro["r"], label="r (yaw rate)")
            axs[0].set_title("Scaled Gyroscope")
            axs[0].set_ylabel("Angular Rate (rad/s)")
            axs[0].legend()
            axs[0].grid(True)
        else:
            axs[0].set_title("Scaled Gyroscope - No Data")
            axs[0].grid(True)

        # Plot accelerometer data if available.
        if self.has_accel:
            df_accel = self.all_data["IMU_ACCEL_SCALED"]
            axs[1].plot(df_accel["timestamp"], df_accel["ax"], label="ax")
            axs[1].plot(df_accel["timestamp"], df_accel["ay"], label="ay")
            axs[1].plot(df_accel["timestamp"], df_accel["az"], label="az")
            axs[1].set_title("Scaled Accelerometer")
            axs[1].set_ylabel("Acceleration (m/s^2)")
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].set_title("Scaled Accelerometer - No Data")
            axs[1].grid(True)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
