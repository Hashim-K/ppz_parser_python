import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import numpy as np

from .base_tool import BaseTool
from tools import register_tool
from utils.spectral import get_psd


@register_tool("IMU FFT")
class IMUFFTTool(BaseTool):
    """
    Performs and plots a Fast Fourier Transform (FFT) on IMU data.
    """

    @property
    def required_messages(self) -> List[str]:
        return []  # Checked manually

    def check_required_messages(self) -> bool:
        """
        Check if at least one of the raw IMU messages is available.
        """
        self.has_accel = "IMU_ACCEL" in self.all_data
        self.has_gyro = "IMU_GYRO" in self.all_data
        return self.has_accel or self.has_gyro

    def parse(self, *args, **kwargs):
        """
        Calculate the Power Spectral Density for each IMU axis.
        """
        self.parsed_data = {"gyro": {}, "accel": {}}

        # Process Gyro data if available
        if self.has_gyro:
            df_gyro = self.all_data["IMU_GYRO"]
            if len(df_gyro) > 1:
                dt = np.mean(np.diff(df_gyro["timestamp"]))
                for axis in ["p", "q", "r"]:
                    f, Pxx = get_psd(df_gyro[axis].to_numpy(), dt)
                    self.parsed_data["gyro"][axis] = (f, Pxx)

        # Process Accel data if available
        if self.has_accel:
            df_accel = self.all_data["IMU_ACCEL"]
            if len(df_accel) > 1:
                dt = np.mean(np.diff(df_accel["timestamp"]))
                for axis in ["ax", "ay", "az"]:
                    f, Pxx = get_psd(df_accel[axis].to_numpy(), dt)
                    self.parsed_data["accel"][axis] = (f, Pxx)

    def plot(self):
        """
        Generates PSD plots for gyro and accelerometer data.
        """
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(
            2, 1, figsize=(15, 12), sharex=True, num="IMU FFT (PSD)"
        )
        fig.suptitle("IMU Power Spectral Density", fontsize=16)

        # Plot Gyro PSD
        ax = axs[0]
        if self.parsed_data["gyro"]:
            for axis, (f, Pxx) in self.parsed_data["gyro"].items():
                ax.semilogy(f, Pxx, label=axis)
            ax.set_title("Gyroscope PSD")
            ax.set_ylabel("Power/Frequency (dB/Hz)")
            ax.legend()
        else:
            ax.set_title("Gyroscope PSD - No Data")
        ax.grid(True)

        # Plot Accel PSD
        ax = axs[1]
        if self.parsed_data["accel"]:
            for axis, (f, Pxx) in self.parsed_data["accel"].items():
                ax.semilogy(f, Pxx, label=axis)
            ax.set_title("Accelerometer PSD")
            ax.set_ylabel("Power/Frequency (dB/Hz)")
            ax.legend()
        else:
            ax.set_title("Accelerometer PSD - No Data")
        ax.grid(True)

        plt.xlabel("Frequency (Hz)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
