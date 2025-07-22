import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool

@register_tool("EKF2")
class EKF2Tool(BaseTool):
    """
    Plots state, covariance, and innovations from the EKF2 messages.
    """
    @property
    def required_messages(self) -> List[str]:
        return [] # Checked manually

    def check_required_messages(self) -> bool:
        """
        Checks if the required EKF2 messages are present.
        """
        self.has_state = "EKF2_STATE" in self.all_data
        self.has_p_diag = "EKF2_P_DIAG" in self.all_data
        self.has_innov = "EKF2_INNOV" in self.all_data
        return self.has_state or self.has_p_diag or self.has_innov

    def parse(self, *args, **kwargs):
        """
        Data is accessed directly in the plot methods.
        """
        pass

    def plot(self):
        """
        Calls the individual plot functions for state, covariance, and innovations.
        """
        if self.has_state:
            self._plot_state()
        if self.has_p_diag:
            self._plot_p_diag()
        if self.has_innov:
            self._plot_innovations()

    def _plot_state(self):
        """
        Plots the EKF2 state variables.
        """
        df = self.all_data["EKF2_STATE"]
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(4, 1, figsize=(15, 14), sharex=True, num='EKF2 State')
        fig.suptitle('EKF2 State', fontsize=16)

        # Position
        axs[0].plot(df['timestamp'], df['px'], label='px')
        axs[0].plot(df['timestamp'], df['py'], label='py')
        axs[0].plot(df['timestamp'], df['pz'], label='pz')
        axs[0].set_title('Position')
        axs[0].set_ylabel('m')
        axs[0].legend()
        axs[0].grid(True)

        # Velocity
        axs[1].plot(df['timestamp'], df['vx'], label='vx')
        axs[1].plot(df['timestamp'], df['vy'], label='vy')
        axs[1].plot(df['timestamp'], df['vz'], label='vz')
        axs[1].set_title('Velocity')
        axs[1].set_ylabel('m/s')
        axs[1].legend()
        axs[1].grid(True)

        # Quaternions
        axs[2].plot(df['timestamp'], df['q0'], label='q0 (w)')
        axs[2].plot(df['timestamp'], df['q1'], label='q1 (x)')
        axs[2].plot(df['timestamp'], df['q2'], label='q2 (y)')
        axs[2].plot(df['timestamp'], df['q3'], label='q3 (z)')
        axs[2].set_title('Quaternions')
        axs[2].legend()
        axs[2].grid(True)

        # Gyro Biases
        axs[3].plot(df['timestamp'], df['gbx'], label='gbx')
        axs[3].plot(df['timestamp'], df['gby'], label='gby')
        axs[3].plot(df['timestamp'], df['gbz'], label='gbz')
        axs[3].set_title('Gyro Biases')
        axs[3].set_ylabel('rad/s')
        axs[3].legend()
        axs[3].grid(True)

        plt.xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    def _plot_p_diag(self):
        """
        Plots the diagonal elements of the EKF2 covariance matrix.
        """
        df = self.all_data["EKF2_P_DIAG"]
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True, num='EKF2 Covariance')
        fig.suptitle('EKF2 Covariance (P Diagonal)', fontsize=16)

        # Position Variance
        axs[0].plot(df['timestamp'], df['P_px'], label='P_px')
        axs[0].plot(df['timestamp'], df['P_py'], label='P_py')
        axs[0].plot(df['timestamp'], df['P_pz'], label='P_pz')
        axs[0].set_title('Position Variance')
        axs[0].legend()
        axs[0].grid(True)

        # Velocity Variance
        axs[1].plot(df['timestamp'], df['P_vx'], label='P_vx')
        axs[1].plot(df['timestamp'], df['P_vy'], label='P_vy')
        axs[1].plot(df['timestamp'], df['P_vz'], label='P_vz')
        axs[1].set_title('Velocity Variance')
        axs[1].legend()
        axs[1].grid(True)

        # Gyro Bias Variance
        axs[2].plot(df['timestamp'], df['P_gbx'], label='P_gbx')
        axs[2].plot(df['timestamp'], df['P_gby'], label='P_gby')
        axs[2].plot(df['timestamp'], df['P_gbz'], label='P_gbz')
        axs[2].set_title('Gyro Bias Variance')
        axs[2].legend()
        axs[2].grid(True)

        plt.xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    def _plot_innovations(self):
        """
        Plots the EKF2 innovations.
        """
        df = self.all_data["EKF2_INNOV"]
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True, num='EKF2 Innovations')
        fig.suptitle('EKF2 Innovations', fontsize=16)

        # Position Innovations
        axs[0].plot(df['timestamp'], df['innov_px'], label='innov_px')
        axs[0].plot(df['timestamp'], df['innov_py'], label='innov_py')
        axs[0].plot(df['timestamp'], df['innov_pz'], label='innov_pz')
        axs[0].set_title('Position Innovations')
        axs[0].legend()
        axs[0].grid(True)

        # Velocity Innovations
        axs[1].plot(df['timestamp'], df['innov_vx'], label='innov_vx')
        axs[1].plot(df['timestamp'], df['innov_vy'], label='innov_vy')
        axs[1].plot(df['timestamp'], df['innov_vz'], label='innov_vz')
        axs[1].set_title('Velocity Innovations')
        axs[1].legend()
        axs[1].grid(True)

        plt.xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
