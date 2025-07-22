import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool


@register_tool("Actuators")
class ActuatorsTool(BaseTool):
    """
    Plots the output of specified actuator channels.
    """

    @property
    def required_messages(self) -> List[str]:
        return ["ACTUATORS"]

    def parse(self, channels: List[int] = None):
        """
        Parses the data for the specified actuator channels.
        """
        df = self.all_data["ACTUATORS"]

        # If no channels are specified, do not plot anything by default.
        if not channels:
            self.is_valid = False
            return

        # Find which of the requested channel columns actually exist in the dataframe.
        actuator_cols = [f"values_{i}" for i in channels if f"values_{i}" in df.columns]

        if not actuator_cols:
            # If none of the requested channels are found, the tool is not valid.
            self.is_valid = False
            return

        # Keep timestamp and only the requested, valid channels.
        self.parsed_data = df[["timestamp"] + actuator_cols]
        # Create a dynamic name for the plot window based on the channels.
        self.figure_name = f"Actuators {channels}"

    def plot(self):
        """
        Generates a plot with a separate subplot for each actuator channel.
        """
        df = self.parsed_data
        # Get the list of actuator columns to plot from the parsed data.
        actuator_cols = [col for col in df.columns if "values" in str(col)]

        if not actuator_cols:
            print(
                f"Skipping actuators plot '{self.figure_name}': No valid channels found."
            )
            return

        sns.set_theme(style="whitegrid")
        # Create a figure with a number of subplots equal to the number of channels.
        fig, axs = plt.subplots(
            len(actuator_cols),
            1,
            figsize=(12, 2 * len(actuator_cols)),
            sharex=True,
            num=self.figure_name,
        )

        # If there's only one subplot, axs is not a list, so we make it one.
        if len(actuator_cols) == 1:
            axs = [axs]

        # Plot each actuator output on its own subplot.
        for i, col in enumerate(actuator_cols):
            axs[i].plot(df["timestamp"], df[col])
            axs[i].set_title(f"Channel: {col}")
            axs[i].set_ylabel("Command")
            axs[i].grid(True)

        plt.xlabel("Time (s)")
        plt.suptitle("Actuator Outputs", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
