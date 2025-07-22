import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from .base_tool import BaseTool
from tools import register_tool

@register_tool("System Errors")
class ErrorsTool(BaseTool):
    """
    Plots I2C error counts from the I2C_ERRORS message.
    """
    @property
    def required_messages(self) -> List[str]:
        return ["I2C_ERRORS"]

    def parse(self, *args, **kwargs):
        """
        Parses the error data, grouping by I2C bus number.
        """
        i2c_errors_df = self.all_data["I2C_ERRORS"]
        # Group data by the bus number to handle multiple I2C busses.
        self.parsed_data = dict(tuple(i2c_errors_df.groupby('bus_number')))

    def plot(self):
        """
        Generates a separate plot for each I2C bus's errors.
        """
        sns.set_theme(style="whitegrid")
        
        # Create a plot for each bus found in the data.
        for bus_number, df in self.parsed_data.items():
            # Identify all error counter columns in the dataframe.
            error_cols = [col for col in df.columns if 'cnt' in col]
            
            if not error_cols:
                continue

            plt.figure(figsize=(15, 8), num=f'I2C Bus {int(bus_number)} Errors')
            
            # Plot each error counter over time.
            for col in error_cols:
                # Plot only if there are actual errors to show.
                if df[col].sum() > 0:
                    plt.plot(df['timestamp'], df[col], label=col)
            
            plt.title(f'I2C Bus {int(bus_number)} Error Counts')
            plt.xlabel('Time (s)')
            plt.ylabel('Cumulative Count')
            # Only show legend if any errors were plotted.
            if plt.gca().has_data():
                plt.legend()
            plt.grid(True)
