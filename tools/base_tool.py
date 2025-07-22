import pandas as pd
from typing import Dict, List
from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Abstract Base Class for all plotting tools.

    Each tool is responsible for plotting a specific aspect of the flight data.
    """

    def __init__(self, data: Dict[str, pd.DataFrame], *args, **kwargs):
        """
        Initializes the tool with the full dataset and any specific parameters.
        """
        self.all_data = data
        self.parsed_data = None
        # Check if the tool has the data it needs before trying to parse.
        self.is_valid = self.check_required_messages()
        if self.is_valid:
            # If the data is available, parse it.
            self.parse(*args, **kwargs)

    @property
    @abstractmethod
    def required_messages(self) -> List[str]:
        """
        A list of Paparazzi message names that are required for this tool to function.
        If the tool can use one of several messages, this can return an empty list,
        and the logic should be handled in check_required_messages.
        """
        pass

    def check_required_messages(self) -> bool:
        """
        Checks if all required messages are present in the dataset.
        This can be overridden for more complex checks (e.g., "message A" OR "message B").
        """
        # The default check ensures every message in the required_messages list is present.
        return all(msg in self.all_data for msg in self.required_messages)

    @abstractmethod
    def parse(self, *args, **kwargs):
        """
        Parses/extracts the necessary data from the main dictionary.
        This method should populate self.parsed_data.
        """
        pass

    @abstractmethod
    def plot(self):
        """
        Generates the plot. This method should only be called if self.is_valid is True.
        """
        pass
