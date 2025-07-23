import os
import re
import json
from typing import Dict, Any

import pandas as pd
import numpy as np

# Import from the same package.
from .message_parser import parse_message_definitions


class PaparazziLogParser:
    """
    A class to parse Paparazzi log files (.data and .log pairs).
    """

    def __init__(self, log_file: str, data_file: str):
        # Store file paths for reference.
        self.log_file = log_file
        self.data_file = data_file
        # Parse the message definitions from the .log file immediately.
        self.message_definitions = parse_message_definitions(self.log_file)
        # Parse the data from the .data file and store it.
        self.data: Dict[str, pd.DataFrame] = self._parse_data()

    def _parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parses the .data file and returns a dictionary of Pandas DataFrames.
        This function is optimized to handle large log files and array fields.
        """
        # Read all lines from the data file into memory.
        with open(self.data_file, "r") as f:
            lines = f.readlines()

        parsed_lines = []
        for line in lines:
            # Split only up to 3 times to separate timestamp, ac_id, message_name, and the values string.
            parts = line.strip().split(" ", 3)
            if len(parts) == 4:
                parsed_lines.append(parts)
            elif len(parts) == 3:
                # Handle messages that have no values payload by adding an empty string.
                parsed_lines.append(parts + [""])

        # If the file is empty or couldn't be parsed, return an empty dictionary.
        if not parsed_lines:
            return {}

        # Create a DataFrame from the parsed lines for efficient processing.
        df = pd.DataFrame(
            parsed_lines, columns=["timestamp", "ac_id", "message_name", "values"]
        )
        # Convert timestamp and ac_id columns to numeric types.
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["ac_id"] = pd.to_numeric(df["ac_id"], errors="coerce")

        parsed_data = {}
        # Group the DataFrame by message name to process each message type separately.
        for msg_name, group in df.groupby("message_name"):
            if msg_name not in self.message_definitions:
                # Create a generic message entry for undefined messages
                parsed_data[msg_name] = group[["timestamp", "ac_id"]].copy()
                # Add the raw values for potential generic event detection
                parsed_data[msg_name]["raw_values"] = group["values"]
                continue

            msg_def = self.message_definitions[msg_name]
            field_names = msg_def["field_names"]

            # If the message has no fields defined, just store timestamp and ac_id.
            if not field_names:
                parsed_data[msg_name] = group[["timestamp", "ac_id"]].copy()
                continue

            # Split the 'values' string into separate columns for each field.
            # This regex splits by comma or space, but correctly handles quoted strings.
            values_split = group["values"].str.split(
                r'[ ,]+(?=(?:[^"]*"[^"]*")*[^"]*$)', expand=True
            )

            # Handle cases where the number of values doesn't match the number of fields,
            # which often indicates an array field (like 'values' in ACTUATORS).
            if values_split.shape[1] > len(field_names) and len(field_names) > 0:
                # Assume the last defined field is an array that consumes the rest of the values.
                num_scalar_fields = len(field_names) - 1
                scalar_cols = field_names[:-1]
                array_base_name = field_names[-1]

                # Create a DataFrame for the scalar fields.
                scalar_df = values_split.iloc[:, :num_scalar_fields]
                scalar_df.columns = scalar_cols

                # Create a DataFrame for the array field, naming columns like 'values_0', 'values_1', etc.
                array_df = values_split.iloc[:, num_scalar_fields:]
                array_df.columns = [
                    f"{array_base_name}_{i}" for i in range(array_df.shape[1])
                ]

                # Combine the scalar and array DataFrames.
                values_df = pd.concat([scalar_df, array_df], axis=1)
            else:
                # For standard messages, assign field names to the split columns.
                num_to_take = min(values_split.shape[1], len(field_names))
                values_df = values_split.iloc[:, :num_to_take]
                values_df.columns = field_names[:num_to_take]

            # Convert all field columns to numeric types where possible.
            # This is a vectorized and much more efficient approach.
            for col in values_df.columns:
                # Using errors='coerce' will turn any values that cannot be converted into NaN (Not a Number).
                values_df[col] = pd.to_numeric(values_df[col], errors="coerce")

            # Combine the parsed values with the timestamp and ac_id columns.
            msg_df = pd.concat(
                [
                    group[["timestamp", "ac_id"]].reset_index(drop=True),
                    values_df.reset_index(drop=True),
                ],
                axis=1,
            )

            parsed_data[msg_name] = msg_df

        return parsed_data

    def to_json(self, output_dir: str):
        """
        Saves the parsed data into JSON files in the specified directory.
        """
        # Create the output directory if it doesn't already exist.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # These groupings determine how messages are bundled into JSON files.
        # This can be customized based on which data you want to access together.
        functional_groups = {
            "attitude": [
                "ATTITUDE",
                "STAB_ATTITUDE",
                "AHRS_REF_QUAT",
                "IMU_GYRO_SCALED",
                "IMU_ACCEL_SCALED",
                "IMU_GYRO",
                "IMU_ACCEL",
            ],
            "position": [
                "GPS",
                "INS",
                "ROTORCRAFT_FP",
                "GUIDANCE_INDI_HYBRID",
                "GUIDANCE",
                "EXTERNAL_POSE_DOWN",
            ],
            "actuators": [
                "ACTUATORS",
                "ROTORCRAFT_CMD",
                "SERIAL_ACT_T4_IN",
                "SERIAL_ACT_T4_OUT",
            ],
            "power": [
                "ENERGY",
                "POWER_DEVICE",
                "ESC",
                "POWER",
                "CURRENT_SPIKE",
                "POWER_DISTRIBUTION",
                "CHARGING_STATUS",
                "CELL_BALANCE",
                "THERMAL_THROTTLE",
            ],
            "system": [
                "ROTORCRAFT_STATUS",
                "I2C_ERRORS",
                "AUTOPILOT_VERSION",
                "GROUND_DETECT",
            ],
            "control": ["INDI_ROTWING", "EFF_MAT", "ROTWING_STATE"],
            "sensors": ["AIRSPEED"],
            "ekf": ["EKF2_STATE", "EKF2_P_DIAG", "EKF2_INNOV"],
            "safety": [
                "EMERGENCY",
                "GEOFENCE_BREACH",
                "COLLISION_AVOIDANCE",
                "TRAFFIC",
                "TERRAIN_FOLLOWING",
                "OBSTACLE_DETECTION",
                "LOSS_OF_CONTROL",
                "STALL_WARNING",
                "OVER_SPEED",
                "ALTITUDE_LIMIT",
            ],
            "communications": [
                "TELEMETRY_STATUS",
                "RADIO_STATUS",
                "MODEM_STATUS",
                "LINK_QUALITY",
                "PACKET_LOSS",
                "RSSI_LOW",
            ],
            "mission": [
                "MISSION_ITEM",
                "HOME_POSITION",
                "RALLY_POINT",
                "SURVEY_STATUS",
                "LANDING_SEQUENCE",
                "APPROACH",
            ],
        }

        for group_name, msg_list in functional_groups.items():
            group_data = {}
            for msg_name in msg_list:
                if msg_name in self.data:
                    # Convert dataframe to a list of records (dictionaries).
                    # NaN values from parsing are converted to None (null in JSON), which is standard.
                    group_data[msg_name] = (
                        self.data[msg_name].replace({np.nan: None}).to_dict("records")
                    )

            if group_data:
                file_path = os.path.join(output_dir, f"{group_name}.json")
                # Use Python's built-in json library for a clean and standard JSON output.
                with open(file_path, "w") as f:
                    json.dump(group_data, f, indent=4)
                print(f"Saved {group_name}.json to {output_dir}")
