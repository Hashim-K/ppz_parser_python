import os
import json
import argparse
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt

# Import the parser and the TOOL_REGISTRY
from parser.log_parser import PaparazziLogParser
from tools import TOOL_REGISTRY


def load_data_from_json(json_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Loads all JSON files from a directory and reconstructs the data dictionary
    of pandas DataFrames.
    """
    reconstructed_data = {}
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(json_dir, filename)
            with open(file_path, "r") as f:
                functional_group = json.load(f)
                for msg_name, records in functional_group.items():
                    reconstructed_data[msg_name] = pd.DataFrame.from_records(records)
    return reconstructed_data


def main():
    """
    Main function to parse a log file and generate plots using a modular tool system.
    """
    parser = argparse.ArgumentParser(
        description="Parse and plot a Paparazzi log file using a modular tool system."
    )
    parser.add_argument(
        "log_filename",
        help="The name of the .log file (e.g., '25_07_09__15_42_36.log').",
    )
    parser.add_argument(
        "--input_dir", default="input_logs", help="Directory for raw log files."
    )
    parser.add_argument(
        "--output_dir",
        default="processed_logs",
        help="Directory for processed JSON data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-parsing, ignoring any cached JSON.",
    )
    args = parser.parse_args()

    log_name = os.path.splitext(args.log_filename)[0]
    log_output_dir = os.path.join(args.output_dir, log_name)

    log_data = None

    # --- Caching Logic ---
    if not args.force and os.path.exists(log_output_dir):
        print(f"Found existing data in '{log_output_dir}'. Loading from JSON cache.")
        try:
            log_data = load_data_from_json(log_output_dir)
            if not log_data:
                print("Warning: Cache directory is empty. Will re-process.")
        except Exception as e:
            print(
                f"Warning: Could not load data from JSON cache ({e}). Will re-process."
            )

    if log_data is None:
        print("Parsing raw log files...")
        log_file = os.path.join(args.input_dir, args.log_filename)
        data_file = log_file.replace(".log", ".data")

        if not (os.path.exists(log_file) and os.path.exists(data_file)):
            print(f"Error: Log or data file not found for {args.log_filename}")
            return

        try:
            parser_instance = PaparazziLogParser(log_file, data_file)
            log_data = parser_instance.data
            print(f"Saving parsed data to JSON in '{log_output_dir}'...")
            parser_instance.to_json(log_output_dir)
        except Exception as e:
            print(f"An error occurred while processing {args.log_filename}: {e}")
            return

    if not log_data:
        print("Could not load or parse any data to plot.")
        return

    # --- Tool Execution ---
    print("\nAvailable tools:", sorted(list(TOOL_REGISTRY.keys())))
    print("-" * 20)

    # This is the complete list of all implemented tools.
    tool_configurations = [
        {"name": "GPS Data"},
        {"name": "Euler Angles"},
        {"name": "Energy"},
        {"name": "Airspeed"},
        {"name": "EKF2"},
        {"name": "Effectiveness Matrix"},
        {"name": "System Errors"},
        {"name": "Ground Detection"},
        {"name": "Hybrid Guidance"},
        {"name": "Hover Loop"},
        {"name": "INDI Rotwing"},
        {"name": "IMU Raw"},
        {"name": "IMU Scaled"},
        {"name": "IMU FFT"},
        {"name": "Device Powers"},
        {"name": "Quaternions"},
        {"name": "Stabilization Attitude"},
        {"name": "Rotorcraft CMD"},
        {"name": "Rotorcraft FP"},
        {"name": "Rotorcraft Status"},
        {"name": "Rotwing Flight Envelope"},
        {"name": "Rotwing State"},
        # Parametric tool examples:
        {"name": "Actuators", "params": {"channels": [1, 2, 3, 4]}},
        {"name": "ESC Telemetry", "params": {"motor_ids": [0, 1, 2, 3]}},
    ]

    active_tools = []
    for config in tool_configurations:
        tool_name = config["name"]
        if tool_name in TOOL_REGISTRY:
            ToolClass = TOOL_REGISTRY[tool_name]
            params = config.get("params", {})
            # Instantiate the tool. It will automatically check if it's valid.
            tool_instance = ToolClass(log_data, **params)
            if tool_instance.is_valid:
                active_tools.append(tool_instance)
                print(f"Activating tool: {tool_name} with params: {params}")
            else:
                print(
                    f"Skipping tool '{tool_name}': Required messages not found or parameters invalid."
                )
        else:
            print(f"Warning: Tool '{tool_name}' not found in registry.")

    if not active_tools:
        print("\nNo active tools to run for this log file.")
        return

    print("-" * 20)
    print("Generating plots...")

    # Call the plot method for every active tool.
    for tool in active_tools:
        tool.plot()

    # Show all the plot windows that were created.
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
