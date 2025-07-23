import os
import argparse
import json
from datetime import datetime

# Import the main log parser and the new event parser.
from parser.log_parser import PaparazziLogParser
from parser.event_parser import EventParser, EventSeverity, FlightEvent


def main():
    """
    Main function to parse a log file and generate a chronological event log.
    """
    # Set up a command-line argument parser.
    parser = argparse.ArgumentParser(
        description="Generate a chronological event log from a Paparazzi log file."
    )

    # The log file name is a required argument.
    parser.add_argument(
        "log_filename",
        help="The name of the .log file (e.g., '25_07_09__15_42_36.log').",
    )

    # Optional argument for the input directory.
    parser.add_argument(
        "--input_dir",
        default="input_logs",
        help="The directory where raw log files are stored.",
    )

    # Optional argument for the output directory.
    parser.add_argument(
        "--output_dir",
        default="processed_logs",
        help="The directory where processed event files are stored.",
    )

    # Optional argument to force reprocessing.
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if cached events.json exists.",
    )

    # Optional argument for log level filtering.
    parser.add_argument(
        "--log_level",
        choices=["debug", "info", "warning", "error"],
        default="debug",
        help="Minimum log level to display (default: debug).",
    )

    args = parser.parse_args()

    # Convert string log level to EventSeverity enum
    log_level_map = {
        "debug": EventSeverity.DEBUG,
        "info": EventSeverity.INFO,
        "warning": EventSeverity.WARNING,
        "error": EventSeverity.ERROR,
    }
    min_log_level = log_level_map[args.log_level]

    # Construct the full paths for the .log and .data files.
    log_file = os.path.join(args.input_dir, args.log_filename)
    data_file = log_file.replace(".log", ".data")

    # Create output directory structure
    log_base_name = args.log_filename.replace(".log", "")
    output_subdir = os.path.join(args.output_dir, log_base_name)
    events_json_path = os.path.join(output_subdir, "events.json")

    # Check that both required files exist.
    if not (os.path.exists(log_file) and os.path.exists(data_file)):
        print(f"Error: Log or data file not found for {args.log_filename}")
        return

    # Check if cached events already exist
    if os.path.exists(events_json_path) and not args.force:
        print(f"Found existing events in '{events_json_path}'. Loading from cache.")
        try:
            with open(events_json_path, "r") as f:
                cached_data = json.load(f)
                all_cached_events = cached_data.get("events", [])

            print(f"Loaded {len(all_cached_events)} cached events.")

            # Convert cached events back to FlightEvent objects for filtering
            cached_flight_events = []
            for event_data in all_cached_events:
                cached_flight_events.append(
                    FlightEvent(
                        timestamp=event_data["timestamp"],
                        level=EventSeverity(event_data["level"]),
                        message=event_data["message"],
                    )
                )

            # Filter events for display based on selected log level
            filtered_events = EventParser.filter_events_by_level_static(
                cached_flight_events, min_log_level
            )

            # Print the results
            print(f"\n--- Flight Event Log (Level: {min_log_level.value}) ---")
            if not filtered_events:
                print("No events found for the selected log level.")
            else:
                for event in filtered_events:
                    print(event)
            print("--- End of Log ---")
            return

        except Exception as e:
            print(f"Error loading cached events: {e}. Reprocessing...")
            # Continue to reprocess if cache loading fails

    try:
        # --- Step 1: Parse the entire log file into dataframes ---
        print(f"Parsing raw log file: {args.log_filename}...")
        # We don't use the JSON cache here because we need all message data.
        parser_instance = PaparazziLogParser(log_file, data_file)
        log_data = parser_instance.data
        print("Log file parsed successfully.")

        # --- Step 2: Initialize and run the event parser ---
        print("\nAnalyzing flight events chronologically...")
        event_parser = EventParser(log_data)  # Remove min_log_level parameter
        flight_events = event_parser.parse_events()  # Get ALL events
        print(f"Found {len(flight_events)} significant events.")

        # --- Step 3: Cache ALL events to JSON files ---
        print(f"\nCaching events to '{events_json_path}'...")

        # Create output directory if it doesn't exist
        os.makedirs(output_subdir, exist_ok=True)

        # Convert events to serializable format
        events_data = []
        for event in flight_events:
            events_data.append(
                {
                    "timestamp": event.timestamp,
                    "level": event.level.value,  # Convert enum to string
                    "message": event.message,
                }
            )

        # Save ALL events to main events.json
        cache_data = {
            "metadata": {
                "log_file": args.log_filename,
                "processed_at": datetime.now().isoformat(),
                "total_events": len(flight_events),
            },
            "events": events_data,
        }

        with open(events_json_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        print(f"Events cached successfully to '{events_json_path}'")

        # --- Step 4: Create separate log level files ---
        log_levels = [
            EventSeverity.DEBUG,
            EventSeverity.INFO,
            EventSeverity.WARNING,
            EventSeverity.ERROR,
        ]
        for level in log_levels:
            # Use exact level filtering for individual log files
            filtered_events = EventParser.filter_events_by_exact_level(
                flight_events, level
            )
            level_file_path = os.path.join(
                output_subdir, f"log_{level.value.lower()}.json"
            )

            level_events_data = []
            for event in filtered_events:
                level_events_data.append(
                    {
                        "timestamp": event.timestamp,
                        "level": event.level.value,
                        "message": event.message,
                    }
                )

            level_cache_data = {
                "metadata": {
                    "log_file": args.log_filename,
                    "log_level": level.value,
                    "processed_at": datetime.now().isoformat(),
                    "total_events": len(filtered_events),
                },
                "events": level_events_data,
            }

            with open(level_file_path, "w") as f:
                json.dump(level_cache_data, f, indent=2)

            print(
                f"Level {level.value} events saved to '{level_file_path}' ({len(filtered_events)} events)"
            )

        # --- Step 5: Display events based on selected log level ---
        display_events = event_parser.filter_events_by_level(
            flight_events, min_log_level
        )
        print(f"\n--- Flight Event Log (Level: {min_log_level.value}) ---")
        if not display_events:
            print("No events found for the selected log level.")
        else:
            for event in display_events:
                print(event)
        print("--- End of Log ---")

    except Exception as e:
        # Catch and print any errors that occur during parsing.
        print(f"An error occurred while processing {args.log_filename}: {e}")
        return


# This ensures the main function is called when the script is executed.
if __name__ == "__main__":
    main()
