import os
from .log_parser import PaparazziLogParser


def process_logs(input_dir: str, output_dir: str):
    """
    Scans for .log files in the input directory, finds their corresponding
    .data files, and processes them.
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".log"):
            log_file = os.path.join(input_dir, filename)
            data_file = os.path.join(input_dir, filename.replace(".log", ".data"))

            if os.path.exists(data_file):
                print(
                    f"Processing log pair: {filename} and {os.path.basename(data_file)}"
                )

                try:
                    # Create a parser instance
                    parser = PaparazziLogParser(log_file, data_file)

                    # Create a directory for the output based on the log name
                    log_name = os.path.splitext(filename)[0]
                    log_output_dir = os.path.join(output_dir, log_name)

                    # Save the parsed data to JSON
                    parser.to_json(log_output_dir)

                    # Example of generating a plot (uncomment to use)
                    # print("Plotting attitude...")
                    # parser.plot_attitude()

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            else:
                print(f"Warning: Found {filename} but no corresponding .data file.")


if __name__ == "__main__":
    # Define the input and output directories
    # These could also be passed as command-line arguments
    INPUT_LOG_DIR = "input_logs"
    PROCESSED_LOG_DIR = "processed_logs"

    # Create the directories if they don't exist
    if not os.path.exists(INPUT_LOG_DIR):
        os.makedirs(INPUT_LOG_DIR)
    if not os.path.exists(PROCESSED_LOG_DIR):
        os.makedirs(PROCESSED_LOG_DIR)

    process_logs(INPUT_LOG_DIR, PROCESSED_LOG_DIR)
    print("Log processing complete.")
