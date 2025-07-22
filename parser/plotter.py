import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import pandas as pd


def plot_attitude(data: Dict[str, pd.DataFrame], start_time=None, end_time=None):
    """
    Plots attitude data (roll, pitch, yaw) from the ATTITUDE message.
    """
    # Check if the necessary data is available
    if "ATTITUDE" not in data:
        print("Skipping attitude plot: ATTITUDE message not found.")
        return

    df = data["ATTITUDE"]
    # Filter data by the specified time range if provided
    if start_time:
        df = df[df["timestamp"] >= start_time]
    if end_time:
        df = df[df["timestamp"] <= end_time]

    # Configure plot aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8))

    # Plot roll, pitch, and yaw against time
    plt.plot(df["timestamp"], df["phi"], label="Roll (phi)")
    plt.plot(df["timestamp"], df["theta"], label="Pitch (theta)")
    plt.plot(df["timestamp"], df["psi"], label="Yaw (psi)")

    # Set plot titles and labels
    plt.title("Attitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)


def plot_position(data: Dict[str, pd.DataFrame], start_time=None, end_time=None):
    """
    Plots position, velocity, and angle data from the ROTORCRAFT_FP message.
    """
    # Check if the necessary data is available
    if "ROTORCRAFT_FP" not in data:
        print("Skipping position plot: ROTORCRAFT_FP message not found.")
        return

    df = data["ROTORCRAFT_FP"]
    # Filter data by the specified time range if provided
    if start_time:
        df = df[df["timestamp"] >= start_time]
    if end_time:
        df = df[df["timestamp"] <= end_time]

    # Configure plot aesthetics
    sns.set_theme(style="whitegrid")
    # Create a figure with 3 subplots, sharing the x-axis
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Plot position data on the first subplot
    axs[0].plot(df["timestamp"], df["east"], label="East")
    axs[0].plot(df["timestamp"], df["north"], label="North")
    axs[0].plot(df["timestamp"], df["up"], label="Up")
    axs[0].set_title("Position")
    axs[0].set_ylabel("Position (m)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot velocity data on the second subplot
    axs[1].plot(df["timestamp"], df["veast"], label="VEast")
    axs[1].plot(df["timestamp"], df["vnorth"], label="VNorth")
    axs[1].plot(df["timestamp"], df["vup"], label="VUp")
    axs[1].set_title("Velocity")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()
    axs[1].grid(True)

    # Plot angle data on the third subplot
    axs[2].plot(df["timestamp"], df["phi"], label="Phi")
    axs[2].plot(df["timestamp"], df["theta"], label="Theta")
    axs[2].plot(df["timestamp"], df["psi"], label="Psi")
    axs[2].set_title("Angles")
    axs[2].set_ylabel("Angle (deg)")
    axs[2].legend()
    axs[2].grid(True)

    plt.xlabel("Time (s)")
    plt.tight_layout()


def plot_actuators(data: Dict[str, pd.DataFrame], start_time=None, end_time=None):
    """
    Plots actuator outputs from the ACTUATORS message.
    """
    # Check if the necessary data is available
    if "ACTUATORS" not in data:
        print("Skipping actuators plot: ACTUATORS message not found.")
        return

    df = data["ACTUATORS"]
    # Filter data by the specified time range if provided
    if start_time:
        df = df[df["timestamp"] >= start_time]
    if end_time:
        df = df[df["timestamp"] <= end_time]

    # Find columns that represent actuator values (e.g., values_0, values_1, ...)
    actuator_cols = [col for col in df.columns if "values" in str(col)]
    if not actuator_cols:
        print("Skipping actuators plot: No actuator 'values' columns found.")
        return

    # Configure plot aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8))

    # Plot each actuator output
    for col in actuator_cols:
        plt.plot(df["timestamp"], df[col], label=col)

    # Set plot titles and labels
    plt.title("Actuator Outputs")
    plt.xlabel("Time (s)")
    plt.ylabel("Command Value")
    plt.legend()
    plt.grid(True)


def plot_gps(data: Dict[str, pd.DataFrame], start_time=None, end_time=None):
    """
    Plots key GPS data from the GPS message.
    """
    # Check if the necessary data is available
    if "GPS" not in data:
        print("Skipping GPS plot: GPS message not found.")
        return

    df = data["GPS"]
    # Filter data by the specified time range if provided
    if start_time:
        df = df[df["timestamp"] >= start_time]
    if end_time:
        df = df[df["timestamp"] <= end_time]

    # Configure plot aesthetics
    sns.set_theme(style="whitegrid")
    # Create a figure with 3 subplots, sharing the x-axis
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Plot position data on the first subplot
    axs[0].plot(df["timestamp"], df["utm_east"], label="UTM East")
    axs[0].plot(df["timestamp"], df["utm_north"], label="UTM North")
    axs[0].set_title("GPS Position")
    axs[0].set_ylabel("Position (m)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot velocity data on the second subplot
    axs[1].plot(df["timestamp"], df["gspeed"], label="Ground Speed")
    axs[1].plot(df["timestamp"], df["climb"], label="Climb Rate")
    axs[1].set_title("GPS Velocity")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()
    axs[1].grid(True)

    # Plot GPS status data on the third subplot
    axs[2].plot(df["timestamp"], df["nb_sats"], label="Number of Satellites")
    axs[2].set_title("GPS Status")
    axs[2].set_ylabel("Count")
    axs[2].legend()
    axs[2].grid(True)

    plt.xlabel("Time (s)")
    plt.tight_layout()


def plot_imu(data: Dict[str, pd.DataFrame], start_time=None, end_time=None):
    """
    Plots scaled IMU data from IMU_ACCEL_SCALED and IMU_GYRO_SCALED.
    """
    # Check if the necessary data is available
    has_accel = "IMU_ACCEL_SCALED" in data
    has_gyro = "IMU_GYRO_SCALED" in data
    if not has_accel and not has_gyro:
        print("Skipping IMU plot: No scaled IMU messages found.")
        return

    # Configure plot aesthetics
    sns.set_theme(style="whitegrid")
    # Create a figure with 2 subplots, sharing the x-axis
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot accelerometer data if available
    if has_accel:
        df_accel = data["IMU_ACCEL_SCALED"]
        axs[0].plot(df_accel["timestamp"], df_accel["ax"], label="ax")
        axs[0].plot(df_accel["timestamp"], df_accel["ay"], label="ay")
        axs[0].plot(df_accel["timestamp"], df_accel["az"], label="az")
        axs[0].set_title("Scaled Accelerometer")
        axs[0].set_ylabel("Acceleration (m/s^2)")
        axs[0].legend()
        axs[0].grid(True)
    else:
        axs[0].set_title("Scaled Accelerometer (No Data)")

    # Plot gyroscope data if available
    if has_gyro:
        df_gyro = data["IMU_GYRO_SCALED"]
        axs[1].plot(df_gyro["timestamp"], df_gyro["p"], label="p")
        axs[1].plot(df_gyro["timestamp"], df_gyro["q"], label="q")
        axs[1].plot(df_gyro["timestamp"], df_gyro["r"], label="r")
        axs[1].set_title("Scaled Gyroscope")
        axs[1].set_ylabel("Angular Rate (rad/s)")
        axs[1].legend()
        axs[1].grid(True)
    else:
        axs[1].set_title("Scaled Gyroscope (No Data)")

    plt.xlabel("Time (s)")
    plt.tight_layout()


def plot_all(data: Dict[str, pd.DataFrame], start_time=None, end_time=None):
    """
    Calls all available plotting functions to generate a full flight report.
    """
    # Create a list of all plotting functions
    all_plot_functions = [
        plot_attitude,
        plot_position,
        plot_actuators,
        plot_gps,
        plot_imu,
    ]

    # Call each plotting function with the provided data and time range
    for plot_func in all_plot_functions:
        try:
            plot_func(data, start_time, end_time)
        except Exception as e:
            # Print an error if a specific plot fails, but continue with the rest
            print(f"Could not generate plot '{plot_func.__name__}': {e}")

    # Display all the generated plot windows
    plt.show()
