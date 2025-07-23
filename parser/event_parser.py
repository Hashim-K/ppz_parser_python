from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Callable, Optional, Any
import pandas as pd
import numpy as np

# Import our new data structures.
from .datastructures import (
    MESSAGE_TO_CLASS_MAP,
    Energy,
    RotorcraftStatus,
    I2CErrors,
    IMU,
    GPS,
    GPSInt,
    RotorcraftFP,
    AirData,
    Actuators,
    BebopActuators,
    DatalinkReport,
    AutopilotVersion,
    Alive,
    DLValue,
    RotorcraftNavStatus,
    WPMoved,
    StateFilterStatus,
    UartErrors,
    InsRef,
    Ins,
    Survey,
    GuidanceIndiHybrid,
    Guidance,
    ExternalPoseDown,
    SerialActT4In,
    SerialActT4Out,
    PowerDevice,
    EffMat,
    EKF2PDiag,
    EKF2Innov,
    # Additional flight monitoring messages
    StabilizationAttitude,
    NavStatus,
    Waypoint,
    FlightPlan,
    RCLost,
    DatalinkLost,
    Geofence,
    Weather,
    WindEstimation,
    BatteryStatus,
    MotorStatus,
    Vibration,
    CompassCal,
    Barometer,
    Temperature,
    # Safety-critical message types
    Emergency,
    GeofenceBreach,
    CollisionAvoidance,
    Traffic,
    TerrainFollowing,
    ObstacleDetection,
    LossOfControl,
    StallWarning,
    OverSpeed,
    AltitudeLimit,
    # Communication and telemetry message types
    TelemetryStatus,
    RadioStatus,
    ModemStatus,
    LinkQuality,
    PacketLoss,
    RSSILow,
    # Power management message types
    CurrentSpike,
    PowerDistribution,
    ChargingStatus,
    CellBalance,
    ThermalThrottle,
    # Mission and navigation message types
    MissionItem,
    HomePosition,
    RallyPoint,
    SurveyStatus,
    LandingSequence,
    Approach,
)

# Note: We'll add this rule for any message type not explicitly handled above
# This is done in the parse_events method when GenericMessage is used


class EventSeverity(Enum):
    """
    Defines the severity level for a flight event.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

    @property
    def priority(self) -> int:
        """Return the priority level for filtering (higher number = higher priority)."""
        priorities = {
            EventSeverity.DEBUG: 0,
            EventSeverity.INFO: 1,
            EventSeverity.WARNING: 2,
            EventSeverity.ERROR: 3,
        }
        return priorities[self]

    def __ge__(self, other):
        """Support >= comparison for log level filtering."""
        if not isinstance(other, EventSeverity):
            return NotImplemented
        return self.priority >= other.priority

    def __gt__(self, other):
        """Support > comparison for log level filtering."""
        if not isinstance(other, EventSeverity):
            return NotImplemented
        return self.priority > other.priority

    def __le__(self, other):
        """Support <= comparison for log level filtering."""
        if not isinstance(other, EventSeverity):
            return NotImplemented
        return self.priority <= other.priority

    def __lt__(self, other):
        """Support < comparison for log level filtering."""
        if not isinstance(other, EventSeverity):
            return NotImplemented
        return self.priority < other.priority


@dataclass
class FlightEvent:
    """
    A simple data structure to hold information about a single flight event.
    """

    timestamp: float
    level: EventSeverity
    message: str

    def __repr__(self) -> str:
        # A helper to make printing the event easy to read.
        return f"[{self.timestamp:10.4f}] [{self.level.value:^7}] {self.message}"


def tag_event(timestamp: float, message: str) -> FlightEvent:
    """
    Centralized event tagging function that assigns log levels based on message content.
    Checks in priority order: ERROR -> WARNING -> INFO -> DEBUG (default)
    """
    message_lower = message.lower()

    # ERROR level criteria
    error_keywords = [
        "error",
        "failure",
        "fail",
        "crash",
        "critical",
        "emergency",
        "i2c acknowledge failure",
        "communication error",
        "system failure",
    ]
    for keyword in error_keywords:
        if keyword in message_lower:
            return FlightEvent(timestamp, EventSeverity.ERROR, message)

    # WARNING level criteria
    warning_keywords = [
        "warning",
        "low battery",
        "high g-force",
        "extreme attitude",
        "high velocity",
        "high power consumption",
        "gps fix lost",
    ]
    for keyword in warning_keywords:
        if keyword in message_lower:
            return FlightEvent(timestamp, EventSeverity.WARNING, message)

    # INFO level criteria
    info_keywords = [
        "flight mode changed",
        "takeoff detected",
        "landing detected",
        "mode changed",
        "takeoff",
        "landing",
    ]
    for keyword in info_keywords:
        if keyword in message_lower:
            return FlightEvent(timestamp, EventSeverity.INFO, message)

    # Default to DEBUG level
    return FlightEvent(timestamp, EventSeverity.DEBUG, message)


# A type hint for our new event checking functions that return (timestamp, message) tuples.
EventChecker = Callable[[Any, Optional[Any]], Optional[tuple[float, str]]]


class EventParser:
    """
    Parses a complete log file chronologically and generates a tagged event log
    based on a defined set of rules that operate on data classes.
    """

    def __init__(self, log_data: Dict[str, pd.DataFrame]):
        self.log_data = log_data
        # The registry now stores checkers keyed by message name for efficiency.
        self.event_checkers: Dict[str, List[EventChecker]] = {}

        # Register all the event-checking rules when the class is initialized.
        self._register_event_rules()

    def _register_event_rules(self):
        """
        This is the central registry for all event detection rules.
        """

        # --- Event Detection Functions ---
        def flight_mode_changed(
            current: RotorcraftStatus, prev: Optional[RotorcraftStatus]
        ) -> Optional[tuple[float, str]]:
            if prev is None or current.flight_mode != prev.flight_mode:
                return (
                    current.timestamp,
                    f"Flight Mode changed to {current.flight_mode}",
                )
            return None

        self.add_rule("ROTORCRAFT_STATUS", flight_mode_changed)

        def low_battery_warning(
            current: Energy, prev: Optional[Energy]
        ) -> Optional[tuple[float, str]]:
            # Warning if voltage drops below 11V (for typical LiPo)
            if current.voltage < 11.0 and (prev is None or prev.voltage >= 11.0):
                return (
                    current.timestamp,
                    f"Low battery warning: {current.voltage:.1f}V",
                )
            return None

        # Add GPS acquisition event
        def gps_acquired(
            current: GPSInt, prev: Optional[GPSInt]
        ) -> Optional[tuple[float, str]]:
            if current.mode >= 3 and (prev is None or prev.mode < 3):
                return (
                    current.timestamp,
                    f"GPS 3D fix acquired (satellites: {current.nb_sats})",
                )
            return None

        def gps_lost(
            current: GPSInt, prev: Optional[GPSInt]
        ) -> Optional[tuple[float, str]]:
            if current.mode < 3 and prev is not None and prev.mode >= 3:
                return (current.timestamp, f"GPS fix lost (mode: {current.mode})")
            return None

        # Add altitude change events using RotorcraftFP
        def significant_altitude_change(
            current: RotorcraftFP, prev: Optional[RotorcraftFP]
        ) -> Optional[tuple[float, str]]:
            if prev is not None:
                altitude_diff = abs(current.up - prev.up)
                if altitude_diff > 10.0:  # 10 meter significant change
                    direction = "gained" if current.up > prev.up else "lost"
                    return (
                        current.timestamp,
                        f"Aircraft {direction} {altitude_diff:.1f}m altitude (now at {current.up:.1f}m)",
                    )
            return None

        # Add velocity events using RotorcraftFP
        def high_velocity_warning(
            current: RotorcraftFP, prev: Optional[RotorcraftFP]
        ) -> Optional[tuple[float, str]]:
            velocity = (current.veast**2 + current.vnorth**2 + current.vup**2) ** 0.5
            if velocity > 15.0:  # 15 m/s threshold
                return (
                    current.timestamp,
                    f"High velocity detected: {velocity:.1f} m/s",
                )
            return None

        # Add attitude events
        def extreme_attitude_warning(
            current: RotorcraftFP, prev: Optional[RotorcraftFP]
        ) -> Optional[tuple[float, str]]:
            import math

            roll_deg = abs(math.degrees(current.phi))
            pitch_deg = abs(math.degrees(current.theta))

            if roll_deg > 45 or pitch_deg > 45:
                return (
                    current.timestamp,
                    f"Extreme attitude: roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°",
                )
            return None

        # Add takeoff detection based on altitude
        def takeoff_detected(
            current: RotorcraftFP, prev: Optional[RotorcraftFP]
        ) -> Optional[tuple[float, str]]:
            if prev is not None and prev.up < 2.0 and current.up >= 2.0:
                return (
                    current.timestamp,
                    f"Takeoff detected - altitude: {current.up:.1f}m",
                )
            return None

        def landing_detected(
            current: RotorcraftFP, prev: Optional[RotorcraftFP]
        ) -> Optional[tuple[float, str]]:
            if prev is not None and prev.up >= 2.0 and current.up < 2.0:
                return (
                    current.timestamp,
                    f"Landing detected - altitude: {current.up:.1f}m",
                )
            return None

        # Add power consumption monitoring
        def high_power_consumption(
            current: Energy, prev: Optional[Energy]
        ) -> Optional[tuple[float, str]]:
            power = current.power  # Uses the property we defined
            if power > 500.0:  # 500W threshold
                return (current.timestamp, f"High power consumption: {power:.1f}W")
            return None

        # Register new rules if the message types exist
        if "ENERGY" in self.log_data:
            self.add_rule("ENERGY", low_battery_warning)
            self.add_rule("ENERGY", high_power_consumption)
        if "GPS_INT" in self.log_data:
            self.add_rule("GPS_INT", gps_acquired)
            self.add_rule("GPS_INT", gps_lost)
        if "ROTORCRAFT_FP" in self.log_data:
            self.add_rule("ROTORCRAFT_FP", significant_altitude_change)
            self.add_rule("ROTORCRAFT_FP", high_velocity_warning)
            self.add_rule("ROTORCRAFT_FP", extreme_attitude_warning)
            self.add_rule("ROTORCRAFT_FP", takeoff_detected)
            self.add_rule("ROTORCRAFT_FP", landing_detected)

        # Additional event detectors for new message types
        def datalink_quality_warning(
            current: DatalinkReport, prev: Optional[DatalinkReport]
        ) -> Optional[tuple[float, str]]:
            if current.link_quality < 50.0:  # Below 50% quality
                return (
                    current.timestamp,
                    f"Poor datalink quality: {current.link_quality:.1f}%",
                )
            if current.rssi < -80.0:  # Weak signal
                return (
                    current.timestamp,
                    f"Weak datalink signal: {current.rssi:.1f} dBm",
                )
            return None

        def uart_communication_errors(
            current: UartErrors, prev: Optional[UartErrors]
        ) -> Optional[tuple[float, str]]:
            if prev is not None:
                new_errors = (
                    current.overrun_err
                    - prev.overrun_err
                    + current.framing_err
                    - prev.framing_err
                    + current.noise_err
                    - prev.noise_err
                )
                if new_errors > 0:
                    return (
                        current.timestamp,
                        f"UART communication errors detected: {new_errors} new errors",
                    )
            return None

        def actuator_saturation_warning(
            current: Actuators, prev: Optional[Actuators]
        ) -> Optional[tuple[float, str]]:
            # Check if any motor is at or near maximum
            max_motor = max(
                current.motor1, current.motor2, current.motor3, current.motor4
            )
            if max_motor > 90.0:  # Assuming 0-100 range
                return (
                    current.timestamp,
                    f"Actuator saturation warning: max motor at {max_motor:.1f}%",
                )
            return None

        def bebop_actuator_imbalance(
            current: BebopActuators, prev: Optional[BebopActuators]
        ) -> Optional[tuple[float, str]]:
            motors = [
                current.motor_front_left,
                current.motor_front_right,
                current.motor_back_right,
                current.motor_back_left,
            ]
            max_motor = max(motors)
            min_motor = min(motors)
            imbalance = max_motor - min_motor

            if imbalance > 30.0:  # Large imbalance between motors
                return (
                    current.timestamp,
                    f"Motor imbalance detected: {imbalance:.1f}% difference",
                )
            return None

        def air_data_anomaly(
            current: AirData, prev: Optional[AirData]
        ) -> Optional[tuple[float, str]]:
            if prev is not None:
                # Check for rapid altitude changes
                alt_change = abs(current.altitude - prev.altitude)
                if alt_change > 50.0:  # 50m sudden change
                    return (
                        current.timestamp,
                        f"Rapid altitude change: {alt_change:.1f}m",
                    )

                # Check for unrealistic airspeed
                if current.airspeed > 50.0:  # 50 m/s for rotorcraft
                    return (
                        current.timestamp,
                        f"High airspeed detected: {current.airspeed:.1f} m/s",
                    )
            return None

        def waypoint_reached(
            current: WPMoved, prev: Optional[WPMoved]
        ) -> Optional[tuple[float, str]]:
            return (
                current.timestamp,
                f"Waypoint {current.wp_id} reached/updated at ({current.utm_east:.1f}, {current.utm_north:.1f})",
            )

        def autopilot_version_info(
            current: AutopilotVersion, prev: Optional[AutopilotVersion]
        ) -> Optional[tuple[float, str]]:
            if prev is None:  # Only log on first occurrence
                return (
                    current.timestamp,
                    f"Autopilot version: {current.version}",
                )
            return None

        def state_filter_health_warning(
            current: StateFilterStatus, prev: Optional[StateFilterStatus]
        ) -> Optional[tuple[float, str]]:
            if current.health < 50:  # Assuming 0-100 health scale
                return (
                    current.timestamp,
                    f"State filter health warning: {current.health}%",
                )
            return None

        def navigation_status_update(
            current: RotorcraftNavStatus, prev: Optional[RotorcraftNavStatus]
        ) -> Optional[tuple[float, str]]:
            if prev is not None and current.dist_to_wp < 5.0 and prev.dist_to_wp >= 5.0:
                return (
                    current.timestamp,
                    f"Approaching waypoint: {current.dist_to_wp:.1f}m remaining",
                )
            return None

        # Register the new event detection rules
        if "DATALINK_REPORT" in self.log_data:
            self.add_rule("DATALINK_REPORT", datalink_quality_warning)
        if "UART_ERRORS" in self.log_data:
            self.add_rule("UART_ERRORS", uart_communication_errors)
        if "ACTUATORS" in self.log_data:
            self.add_rule("ACTUATORS", actuator_saturation_warning)
        if "BEBOP_ACTUATORS" in self.log_data:
            self.add_rule("BEBOP_ACTUATORS", bebop_actuator_imbalance)
        if "AIR_DATA" in self.log_data:
            self.add_rule("AIR_DATA", air_data_anomaly)
        if "WP_MOVED" in self.log_data:
            self.add_rule("WP_MOVED", waypoint_reached)
        if "AUTOPILOT_VERSION" in self.log_data:
            self.add_rule("AUTOPILOT_VERSION", autopilot_version_info)
        if "STATE_FILTER_STATUS" in self.log_data:
            self.add_rule("STATE_FILTER_STATUS", state_filter_health_warning)
        if "ROTORCRAFT_NAV_STATUS" in self.log_data:
            self.add_rule("ROTORCRAFT_NAV_STATUS", navigation_status_update)

        def low_battery(reading: Energy, _) -> Optional[tuple[float, str]]:
            if 10.0 < reading.voltage < 14.0:  # Check for a plausible voltage range.
                return (
                    reading.timestamp,
                    f"Low Battery Warning: Voltage is {reading.voltage:.2f}V",
                )
            return None

        self.add_rule("ENERGY", low_battery)

        def high_g_force(imu: IMU, _) -> Optional[tuple[float, str]]:
            # Calculate the magnitude of the acceleration vector
            g_force = np.sqrt(imu.ax**2 + imu.ay**2 + imu.az**2) / 9.81
            if g_force > 4.0:  # Trigger if G-force exceeds 4 G's
                return (imu.timestamp, f"High G-Force Detected: {g_force:.1f} G")
            return None

        self.add_rule("IMU_ACCEL_SCALED", high_g_force)

        # --- ERROR Level Events ---
        def i2c_error_detected(
            current: I2CErrors, prev: Optional[I2CErrors]
        ) -> Optional[tuple[float, str]]:
            if prev is not None and current.ack_failures > prev.ack_failures:
                return (
                    current.timestamp,
                    f"I2C Acknowledge Failure on bus {current.bus_number}",
                )
            return None

        self.add_rule("I2C_ERRORS", i2c_error_detected)

        # === COMPREHENSIVE EVENT DETECTION FOR ALL MESSAGE TYPES ===

        # Additional IMU and attitude events
        def imu_gyro_anomaly(imu: IMU, _) -> Optional[tuple[float, str]]:
            """Detect unusually high gyro rates"""
            if hasattr(imu, "p") and hasattr(imu, "q") and hasattr(imu, "r"):
                max_rate = max(abs(imu.p), abs(imu.q), abs(imu.r))
                if max_rate > 5.0:  # More than 5 rad/s
                    return (imu.timestamp, f"High angular rate: {max_rate:.1f} rad/s")
            return None

        def attitude_extreme_angles(attitude: any, _) -> Optional[tuple[float, str]]:
            """Detect extreme attitude angles"""
            if hasattr(attitude, "phi") and hasattr(attitude, "theta"):
                roll_deg = np.degrees(attitude.phi) if attitude.phi else 0
                pitch_deg = np.degrees(attitude.theta) if attitude.theta else 0
                if abs(roll_deg) > 45 or abs(pitch_deg) > 45:
                    return (
                        attitude.timestamp,
                        f"Extreme attitude: roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°",
                    )
            return None

        # Power and energy monitoring
        def power_consumption_spike(
            energy: Energy, prev_energy
        ) -> Optional[tuple[float, str]]:
            """Detect power consumption spikes"""
            current_power = energy.power
            if current_power > 150:  # More than 150W
                return (
                    energy.timestamp,
                    f"High power consumption: {current_power:.1f}W",
                )
            return None

        def voltage_drop_warning(energy: Energy, _) -> Optional[tuple[float, str]]:
            """Detect voltage drops"""
            if energy.voltage < 11.5:  # Low voltage warning
                return (energy.timestamp, f"Low voltage warning: {energy.voltage:.1f}V")
            elif energy.voltage < 10.5:  # Critical voltage
                return (
                    energy.timestamp,
                    f"Critical low voltage: {energy.voltage:.1f}V",
                )
            return None

        # Motor and ESC monitoring
        def motor_speed_anomaly(esc: any, _) -> Optional[tuple[float, str]]:
            """Detect motor speed anomalies"""
            if hasattr(esc, "rpm") and esc.rpm:
                if esc.rpm > 12000:  # High RPM
                    return (esc.timestamp, f"High motor speed: {esc.rpm} RPM")
                elif esc.rpm < 200 and esc.rpm > 0:  # Unusually low but not zero
                    return (esc.timestamp, f"Low motor speed warning: {esc.rpm} RPM")
            return None

        def esc_temperature_warning(esc: any, _) -> Optional[tuple[float, str]]:
            """Detect ESC overheating"""
            if hasattr(esc, "temperature") and esc.temperature:
                if esc.temperature > 85:  # High temperature
                    return (
                        esc.timestamp,
                        f"ESC overheating warning: {esc.temperature}°C",
                    )
                elif esc.temperature > 95:  # Critical temperature
                    return (
                        esc.timestamp,
                        f"ESC critical temperature: {esc.temperature}°C",
                    )
            return None

        # System status monitoring
        def ground_detect_event(
            ground: any, prev_ground
        ) -> Optional[tuple[float, str]]:
            """Detect ground proximity changes"""
            if (
                hasattr(ground, "ground_proximity")
                and prev_ground
                and hasattr(prev_ground, "ground_proximity")
            ):
                if ground.ground_proximity != prev_ground.ground_proximity:
                    status = "detected" if ground.ground_proximity else "lost"
                    return (ground.timestamp, f"Ground proximity {status}")
            return None

        def rotorcraft_status_change(
            status: RotorcraftStatus, prev_status
        ) -> Optional[tuple[float, str]]:
            """Detect flight mode and status changes"""
            if prev_status:
                if status.flight_mode != prev_status.flight_mode:
                    return (
                        status.timestamp,
                        f"Flight mode changed from {prev_status.flight_mode} to {status.flight_mode}",
                    )
                if status.vehicle_mode != prev_status.vehicle_mode:
                    return (
                        status.timestamp,
                        f"Vehicle mode changed to {status.vehicle_mode}",
                    )
                if status.failsafe_mode != prev_status.failsafe_mode:
                    return (
                        status.timestamp,
                        f"Failsafe mode changed to {status.failsafe_mode}",
                    )
            return None

        # Navigation and control events
        def ahrs_quat_update(ahrs: any, _) -> Optional[tuple[float, str]]:
            """Log AHRS quaternion updates for system monitoring"""
            if hasattr(ahrs, "body_qi"):
                return (ahrs.timestamp, f"AHRS quaternion update")
            return None

        def ekf_position_update(ekf: any, _) -> Optional[tuple[float, str]]:
            """Log EKF position updates"""
            if hasattr(ekf, "px") and hasattr(ekf, "py") and hasattr(ekf, "pz"):
                return (
                    ekf.timestamp,
                    f"EKF position: x={ekf.px:.1f}, y={ekf.py:.1f}, z={ekf.pz:.1f}",
                )
            return None

        def control_cmd_update(cmd: any, _) -> Optional[tuple[float, str]]:
            """Log control command updates"""
            if hasattr(cmd, "thrust") and hasattr(cmd, "roll"):
                return (
                    cmd.timestamp,
                    f"Control command: thrust={cmd.thrust}, roll={cmd.roll}",
                )
            return None

        def rotwing_state_monitoring(
            rotwing: any, prev_rotwing
        ) -> Optional[tuple[float, str]]:
            """Monitor rotwing aircraft state changes"""
            if (
                hasattr(rotwing, "tilt_angle_ref")
                and prev_rotwing
                and hasattr(prev_rotwing, "tilt_angle_ref")
            ):
                angle_change = abs(rotwing.tilt_angle_ref - prev_rotwing.tilt_angle_ref)
                if angle_change > 10.0:  # Significant tilt change
                    return (
                        rotwing.timestamp,
                        f"Rotwing tilt change: {rotwing.tilt_angle_ref:.1f}°",
                    )
            return None

        def airspeed_monitoring(airspeed: any, _) -> Optional[tuple[float, str]]:
            """Monitor airspeed events"""
            if hasattr(airspeed, "airspeed"):
                if airspeed.airspeed > 20.0:  # High airspeed
                    return (
                        airspeed.timestamp,
                        f"High airspeed: {airspeed.airspeed:.1f} m/s",
                    )
                elif (
                    airspeed.airspeed < 1.0 and airspeed.airspeed > 0
                ):  # Very low airspeed
                    return (
                        airspeed.timestamp,
                        f"Low airspeed: {airspeed.airspeed:.1f} m/s",
                    )
            return None

        # Register all additional event detection rules
        if "IMU_GYRO" in self.log_data:
            self.add_rule("IMU_GYRO", imu_gyro_anomaly)
        if "IMU_GYRO_SCALED" in self.log_data:
            self.add_rule("IMU_GYRO_SCALED", imu_gyro_anomaly)

        if "ATTITUDE" in self.log_data:
            self.add_rule("ATTITUDE", attitude_extreme_angles)

        if "ENERGY" in self.log_data:
            self.add_rule("ENERGY", power_consumption_spike)
            self.add_rule("ENERGY", voltage_drop_warning)

        if "ESC" in self.log_data:
            self.add_rule("ESC", motor_speed_anomaly)
            self.add_rule("ESC", esc_temperature_warning)

        if "GROUND_DETECT" in self.log_data:
            self.add_rule("GROUND_DETECT", ground_detect_event)

        if "ROTORCRAFT_STATUS" in self.log_data:
            self.add_rule("ROTORCRAFT_STATUS", rotorcraft_status_change)

        if "AHRS_REF_QUAT" in self.log_data:
            self.add_rule("AHRS_REF_QUAT", ahrs_quat_update)

        if "EKF2_STATE" in self.log_data:
            self.add_rule("EKF2_STATE", ekf_position_update)

        if "ROTORCRAFT_CMD" in self.log_data:
            self.add_rule("ROTORCRAFT_CMD", control_cmd_update)

        if "ROTWING_STATE" in self.log_data:
            self.add_rule("ROTWING_STATE", rotwing_state_monitoring)

        if "AIRSPEED" in self.log_data:
            self.add_rule("AIRSPEED", airspeed_monitoring)

        # === NEW MESSAGE TYPE EVENT DETECTION ===

        # 1. GUIDANCE_INDI_HYBRID - Guidance system events
        def guidance_indi_hybrid_monitoring(
            guidance: GuidanceIndiHybrid, prev_guidance
        ) -> Optional[tuple[float, str]]:
            """Monitor INDI hybrid guidance system events"""
            if (
                hasattr(guidance, "pos_err_x")
                and hasattr(guidance, "pos_err_y")
                and hasattr(guidance, "pos_err_z")
            ):
                pos_error = np.sqrt(
                    guidance.pos_err_x**2
                    + guidance.pos_err_y**2
                    + guidance.pos_err_z**2
                )
                if pos_error > 10.0:  # Large position error
                    return (
                        guidance.timestamp,
                        f"Large guidance position error: {pos_error:.1f}m",
                    )
                elif pos_error > 5.0:  # Moderate position error
                    return (
                        guidance.timestamp,
                        f"Guidance position error: {pos_error:.1f}m",
                    )
            return None

        # 2. GUIDANCE - General guidance events
        def guidance_monitoring(
            guidance: Guidance, prev_guidance
        ) -> Optional[tuple[float, str]]:
            """Monitor general guidance system events"""
            if (
                hasattr(guidance, "indi_cmd_x")
                and hasattr(guidance, "indi_cmd_y")
                and hasattr(guidance, "indi_cmd_z")
            ):
                cmd_magnitude = np.sqrt(
                    guidance.indi_cmd_x**2
                    + guidance.indi_cmd_y**2
                    + guidance.indi_cmd_z**2
                )
                if cmd_magnitude > 15.0:  # High guidance command
                    return (
                        guidance.timestamp,
                        f"High guidance command: {cmd_magnitude:.1f}",
                    )
            return None

        # 3. EXTERNAL_POSE_DOWN - External pose estimation
        def external_pose_monitoring(
            pose: ExternalPoseDown, prev_pose
        ) -> Optional[tuple[float, str]]:
            """Monitor external pose estimation events"""
            if (
                prev_pose
                and hasattr(pose, "x")
                and hasattr(pose, "y")
                and hasattr(pose, "z")
            ):
                pos_change = np.sqrt(
                    (pose.x - prev_pose.x) ** 2
                    + (pose.y - prev_pose.y) ** 2
                    + (pose.z - prev_pose.z) ** 2
                )
                if pos_change > 5.0:  # Large position jump
                    return (
                        pose.timestamp,
                        f"Large external pose change: {pos_change:.1f}m",
                    )
            elif hasattr(pose, "x"):  # First pose update
                return (
                    pose.timestamp,
                    f"External pose initialized: x={pose.x:.1f}, y={pose.y:.1f}, z={pose.z:.1f}",
                )
            return None

        # 4. SERIAL_ACT_T4_IN - Serial actuator input
        def serial_act_t4_in_monitoring(
            act_in: SerialActT4In, prev_act_in
        ) -> Optional[tuple[float, str]]:
            """Monitor serial actuator T4 input events"""
            if hasattr(act_in, "values") and act_in.values:
                max_val = max(act_in.values) if act_in.values else 0
                min_val = min(act_in.values) if act_in.values else 0
                if max_val > 9000:  # High actuator input
                    return (act_in.timestamp, f"High actuator input: {max_val}")
                elif min_val < 1000 and max_val > 1000:  # Mixed high/low values
                    return (
                        act_in.timestamp,
                        f"Actuator input range: {min_val}-{max_val}",
                    )
            return None

        # 5. SERIAL_ACT_T4_OUT - Serial actuator output
        def serial_act_t4_out_monitoring(
            act_out: SerialActT4Out, prev_act_out
        ) -> Optional[tuple[float, str]]:
            """Monitor serial actuator T4 output events"""
            if hasattr(act_out, "values") and act_out.values:
                max_val = max(act_out.values) if act_out.values else 0
                if max_val > 9500:  # Very high actuator output
                    return (act_out.timestamp, f"Maximum actuator output: {max_val}")
                elif max_val > 9000:  # High actuator output
                    return (act_out.timestamp, f"High actuator output: {max_val}")
            return None

        # 6. POWER_DEVICE - Power device monitoring
        def power_device_monitoring(
            power_dev: PowerDevice, prev_power_dev
        ) -> Optional[tuple[float, str]]:
            """Monitor power device events"""
            if (
                hasattr(power_dev, "power") and power_dev.power > 200
            ):  # High power consumption
                return (
                    power_dev.timestamp,
                    f"High power device consumption: {power_dev.power:.1f}W",
                )
            elif (
                hasattr(power_dev, "voltage") and power_dev.voltage < 10.5
            ):  # Low voltage
                return (
                    power_dev.timestamp,
                    f"Low power device voltage: {power_dev.voltage:.1f}V",
                )
            elif (
                hasattr(power_dev, "current") and power_dev.current > 20.0
            ):  # High current
                return (
                    power_dev.timestamp,
                    f"High power device current: {power_dev.current:.1f}A",
                )
            return None

        # 7. EFF_MAT (Effectiveness Matrix) - Control effectiveness
        def eff_mat_monitoring(
            eff_mat: EffMat, prev_eff_mat
        ) -> Optional[tuple[float, str]]:
            """Monitor control effectiveness matrix events"""
            if hasattr(eff_mat, "motor_cmd") and eff_mat.motor_cmd:
                max_cmd = max(eff_mat.motor_cmd) if eff_mat.motor_cmd else 0
                min_cmd = min(eff_mat.motor_cmd) if eff_mat.motor_cmd else 0
                cmd_range = max_cmd - min_cmd
                if cmd_range > 5000:  # Large command range indicates unbalanced control
                    return (
                        eff_mat.timestamp,
                        f"Large motor command range: {cmd_range}",
                    )
                elif max_cmd > 9000:  # High motor command
                    return (eff_mat.timestamp, f"High motor command: {max_cmd}")
            return None

        # 8. EKF2_P_DIAG - EKF covariance diagnostics
        def ekf2_p_diag_monitoring(
            ekf_diag: EKF2PDiag, prev_ekf_diag
        ) -> Optional[tuple[float, str]]:
            """Monitor EKF covariance diagonal events"""
            if (
                hasattr(ekf_diag, "pos_var_x")
                and hasattr(ekf_diag, "pos_var_y")
                and hasattr(ekf_diag, "pos_var_z")
            ):
                max_pos_var = max(
                    ekf_diag.pos_var_x, ekf_diag.pos_var_y, ekf_diag.pos_var_z
                )
                if max_pos_var > 100.0:  # High position variance
                    return (
                        ekf_diag.timestamp,
                        f"High EKF position uncertainty: {max_pos_var:.1f}",
                    )
                elif max_pos_var > 25.0:  # Moderate position variance
                    return (
                        ekf_diag.timestamp,
                        f"EKF position uncertainty: {max_pos_var:.1f}",
                    )
            return None

        # 9. EKF2_INNOV - EKF innovation monitoring
        def ekf2_innov_monitoring(
            ekf_innov: EKF2Innov, prev_ekf_innov
        ) -> Optional[tuple[float, str]]:
            """Monitor EKF innovation events"""
            if (
                hasattr(ekf_innov, "innov_x")
                and hasattr(ekf_innov, "innov_y")
                and hasattr(ekf_innov, "innov_z")
            ):
                max_innov = max(
                    abs(ekf_innov.innov_x),
                    abs(ekf_innov.innov_y),
                    abs(ekf_innov.innov_z),
                )
                if max_innov > 5.0:  # Large innovation
                    return (
                        ekf_innov.timestamp,
                        f"Large EKF innovation: {max_innov:.1f}",
                    )
                elif max_innov > 2.0:  # Moderate innovation
                    return (ekf_innov.timestamp, f"EKF innovation: {max_innov:.1f}")
            return None

        # 10. GPS (basic GPS, not GPS_INT) - Basic GPS events
        def gps_basic_monitoring(gps: GPS, prev_gps) -> Optional[tuple[float, str]]:
            """Monitor basic GPS events"""
            if hasattr(gps, "mode") and prev_gps and hasattr(prev_gps, "mode"):
                if gps.mode != prev_gps.mode:
                    return (
                        gps.timestamp,
                        f"GPS mode changed: {prev_gps.mode} -> {gps.mode}",
                    )
            elif hasattr(gps, "mode") and gps.mode >= 3:  # GPS fix
                return (gps.timestamp, f"GPS fix acquired (mode {gps.mode})")

            if hasattr(gps, "nb_sats") and gps.nb_sats < 6:  # Low satellite count
                return (gps.timestamp, f"Low GPS satellite count: {gps.nb_sats}")
            elif hasattr(gps, "hdop") and gps.hdop > 200:  # High HDOP (poor accuracy)
                return (gps.timestamp, f"Poor GPS accuracy (HDOP: {gps.hdop})")

            return None

        # Register all the new event detection rules
        if "GUIDANCE_INDI_HYBRID" in self.log_data:
            self.add_rule("GUIDANCE_INDI_HYBRID", guidance_indi_hybrid_monitoring)

        if "GUIDANCE" in self.log_data:
            self.add_rule("GUIDANCE", guidance_monitoring)

        if "EXTERNAL_POSE_DOWN" in self.log_data:
            self.add_rule("EXTERNAL_POSE_DOWN", external_pose_monitoring)

        if "SERIAL_ACT_T4_IN" in self.log_data:
            self.add_rule("SERIAL_ACT_T4_IN", serial_act_t4_in_monitoring)

        if "SERIAL_ACT_T4_OUT" in self.log_data:
            self.add_rule("SERIAL_ACT_T4_OUT", serial_act_t4_out_monitoring)

        if "POWER_DEVICE" in self.log_data:
            self.add_rule("POWER_DEVICE", power_device_monitoring)

        if "EFF_MAT" in self.log_data:
            self.add_rule("EFF_MAT", eff_mat_monitoring)

        if "EKF2_P_DIAG" in self.log_data:
            self.add_rule("EKF2_P_DIAG", ekf2_p_diag_monitoring)

        if "EKF2_INNOV" in self.log_data:
            self.add_rule("EKF2_INNOV", ekf2_innov_monitoring)

        if "GPS" in self.log_data:
            self.add_rule("GPS", gps_basic_monitoring)

        # === ADDITIONAL FLIGHT MONITORING MESSAGE TYPES ===

        # 1. STABILIZATION_ATTITUDE - Attitude stabilization control
        def stabilization_attitude_monitoring(
            stab_att: StabilizationAttitude, prev_stab_att
        ) -> Optional[tuple[float, str]]:
            """Monitor attitude stabilization control events"""
            if (
                hasattr(stab_att, "phi_pgain")
                and hasattr(stab_att, "theta_pgain")
                and hasattr(stab_att, "psi_pgain")
            ):
                # Check for extreme gain changes
                if prev_stab_att:
                    phi_change = abs(stab_att.phi_pgain - prev_stab_att.phi_pgain)
                    theta_change = abs(stab_att.theta_pgain - prev_stab_att.theta_pgain)
                    psi_change = abs(stab_att.psi_pgain - prev_stab_att.psi_pgain)

                    if (
                        max(phi_change, theta_change, psi_change) > 100
                    ):  # Large gain change
                        return (
                            stab_att.timestamp,
                            f"Large attitude gain change detected",
                        )

                # Check for extreme gain values
                max_gain = max(
                    abs(stab_att.phi_pgain),
                    abs(stab_att.theta_pgain),
                    abs(stab_att.psi_pgain),
                )
                if max_gain > 1000:  # Very high gains
                    return (
                        stab_att.timestamp,
                        f"High attitude control gains: {max_gain:.1f}",
                    )
            return None

        # 2. NAV_STATUS - Navigation system status
        def nav_status_monitoring(
            nav_stat: NavStatus, prev_nav_stat
        ) -> Optional[tuple[float, str]]:
            """Monitor navigation system status events"""
            if (
                prev_nav_stat
                and hasattr(nav_stat, "cur_block")
                and hasattr(prev_nav_stat, "cur_block")
            ):
                if nav_stat.cur_block != prev_nav_stat.cur_block:
                    return (
                        nav_stat.timestamp,
                        f"Navigation block changed: {prev_nav_stat.cur_block} -> {nav_stat.cur_block}",
                    )

            if (
                prev_nav_stat
                and hasattr(nav_stat, "nav_mode")
                and hasattr(prev_nav_stat, "nav_mode")
            ):
                if nav_stat.nav_mode != prev_nav_stat.nav_mode:
                    return (
                        nav_stat.timestamp,
                        f"Navigation mode changed: {prev_nav_stat.nav_mode} -> {nav_stat.nav_mode}",
                    )

            return None

        # 3. WAYPOINT - Waypoint management
        def waypoint_monitoring(
            waypoint: Waypoint, prev_waypoint
        ) -> Optional[tuple[float, str]]:
            """Monitor waypoint management events"""
            if hasattr(waypoint, "wp_id"):
                if (
                    prev_waypoint
                    and hasattr(prev_waypoint, "wp_id")
                    and waypoint.wp_id != prev_waypoint.wp_id
                ):
                    return (
                        waypoint.timestamp,
                        f"Waypoint changed: WP{prev_waypoint.wp_id} -> WP{waypoint.wp_id}",
                    )

                # Check for waypoint position updates
                if (
                    hasattr(waypoint, "wp_x")
                    and hasattr(waypoint, "wp_y")
                    and hasattr(waypoint, "wp_z")
                ):
                    if prev_waypoint and hasattr(prev_waypoint, "wp_x"):
                        distance_moved = np.sqrt(
                            (waypoint.wp_x - prev_waypoint.wp_x) ** 2
                            + (waypoint.wp_y - prev_waypoint.wp_y) ** 2
                            + (waypoint.wp_z - prev_waypoint.wp_z) ** 2
                        )
                        if distance_moved > 10.0:  # Significant waypoint movement
                            return (
                                waypoint.timestamp,
                                f"Waypoint WP{waypoint.wp_id} moved {distance_moved:.1f}m",
                            )

            return None

        # 4. FLIGHT_PLAN - Flight plan execution
        def flight_plan_monitoring(
            fp: FlightPlan, prev_fp
        ) -> Optional[tuple[float, str]]:
            """Monitor flight plan execution events"""
            if prev_fp:
                if (
                    hasattr(fp, "fp_block")
                    and hasattr(prev_fp, "fp_block")
                    and fp.fp_block != prev_fp.fp_block
                ):
                    return (
                        fp.timestamp,
                        f"Flight plan block changed: {prev_fp.fp_block} -> {fp.fp_block}",
                    )

                if (
                    hasattr(fp, "fp_mode")
                    and hasattr(prev_fp, "fp_mode")
                    and fp.fp_mode != prev_fp.fp_mode
                ):
                    return (
                        fp.timestamp,
                        f"Flight plan mode changed: {prev_fp.fp_mode} -> {fp.fp_mode}",
                    )

                if (
                    hasattr(fp, "fp_kill")
                    and hasattr(prev_fp, "fp_kill")
                    and fp.fp_kill != prev_fp.fp_kill
                ):
                    kill_status = "ENABLED" if fp.fp_kill else "DISABLED"
                    return (fp.timestamp, f"Flight plan kill switch {kill_status}")

            return None

        # 5. RC_LOST - RC signal loss detection
        def rc_lost_monitoring(
            rc_lost: RCLost, prev_rc_lost
        ) -> Optional[tuple[float, str]]:
            """Monitor RC signal loss events"""
            if (
                prev_rc_lost
                and hasattr(rc_lost, "rc_lost")
                and hasattr(prev_rc_lost, "rc_lost")
            ):
                if rc_lost.rc_lost != prev_rc_lost.rc_lost:
                    if rc_lost.rc_lost:
                        timeout = (
                            rc_lost.rc_timeout if hasattr(rc_lost, "rc_timeout") else 0
                        )
                        return (
                            rc_lost.timestamp,
                            f"RC SIGNAL LOST (timeout: {timeout:.1f}s)",
                        )
                    else:
                        return (rc_lost.timestamp, f"RC signal recovered")

            elif hasattr(rc_lost, "rc_lost") and rc_lost.rc_lost:
                return (rc_lost.timestamp, f"RC signal loss detected")

            return None

        # 6. DATALINK_LOST - Datalink communication loss
        def datalink_lost_monitoring(
            dl_lost: DatalinkLost, prev_dl_lost
        ) -> Optional[tuple[float, str]]:
            """Monitor datalink communication loss events"""
            if (
                prev_dl_lost
                and hasattr(dl_lost, "dl_lost")
                and hasattr(prev_dl_lost, "dl_lost")
            ):
                if dl_lost.dl_lost != prev_dl_lost.dl_lost:
                    if dl_lost.dl_lost:
                        timeout = (
                            dl_lost.dl_timeout if hasattr(dl_lost, "dl_timeout") else 0
                        )
                        return (
                            dl_lost.timestamp,
                            f"DATALINK LOST (timeout: {timeout:.1f}s)",
                        )
                    else:
                        return (dl_lost.timestamp, f"Datalink recovered")

            elif hasattr(dl_lost, "dl_lost") and dl_lost.dl_lost:
                return (dl_lost.timestamp, f"Datalink loss detected")

            return None

        # 7. GEOFENCE - Geofence violation monitoring
        def geofence_monitoring(
            geofence: Geofence, prev_geofence
        ) -> Optional[tuple[float, str]]:
            """Monitor geofence violation events"""
            if (
                prev_geofence
                and hasattr(geofence, "fence_violation")
                and hasattr(prev_geofence, "fence_violation")
            ):
                if geofence.fence_violation != prev_geofence.fence_violation:
                    if geofence.fence_violation:
                        fence_type = (
                            geofence.fence_type
                            if hasattr(geofence, "fence_type")
                            else "unknown"
                        )
                        distance = (
                            geofence.distance_to_fence
                            if hasattr(geofence, "distance_to_fence")
                            else 0
                        )
                        return (
                            geofence.timestamp,
                            f"GEOFENCE VIOLATION ({fence_type}) - distance: {distance:.1f}m",
                        )
                    else:
                        return (geofence.timestamp, f"Geofence violation cleared")

            if (
                hasattr(geofence, "distance_to_fence")
                and geofence.distance_to_fence < 10.0
            ):
                fence_type = (
                    geofence.fence_type
                    if hasattr(geofence, "fence_type")
                    else "unknown"
                )
                return (
                    geofence.timestamp,
                    f"Approaching geofence ({fence_type}): {geofence.distance_to_fence:.1f}m",
                )

            return None

        # 8. WEATHER - Weather monitoring
        def weather_monitoring(
            weather: Weather, prev_weather
        ) -> Optional[tuple[float, str]]:
            """Monitor weather condition events"""
            if (
                hasattr(weather, "wind_speed") and weather.wind_speed > 15.0
            ):  # Strong wind
                wind_dir = weather.wind_dir if hasattr(weather, "wind_dir") else 0
                return (
                    weather.timestamp,
                    f"Strong wind detected: {weather.wind_speed:.1f} m/s @ {wind_dir:.0f}°",
                )

            if hasattr(weather, "temperature") and (
                weather.temperature < -10.0 or weather.temperature > 50.0
            ):
                return (
                    weather.timestamp,
                    f"Extreme temperature: {weather.temperature:.1f}°C",
                )

            if (
                hasattr(weather, "pressure") and weather.pressure < 900.0
            ):  # Very low pressure
                return (
                    weather.timestamp,
                    f"Low atmospheric pressure: {weather.pressure:.1f} hPa",
                )

            return None

        # 9. WIND_ESTIMATION - Wind estimation monitoring
        def wind_estimation_monitoring(
            wind_est: WindEstimation, prev_wind_est
        ) -> Optional[tuple[float, str]]:
            """Monitor wind estimation events"""
            if hasattr(wind_est, "wind_north") and hasattr(wind_est, "wind_east"):
                wind_speed = np.sqrt(wind_est.wind_north**2 + wind_est.wind_east**2)
                if wind_speed > 20.0:  # Very strong estimated wind
                    wind_dir = np.degrees(
                        np.arctan2(wind_est.wind_east, wind_est.wind_north)
                    )
                    confidence = (
                        wind_est.wind_confidence
                        if hasattr(wind_est, "wind_confidence")
                        else 0
                    )
                    return (
                        wind_est.timestamp,
                        f"Strong wind estimated: {wind_speed:.1f} m/s @ {wind_dir:.0f}° (conf: {confidence:.2f})",
                    )

            if hasattr(wind_est, "wind_confidence") and wind_est.wind_confidence < 0.3:
                return (
                    wind_est.timestamp,
                    f"Low wind estimation confidence: {wind_est.wind_confidence:.2f}",
                )

            return None

        # 10. BATTERY_STATUS - Detailed battery monitoring
        def battery_status_monitoring(
            battery: BatteryStatus, prev_battery
        ) -> Optional[tuple[float, str]]:
            """Monitor detailed battery status events"""
            if hasattr(battery, "cell1_voltage") and hasattr(battery, "cell2_voltage"):
                # Check individual cell voltages
                cell_voltages = []
                for i, attr in enumerate(
                    [
                        "cell1_voltage",
                        "cell2_voltage",
                        "cell3_voltage",
                        "cell4_voltage",
                    ],
                    1,
                ):
                    if hasattr(battery, attr):
                        voltage = getattr(battery, attr)
                        if voltage > 0:  # Only check active cells
                            cell_voltages.append((i, voltage))
                            if voltage < 3.0:  # Critical cell voltage
                                return (
                                    battery.timestamp,
                                    f"CRITICAL: Cell {i} voltage: {voltage:.2f}V",
                                )
                            elif voltage < 3.3:  # Low cell voltage
                                return (
                                    battery.timestamp,
                                    f"Low cell {i} voltage: {voltage:.2f}V",
                                )

                # Check cell imbalance
                if len(cell_voltages) > 1:
                    voltages = [v[1] for v in cell_voltages]
                    voltage_spread = max(voltages) - min(voltages)
                    if voltage_spread > 0.2:  # Significant imbalance
                        return (
                            battery.timestamp,
                            f"Battery cell imbalance: {voltage_spread:.3f}V spread",
                        )

            if hasattr(battery, "temperature") and battery.temperature > 60.0:
                return (
                    battery.timestamp,
                    f"High battery temperature: {battery.temperature:.1f}°C",
                )

            if hasattr(battery, "remaining") and battery.remaining < 20.0:
                return (
                    battery.timestamp,
                    f"Low battery remaining: {battery.remaining:.1f}%",
                )

            return None

        # 11. MOTOR_STATUS - Individual motor health monitoring
        def motor_status_monitoring(
            motor: MotorStatus, prev_motor
        ) -> Optional[tuple[float, str]]:
            """Monitor individual motor health events"""
            motor_id = motor.motor_id if hasattr(motor, "motor_id") else 0

            if hasattr(motor, "temperature") and motor.temperature > 90.0:
                return (
                    motor.timestamp,
                    f"Motor {motor_id} overheating: {motor.temperature:.1f}°C",
                )

            if hasattr(motor, "rpm") and motor.rpm > 15000:
                return (motor.timestamp, f"Motor {motor_id} high RPM: {motor.rpm}")

            if hasattr(motor, "vibration") and motor.vibration > 5.0:
                return (
                    motor.timestamp,
                    f"Motor {motor_id} high vibration: {motor.vibration:.1f}",
                )

            if hasattr(motor, "error_flags") and motor.error_flags > 0:
                return (
                    motor.timestamp,
                    f"Motor {motor_id} error flags: 0x{motor.error_flags:04X}",
                )

            return None

        # 12. VIBRATION - System vibration monitoring
        def vibration_monitoring(
            vibration: Vibration, prev_vibration
        ) -> Optional[tuple[float, str]]:
            """Monitor system vibration events"""
            if (
                hasattr(vibration, "accel_x_rms")
                and hasattr(vibration, "accel_y_rms")
                and hasattr(vibration, "accel_z_rms")
            ):
                max_accel_rms = max(
                    vibration.accel_x_rms, vibration.accel_y_rms, vibration.accel_z_rms
                )
                if max_accel_rms > 30.0:  # High vibration threshold
                    return (
                        vibration.timestamp,
                        f"High acceleration vibration: {max_accel_rms:.1f} m/s²",
                    )

            if (
                hasattr(vibration, "gyro_x_rms")
                and hasattr(vibration, "gyro_y_rms")
                and hasattr(vibration, "gyro_z_rms")
            ):
                max_gyro_rms = max(
                    vibration.gyro_x_rms, vibration.gyro_y_rms, vibration.gyro_z_rms
                )
                if max_gyro_rms > 50.0:  # High gyro vibration threshold
                    return (
                        vibration.timestamp,
                        f"High gyroscope vibration: {max_gyro_rms:.1f} rad/s",
                    )

            if hasattr(vibration, "clip_count") and vibration.clip_count > 0:
                return (
                    vibration.timestamp,
                    f"IMU clipping detected: {vibration.clip_count} samples",
                )

            return None

        # 13. COMPASS_CAL - Compass calibration monitoring
        def compass_cal_monitoring(
            compass_cal: CompassCal, prev_compass_cal
        ) -> Optional[tuple[float, str]]:
            """Monitor compass calibration events"""
            if (
                prev_compass_cal
                and hasattr(compass_cal, "cal_progress")
                and hasattr(prev_compass_cal, "cal_progress")
            ):
                if (
                    compass_cal.cal_progress > prev_compass_cal.cal_progress + 10
                ):  # Significant progress
                    return (
                        compass_cal.timestamp,
                        f"Compass calibration progress: {compass_cal.cal_progress:.1f}%",
                    )

            if (
                hasattr(compass_cal, "cal_progress")
                and compass_cal.cal_progress >= 100.0
            ):
                fitness = (
                    compass_cal.cal_fitness
                    if hasattr(compass_cal, "cal_fitness")
                    else 0
                )
                return (
                    compass_cal.timestamp,
                    f"Compass calibration complete (fitness: {fitness:.2f})",
                )

            if hasattr(compass_cal, "cal_fitness") and compass_cal.cal_fitness < 0.5:
                return (
                    compass_cal.timestamp,
                    f"Poor compass calibration fitness: {compass_cal.cal_fitness:.2f}",
                )

            return None

        # 14. BAROMETER - Barometric pressure monitoring
        def barometer_monitoring(
            baro: Barometer, prev_baro
        ) -> Optional[tuple[float, str]]:
            """Monitor barometric pressure events"""
            if (
                prev_baro
                and hasattr(baro, "altitude")
                and hasattr(prev_baro, "altitude")
            ):
                altitude_change = abs(baro.altitude - prev_baro.altitude)
                if altitude_change > 50.0:  # Large altitude change
                    direction = (
                        "climbed" if baro.altitude > prev_baro.altitude else "descended"
                    )
                    return (
                        baro.timestamp,
                        f"Aircraft {direction} {altitude_change:.1f}m (baro altitude now {baro.altitude:.1f}m)",
                    )

            if hasattr(baro, "pressure") and baro.pressure < 800.0:  # Very low pressure
                return (
                    baro.timestamp,
                    f"Very low barometric pressure: {baro.pressure:.1f} hPa",
                )

            return None

        # 15. TEMPERATURE - Temperature monitoring system
        def temperature_monitoring(
            temp: Temperature, prev_temp
        ) -> Optional[tuple[float, str]]:
            """Monitor system temperature events"""
            if hasattr(temp, "cpu_temp") and temp.cpu_temp > 85.0:
                return (temp.timestamp, f"High CPU temperature: {temp.cpu_temp:.1f}°C")

            if hasattr(temp, "imu_temp"):
                if temp.imu_temp > 75.0:
                    return (
                        temp.timestamp,
                        f"High IMU temperature: {temp.imu_temp:.1f}°C",
                    )
                elif temp.imu_temp < -20.0:
                    return (
                        temp.timestamp,
                        f"Low IMU temperature: {temp.imu_temp:.1f}°C",
                    )

            if hasattr(temp, "motor_temp") and temp.motor_temp > 100.0:
                return (
                    temp.timestamp,
                    f"High motor temperature: {temp.motor_temp:.1f}°C",
                )

            return None

        # Register all the additional flight monitoring rules
        if "STABILIZATION_ATTITUDE" in self.log_data:
            self.add_rule("STABILIZATION_ATTITUDE", stabilization_attitude_monitoring)

        if "NAV_STATUS" in self.log_data:
            self.add_rule("NAV_STATUS", nav_status_monitoring)

        if "WAYPOINT" in self.log_data:
            self.add_rule("WAYPOINT", waypoint_monitoring)

        if "FLIGHT_PLAN" in self.log_data:
            self.add_rule("FLIGHT_PLAN", flight_plan_monitoring)

        if "RC_LOST" in self.log_data:
            self.add_rule("RC_LOST", rc_lost_monitoring)

        if "DATALINK_LOST" in self.log_data:
            self.add_rule("DATALINK_LOST", datalink_lost_monitoring)

        if "GEOFENCE" in self.log_data:
            self.add_rule("GEOFENCE", geofence_monitoring)

        if "WEATHER" in self.log_data:
            self.add_rule("WEATHER", weather_monitoring)

        if "WIND_ESTIMATION" in self.log_data:
            self.add_rule("WIND_ESTIMATION", wind_estimation_monitoring)

        if "BATTERY_STATUS" in self.log_data:
            self.add_rule("BATTERY_STATUS", battery_status_monitoring)

        if "MOTOR_STATUS" in self.log_data:
            self.add_rule("MOTOR_STATUS", motor_status_monitoring)

        if "VIBRATION" in self.log_data:
            self.add_rule("VIBRATION", vibration_monitoring)

        if "COMPASS_CAL" in self.log_data:
            self.add_rule("COMPASS_CAL", compass_cal_monitoring)

        if "BAROMETER" in self.log_data:
            self.add_rule("BAROMETER", barometer_monitoring)

        if "TEMPERATURE" in self.log_data:
            self.add_rule("TEMPERATURE", temperature_monitoring)

        # === SAFETY-CRITICAL MESSAGE TYPES ===

        # 1. EMERGENCY - Emergency situations (CRITICAL - always ERROR level)
        def emergency_monitoring(
            emergency: Emergency, prev_emergency
        ) -> Optional[tuple[float, str]]:
            """Monitor emergency situations - always critical"""
            emergency_type = (
                emergency.emergency_type
                if hasattr(emergency, "emergency_type")
                else "UNKNOWN"
            )
            severity = emergency.severity if hasattr(emergency, "severity") else 0
            description = (
                emergency.description if hasattr(emergency, "description") else ""
            )

            if severity >= 3:  # Critical emergency
                return (
                    emergency.timestamp,
                    f"CRITICAL EMERGENCY ({emergency_type}): {description}",
                )
            elif severity >= 2:  # Major emergency
                return (
                    emergency.timestamp,
                    f"MAJOR EMERGENCY ({emergency_type}): {description}",
                )
            else:  # Minor emergency
                return (
                    emergency.timestamp,
                    f"EMERGENCY ({emergency_type}): {description}",
                )

        # 2. GEOFENCE_BREACH - Geofence violations (CRITICAL)
        def geofence_breach_monitoring(
            breach: GeofenceBreach, prev_breach
        ) -> Optional[tuple[float, str]]:
            """Monitor geofence breach events - critical safety violations"""
            breach_type = (
                breach.breach_type if hasattr(breach, "breach_type") else "UNKNOWN"
            )
            distance = (
                breach.distance_from_fence
                if hasattr(breach, "distance_from_fence")
                else 0
            )
            fence_id = breach.fence_id if hasattr(breach, "fence_id") else 0
            recovery = (
                breach.recovery_required
                if hasattr(breach, "recovery_required")
                else False
            )

            if recovery:
                return (
                    breach.timestamp,
                    f"CRITICAL GEOFENCE BREACH ({breach_type}) - Fence {fence_id}: {distance:.1f}m - RECOVERY REQUIRED",
                )
            else:
                return (
                    breach.timestamp,
                    f"GEOFENCE BREACH ({breach_type}) - Fence {fence_id}: {distance:.1f}m from boundary",
                )

        # 3. COLLISION_AVOIDANCE - Collision detection (CRITICAL)
        def collision_avoidance_monitoring(
            collision: CollisionAvoidance, prev_collision
        ) -> Optional[tuple[float, str]]:
            """Monitor collision avoidance events - critical safety"""
            threat_type = (
                collision.threat_type
                if hasattr(collision, "threat_type")
                else "UNKNOWN"
            )
            distance = (
                collision.threat_distance
                if hasattr(collision, "threat_distance")
                else 0
            )
            bearing = (
                collision.threat_bearing if hasattr(collision, "threat_bearing") else 0
            )
            action = (
                collision.avoidance_action
                if hasattr(collision, "avoidance_action")
                else ""
            )
            threat_id = collision.threat_id if hasattr(collision, "threat_id") else ""

            if distance < 100:  # Very close threat
                return (
                    collision.timestamp,
                    f"CRITICAL COLLISION THREAT ({threat_type}) {threat_id}: {distance:.1f}m @ {bearing:.0f}° - ACTION: {action}",
                )
            elif distance < 500:  # Close threat
                return (
                    collision.timestamp,
                    f"COLLISION THREAT ({threat_type}) {threat_id}: {distance:.1f}m @ {bearing:.0f}° - ACTION: {action}",
                )
            else:
                return (
                    collision.timestamp,
                    f"Collision avoidance active ({threat_type}): {distance:.1f}m",
                )

        # 4. TRAFFIC - Traffic advisory (WARNING/INFO)
        def traffic_monitoring(
            traffic: Traffic, prev_traffic
        ) -> Optional[tuple[float, str]]:
            """Monitor traffic advisory events"""
            traffic_id = (
                traffic.traffic_id if hasattr(traffic, "traffic_id") else "UNKNOWN"
            )
            distance = (
                traffic.relative_distance
                if hasattr(traffic, "relative_distance")
                else 0
            )
            bearing = (
                traffic.relative_bearing if hasattr(traffic, "relative_bearing") else 0
            )
            altitude = (
                traffic.relative_altitude
                if hasattr(traffic, "relative_altitude")
                else 0
            )
            level = traffic.advisory_level if hasattr(traffic, "advisory_level") else 0
            aircraft_type = (
                traffic.aircraft_type if hasattr(traffic, "aircraft_type") else ""
            )

            if level >= 3:  # High priority traffic
                return (
                    traffic.timestamp,
                    f"HIGH PRIORITY TRAFFIC ({aircraft_type}) {traffic_id}: {distance:.1f}m @ {bearing:.0f}°, alt {altitude:+.0f}m",
                )
            elif level >= 2:  # Medium priority traffic
                return (
                    traffic.timestamp,
                    f"Traffic advisory ({aircraft_type}) {traffic_id}: {distance:.1f}m @ {bearing:.0f}°",
                )
            elif distance < 1000:  # Close traffic
                return (
                    traffic.timestamp,
                    f"Traffic detected ({aircraft_type}): {distance:.1f}m",
                )

            return None

        # 5. TERRAIN_FOLLOWING - Terrain awareness (WARNING)
        def terrain_following_monitoring(
            terrain: TerrainFollowing, prev_terrain
        ) -> Optional[tuple[float, str]]:
            """Monitor terrain following and awareness events"""
            clearance = (
                terrain.ground_clearance if hasattr(terrain, "ground_clearance") else 0
            )
            terrain_height = (
                terrain.terrain_height if hasattr(terrain, "terrain_height") else 0
            )
            warning = (
                terrain.terrain_warning
                if hasattr(terrain, "terrain_warning")
                else False
            )
            min_clearance = (
                terrain.minimum_clearance
                if hasattr(terrain, "minimum_clearance")
                else 0
            )
            follow_mode = terrain.follow_mode if hasattr(terrain, "follow_mode") else ""

            if warning:
                return (
                    terrain.timestamp,
                    f"TERRAIN WARNING: {clearance:.1f}m clearance (min: {min_clearance:.1f}m)",
                )
            elif clearance < 50 and clearance > 0:  # Low terrain clearance
                return (
                    terrain.timestamp,
                    f"Low terrain clearance: {clearance:.1f}m ({follow_mode})",
                )
            elif (
                prev_terrain
                and hasattr(prev_terrain, "follow_mode")
                and terrain.follow_mode != prev_terrain.follow_mode
            ):
                return (
                    terrain.timestamp,
                    f"Terrain following mode changed: {prev_terrain.follow_mode} -> {terrain.follow_mode}",
                )

            return None

        # 6. OBSTACLE_DETECTION - Obstacle avoidance (WARNING/ERROR)
        def obstacle_detection_monitoring(
            obstacle: ObstacleDetection, prev_obstacle
        ) -> Optional[tuple[float, str]]:
            """Monitor obstacle detection events"""
            distance = (
                obstacle.obstacle_distance
                if hasattr(obstacle, "obstacle_distance")
                else 0
            )
            bearing = (
                obstacle.obstacle_bearing
                if hasattr(obstacle, "obstacle_bearing")
                else 0
            )
            height = (
                obstacle.obstacle_height if hasattr(obstacle, "obstacle_height") else 0
            )
            obstacle_type = (
                obstacle.obstacle_type
                if hasattr(obstacle, "obstacle_type")
                else "UNKNOWN"
            )
            avoidance_required = (
                obstacle.avoidance_required
                if hasattr(obstacle, "avoidance_required")
                else False
            )
            sensor_type = (
                obstacle.sensor_type if hasattr(obstacle, "sensor_type") else ""
            )

            if avoidance_required and distance < 50:
                return (
                    obstacle.timestamp,
                    f"CRITICAL OBSTACLE ({obstacle_type}): {distance:.1f}m @ {bearing:.0f}° - IMMEDIATE AVOIDANCE REQUIRED",
                )
            elif avoidance_required:
                return (
                    obstacle.timestamp,
                    f"OBSTACLE AVOIDANCE ({obstacle_type}): {distance:.1f}m @ {bearing:.0f}° h:{height:.1f}m",
                )
            elif distance < 100:
                return (
                    obstacle.timestamp,
                    f"Obstacle detected ({obstacle_type}, {sensor_type}): {distance:.1f}m @ {bearing:.0f}°",
                )

            return None

        # 7. LOSS_OF_CONTROL - Control loss detection (CRITICAL)
        def loss_of_control_monitoring(
            control_loss: LossOfControl, prev_control_loss
        ) -> Optional[tuple[float, str]]:
            """Monitor control loss events - critical safety"""
            loss_type = (
                control_loss.control_loss_type
                if hasattr(control_loss, "control_loss_type")
                else "UNKNOWN"
            )
            affected_axis = (
                control_loss.affected_axis
                if hasattr(control_loss, "affected_axis")
                else ""
            )
            magnitude = (
                control_loss.loss_magnitude
                if hasattr(control_loss, "loss_magnitude")
                else 0
            )
            recovery_attempted = (
                control_loss.recovery_attempted
                if hasattr(control_loss, "recovery_attempted")
                else False
            )
            backup_active = (
                control_loss.backup_system_active
                if hasattr(control_loss, "backup_system_active")
                else False
            )

            if magnitude > 0.8:  # Severe control loss
                status = "BACKUP ACTIVE" if backup_active else "NO BACKUP"
                return (
                    control_loss.timestamp,
                    f"SEVERE CONTROL LOSS ({loss_type}) {affected_axis}: {magnitude:.2f} - {status}",
                )
            elif magnitude > 0.5:  # Moderate control loss
                recovery_status = (
                    "RECOVERY ATTEMPTED" if recovery_attempted else "NO RECOVERY"
                )
                return (
                    control_loss.timestamp,
                    f"CONTROL LOSS ({loss_type}) {affected_axis}: {magnitude:.2f} - {recovery_status}",
                )
            else:
                return (
                    control_loss.timestamp,
                    f"Control degradation ({loss_type}) {affected_axis}: {magnitude:.2f}",
                )

        # 8. STALL_WARNING - Aerodynamic stall (CRITICAL)
        def stall_warning_monitoring(
            stall: StallWarning, prev_stall
        ) -> Optional[tuple[float, str]]:
            """Monitor stall warning events - critical flight safety"""
            probability = (
                stall.stall_probability if hasattr(stall, "stall_probability") else 0
            )
            aoa = stall.angle_of_attack if hasattr(stall, "angle_of_attack") else 0
            airspeed = stall.airspeed if hasattr(stall, "airspeed") else 0
            margin = stall.stall_margin if hasattr(stall, "stall_margin") else 0
            level = stall.warning_level if hasattr(stall, "warning_level") else 0

            if probability > 0.8 or level >= 3:  # Imminent stall
                return (
                    stall.timestamp,
                    f"IMMINENT STALL WARNING: {probability:.1%} prob, AoA:{aoa:.1f}°, AS:{airspeed:.1f}m/s",
                )
            elif probability > 0.5 or level >= 2:  # High stall risk
                return (
                    stall.timestamp,
                    f"HIGH STALL RISK: {probability:.1%} prob, margin:{margin:.1f}°, AS:{airspeed:.1f}m/s",
                )
            elif probability > 0.2 or level >= 1:  # Stall warning
                return (
                    stall.timestamp,
                    f"Stall warning: {probability:.1%} probability, AoA:{aoa:.1f}°",
                )

            return None

        # 9. OVER_SPEED - Speed limit violations (WARNING)
        def over_speed_monitoring(
            overspeed: OverSpeed, prev_overspeed
        ) -> Optional[tuple[float, str]]:
            """Monitor overspeed events"""
            current_speed = (
                overspeed.current_speed if hasattr(overspeed, "current_speed") else 0
            )
            speed_limit = (
                overspeed.speed_limit if hasattr(overspeed, "speed_limit") else 0
            )
            margin = (
                overspeed.overspeed_margin
                if hasattr(overspeed, "overspeed_margin")
                else 0
            )
            limit_type = (
                overspeed.limit_type if hasattr(overspeed, "limit_type") else ""
            )
            level = (
                overspeed.warning_level if hasattr(overspeed, "warning_level") else 0
            )

            if level >= 3 or margin > 50:  # Severe overspeed
                return (
                    overspeed.timestamp,
                    f"SEVERE OVERSPEED ({limit_type}): {current_speed:.1f}m/s (limit: {speed_limit:.1f}m/s, +{margin:.1f}m/s)",
                )
            elif level >= 2 or margin > 20:  # Moderate overspeed
                return (
                    overspeed.timestamp,
                    f"OVERSPEED ({limit_type}): {current_speed:.1f}m/s (limit: {speed_limit:.1f}m/s)",
                )
            elif level >= 1 or margin > 5:  # Minor overspeed
                return (
                    overspeed.timestamp,
                    f"Speed limit exceeded ({limit_type}): {current_speed:.1f}m/s",
                )

            return None

        # 10. ALTITUDE_LIMIT - Altitude restrictions (WARNING)
        def altitude_limit_monitoring(
            alt_limit: AltitudeLimit, prev_alt_limit
        ) -> Optional[tuple[float, str]]:
            """Monitor altitude limit events"""
            current_alt = (
                alt_limit.current_altitude
                if hasattr(alt_limit, "current_altitude")
                else 0
            )
            altitude_limit = (
                alt_limit.altitude_limit if hasattr(alt_limit, "altitude_limit") else 0
            )
            margin = (
                alt_limit.altitude_margin
                if hasattr(alt_limit, "altitude_margin")
                else 0
            )
            limit_type = (
                alt_limit.limit_type if hasattr(alt_limit, "limit_type") else ""
            )
            severity = (
                alt_limit.violation_severity
                if hasattr(alt_limit, "violation_severity")
                else 0
            )

            if severity >= 3 or abs(margin) > 100:  # Severe altitude violation
                violation_dir = "above" if margin > 0 else "below"
                return (
                    alt_limit.timestamp,
                    f"SEVERE ALTITUDE VIOLATION ({limit_type}): {current_alt:.1f}m ({abs(margin):.1f}m {violation_dir} limit)",
                )
            elif severity >= 2 or abs(margin) > 50:  # Moderate altitude violation
                violation_dir = "above" if margin > 0 else "below"
                return (
                    alt_limit.timestamp,
                    f"ALTITUDE VIOLATION ({limit_type}): {current_alt:.1f}m ({abs(margin):.1f}m {violation_dir} limit)",
                )
            elif severity >= 1 or abs(margin) > 10:  # Minor altitude violation
                return (
                    alt_limit.timestamp,
                    f"Altitude limit approached ({limit_type}): {current_alt:.1f}m",
                )

            return None

        # Register all safety-critical event detection rules
        if "EMERGENCY" in self.log_data:
            self.add_rule("EMERGENCY", emergency_monitoring)

        if "GEOFENCE_BREACH" in self.log_data:
            self.add_rule("GEOFENCE_BREACH", geofence_breach_monitoring)

        if "COLLISION_AVOIDANCE" in self.log_data:
            self.add_rule("COLLISION_AVOIDANCE", collision_avoidance_monitoring)

        if "TRAFFIC" in self.log_data:
            self.add_rule("TRAFFIC", traffic_monitoring)

        if "TERRAIN_FOLLOWING" in self.log_data:
            self.add_rule("TERRAIN_FOLLOWING", terrain_following_monitoring)

        if "OBSTACLE_DETECTION" in self.log_data:
            self.add_rule("OBSTACLE_DETECTION", obstacle_detection_monitoring)

        if "LOSS_OF_CONTROL" in self.log_data:
            self.add_rule("LOSS_OF_CONTROL", loss_of_control_monitoring)

        if "STALL_WARNING" in self.log_data:
            self.add_rule("STALL_WARNING", stall_warning_monitoring)

        if "OVER_SPEED" in self.log_data:
            self.add_rule("OVER_SPEED", over_speed_monitoring)

        if "ALTITUDE_LIMIT" in self.log_data:
            self.add_rule("ALTITUDE_LIMIT", altitude_limit_monitoring)

        # === COMMUNICATION AND TELEMETRY MESSAGE TYPES ===

        # 1. TELEMETRY_STATUS - Telemetry health monitoring
        def telemetry_status_monitoring(
            telemetry: TelemetryStatus, prev_telemetry
        ) -> Optional[tuple[float, str]]:
            """Monitor telemetry health and communication status"""
            health_status = (
                telemetry.health_status if hasattr(telemetry, "health_status") else ""
            )
            connection_status = (
                telemetry.connection_status
                if hasattr(telemetry, "connection_status")
                else ""
            )
            rate = (
                telemetry.telemetry_rate if hasattr(telemetry, "telemetry_rate") else 0
            )
            expected_rate = (
                telemetry.expected_rate if hasattr(telemetry, "expected_rate") else 0
            )
            error_count = (
                telemetry.error_count if hasattr(telemetry, "error_count") else 0
            )

            # Check for connection issues
            if connection_status.lower() in ["disconnected", "failed", "error"]:
                return (
                    telemetry.timestamp,
                    f"TELEMETRY CONNECTION LOST: {connection_status}",
                )
            elif health_status.lower() in ["critical", "error", "failed"]:
                return (
                    telemetry.timestamp,
                    f"TELEMETRY HEALTH CRITICAL: {health_status}",
                )
            elif error_count > 10:
                return (
                    telemetry.timestamp,
                    f"High telemetry error count: {error_count}",
                )
            elif (
                expected_rate > 0 and rate < expected_rate * 0.5
            ):  # Less than 50% expected rate
                return (
                    telemetry.timestamp,
                    f"Low telemetry rate: {rate:.1f}Hz (expected: {expected_rate:.1f}Hz)",
                )
            elif (
                prev_telemetry
                and hasattr(prev_telemetry, "connection_status")
                and connection_status != prev_telemetry.connection_status
            ):
                return (
                    telemetry.timestamp,
                    f"Telemetry status changed: {prev_telemetry.connection_status} -> {connection_status}",
                )

            return None

        # 2. RADIO_STATUS - Radio communication monitoring
        def radio_status_monitoring(
            radio: RadioStatus, prev_radio
        ) -> Optional[tuple[float, str]]:
            """Monitor radio communication status"""
            channel_status = (
                radio.channel_status if hasattr(radio, "channel_status") else ""
            )
            error_rate = radio.error_rate if hasattr(radio, "error_rate") else 0
            tx_power = radio.tx_power if hasattr(radio, "tx_power") else 0
            rx_power = radio.rx_power if hasattr(radio, "rx_power") else 0
            noise_floor = radio.noise_floor if hasattr(radio, "noise_floor") else 0
            frequency = (
                radio.radio_frequency if hasattr(radio, "radio_frequency") else 0
            )

            # Check for radio issues
            if channel_status.lower() in ["blocked", "interference", "failed"]:
                return (
                    radio.timestamp,
                    f"RADIO CHANNEL ISSUE: {channel_status} @ {frequency:.1f}MHz",
                )
            elif error_rate > 0.1:  # 10% error rate
                return (
                    radio.timestamp,
                    f"High radio error rate: {error_rate:.1%} @ {frequency:.1f}MHz",
                )
            elif (
                rx_power > 0 and noise_floor > 0 and (rx_power - noise_floor) < 10
            ):  # Poor SNR
                snr = rx_power - noise_floor
                return (
                    radio.timestamp,
                    f"Poor radio SNR: {snr:.1f}dB (RX: {rx_power:.1f}dBm, Noise: {noise_floor:.1f}dBm)",
                )
            elif tx_power > 30:  # High transmit power might indicate range issues
                return (radio.timestamp, f"High radio TX power: {tx_power:.1f}dBm")
            elif (
                prev_radio
                and hasattr(prev_radio, "channel_status")
                and channel_status != prev_radio.channel_status
            ):
                return (
                    radio.timestamp,
                    f"Radio channel status changed: {prev_radio.channel_status} -> {channel_status}",
                )

            return None

        # 3. MODEM_STATUS - Modem connectivity monitoring
        def modem_status_monitoring(
            modem: ModemStatus, prev_modem
        ) -> Optional[tuple[float, str]]:
            """Monitor modem connectivity status"""
            connection_state = (
                modem.connection_state if hasattr(modem, "connection_state") else ""
            )
            signal_quality = (
                modem.signal_quality if hasattr(modem, "signal_quality") else 0
            )
            network_registration = (
                modem.network_registration
                if hasattr(modem, "network_registration")
                else ""
            )
            data_session_status = (
                modem.data_session_status
                if hasattr(modem, "data_session_status")
                else ""
            )
            modem_type = modem.modem_type if hasattr(modem, "modem_type") else ""

            # Check for modem connectivity issues
            if connection_state.lower() in [
                "disconnected",
                "failed",
                "error",
                "offline",
            ]:
                return (
                    modem.timestamp,
                    f"MODEM DISCONNECTED ({modem_type}): {connection_state}",
                )
            elif network_registration.lower() in [
                "denied",
                "failed",
                "roaming_not_allowed",
            ]:
                return (
                    modem.timestamp,
                    f"MODEM NETWORK REGISTRATION FAILED: {network_registration}",
                )
            elif data_session_status.lower() in ["failed", "disconnected", "suspended"]:
                return (
                    modem.timestamp,
                    f"MODEM DATA SESSION ISSUE: {data_session_status}",
                )
            elif signal_quality < 20:  # Poor signal quality (assuming 0-100 scale)
                return (
                    modem.timestamp,
                    f"Poor modem signal quality: {signal_quality}%",
                )
            elif (
                prev_modem
                and hasattr(prev_modem, "connection_state")
                and connection_state != prev_modem.connection_state
            ):
                return (
                    modem.timestamp,
                    f"Modem connection changed: {prev_modem.connection_state} -> {connection_state}",
                )

            return None

        # 4. LINK_QUALITY - Communication quality monitoring
        def link_quality_monitoring(
            link: LinkQuality, prev_link
        ) -> Optional[tuple[float, str]]:
            """Monitor communication link quality"""
            quality_percent = (
                link.link_quality_percent
                if hasattr(link, "link_quality_percent")
                else 0
            )
            snr = link.signal_to_noise if hasattr(link, "signal_to_noise") else 0
            ber = link.bit_error_rate if hasattr(link, "bit_error_rate") else 0
            fer = link.frame_error_rate if hasattr(link, "frame_error_rate") else 0
            latency = link.latency_ms if hasattr(link, "latency_ms") else 0
            jitter = link.jitter_ms if hasattr(link, "jitter_ms") else 0
            throughput = link.throughput_bps if hasattr(link, "throughput_bps") else 0

            # Check for link quality issues
            if quality_percent < 30:  # Very poor quality
                return (
                    link.timestamp,
                    f"CRITICAL LINK QUALITY: {quality_percent:.1f}% (SNR: {snr:.1f}dB)",
                )
            elif quality_percent < 50:  # Poor quality
                return (
                    link.timestamp,
                    f"Poor link quality: {quality_percent:.1f}% (BER: {ber:.2e}, FER: {fer:.2e})",
                )
            elif ber > 1e-3:  # High bit error rate
                return (
                    link.timestamp,
                    f"High bit error rate: {ber:.2e} (quality: {quality_percent:.1f}%)",
                )
            elif latency > 1000:  # High latency
                return (
                    link.timestamp,
                    f"High link latency: {latency:.0f}ms (jitter: {jitter:.0f}ms)",
                )
            elif throughput > 0 and prev_link and hasattr(prev_link, "throughput_bps"):
                if (
                    prev_link.throughput_bps > 0
                    and throughput < prev_link.throughput_bps * 0.5
                ):
                    return (
                        link.timestamp,
                        f"Throughput degradation: {throughput:.0f}bps (was {prev_link.throughput_bps:.0f}bps)",
                    )

            return None

        # 5. PACKET_LOSS - Data packet loss monitoring
        def packet_loss_monitoring(
            packet_loss: PacketLoss, prev_packet_loss
        ) -> Optional[tuple[float, str]]:
            """Monitor data packet loss"""
            loss_percentage = (
                packet_loss.loss_percentage
                if hasattr(packet_loss, "loss_percentage")
                else 0
            )
            consecutive_losses = (
                packet_loss.consecutive_losses
                if hasattr(packet_loss, "consecutive_losses")
                else 0
            )
            packets_sent = (
                packet_loss.packets_sent if hasattr(packet_loss, "packets_sent") else 0
            )
            packets_received = (
                packet_loss.packets_received
                if hasattr(packet_loss, "packets_received")
                else 0
            )
            packets_lost = (
                packet_loss.packets_lost if hasattr(packet_loss, "packets_lost") else 0
            )
            recovery_time = (
                packet_loss.recovery_time
                if hasattr(packet_loss, "recovery_time")
                else 0
            )

            # Check for packet loss issues
            if loss_percentage > 20:  # Severe packet loss
                return (
                    packet_loss.timestamp,
                    f"SEVERE PACKET LOSS: {loss_percentage:.1f}% ({packets_lost}/{packets_sent} packets)",
                )
            elif loss_percentage > 10:  # Moderate packet loss
                return (
                    packet_loss.timestamp,
                    f"High packet loss: {loss_percentage:.1f}% ({packets_lost}/{packets_sent} packets)",
                )
            elif consecutive_losses > 10:  # Many consecutive losses
                return (
                    packet_loss.timestamp,
                    f"Consecutive packet losses: {consecutive_losses} packets",
                )
            elif loss_percentage > 5:  # Some packet loss
                return (
                    packet_loss.timestamp,
                    f"Packet loss detected: {loss_percentage:.1f}%",
                )
            elif recovery_time > 0:  # Recovery from packet loss
                return (
                    packet_loss.timestamp,
                    f"Packet loss recovery: {recovery_time:.1f}s recovery time",
                )

            return None

        # 6. RSSI_LOW - Signal strength warnings
        def rssi_low_monitoring(
            rssi: RSSILow, prev_rssi
        ) -> Optional[tuple[float, str]]:
            """Monitor signal strength warnings"""
            current_rssi = rssi.current_rssi if hasattr(rssi, "current_rssi") else 0
            minimum_rssi = rssi.minimum_rssi if hasattr(rssi, "minimum_rssi") else 0
            margin = rssi.rssi_margin if hasattr(rssi, "rssi_margin") else 0
            warning_level = rssi.warning_level if hasattr(rssi, "warning_level") else 0
            frequency_band = (
                rssi.frequency_band if hasattr(rssi, "frequency_band") else ""
            )
            antenna_status = (
                rssi.antenna_status if hasattr(rssi, "antenna_status") else ""
            )

            # Check for RSSI issues
            if warning_level >= 3 or current_rssi < -100:  # Critical RSSI
                return (
                    rssi.timestamp,
                    f"CRITICAL LOW RSSI: {current_rssi:.1f}dBm ({frequency_band}) - LINK LOSS IMMINENT",
                )
            elif warning_level >= 2 or current_rssi < -90:  # Severe RSSI warning
                return (
                    rssi.timestamp,
                    f"SEVERE LOW RSSI: {current_rssi:.1f}dBm ({frequency_band}) margin: {margin:.1f}dB",
                )
            elif warning_level >= 1 or current_rssi < -80:  # RSSI warning
                return (
                    rssi.timestamp,
                    f"Low RSSI warning: {current_rssi:.1f}dBm ({frequency_band})",
                )
            elif antenna_status.lower() in ["damaged", "disconnected", "failed"]:
                return (
                    rssi.timestamp,
                    f"ANTENNA ISSUE: {antenna_status} - RSSI: {current_rssi:.1f}dBm",
                )
            elif prev_rssi and hasattr(prev_rssi, "current_rssi"):
                rssi_drop = prev_rssi.current_rssi - current_rssi
                if rssi_drop > 20:  # Significant RSSI drop
                    return (
                        rssi.timestamp,
                        f"Large RSSI drop: -{rssi_drop:.1f}dB (now {current_rssi:.1f}dBm)",
                    )

            return None

        # Register all communication and telemetry event detection rules
        if "TELEMETRY_STATUS" in self.log_data:
            self.add_rule("TELEMETRY_STATUS", telemetry_status_monitoring)

        if "RADIO_STATUS" in self.log_data:
            self.add_rule("RADIO_STATUS", radio_status_monitoring)

        if "MODEM_STATUS" in self.log_data:
            self.add_rule("MODEM_STATUS", modem_status_monitoring)

        if "LINK_QUALITY" in self.log_data:
            self.add_rule("LINK_QUALITY", link_quality_monitoring)

        if "PACKET_LOSS" in self.log_data:
            self.add_rule("PACKET_LOSS", packet_loss_monitoring)

        if "RSSI_LOW" in self.log_data:
            self.add_rule("RSSI_LOW", rssi_low_monitoring)

        # === POWER MANAGEMENT MESSAGE TYPES ===

        def current_spike_monitoring(
            spike: CurrentSpike, prev_spike
        ) -> Optional[tuple[float, str]]:
            """Monitor current consumption spikes"""
            current = spike.current if hasattr(spike, "current") else 0.0
            peak_current = spike.peak_current if hasattr(spike, "peak_current") else 0.0
            duration = spike.spike_duration if hasattr(spike, "spike_duration") else 0.0
            threshold = spike.threshold if hasattr(spike, "threshold") else 0.0
            source_device = (
                spike.source_device if hasattr(spike, "source_device") else ""
            )
            power_rail = spike.power_rail if hasattr(spike, "power_rail") else ""

            # Check for dangerous current spikes
            if peak_current > 50.0:  # Critical current spike
                return (
                    spike.timestamp,
                    f"CRITICAL CURRENT SPIKE: {peak_current:.1f}A from {source_device} on {power_rail} rail",
                )
            elif peak_current > 30.0:  # High current spike
                return (
                    spike.timestamp,
                    f"HIGH CURRENT SPIKE: {peak_current:.1f}A duration: {duration:.2f}s ({source_device})",
                )
            elif peak_current > threshold and threshold > 0:  # Threshold exceeded
                return (
                    spike.timestamp,
                    f"Current spike threshold exceeded: {peak_current:.1f}A (threshold: {threshold:.1f}A)",
                )
            elif duration > 1.0 and peak_current > 15.0:  # Sustained spike
                return (
                    spike.timestamp,
                    f"Sustained current spike: {peak_current:.1f}A for {duration:.2f}s",
                )
            elif prev_spike and hasattr(prev_spike, "peak_current"):
                spike_increase = peak_current - prev_spike.peak_current
                if spike_increase > 10.0:  # Large spike increase
                    return (
                        spike.timestamp,
                        f"Current spike increase: +{spike_increase:.1f}A (now {peak_current:.1f}A)",
                    )

            return None

        def power_distribution_monitoring(
            power_dist: PowerDistribution, prev_power_dist
        ) -> Optional[tuple[float, str]]:
            """Monitor power rail distribution"""
            rail_3v3_voltage = (
                power_dist.rail_3v3_voltage
                if hasattr(power_dist, "rail_3v3_voltage")
                else 0.0
            )
            rail_5v_voltage = (
                power_dist.rail_5v_voltage
                if hasattr(power_dist, "rail_5v_voltage")
                else 0.0
            )
            rail_12v_voltage = (
                power_dist.rail_12v_voltage
                if hasattr(power_dist, "rail_12v_voltage")
                else 0.0
            )
            total_power = (
                power_dist.total_power if hasattr(power_dist, "total_power") else 0.0
            )
            efficiency = (
                power_dist.efficiency if hasattr(power_dist, "efficiency") else 0.0
            )

            # Check for power rail issues
            if (
                rail_3v3_voltage < 3.1 or rail_3v3_voltage > 3.5
            ):  # 3.3V rail out of spec
                return (
                    power_dist.timestamp,
                    f"POWER RAIL ERROR: 3.3V rail at {rail_3v3_voltage:.2f}V (spec: 3.1-3.5V)",
                )
            elif rail_5v_voltage < 4.7 or rail_5v_voltage > 5.3:  # 5V rail out of spec
                return (
                    power_dist.timestamp,
                    f"POWER RAIL ERROR: 5V rail at {rail_5v_voltage:.2f}V (spec: 4.7-5.3V)",
                )
            elif (
                rail_12v_voltage < 11.0 or rail_12v_voltage > 13.0
            ):  # 12V rail out of spec
                return (
                    power_dist.timestamp,
                    f"POWER RAIL ERROR: 12V rail at {rail_12v_voltage:.2f}V (spec: 11.0-13.0V)",
                )
            elif total_power > 500.0:  # High power consumption
                return (
                    power_dist.timestamp,
                    f"HIGH POWER CONSUMPTION: {total_power:.1f}W efficiency: {efficiency:.1f}%",
                )
            elif efficiency < 70.0 and total_power > 50.0:  # Poor efficiency
                return (
                    power_dist.timestamp,
                    f"LOW POWER EFFICIENCY: {efficiency:.1f}% at {total_power:.1f}W",
                )
            elif prev_power_dist and hasattr(prev_power_dist, "total_power"):
                power_increase = total_power - prev_power_dist.total_power
                if power_increase > 100.0:  # Large power increase
                    return (
                        power_dist.timestamp,
                        f"Power consumption increase: +{power_increase:.1f}W (now {total_power:.1f}W)",
                    )

            return None

        def charging_status_monitoring(
            charging: ChargingStatus, prev_charging
        ) -> Optional[tuple[float, str]]:
            """Monitor battery charging status"""
            charging_state = (
                charging.charging_state if hasattr(charging, "charging_state") else ""
            )
            charge_current = (
                charging.charge_current if hasattr(charging, "charge_current") else 0.0
            )
            charge_voltage = (
                charging.charge_voltage if hasattr(charging, "charge_voltage") else 0.0
            )
            charge_percentage = (
                charging.charge_percentage
                if hasattr(charging, "charge_percentage")
                else 0.0
            )
            charger_temperature = (
                charging.charger_temperature
                if hasattr(charging, "charger_temperature")
                else 0.0
            )
            charging_power = (
                charging.charging_power if hasattr(charging, "charging_power") else 0.0
            )

            # Check for charging issues
            if charging_state.lower() in ["error", "fault", "failed"]:
                return (
                    charging.timestamp,
                    f"CHARGING ERROR: {charging_state} - voltage: {charge_voltage:.1f}V",
                )
            elif charger_temperature > 70.0:  # Hot charger
                return (
                    charging.timestamp,
                    f"CHARGER OVERHEATING: {charger_temperature:.1f}°C - power: {charging_power:.1f}W",
                )
            elif (
                charging_state.lower() == "charging" and charge_current < 0.1
            ):  # Charging but no current
                return (
                    charging.timestamp,
                    f"CHARGING ANOMALY: state={charging_state} but current={charge_current:.2f}A",
                )
            elif charge_voltage > 60.0 or (
                charge_voltage > 30.0 and charge_current > 10.0
            ):  # Dangerous charging
                return (
                    charging.timestamp,
                    f"DANGEROUS CHARGING: {charge_voltage:.1f}V {charge_current:.1f}A",
                )
            elif prev_charging and hasattr(prev_charging, "charging_state"):
                if prev_charging.charging_state != charging_state:  # State change
                    return (
                        charging.timestamp,
                        f"Charging state change: {prev_charging.charging_state} → {charging_state}",
                    )

            return None

        def cell_balance_monitoring(
            cell_balance: CellBalance, prev_cell_balance
        ) -> Optional[tuple[float, str]]:
            """Monitor battery cell balancing"""
            cell_count = (
                cell_balance.cell_count if hasattr(cell_balance, "cell_count") else 0
            )
            voltage_min = (
                cell_balance.cell_voltage_min
                if hasattr(cell_balance, "cell_voltage_min")
                else 0.0
            )
            voltage_max = (
                cell_balance.cell_voltage_max
                if hasattr(cell_balance, "cell_voltage_max")
                else 0.0
            )
            voltage_imbalance = (
                cell_balance.voltage_imbalance
                if hasattr(cell_balance, "voltage_imbalance")
                else 0.0
            )
            balancing_active = (
                cell_balance.balancing_active
                if hasattr(cell_balance, "balancing_active")
                else False
            )
            balance_current = (
                cell_balance.balance_current
                if hasattr(cell_balance, "balance_current")
                else 0.0
            )

            # Check for cell balance issues
            if voltage_imbalance > 0.5:  # Critical imbalance
                return (
                    cell_balance.timestamp,
                    f"CRITICAL CELL IMBALANCE: {voltage_imbalance:.3f}V ({voltage_min:.2f}-{voltage_max:.2f}V)",
                )
            elif voltage_imbalance > 0.2:  # High imbalance
                return (
                    cell_balance.timestamp,
                    f"HIGH CELL IMBALANCE: {voltage_imbalance:.3f}V range: {voltage_min:.2f}-{voltage_max:.2f}V",
                )
            elif voltage_min < 3.0:  # Low cell voltage
                return (
                    cell_balance.timestamp,
                    f"LOW CELL VOLTAGE: {voltage_min:.2f}V (minimum cell)",
                )
            elif voltage_max > 4.3:  # High cell voltage
                return (
                    cell_balance.timestamp,
                    f"HIGH CELL VOLTAGE: {voltage_max:.2f}V (maximum cell)",
                )
            elif balancing_active and balance_current > 1.0:  # Active balancing
                return (
                    cell_balance.timestamp,
                    f"Cell balancing active: {balance_current:.2f}A imbalance: {voltage_imbalance:.3f}V",
                )
            elif prev_cell_balance and hasattr(prev_cell_balance, "voltage_imbalance"):
                imbalance_change = (
                    voltage_imbalance - prev_cell_balance.voltage_imbalance
                )
                if imbalance_change > 0.1:  # Worsening imbalance
                    return (
                        cell_balance.timestamp,
                        f"Cell imbalance worsening: +{imbalance_change:.3f}V (now {voltage_imbalance:.3f}V)",
                    )

            return None

        def thermal_throttle_monitoring(
            thermal: ThermalThrottle, prev_thermal
        ) -> Optional[tuple[float, str]]:
            """Monitor thermal throttling"""
            temperature = (
                thermal.temperature if hasattr(thermal, "temperature") else 0.0
            )
            throttle_level = (
                thermal.throttle_level if hasattr(thermal, "throttle_level") else 0.0
            )
            thermal_state = (
                thermal.thermal_state if hasattr(thermal, "thermal_state") else ""
            )
            max_temperature = (
                thermal.max_temperature if hasattr(thermal, "max_temperature") else 0.0
            )
            cooling_active = (
                thermal.cooling_active if hasattr(thermal, "cooling_active") else False
            )
            throttle_reason = (
                thermal.throttle_reason if hasattr(thermal, "throttle_reason") else ""
            )

            # Check for thermal issues
            if temperature > 85.0:  # Critical temperature
                return (
                    thermal.timestamp,
                    f"CRITICAL TEMPERATURE: {temperature:.1f}°C throttle: {throttle_level:.0f}% ({throttle_reason})",
                )
            elif temperature > 75.0:  # High temperature warning
                return (
                    thermal.timestamp,
                    f"HIGH TEMPERATURE WARNING: {temperature:.1f}°C max: {max_temperature:.1f}°C",
                )
            elif throttle_level > 50.0:  # Significant throttling
                return (
                    thermal.timestamp,
                    f"THERMAL THROTTLING: {throttle_level:.0f}% at {temperature:.1f}°C ({throttle_reason})",
                )
            elif thermal_state.lower() in ["emergency", "critical", "shutdown"]:
                return (
                    thermal.timestamp,
                    f"THERMAL EMERGENCY: {thermal_state} temp: {temperature:.1f}°C",
                )
            elif cooling_active and temperature > 60.0:  # Active cooling
                return (
                    thermal.timestamp,
                    f"Active cooling engaged: {temperature:.1f}°C throttle: {throttle_level:.0f}%",
                )
            elif prev_thermal and hasattr(prev_thermal, "temperature"):
                temp_rise = temperature - prev_thermal.temperature
                if temp_rise > 10.0:  # Rapid temperature rise
                    return (
                        thermal.timestamp,
                        f"Rapid temperature rise: +{temp_rise:.1f}°C (now {temperature:.1f}°C)",
                    )
                elif prev_thermal.throttle_level != throttle_level:  # Throttle change
                    return (
                        thermal.timestamp,
                        f"Thermal throttle change: {prev_thermal.throttle_level:.0f}% → {throttle_level:.0f}%",
                    )

            return None

        # Register all power management event detection rules
        if "CURRENT_SPIKE" in self.log_data:
            self.add_rule("CURRENT_SPIKE", current_spike_monitoring)

        if "POWER_DISTRIBUTION" in self.log_data:
            self.add_rule("POWER_DISTRIBUTION", power_distribution_monitoring)

        if "CHARGING_STATUS" in self.log_data:
            self.add_rule("CHARGING_STATUS", charging_status_monitoring)

        if "CELL_BALANCE" in self.log_data:
            self.add_rule("CELL_BALANCE", cell_balance_monitoring)

        if "THERMAL_THROTTLE" in self.log_data:
            self.add_rule("THERMAL_THROTTLE", thermal_throttle_monitoring)

        # === END POWER MANAGEMENT MESSAGE TYPES ===

        # === MISSION AND NAVIGATION MESSAGE TYPES ===

        def mission_item_monitoring(
            mission: MissionItem, prev_mission
        ) -> Optional[tuple[float, str]]:
            """Monitor mission execution"""
            waypoint_id = mission.waypoint_id if hasattr(mission, "waypoint_id") else 0
            mission_type = (
                mission.mission_type if hasattr(mission, "mission_type") else ""
            )
            completion_status = (
                mission.completion_status
                if hasattr(mission, "completion_status")
                else ""
            )
            latitude = mission.latitude if hasattr(mission, "latitude") else 0.0
            longitude = mission.longitude if hasattr(mission, "longitude") else 0.0
            altitude = mission.altitude if hasattr(mission, "altitude") else 0.0
            execution_time = (
                mission.execution_time if hasattr(mission, "execution_time") else 0.0
            )

            # Check for mission execution issues
            if completion_status.lower() in ["failed", "error", "aborted"]:
                return (
                    mission.timestamp,
                    f"MISSION FAILURE: WP{waypoint_id} ({mission_type}) - {completion_status}",
                )
            elif completion_status.lower() == "timeout":
                return (
                    mission.timestamp,
                    f"MISSION TIMEOUT: WP{waypoint_id} execution exceeded time limit",
                )
            elif completion_status.lower() == "completed":
                return (
                    mission.timestamp,
                    f"Mission waypoint completed: WP{waypoint_id} ({mission_type}) in {execution_time:.1f}s",
                )
            elif execution_time > 300.0:  # Long execution time (5 minutes)
                return (
                    mission.timestamp,
                    f"Long mission execution: WP{waypoint_id} running {execution_time:.1f}s",
                )
            elif prev_mission and hasattr(prev_mission, "waypoint_id"):
                if prev_mission.waypoint_id != waypoint_id:  # Waypoint change
                    return (
                        mission.timestamp,
                        f"Mission waypoint change: WP{prev_mission.waypoint_id} → WP{waypoint_id} ({mission_type})",
                    )

            return None

        def home_position_monitoring(
            home: HomePosition, prev_home
        ) -> Optional[tuple[float, str]]:
            """Monitor home position updates"""
            home_lat = home.home_latitude if hasattr(home, "home_latitude") else 0.0
            home_lon = home.home_longitude if hasattr(home, "home_longitude") else 0.0
            home_alt = home.home_altitude if hasattr(home, "home_altitude") else 0.0
            position_source = (
                home.position_source if hasattr(home, "position_source") else ""
            )
            accuracy = home.accuracy if hasattr(home, "accuracy") else 0.0
            update_reason = home.update_reason if hasattr(home, "update_reason") else ""

            # Check for home position issues
            if update_reason.lower() in [
                "gps_lost",
                "position_error",
                "manual_override",
            ]:
                return (
                    home.timestamp,
                    f"HOME POSITION UPDATE: {update_reason} - accuracy: {accuracy:.1f}m",
                )
            elif accuracy > 50.0:  # Poor accuracy
                return (
                    home.timestamp,
                    f"Poor home position accuracy: {accuracy:.1f}m ({position_source})",
                )
            elif prev_home and hasattr(prev_home, "home_latitude"):
                # Calculate distance moved
                lat_diff = (
                    abs(home_lat - prev_home.home_latitude) * 111000
                )  # Rough conversion to meters
                lon_diff = abs(home_lon - prev_home.home_longitude) * 111000
                distance_moved = (lat_diff**2 + lon_diff**2) ** 0.5
                if distance_moved > 1000.0:  # Home moved more than 1km
                    return (
                        home.timestamp,
                        f"HOME POSITION MOVED: {distance_moved:.0f}m from previous location",
                    )
                elif distance_moved > 100.0:  # Significant home movement
                    return (
                        home.timestamp,
                        f"Home position updated: moved {distance_moved:.0f}m ({update_reason})",
                    )

            return None

        def rally_point_monitoring(
            rally: RallyPoint, prev_rally
        ) -> Optional[tuple[float, str]]:
            """Monitor rally point management"""
            rally_id = rally.rally_id if hasattr(rally, "rally_id") else 0
            rally_type = rally.rally_type if hasattr(rally, "rally_type") else ""
            rally_status = rally.rally_status if hasattr(rally, "rally_status") else ""
            rally_lat = (
                rally.rally_latitude if hasattr(rally, "rally_latitude") else 0.0
            )
            rally_lon = (
                rally.rally_longitude if hasattr(rally, "rally_longitude") else 0.0
            )
            approach_direction = (
                rally.approach_direction
                if hasattr(rally, "approach_direction")
                else 0.0
            )

            # Check for rally point issues
            if rally_status.lower() in ["unreachable", "blocked", "invalid"]:
                return (
                    rally.timestamp,
                    f"RALLY POINT ISSUE: Rally {rally_id} ({rally_type}) - {rally_status}",
                )
            elif rally_status.lower() == "activated":
                return (
                    rally.timestamp,
                    f"Rally point activated: Rally {rally_id} ({rally_type}) approach: {approach_direction:.0f}°",
                )
            elif rally_status.lower() in ["added", "updated"]:
                return (
                    rally.timestamp,
                    f"Rally point {rally_status}: Rally {rally_id} at {rally_lat:.5f},{rally_lon:.5f}",
                )
            elif prev_rally and hasattr(prev_rally, "rally_status"):
                if prev_rally.rally_status != rally_status:  # Status change
                    return (
                        rally.timestamp,
                        f"Rally {rally_id} status change: {prev_rally.rally_status} → {rally_status}",
                    )

            return None

        def survey_status_monitoring(
            survey: SurveyStatus, prev_survey
        ) -> Optional[tuple[float, str]]:
            """Monitor survey mission progress"""
            survey_id = survey.survey_id if hasattr(survey, "survey_id") else 0
            current_leg = survey.current_leg if hasattr(survey, "current_leg") else 0
            total_legs = survey.total_legs if hasattr(survey, "total_legs") else 0
            progress = (
                survey.progress_percentage
                if hasattr(survey, "progress_percentage")
                else 0.0
            )
            coverage_area = (
                survey.coverage_area if hasattr(survey, "coverage_area") else 0.0
            )
            overlap = (
                survey.overlap_percentage
                if hasattr(survey, "overlap_percentage")
                else 0.0
            )
            survey_state = (
                survey.survey_state if hasattr(survey, "survey_state") else ""
            )

            # Check for survey mission issues
            if survey_state.lower() in ["failed", "aborted", "error"]:
                return (
                    survey.timestamp,
                    f"SURVEY FAILURE: Survey {survey_id} - {survey_state} at leg {current_leg}/{total_legs}",
                )
            elif survey_state.lower() == "completed":
                return (
                    survey.timestamp,
                    f"Survey completed: Survey {survey_id} - {coverage_area:.1f}m² with {overlap:.1f}% overlap",
                )
            elif (
                progress >= 50.0
                and prev_survey
                and hasattr(prev_survey, "progress_percentage")
            ):
                if prev_survey.progress_percentage < 50.0:  # Halfway milestone
                    return (
                        survey.timestamp,
                        f"Survey halfway complete: {progress:.1f}% (leg {current_leg}/{total_legs})",
                    )
            elif (
                current_leg != prev_survey.current_leg
                if prev_survey and hasattr(prev_survey, "current_leg")
                else False
            ):
                return (
                    survey.timestamp,
                    f"Survey leg progress: {current_leg}/{total_legs} ({progress:.1f}% complete)",
                )
            elif overlap < 20.0 and coverage_area > 1000.0:  # Low overlap warning
                return (
                    survey.timestamp,
                    f"Low survey overlap: {overlap:.1f}% (recommended >20%)",
                )

            return None

        def landing_sequence_monitoring(
            landing: LandingSequence, prev_landing
        ) -> Optional[tuple[float, str]]:
            """Monitor landing procedures"""
            landing_phase = (
                landing.landing_phase if hasattr(landing, "landing_phase") else ""
            )
            sequence_status = (
                landing.sequence_status if hasattr(landing, "sequence_status") else ""
            )
            approach_speed = (
                landing.approach_speed if hasattr(landing, "approach_speed") else 0.0
            )
            descent_rate = (
                landing.descent_rate if hasattr(landing, "descent_rate") else 0.0
            )
            target_alt = (
                landing.target_altitude if hasattr(landing, "target_altitude") else 0.0
            )
            landing_type = (
                landing.landing_type if hasattr(landing, "landing_type") else ""
            )

            # Check for landing sequence issues
            if sequence_status.lower() in ["aborted", "failed", "emergency"]:
                return (
                    landing.timestamp,
                    f"LANDING ABORT: {sequence_status} during {landing_phase} - {landing_type} landing",
                )
            elif (
                landing_phase.lower() == "final" and descent_rate > 10.0
            ):  # High descent rate on final
                return (
                    landing.timestamp,
                    f"HIGH DESCENT RATE: {descent_rate:.1f}m/s on final approach (max recommended: 10m/s)",
                )
            elif (
                landing_phase.lower() == "approach" and approach_speed > 50.0
            ):  # High approach speed
                return (
                    landing.timestamp,
                    f"HIGH APPROACH SPEED: {approach_speed:.1f}m/s ({landing_type} landing)",
                )
            elif sequence_status.lower() == "completed":
                return (
                    landing.timestamp,
                    f"Landing completed: {landing_type} landing at {approach_speed:.1f}m/s",
                )
            elif prev_landing and hasattr(prev_landing, "landing_phase"):
                if prev_landing.landing_phase != landing_phase:  # Phase change
                    return (
                        landing.timestamp,
                        f"Landing phase: {prev_landing.landing_phase} → {landing_phase} ({landing_type})",
                    )

            return None

        def approach_monitoring(
            approach: Approach, prev_approach
        ) -> Optional[tuple[float, str]]:
            """Monitor approach procedures"""
            approach_type = (
                approach.approach_type if hasattr(approach, "approach_type") else ""
            )
            approach_phase = (
                approach.approach_phase if hasattr(approach, "approach_phase") else ""
            )
            deviation_lateral = (
                approach.deviation_lateral
                if hasattr(approach, "deviation_lateral")
                else 0.0
            )
            deviation_vertical = (
                approach.deviation_vertical
                if hasattr(approach, "deviation_vertical")
                else 0.0
            )
            distance_to_threshold = (
                approach.distance_to_threshold
                if hasattr(approach, "distance_to_threshold")
                else 0.0
            )
            glide_slope = (
                approach.glide_slope if hasattr(approach, "glide_slope") else 0.0
            )
            approach_speed = (
                approach.approach_speed if hasattr(approach, "approach_speed") else 0.0
            )

            # Check for approach procedure issues
            if abs(deviation_lateral) > 50.0:  # Large lateral deviation
                return (
                    approach.timestamp,
                    f"LARGE LATERAL DEVIATION: {deviation_lateral:.1f}m on {approach_type} approach",
                )
            elif abs(deviation_vertical) > 20.0:  # Large vertical deviation
                return (
                    approach.timestamp,
                    f"LARGE VERTICAL DEVIATION: {deviation_vertical:.1f}m from glide slope",
                )
            elif approach_phase.lower() == "final" and abs(deviation_lateral) > 20.0:
                return (
                    approach.timestamp,
                    f"Final approach deviation: {deviation_lateral:.1f}m lateral, {deviation_vertical:.1f}m vertical",
                )
            elif distance_to_threshold < 500.0 and approach_phase.lower() != "final":
                return (
                    approach.timestamp,
                    f"Approaching threshold: {distance_to_threshold:.0f}m - entering final phase",
                )
            elif (
                glide_slope > 10.0 and approach_type.lower() != "steep"
            ):  # Steep glide slope
                return (
                    approach.timestamp,
                    f"Steep glide slope: {glide_slope:.1f}° ({approach_type} approach)",
                )
            elif prev_approach and hasattr(prev_approach, "approach_phase"):
                if prev_approach.approach_phase != approach_phase:  # Phase change
                    return (
                        approach.timestamp,
                        f"Approach phase: {prev_approach.approach_phase} → {approach_phase}",
                    )

            return None

        # Register all mission and navigation event detection rules
        if "MISSION_ITEM" in self.log_data:
            self.add_rule("MISSION_ITEM", mission_item_monitoring)

        if "HOME_POSITION" in self.log_data:
            self.add_rule("HOME_POSITION", home_position_monitoring)

        if "RALLY_POINT" in self.log_data:
            self.add_rule("RALLY_POINT", rally_point_monitoring)

        if "SURVEY_STATUS" in self.log_data:
            self.add_rule("SURVEY_STATUS", survey_status_monitoring)

        if "LANDING_SEQUENCE" in self.log_data:
            self.add_rule("LANDING_SEQUENCE", landing_sequence_monitoring)

        if "APPROACH" in self.log_data:
            self.add_rule("APPROACH", approach_monitoring)

        # === END MISSION AND NAVIGATION MESSAGE TYPES ===

        # === END COMMUNICATION AND TELEMETRY MESSAGE TYPES ===

        # === END SAFETY-CRITICAL MESSAGE TYPES ===

        # === END ADDITIONAL FLIGHT MONITORING MESSAGE TYPES ===

        # Add detection for additional message types that might exist
        def generic_alive_monitoring(alive: any, _) -> Optional[tuple[float, str]]:
            """Monitor system alive messages"""
            if hasattr(alive, "counter"):
                return (alive.timestamp, f"System heartbeat: {alive.counter}")
            return None

        def generic_dl_value_monitoring(dl_val: any, _) -> Optional[tuple[float, str]]:
            """Monitor downlink values"""
            if hasattr(dl_val, "id") and hasattr(dl_val, "value"):
                return (dl_val.timestamp, f"DL Value {dl_val.id}: {dl_val.value}")
            return None

        def power_monitoring(power: any, _) -> Optional[tuple[float, str]]:
            """Monitor power distribution"""
            if hasattr(power, "values") and power.values:
                max_power = max(power.values) if power.values else 0
                if max_power > 8000:  # High power value
                    return (power.timestamp, f"High power output detected: {max_power}")
            return None

        if "ALIVE" in self.log_data:
            self.add_rule("ALIVE", generic_alive_monitoring)

        if "DL_VALUE" in self.log_data:
            self.add_rule("DL_VALUE", generic_dl_value_monitoring)

        if "POWER" in self.log_data:
            self.add_rule("POWER", power_monitoring)

        # Add catch-all monitoring for any message type that has a GenericMessage handler
        def generic_message_monitoring(msg: any, _) -> Optional[tuple[float, str]]:
            """Monitor generic messages for debugging"""
            if hasattr(msg, "message_type") and hasattr(msg, "data"):
                return (msg.timestamp, f"Generic message: {msg.message_type}")
            return None

        # Note: We'll add this rule for any message type not explicitly handled above
        # This is done in the parse_events method when GenericMessage is used

    def add_rule(self, message_name: str, checker_func: EventChecker):
        """
        Registers a new event checker function for a specific message type.
        """
        if message_name not in self.event_checkers:
            self.event_checkers[message_name] = []
        self.event_checkers[message_name].append(checker_func)

    def parse_events(self) -> List[FlightEvent]:
        """
        Processes all messages chronologically and returns ALL detected events.
        Filtering is now only applied during display, not during event collection.
        """
        # 1. Combine all individual message dataframes into one large dataframe.
        all_messages = []
        for name, df in self.log_data.items():
            temp_df = df.copy()
            temp_df.reset_index(
                drop=True, inplace=True
            )  # Reset index to avoid duplicates

            # Remove duplicate columns (keep first occurrence)
            temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]

            temp_df["message_name"] = name
            all_messages.append(temp_df)

        if not all_messages:
            return []

        # Concatenate all dataframes
        combined_df = pd.concat(all_messages, ignore_index=True)
        combined_df.sort_values(by="timestamp", inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        # 2. Iterate through each message and apply the registered checker functions.
        events: List[FlightEvent] = []
        # Store the last seen object of each type for stateful comparisons.
        last_seen: Dict[str, Any] = {}

        # Process in chunks to avoid memory issues
        chunk_size = 10000
        total_rows = len(combined_df)

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = combined_df.iloc[start_idx:end_idx]

            for _, row in chunk.iterrows():
                message_name = row["message_name"]

                # Auto-register comprehensive monitoring for all message types
                if message_name not in self.event_checkers:
                    # Create intelligent event checkers based on message type patterns
                    def create_smart_checker(msg_type):
                        def smart_checker(
                            current_obj, prev_obj
                        ) -> Optional[tuple[float, str]]:
                            if hasattr(current_obj, "timestamp"):
                                # Different strategies for different message types
                                if any(
                                    keyword in msg_type.lower()
                                    for keyword in ["error", "fault", "fail", "warn"]
                                ):
                                    # Always log error/warning related messages
                                    return (
                                        current_obj.timestamp,
                                        f"{msg_type} event detected",
                                    )
                                elif any(
                                    keyword in msg_type.lower()
                                    for keyword in ["mode", "state", "status"]
                                ):
                                    # Log state changes with comparison to previous
                                    if (
                                        prev_obj
                                        and hasattr(current_obj, "data")
                                        and hasattr(prev_obj, "data")
                                    ):
                                        if current_obj.data != prev_obj.data:
                                            return (
                                                current_obj.timestamp,
                                                f"{msg_type} state change",
                                            )
                                    else:
                                        # Log first occurrence of state messages
                                        return (
                                            current_obj.timestamp,
                                            f"{msg_type} status update",
                                        )
                                elif any(
                                    keyword in msg_type.lower()
                                    for keyword in ["cmd", "actuator", "motor", "servo"]
                                ):
                                    # Sample control and actuator messages occasionally
                                    if (
                                        int(current_obj.timestamp * 10) % 50 == 0
                                    ):  # Every 5 seconds approx
                                        return (
                                            current_obj.timestamp,
                                            f"{msg_type} command update",
                                        )
                                elif any(
                                    keyword in msg_type.lower()
                                    for keyword in ["imu", "gyro", "accel", "mag"]
                                ):
                                    # Sample sensor data less frequently
                                    if (
                                        int(current_obj.timestamp * 10) % 100 == 0
                                    ):  # Every 10 seconds approx
                                        return (
                                            current_obj.timestamp,
                                            f"{msg_type} sensor data",
                                        )
                                elif any(
                                    keyword in msg_type.lower()
                                    for keyword in ["gps", "position", "nav"]
                                ):
                                    # Sample navigation data moderately
                                    if (
                                        int(current_obj.timestamp * 10) % 30 == 0
                                    ):  # Every 3 seconds approx
                                        return (
                                            current_obj.timestamp,
                                            f"{msg_type} navigation update",
                                        )
                                else:
                                    # For any other message type, sample very occasionally
                                    if (
                                        int(current_obj.timestamp * 10) % 200 == 0
                                    ):  # Every 20 seconds approx
                                        return (
                                            current_obj.timestamp,
                                            f"{msg_type} system message",
                                        )
                            return None

                        return smart_checker

                    self.add_rule(message_name, create_smart_checker(message_name))

                # Check if we have checkers registered for this message type.
                if message_name in self.event_checkers:
                    try:
                        # Get the corresponding data class constructor.
                        if message_name in MESSAGE_TO_CLASS_MAP:
                            DataClass = MESSAGE_TO_CLASS_MAP[message_name]
                            current_obj = DataClass.from_series(row)
                        else:
                            # Use GenericMessage for unknown message types
                            from .datastructures import GenericMessage

                            DataClass = GenericMessage
                            current_obj = DataClass.from_series(row, message_name)

                        # Get the previous object of the same type.
                        prev_obj = last_seen.get(message_name)

                        # Run all checker functions registered for this message type.
                        for checker in self.event_checkers[message_name]:
                            result = checker(current_obj, prev_obj)
                            if result:
                                timestamp, message = result
                                # Use the centralized tagging function to assign log level
                                event = tag_event(timestamp, message)
                                events.append(event)

                        # Update the last seen object for this message type.
                        last_seen[message_name] = current_obj
                    except Exception as e:
                        # Skip problematic rows but continue processing
                        continue

        return events

    def filter_events_by_level(
        self, events: List[FlightEvent], min_level: EventSeverity
    ) -> List[FlightEvent]:
        """
        Filter events based on log level hierarchy:
        - DEBUG: shows all events
        - INFO: shows INFO, WARNING, ERROR
        - WARNING: shows WARNING, ERROR
        - ERROR: shows only ERROR
        """
        if min_level == EventSeverity.DEBUG:
            return events
        elif min_level == EventSeverity.INFO:
            return [
                e
                for e in events
                if e.level
                in [EventSeverity.INFO, EventSeverity.WARNING, EventSeverity.ERROR]
            ]
        elif min_level == EventSeverity.WARNING:
            return [
                e
                for e in events
                if e.level in [EventSeverity.WARNING, EventSeverity.ERROR]
            ]
        elif min_level == EventSeverity.ERROR:
            return [e for e in events if e.level == EventSeverity.ERROR]
        else:
            return events

    @staticmethod
    def filter_events_by_level_static(
        events: List[FlightEvent], min_level: EventSeverity
    ) -> List[FlightEvent]:
        """
        Static version of filter_events_by_level for use without EventParser instance.
        """
        if min_level == EventSeverity.DEBUG:
            return events
        elif min_level == EventSeverity.INFO:
            return [
                e
                for e in events
                if e.level
                in [EventSeverity.INFO, EventSeverity.WARNING, EventSeverity.ERROR]
            ]
        elif min_level == EventSeverity.WARNING:
            return [
                e
                for e in events
                if e.level in [EventSeverity.WARNING, EventSeverity.ERROR]
            ]
        elif min_level == EventSeverity.ERROR:
            return [e for e in events if e.level == EventSeverity.ERROR]
        else:
            return events

    @staticmethod
    def filter_events_by_exact_level(
        events: List[FlightEvent], exact_level: EventSeverity
    ) -> List[FlightEvent]:
        """
        Filter events to show only events of the exact specified level.
        Used for individual log level files (log_debug.json, log_info.json, etc.)
        """
        return [e for e in events if e.level == exact_level]
