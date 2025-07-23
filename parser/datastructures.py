from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd


# Base class to ensure all data structures have a timestamp
@dataclass
class PaparazziMessage:
    timestamp: float


# --- Sensor Messages ---


@dataclass
class Attitude(PaparazziMessage):
    phi: float
    theta: float
    psi: float

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            phi=data.get("phi"),
            theta=data.get("theta"),
            psi=data.get("psi"),
        )


@dataclass
class GPS(PaparazziMessage):
    mode: int
    utm_east: float
    utm_north: float
    course: int
    alt: int
    speed: int
    climb: int
    week: int
    itow: int
    hmsl: int
    h_acc: int
    v_acc: int
    nb_sats: int
    gspeed: float
    pdop: int
    hdop: int
    vdop: int
    hmsl_acc: float
    vel_acc: float


@dataclass
class GPSInt(PaparazziMessage):
    ecef_x: float
    ecef_y: float
    ecef_z: float
    lat: float
    lon: float
    alt: float
    hmsl: float
    ecef_xd: float
    ecef_yd: float
    ecef_zd: float
    pacc: float
    sacc: float
    tow: float
    pdop: float
    numsv: int
    fix: int
    comp_id: int

    @property
    def mode(self) -> int:
        """Compatibility property for fix mode."""
        return self.fix

    @property
    def nb_sats(self) -> int:
        """Compatibility property for number of satellites."""
        return self.numsv

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            ecef_x=data.get("ecef_x"),
            ecef_y=data.get("ecef_y"),
            ecef_z=data.get("ecef_z"),
            lat=data.get("lat"),
            lon=data.get("lon"),
            alt=data.get("alt"),
            hmsl=data.get("hmsl"),
            ecef_xd=data.get("ecef_xd"),
            ecef_yd=data.get("ecef_yd"),
            ecef_zd=data.get("ecef_zd"),
            pacc=data.get("pacc"),
            sacc=data.get("sacc"),
            tow=data.get("tow"),
            pdop=data.get("pdop"),
            numsv=data.get("numsv"),
            fix=data.get("fix"),
            comp_id=data.get("comp_id"),
        )


@dataclass
class IMU(PaparazziMessage):
    p: float
    q: float
    r: float
    ax: float
    ay: float
    az: float

    @classmethod
    def from_series(cls, data: pd.Series, prefix=""):
        return cls(
            timestamp=data.get("timestamp"),
            p=data.get(f"{prefix}p"),
            q=data.get(f"{prefix}q"),
            r=data.get(f"{prefix}r"),
            ax=data.get(f"{prefix}ax"),
            ay=data.get(f"{prefix}ay"),
            az=data.get(f"{prefix}az"),
        )


@dataclass
class Airspeed(PaparazziMessage):
    airspeed: float
    differential_pressure: float
    scaled_pressure: float
    temperature: float

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(**{k: data.get(k) for k in cls.__annotations__})


# --- Power and System Messages ---


@dataclass
class Energy(PaparazziMessage):
    voltage: float
    current: float

    @property
    def power(self) -> float:
        return self.voltage * self.current

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            voltage=data.get("voltage", 0.0),
            current=data.get("current", 0.0),
        )


@dataclass
class Power(PaparazziMessage):
    values: List[int] = field(default_factory=list)

    @classmethod
    def from_series(cls, data: pd.Series):
        power_values = [
            int(data.get(f"p{i+1}", 0)) for i in range(8)
        ]  # Assuming up to p8
        return cls(timestamp=data.get("timestamp"), values=power_values)


@dataclass
class ESC(PaparazziMessage):
    motor_id: int
    rpm: int
    motor_volts: float
    amps: float
    power: int
    temperature: int

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(**{k: data.get(k) for k in cls.__annotations__})


@dataclass
class I2CErrors(PaparazziMessage):
    bus_number: int
    ack_failures: int

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            bus_number=int(data.get("bus_number", -1)),
            ack_failures=int(data.get("acknowledge_failure_cnt", 0)),
        )


@dataclass
class RotorcraftStatus(PaparazziMessage):
    flight_mode: int
    vehicle_mode: int
    failsafe_mode: int

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            flight_mode=int(data.get("flight_mode", 0)),
            vehicle_mode=int(data.get("vehicle_mode", 0)),
            failsafe_mode=int(data.get("failsafe_mode", 0)),
        )


@dataclass
class GroundDetect(PaparazziMessage):
    ground_proximity: int
    throttle_low: bool

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            ground_proximity=data.get("ground_proximity"),
            throttle_low=bool(data.get("throttle_low")),
        )


# --- Estimation and Control Messages ---


@dataclass
class AHRSQuat(PaparazziMessage):
    body_qi: float
    body_qx: float
    body_qy: float
    body_qz: float
    ref_qi: Optional[float] = None
    ref_qx: Optional[float] = None
    ref_qy: Optional[float] = None
    ref_qz: Optional[float] = None

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(**{k: data.get(k) for k in cls.__annotations__})


@dataclass
class EKF2State(PaparazziMessage):
    px: float
    py: float
    pz: float
    vx: float
    vy: float
    vz: float
    q0: float
    q1: float
    q2: float
    q3: float
    gbx: float
    gby: float
    gbz: float

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(**{k: data.get(k) for k in cls.__annotations__})


@dataclass
class StabAttitude(PaparazziMessage):
    roll: float
    pitch: float
    yaw: float
    roll_ref: float
    pitch_ref: float
    yaw_ref: float

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(**{k: data.get(k) for k in cls.__annotations__})


@dataclass
class RotorcraftCmd(PaparazziMessage):
    roll: int
    pitch: int
    yaw: int
    thrust: int

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(**{k: data.get(k) for k in cls.__annotations__})


@dataclass
class RotorcraftFP(PaparazziMessage):
    east: float
    north: float
    up: float
    veast: float
    vnorth: float
    vup: float
    phi: float
    theta: float
    psi: float
    pe: float
    pn: float
    pup: float
    pde: float
    pdn: float
    pdup: float
    ae: float
    an: float
    aup: float
    pdpsi: float

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            east=data.get("east"),
            north=data.get("north"),
            up=data.get("up"),
            veast=data.get("veast"),
            vnorth=data.get("vnorth"),
            vup=data.get("vup"),
            phi=data.get("phi"),
            theta=data.get("theta"),
            psi=data.get("psi"),
            pe=data.get("pe"),
            pn=data.get("pn"),
            pup=data.get("pup"),
            pde=data.get("pde"),
            pdn=data.get("pdn"),
            pdup=data.get("pdup"),
            ae=data.get("ae"),
            an=data.get("an"),
            aup=data.get("aup"),
            pdpsi=data.get("pdpsi"),
        )


@dataclass
class IndiRotwing(PaparazziMessage):
    p_dot_ref: float
    q_dot_ref: float
    r_dot_ref: float
    p_dot_mes: float
    q_dot_mes: float
    r_dot_mes: float
    u_l: float
    u_m: float
    u_n: float
    delta_u_l: float
    delta_u_m: float
    delta_u_n: float

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(**{k: data.get(k) for k in cls.__annotations__})


@dataclass
class RotwingState(PaparazziMessage):
    rpm_ref: float
    rpm_mes: float
    tilt_angle_ref: float
    tilt_angle_mes: float
    thrust_ref: float
    thrust_mes: float
    q_slash_q_max: float
    airspeed: float
    theta_mes: float

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(**{k: data.get(k) for k in cls.__annotations__})


# --- Additional Message Types ---


@dataclass
class AirData(PaparazziMessage):
    """Air data sensor readings"""

    pressure: float = 0.0
    temperature: float = 0.0
    airspeed: float = 0.0
    altitude: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            pressure=data.get("pressure", 0.0),
            temperature=data.get("temperature", 0.0),
            airspeed=data.get("airspeed", 0.0),
            altitude=data.get("altitude", 0.0),
        )


@dataclass
class Actuators(PaparazziMessage):
    """Actuator commands and feedback"""

    motor1: float = 0.0
    motor2: float = 0.0
    motor3: float = 0.0
    motor4: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            motor1=data.get("motor1", 0.0),
            motor2=data.get("motor2", 0.0),
            motor3=data.get("motor3", 0.0),
            motor4=data.get("motor4", 0.0),
        )


@dataclass
class BebopActuators(PaparazziMessage):
    """Bebop-specific actuator commands"""

    motor_front_left: float = 0.0
    motor_front_right: float = 0.0
    motor_back_right: float = 0.0
    motor_back_left: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            motor_front_left=data.get("motor_front_left", 0.0),
            motor_front_right=data.get("motor_front_right", 0.0),
            motor_back_right=data.get("motor_back_right", 0.0),
            motor_back_left=data.get("motor_back_left", 0.0),
        )


@dataclass
class DatalinkReport(PaparazziMessage):
    """Datalink quality and status"""

    rx_bytes: int = 0
    tx_bytes: int = 0
    link_quality: float = 0.0
    rssi: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            rx_bytes=data.get("rx_bytes", 0),
            tx_bytes=data.get("tx_bytes", 0),
            link_quality=data.get("link_quality", 0.0),
            rssi=data.get("rssi", 0.0),
        )


@dataclass
class AutopilotVersion(PaparazziMessage):
    """Autopilot version information"""

    version: str = ""
    capabilities: int = 0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            version=str(data.get("version", "")),
            capabilities=data.get("capabilities", 0),
        )


@dataclass
class Alive(PaparazziMessage):
    """Heartbeat/alive message"""

    counter: int = 0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            counter=data.get("counter", 0),
        )


@dataclass
class DLValue(PaparazziMessage):
    """Downlink value message"""

    id: int = 0
    value: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            id=data.get("id", 0),
            value=data.get("value", 0.0),
        )


@dataclass
class RotorcraftNavStatus(PaparazziMessage):
    """Navigation status for rotorcraft"""

    block_time: float = 0.0
    stage_time: float = 0.0
    dist_to_wp: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            block_time=data.get("block_time", 0.0),
            stage_time=data.get("stage_time", 0.0),
            dist_to_wp=data.get("dist_to_wp", 0.0),
        )


@dataclass
class WPMoved(PaparazziMessage):
    """Waypoint moved message"""

    wp_id: int = 0
    utm_east: float = 0.0
    utm_north: float = 0.0
    alt: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            wp_id=data.get("wp_id", 0),
            utm_east=data.get("utm_east", 0.0),
            utm_north=data.get("utm_north", 0.0),
            alt=data.get("alt", 0.0),
        )


@dataclass
class StateFilterStatus(PaparazziMessage):
    """State filter status"""

    id: int = 0
    health: int = 0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            id=data.get("id", 0),
            health=data.get("health", 0),
        )


@dataclass
class UartErrors(PaparazziMessage):
    """UART communication errors"""

    overrun_err: int = 0
    framing_err: int = 0
    noise_err: int = 0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            overrun_err=data.get("overrun_err", 0),
            framing_err=data.get("framing_err", 0),
            noise_err=data.get("noise_err", 0),
        )


@dataclass
class InsRef(PaparazziMessage):
    """INS reference frame"""

    ecef_x: float = 0.0
    ecef_y: float = 0.0
    ecef_z: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            ecef_x=data.get("ecef_x", 0.0),
            ecef_y=data.get("ecef_y", 0.0),
            ecef_z=data.get("ecef_z", 0.0),
        )


@dataclass
class Ins(PaparazziMessage):
    """INS state estimates"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0),
            vx=data.get("vx", 0.0),
            vy=data.get("vy", 0.0),
            vz=data.get("vz", 0.0),
        )


@dataclass
class Survey(PaparazziMessage):
    """Survey mission data"""

    east: float = 0.0
    north: float = 0.0
    west: float = 0.0
    south: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            east=data.get("east", 0.0),
            north=data.get("north", 0.0),
            west=data.get("west", 0.0),
            south=data.get("south", 0.0),
        )


@dataclass
class GuidanceIndiHybrid(PaparazziMessage):
    """INDI hybrid guidance system data"""

    pos_err_x: float = 0.0
    pos_err_y: float = 0.0
    pos_err_z: float = 0.0
    vel_ref_x: float = 0.0
    vel_ref_y: float = 0.0
    vel_ref_z: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            pos_err_x=data.get("pos_err_x", 0.0),
            pos_err_y=data.get("pos_err_y", 0.0),
            pos_err_z=data.get("pos_err_z", 0.0),
            vel_ref_x=data.get("vel_ref_x", 0.0),
            vel_ref_y=data.get("vel_ref_y", 0.0),
            vel_ref_z=data.get("vel_ref_z", 0.0),
        )


@dataclass
class Guidance(PaparazziMessage):
    """General guidance system data"""

    indi_cmd_x: float = 0.0
    indi_cmd_y: float = 0.0
    indi_cmd_z: float = 0.0
    ref_x: float = 0.0
    ref_y: float = 0.0
    ref_z: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            indi_cmd_x=data.get("indi_cmd_x", 0.0),
            indi_cmd_y=data.get("indi_cmd_y", 0.0),
            indi_cmd_z=data.get("indi_cmd_z", 0.0),
            ref_x=data.get("ref_x", 0.0),
            ref_y=data.get("ref_y", 0.0),
            ref_z=data.get("ref_z", 0.0),
        )


@dataclass
class ExternalPoseDown(PaparazziMessage):
    """External pose estimation data"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0),
            qx=data.get("qx", 0.0),
            qy=data.get("qy", 0.0),
            qz=data.get("qz", 0.0),
            qw=data.get("qw", 1.0),
        )


@dataclass
class SerialActT4In(PaparazziMessage):
    """Serial actuator T4 input data"""

    values: List[int] = field(default_factory=list)

    @classmethod
    def from_series(cls, data: pd.Series):
        # Handle array values
        values = []
        for i in range(10):  # Assume up to 10 values
            val = data.get(f"values_{i}")
            if val is not None:
                values.append(int(val))
        return cls(timestamp=data.get("timestamp"), values=values)


@dataclass
class SerialActT4Out(PaparazziMessage):
    """Serial actuator T4 output data"""

    values: List[int] = field(default_factory=list)

    @classmethod
    def from_series(cls, data: pd.Series):
        # Handle array values
        values = []
        for i in range(10):  # Assume up to 10 values
            val = data.get(f"values_{i}")
            if val is not None:
                values.append(int(val))
        return cls(timestamp=data.get("timestamp"), values=values)


@dataclass
class PowerDevice(PaparazziMessage):
    """Power device monitoring data"""

    current: float = 0.0
    voltage: float = 0.0
    power: float = 0.0
    energy: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            current=data.get("current", 0.0),
            voltage=data.get("voltage", 0.0),
            power=data.get("power", 0.0),
            energy=data.get("energy", 0.0),
        )


@dataclass
class EffMat(PaparazziMessage):
    """Control effectiveness matrix data"""

    motor_cmd: List[float] = field(default_factory=list)
    motor_rpm: List[float] = field(default_factory=list)

    @classmethod
    def from_series(cls, data: pd.Series):
        # Handle effectiveness matrix values
        motor_cmd = []
        motor_rpm = []
        for i in range(8):  # Assume up to 8 motors
            cmd_val = data.get(f"motor_cmd_{i}")
            rpm_val = data.get(f"motor_rpm_{i}")
            if cmd_val is not None:
                motor_cmd.append(float(cmd_val))
            if rpm_val is not None:
                motor_rpm.append(float(rpm_val))
        return cls(
            timestamp=data.get("timestamp"), motor_cmd=motor_cmd, motor_rpm=motor_rpm
        )


@dataclass
class EKF2PDiag(PaparazziMessage):
    """EKF2 covariance diagonal elements"""

    pos_var_x: float = 0.0
    pos_var_y: float = 0.0
    pos_var_z: float = 0.0
    vel_var_x: float = 0.0
    vel_var_y: float = 0.0
    vel_var_z: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            pos_var_x=data.get("pos_var_x", 0.0),
            pos_var_y=data.get("pos_var_y", 0.0),
            pos_var_z=data.get("pos_var_z", 0.0),
            vel_var_x=data.get("vel_var_x", 0.0),
            vel_var_y=data.get("vel_var_y", 0.0),
            vel_var_z=data.get("vel_var_z", 0.0),
        )


@dataclass
class EKF2Innov(PaparazziMessage):
    """EKF2 innovation monitoring"""

    innov_x: float = 0.0
    innov_y: float = 0.0
    innov_z: float = 0.0
    innov_vx: float = 0.0
    innov_vy: float = 0.0
    innov_vz: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            innov_x=data.get("innov_x", 0.0),
            innov_y=data.get("innov_y", 0.0),
            innov_z=data.get("innov_z", 0.0),
            innov_vx=data.get("innov_vx", 0.0),
            innov_vy=data.get("innov_vy", 0.0),
            innov_vz=data.get("innov_vz", 0.0),
        )


# --- Additional Flight Monitoring Messages ---


@dataclass
class StabilizationAttitude(PaparazziMessage):
    """Attitude stabilization control data"""

    phi_pgain: float = 0.0
    theta_pgain: float = 0.0
    psi_pgain: float = 0.0
    phi_dgain: float = 0.0
    theta_dgain: float = 0.0
    psi_dgain: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            phi_pgain=data.get("phi_pgain", 0.0),
            theta_pgain=data.get("theta_pgain", 0.0),
            psi_pgain=data.get("psi_pgain", 0.0),
            phi_dgain=data.get("phi_dgain", 0.0),
            theta_dgain=data.get("theta_dgain", 0.0),
            psi_dgain=data.get("psi_dgain", 0.0),
        )


@dataclass
class NavStatus(PaparazziMessage):
    """Navigation system status"""

    cur_block: int = 0
    cur_stage: int = 0
    block_time: float = 0.0
    stage_time: float = 0.0
    nav_mode: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            cur_block=data.get("cur_block", 0),
            cur_stage=data.get("cur_stage", 0),
            block_time=data.get("block_time", 0.0),
            stage_time=data.get("stage_time", 0.0),
            nav_mode=data.get("nav_mode", ""),
        )


@dataclass
class Waypoint(PaparazziMessage):
    """Waypoint management information"""

    wp_id: int = 0
    wp_x: float = 0.0
    wp_y: float = 0.0
    wp_z: float = 0.0
    wp_a: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            wp_id=data.get("wp_id", 0),
            wp_x=data.get("wp_x", 0.0),
            wp_y=data.get("wp_y", 0.0),
            wp_z=data.get("wp_z", 0.0),
            wp_a=data.get("wp_a", 0.0),
        )


@dataclass
class FlightPlan(PaparazziMessage):
    """Flight plan execution data"""

    fp_block: int = 0
    fp_stage: int = 0
    fp_kill: bool = False
    fp_mode: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            fp_block=data.get("fp_block", 0),
            fp_stage=data.get("fp_stage", 0),
            fp_kill=data.get("fp_kill", False),
            fp_mode=data.get("fp_mode", ""),
        )


@dataclass
class RCLost(PaparazziMessage):
    """RC signal loss detection"""

    rc_lost: bool = False
    rc_timeout: float = 0.0
    last_valid_rc: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            rc_lost=data.get("rc_lost", False),
            rc_timeout=data.get("rc_timeout", 0.0),
            last_valid_rc=data.get("last_valid_rc", 0.0),
        )


@dataclass
class DatalinkLost(PaparazziMessage):
    """Datalink communication loss"""

    dl_lost: bool = False
    dl_timeout: float = 0.0
    last_valid_dl: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            dl_lost=data.get("dl_lost", False),
            dl_timeout=data.get("dl_timeout", 0.0),
            last_valid_dl=data.get("last_valid_dl", 0.0),
        )


@dataclass
class Geofence(PaparazziMessage):
    """Geofence violation monitoring"""

    inside_fence: bool = True
    fence_violation: bool = False
    distance_to_fence: float = 0.0
    fence_type: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            inside_fence=data.get("inside_fence", True),
            fence_violation=data.get("fence_violation", False),
            distance_to_fence=data.get("distance_to_fence", 0.0),
            fence_type=data.get("fence_type", ""),
        )


@dataclass
class Weather(PaparazziMessage):
    """Weather monitoring data"""

    wind_speed: float = 0.0
    wind_dir: float = 0.0
    temperature: float = 0.0
    humidity: float = 0.0
    pressure: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            wind_speed=data.get("wind_speed", 0.0),
            wind_dir=data.get("wind_dir", 0.0),
            temperature=data.get("temperature", 0.0),
            humidity=data.get("humidity", 0.0),
            pressure=data.get("pressure", 0.0),
        )


@dataclass
class WindEstimation(PaparazziMessage):
    """Wind estimation algorithms"""

    wind_north: float = 0.0
    wind_east: float = 0.0
    wind_up: float = 0.0
    wind_confidence: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            wind_north=data.get("wind_north", 0.0),
            wind_east=data.get("wind_east", 0.0),
            wind_up=data.get("wind_up", 0.0),
            wind_confidence=data.get("wind_confidence", 0.0),
        )


@dataclass
class BatteryStatus(PaparazziMessage):
    """Detailed battery monitoring"""

    cell1_voltage: float = 0.0
    cell2_voltage: float = 0.0
    cell3_voltage: float = 0.0
    cell4_voltage: float = 0.0
    total_voltage: float = 0.0
    current: float = 0.0
    consumed: float = 0.0
    remaining: float = 0.0
    temperature: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            cell1_voltage=data.get("cell1_voltage", 0.0),
            cell2_voltage=data.get("cell2_voltage", 0.0),
            cell3_voltage=data.get("cell3_voltage", 0.0),
            cell4_voltage=data.get("cell4_voltage", 0.0),
            total_voltage=data.get("total_voltage", 0.0),
            current=data.get("current", 0.0),
            consumed=data.get("consumed", 0.0),
            remaining=data.get("remaining", 0.0),
            temperature=data.get("temperature", 0.0),
        )


@dataclass
class MotorStatus(PaparazziMessage):
    """Individual motor health monitoring"""

    motor_id: int = 0
    rpm: float = 0.0
    temperature: float = 0.0
    current: float = 0.0
    voltage: float = 0.0
    vibration: float = 0.0
    error_flags: int = 0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            motor_id=data.get("motor_id", 0),
            rpm=data.get("rpm", 0.0),
            temperature=data.get("temperature", 0.0),
            current=data.get("current", 0.0),
            voltage=data.get("voltage", 0.0),
            vibration=data.get("vibration", 0.0),
            error_flags=data.get("error_flags", 0),
        )


@dataclass
class Vibration(PaparazziMessage):
    """Vibration monitoring system"""

    accel_x_rms: float = 0.0
    accel_y_rms: float = 0.0
    accel_z_rms: float = 0.0
    gyro_x_rms: float = 0.0
    gyro_y_rms: float = 0.0
    gyro_z_rms: float = 0.0
    clip_count: int = 0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            accel_x_rms=data.get("accel_x_rms", 0.0),
            accel_y_rms=data.get("accel_y_rms", 0.0),
            accel_z_rms=data.get("accel_z_rms", 0.0),
            gyro_x_rms=data.get("gyro_x_rms", 0.0),
            gyro_y_rms=data.get("gyro_y_rms", 0.0),
            gyro_z_rms=data.get("gyro_z_rms", 0.0),
            clip_count=data.get("clip_count", 0),
        )


@dataclass
class CompassCal(PaparazziMessage):
    """Compass calibration status"""

    cal_progress: float = 0.0
    cal_fitness: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            cal_progress=data.get("cal_progress", 0.0),
            cal_fitness=data.get("cal_fitness", 0.0),
            offset_x=data.get("offset_x", 0.0),
            offset_y=data.get("offset_y", 0.0),
            offset_z=data.get("offset_z", 0.0),
            scale_x=data.get("scale_x", 1.0),
            scale_y=data.get("scale_y", 1.0),
            scale_z=data.get("scale_z", 1.0),
        )


@dataclass
class Barometer(PaparazziMessage):
    """Barometric pressure monitoring"""

    pressure: float = 0.0
    altitude: float = 0.0
    temperature: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            pressure=data.get("pressure", 0.0),
            altitude=data.get("altitude", 0.0),
            temperature=data.get("temperature", 0.0),
        )


@dataclass
class Temperature(PaparazziMessage):
    """Temperature monitoring system"""

    cpu_temp: float = 0.0
    imu_temp: float = 0.0
    baro_temp: float = 0.0
    motor_temp: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            cpu_temp=data.get("cpu_temp", 0.0),
            imu_temp=data.get("imu_temp", 0.0),
            baro_temp=data.get("baro_temp", 0.0),
            motor_temp=data.get("motor_temp", 0.0),
        )


# --- Safety-Critical Message Types ---


@dataclass
class Emergency(PaparazziMessage):
    """Emergency situations"""

    emergency_type: str = ""
    severity: int = 0
    description: str = ""
    recovery_action: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            emergency_type=data.get("emergency_type", ""),
            severity=data.get("severity", 0),
            description=data.get("description", ""),
            recovery_action=data.get("recovery_action", ""),
        )


@dataclass
class GeofenceBreach(PaparazziMessage):
    """Geofence violations"""

    breach_type: str = ""
    fence_id: int = 0
    distance_from_fence: float = 0.0
    breach_time: float = 0.0
    recovery_required: bool = False

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            breach_type=data.get("breach_type", ""),
            fence_id=data.get("fence_id", 0),
            distance_from_fence=data.get("distance_from_fence", 0.0),
            breach_time=data.get("breach_time", 0.0),
            recovery_required=data.get("recovery_required", False),
        )


@dataclass
class CollisionAvoidance(PaparazziMessage):
    """Collision detection and avoidance"""

    threat_type: str = ""
    threat_distance: float = 0.0
    threat_bearing: float = 0.0
    threat_altitude: float = 0.0
    avoidance_action: str = ""
    threat_id: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            threat_type=data.get("threat_type", ""),
            threat_distance=data.get("threat_distance", 0.0),
            threat_bearing=data.get("threat_bearing", 0.0),
            threat_altitude=data.get("threat_altitude", 0.0),
            avoidance_action=data.get("avoidance_action", ""),
            threat_id=data.get("threat_id", ""),
        )


@dataclass
class Traffic(PaparazziMessage):
    """Traffic advisory system"""

    traffic_id: str = ""
    relative_distance: float = 0.0
    relative_bearing: float = 0.0
    relative_altitude: float = 0.0
    advisory_level: int = 0
    aircraft_type: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            traffic_id=data.get("traffic_id", ""),
            relative_distance=data.get("relative_distance", 0.0),
            relative_bearing=data.get("relative_bearing", 0.0),
            relative_altitude=data.get("relative_altitude", 0.0),
            advisory_level=data.get("advisory_level", 0),
            aircraft_type=data.get("aircraft_type", ""),
        )


@dataclass
class TerrainFollowing(PaparazziMessage):
    """Terrain awareness and following"""

    ground_clearance: float = 0.0
    terrain_height: float = 0.0
    follow_mode: str = ""
    terrain_warning: bool = False
    minimum_clearance: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            ground_clearance=data.get("ground_clearance", 0.0),
            terrain_height=data.get("terrain_height", 0.0),
            follow_mode=data.get("follow_mode", ""),
            terrain_warning=data.get("terrain_warning", False),
            minimum_clearance=data.get("minimum_clearance", 0.0),
        )


@dataclass
class ObstacleDetection(PaparazziMessage):
    """Obstacle detection and avoidance"""

    obstacle_distance: float = 0.0
    obstacle_bearing: float = 0.0
    obstacle_height: float = 0.0
    obstacle_type: str = ""
    avoidance_required: bool = False
    sensor_type: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            obstacle_distance=data.get("obstacle_distance", 0.0),
            obstacle_bearing=data.get("obstacle_bearing", 0.0),
            obstacle_height=data.get("obstacle_height", 0.0),
            obstacle_type=data.get("obstacle_type", ""),
            avoidance_required=data.get("avoidance_required", False),
            sensor_type=data.get("sensor_type", ""),
        )


@dataclass
class LossOfControl(PaparazziMessage):
    """Control loss detection"""

    control_loss_type: str = ""
    affected_axis: str = ""
    loss_magnitude: float = 0.0
    recovery_attempted: bool = False
    backup_system_active: bool = False

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            control_loss_type=data.get("control_loss_type", ""),
            affected_axis=data.get("affected_axis", ""),
            loss_magnitude=data.get("loss_magnitude", 0.0),
            recovery_attempted=data.get("recovery_attempted", False),
            backup_system_active=data.get("backup_system_active", False),
        )


@dataclass
class StallWarning(PaparazziMessage):
    """Aerodynamic stall warning"""

    stall_probability: float = 0.0
    angle_of_attack: float = 0.0
    airspeed: float = 0.0
    stall_margin: float = 0.0
    warning_level: int = 0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            stall_probability=data.get("stall_probability", 0.0),
            angle_of_attack=data.get("angle_of_attack", 0.0),
            airspeed=data.get("airspeed", 0.0),
            stall_margin=data.get("stall_margin", 0.0),
            warning_level=data.get("warning_level", 0),
        )


@dataclass
class OverSpeed(PaparazziMessage):
    """Speed limit violations"""

    current_speed: float = 0.0
    speed_limit: float = 0.0
    overspeed_margin: float = 0.0
    limit_type: str = ""
    warning_level: int = 0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            current_speed=data.get("current_speed", 0.0),
            speed_limit=data.get("speed_limit", 0.0),
            overspeed_margin=data.get("overspeed_margin", 0.0),
            limit_type=data.get("limit_type", ""),
            warning_level=data.get("warning_level", 0),
        )


@dataclass
class AltitudeLimit(PaparazziMessage):
    """Altitude restrictions and violations"""

    current_altitude: float = 0.0
    altitude_limit: float = 0.0
    altitude_margin: float = 0.0
    limit_type: str = ""
    violation_severity: int = 0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            current_altitude=data.get("current_altitude", 0.0),
            altitude_limit=data.get("altitude_limit", 0.0),
            altitude_margin=data.get("altitude_margin", 0.0),
            limit_type=data.get("limit_type", ""),
            violation_severity=data.get("violation_severity", 0),
        )


# --- Communication and Telemetry Message Types ---


@dataclass
class TelemetryStatus(PaparazziMessage):
    """Telemetry health monitoring"""

    telemetry_rate: float = 0.0
    expected_rate: float = 0.0
    health_status: str = ""
    error_count: int = 0
    last_message_time: float = 0.0
    connection_status: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            telemetry_rate=data.get("telemetry_rate", 0.0),
            expected_rate=data.get("expected_rate", 0.0),
            health_status=data.get("health_status", ""),
            error_count=data.get("error_count", 0),
            last_message_time=data.get("last_message_time", 0.0),
            connection_status=data.get("connection_status", ""),
        )


@dataclass
class RadioStatus(PaparazziMessage):
    """Radio communication status"""

    radio_frequency: float = 0.0
    tx_power: float = 0.0
    rx_power: float = 0.0
    noise_floor: float = 0.0
    channel_status: str = ""
    modulation_type: str = ""
    error_rate: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            radio_frequency=data.get("radio_frequency", 0.0),
            tx_power=data.get("tx_power", 0.0),
            rx_power=data.get("rx_power", 0.0),
            noise_floor=data.get("noise_floor", 0.0),
            channel_status=data.get("channel_status", ""),
            modulation_type=data.get("modulation_type", ""),
            error_rate=data.get("error_rate", 0.0),
        )


@dataclass
class ModemStatus(PaparazziMessage):
    """Modem connectivity status"""

    modem_type: str = ""
    connection_state: str = ""
    signal_quality: int = 0
    network_registration: str = ""
    data_session_status: str = ""
    bytes_sent: int = 0
    bytes_received: int = 0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            modem_type=data.get("modem_type", ""),
            connection_state=data.get("connection_state", ""),
            signal_quality=data.get("signal_quality", 0),
            network_registration=data.get("network_registration", ""),
            data_session_status=data.get("data_session_status", ""),
            bytes_sent=data.get("bytes_sent", 0),
            bytes_received=data.get("bytes_received", 0),
        )


@dataclass
class LinkQuality(PaparazziMessage):
    """Communication link quality metrics"""

    link_quality_percent: float = 0.0
    signal_to_noise: float = 0.0
    bit_error_rate: float = 0.0
    frame_error_rate: float = 0.0
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    throughput_bps: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            link_quality_percent=data.get("link_quality_percent", 0.0),
            signal_to_noise=data.get("signal_to_noise", 0.0),
            bit_error_rate=data.get("bit_error_rate", 0.0),
            frame_error_rate=data.get("frame_error_rate", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            jitter_ms=data.get("jitter_ms", 0.0),
            throughput_bps=data.get("throughput_bps", 0.0),
        )


@dataclass
class PacketLoss(PaparazziMessage):
    """Data packet loss monitoring"""

    packets_sent: int = 0
    packets_received: int = 0
    packets_lost: int = 0
    loss_percentage: float = 0.0
    consecutive_losses: int = 0
    recovery_time: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            packets_sent=data.get("packets_sent", 0),
            packets_received=data.get("packets_received", 0),
            packets_lost=data.get("packets_lost", 0),
            loss_percentage=data.get("loss_percentage", 0.0),
            consecutive_losses=data.get("consecutive_losses", 0),
            recovery_time=data.get("recovery_time", 0.0),
        )


@dataclass
class RSSILow(PaparazziMessage):
    """Signal strength warnings"""

    current_rssi: float = 0.0
    minimum_rssi: float = 0.0
    rssi_margin: float = 0.0
    warning_level: int = 0
    frequency_band: str = ""
    antenna_status: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            current_rssi=data.get("current_rssi", 0.0),
            minimum_rssi=data.get("minimum_rssi", 0.0),
            rssi_margin=data.get("rssi_margin", 0.0),
            warning_level=data.get("warning_level", 0),
            frequency_band=data.get("frequency_band", ""),
            antenna_status=data.get("antenna_status", ""),
        )


@dataclass
class CurrentSpike(PaparazziMessage):
    """Current consumption spikes"""

    current: float = 0.0
    peak_current: float = 0.0
    spike_duration: float = 0.0
    threshold: float = 0.0
    source_device: str = ""
    power_rail: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            current=data.get("current", 0.0),
            peak_current=data.get("peak_current", 0.0),
            spike_duration=data.get("spike_duration", 0.0),
            threshold=data.get("threshold", 0.0),
            source_device=data.get("source_device", ""),
            power_rail=data.get("power_rail", ""),
        )


@dataclass
class PowerDistribution(PaparazziMessage):
    """Power rail monitoring"""

    rail_3v3_voltage: float = 0.0
    rail_5v_voltage: float = 0.0
    rail_12v_voltage: float = 0.0
    rail_3v3_current: float = 0.0
    rail_5v_current: float = 0.0
    rail_12v_current: float = 0.0
    total_power: float = 0.0
    efficiency: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            rail_3v3_voltage=data.get("rail_3v3_voltage", 0.0),
            rail_5v_voltage=data.get("rail_5v_voltage", 0.0),
            rail_12v_voltage=data.get("rail_12v_voltage", 0.0),
            rail_3v3_current=data.get("rail_3v3_current", 0.0),
            rail_5v_current=data.get("rail_5v_current", 0.0),
            rail_12v_current=data.get("rail_12v_current", 0.0),
            total_power=data.get("total_power", 0.0),
            efficiency=data.get("efficiency", 0.0),
        )


@dataclass
class ChargingStatus(PaparazziMessage):
    """Battery charging"""

    charging_state: str = ""
    charge_current: float = 0.0
    charge_voltage: float = 0.0
    charge_percentage: float = 0.0
    time_remaining: float = 0.0
    charger_temperature: float = 0.0
    charging_power: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            charging_state=data.get("charging_state", ""),
            charge_current=data.get("charge_current", 0.0),
            charge_voltage=data.get("charge_voltage", 0.0),
            charge_percentage=data.get("charge_percentage", 0.0),
            time_remaining=data.get("time_remaining", 0.0),
            charger_temperature=data.get("charger_temperature", 0.0),
            charging_power=data.get("charging_power", 0.0),
        )


@dataclass
class CellBalance(PaparazziMessage):
    """Battery cell balancing"""

    cell_count: int = 0
    cell_voltage_min: float = 0.0
    cell_voltage_max: float = 0.0
    voltage_imbalance: float = 0.0
    balancing_active: bool = False
    balance_current: float = 0.0
    balance_time: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            cell_count=data.get("cell_count", 0),
            cell_voltage_min=data.get("cell_voltage_min", 0.0),
            cell_voltage_max=data.get("cell_voltage_max", 0.0),
            voltage_imbalance=data.get("voltage_imbalance", 0.0),
            balancing_active=bool(data.get("balancing_active", 0)),
            balance_current=data.get("balance_current", 0.0),
            balance_time=data.get("balance_time", 0.0),
        )


@dataclass
class ThermalThrottle(PaparazziMessage):
    """Thermal protection"""

    temperature: float = 0.0
    throttle_level: float = 0.0
    thermal_state: str = ""
    max_temperature: float = 0.0
    cooling_active: bool = False
    throttle_reason: str = ""
    recovery_time: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            temperature=data.get("temperature", 0.0),
            throttle_level=data.get("throttle_level", 0.0),
            thermal_state=data.get("thermal_state", ""),
            max_temperature=data.get("max_temperature", 0.0),
            cooling_active=bool(data.get("cooling_active", 0)),
            throttle_reason=data.get("throttle_reason", ""),
            recovery_time=data.get("recovery_time", 0.0),
        )


@dataclass
class MissionItem(PaparazziMessage):
    """Mission execution"""

    waypoint_id: int = 0
    mission_type: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    heading: float = 0.0
    completion_status: str = ""
    execution_time: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            waypoint_id=data.get("waypoint_id", 0),
            mission_type=data.get("mission_type", ""),
            latitude=data.get("latitude", 0.0),
            longitude=data.get("longitude", 0.0),
            altitude=data.get("altitude", 0.0),
            heading=data.get("heading", 0.0),
            completion_status=data.get("completion_status", ""),
            execution_time=data.get("execution_time", 0.0),
        )


@dataclass
class HomePosition(PaparazziMessage):
    """Home point updates"""

    home_latitude: float = 0.0
    home_longitude: float = 0.0
    home_altitude: float = 0.0
    home_heading: float = 0.0
    position_source: str = ""
    accuracy: float = 0.0
    update_reason: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            home_latitude=data.get("home_latitude", 0.0),
            home_longitude=data.get("home_longitude", 0.0),
            home_altitude=data.get("home_altitude", 0.0),
            home_heading=data.get("home_heading", 0.0),
            position_source=data.get("position_source", ""),
            accuracy=data.get("accuracy", 0.0),
            update_reason=data.get("update_reason", ""),
        )


@dataclass
class RallyPoint(PaparazziMessage):
    """Rally point management"""

    rally_id: int = 0
    rally_latitude: float = 0.0
    rally_longitude: float = 0.0
    rally_altitude: float = 0.0
    rally_type: str = ""
    approach_direction: float = 0.0
    land_direction: float = 0.0
    rally_status: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            rally_id=data.get("rally_id", 0),
            rally_latitude=data.get("rally_latitude", 0.0),
            rally_longitude=data.get("rally_longitude", 0.0),
            rally_altitude=data.get("rally_altitude", 0.0),
            rally_type=data.get("rally_type", ""),
            approach_direction=data.get("approach_direction", 0.0),
            land_direction=data.get("land_direction", 0.0),
            rally_status=data.get("rally_status", ""),
        )


@dataclass
class SurveyStatus(PaparazziMessage):
    """Survey mission progress"""

    survey_id: int = 0
    current_leg: int = 0
    total_legs: int = 0
    progress_percentage: float = 0.0
    coverage_area: float = 0.0
    overlap_percentage: float = 0.0
    survey_state: str = ""
    estimated_completion: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            survey_id=data.get("survey_id", 0),
            current_leg=data.get("current_leg", 0),
            total_legs=data.get("total_legs", 0),
            progress_percentage=data.get("progress_percentage", 0.0),
            coverage_area=data.get("coverage_area", 0.0),
            overlap_percentage=data.get("overlap_percentage", 0.0),
            survey_state=data.get("survey_state", ""),
            estimated_completion=data.get("estimated_completion", 0.0),
        )


@dataclass
class LandingSequence(PaparazziMessage):
    """Landing procedures"""

    landing_phase: str = ""
    target_latitude: float = 0.0
    target_longitude: float = 0.0
    target_altitude: float = 0.0
    approach_speed: float = 0.0
    descent_rate: float = 0.0
    landing_type: str = ""
    sequence_status: str = ""

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            landing_phase=data.get("landing_phase", ""),
            target_latitude=data.get("target_latitude", 0.0),
            target_longitude=data.get("target_longitude", 0.0),
            target_altitude=data.get("target_altitude", 0.0),
            approach_speed=data.get("approach_speed", 0.0),
            descent_rate=data.get("descent_rate", 0.0),
            landing_type=data.get("landing_type", ""),
            sequence_status=data.get("sequence_status", ""),
        )


@dataclass
class Approach(PaparazziMessage):
    """Approach procedures"""

    approach_type: str = ""
    approach_altitude: float = 0.0
    approach_speed: float = 0.0
    glide_slope: float = 0.0
    runway_heading: float = 0.0
    distance_to_threshold: float = 0.0
    approach_phase: str = ""
    deviation_lateral: float = 0.0
    deviation_vertical: float = 0.0

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(
            timestamp=data.get("timestamp"),
            approach_type=data.get("approach_type", ""),
            approach_altitude=data.get("approach_altitude", 0.0),
            approach_speed=data.get("approach_speed", 0.0),
            glide_slope=data.get("glide_slope", 0.0),
            runway_heading=data.get("runway_heading", 0.0),
            distance_to_threshold=data.get("distance_to_threshold", 0.0),
            approach_phase=data.get("approach_phase", ""),
            deviation_lateral=data.get("deviation_lateral", 0.0),
            deviation_vertical=data.get("deviation_vertical", 0.0),
        )


# --- Generic fallback class for unmapped messages ---


@dataclass
class GenericMessage(PaparazziMessage):
    """Generic message for unmapped message types"""

    message_type: str = ""
    data: dict = field(default_factory=dict)

    @classmethod
    def from_series(cls, data: pd.Series, message_type: str = "UNKNOWN"):
        return cls(
            timestamp=data.get("timestamp"),
            message_type=message_type,
            data=data.to_dict(),
        )


# --- Mapping from message name string to data class ---
MESSAGE_TO_CLASS_MAP = {
    "ATTITUDE": Attitude,
    "GPS": GPS,
    "GPS_INT": GPSInt,
    "IMU_ACCEL": IMU,
    "IMU_GYRO": IMU,
    "IMU_ACCEL_SCALED": IMU,
    "IMU_GYRO_SCALED": IMU,
    "AIRSPEED": Airspeed,
    "ENERGY": Energy,
    "POWER": Power,
    "ESC": ESC,
    "I2C_ERRORS": I2CErrors,
    "ROTORCRAFT_STATUS": RotorcraftStatus,
    "GROUND_DETECT": GroundDetect,
    "AHRS_REF_QUAT": AHRSQuat,
    "EKF2_STATE": EKF2State,
    "STAB_ATTITUDE": StabAttitude,
    "ROTORCRAFT_CMD": RotorcraftCmd,
    "ROTORCRAFT_FP": RotorcraftFP,
    "INDI_ROTWING": IndiRotwing,
    "ROTWING_STATE": RotwingState,
    # New message types
    "AIR_DATA": AirData,
    "ACTUATORS": Actuators,
    "BEBOP_ACTUATORS": BebopActuators,
    "DATALINK_REPORT": DatalinkReport,
    "AUTOPILOT_VERSION": AutopilotVersion,
    "ALIVE": Alive,
    "DL_VALUE": DLValue,
    "ROTORCRAFT_NAV_STATUS": RotorcraftNavStatus,
    "WP_MOVED": WPMoved,
    "STATE_FILTER_STATUS": StateFilterStatus,
    "UART_ERRORS": UartErrors,
    "INS_REF": InsRef,
    "INS": Ins,
    "SURVEY": Survey,
    # Additional message types for comprehensive coverage
    "GUIDANCE_INDI_HYBRID": GuidanceIndiHybrid,
    "GUIDANCE": Guidance,
    "EXTERNAL_POSE_DOWN": ExternalPoseDown,
    "SERIAL_ACT_T4_IN": SerialActT4In,
    "SERIAL_ACT_T4_OUT": SerialActT4Out,
    "POWER_DEVICE": PowerDevice,
    "EFF_MAT": EffMat,
    "EKF2_P_DIAG": EKF2PDiag,
    "EKF2_INNOV": EKF2Innov,
    # Additional flight monitoring message types
    "STABILIZATION_ATTITUDE": StabilizationAttitude,
    "NAV_STATUS": NavStatus,
    "WAYPOINT": Waypoint,
    "FLIGHT_PLAN": FlightPlan,
    "RC_LOST": RCLost,
    "DATALINK_LOST": DatalinkLost,
    "GEOFENCE": Geofence,
    "WEATHER": Weather,
    "WIND_ESTIMATION": WindEstimation,
    "BATTERY_STATUS": BatteryStatus,
    "MOTOR_STATUS": MotorStatus,
    "VIBRATION": Vibration,
    "COMPASS_CAL": CompassCal,
    "BAROMETER": Barometer,
    "TEMPERATURE": Temperature,
    # Safety-critical message types
    "EMERGENCY": Emergency,
    "GEOFENCE_BREACH": GeofenceBreach,
    "COLLISION_AVOIDANCE": CollisionAvoidance,
    "TRAFFIC": Traffic,
    "TERRAIN_FOLLOWING": TerrainFollowing,
    "OBSTACLE_DETECTION": ObstacleDetection,
    "LOSS_OF_CONTROL": LossOfControl,
    "STALL_WARNING": StallWarning,
    "OVER_SPEED": OverSpeed,
    "ALTITUDE_LIMIT": AltitudeLimit,
    # Communication and telemetry message types
    "TELEMETRY_STATUS": TelemetryStatus,
    "RADIO_STATUS": RadioStatus,
    "MODEM_STATUS": ModemStatus,
    "LINK_QUALITY": LinkQuality,
    "PACKET_LOSS": PacketLoss,
    "RSSI_LOW": RSSILow,
    # Power management message types
    "CURRENT_SPIKE": CurrentSpike,
    "POWER_DISTRIBUTION": PowerDistribution,
    "CHARGING_STATUS": ChargingStatus,
    "CELL_BALANCE": CellBalance,
    "THERMAL_THROTTLE": ThermalThrottle,
    # Mission and navigation message types
    "MISSION_ITEM": MissionItem,
    "HOME_POSITION": HomePosition,
    "RALLY_POINT": RallyPoint,
    "SURVEY_STATUS": SurveyStatus,
    "LANDING_SEQUENCE": LandingSequence,
    "APPROACH": Approach,
}
