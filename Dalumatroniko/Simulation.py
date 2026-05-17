import serial
import pybullet as p
import pybullet_data
import time
import math
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, font

# ================= CONFIGURATION =================
SERIAL_PORT = 'COM12'
BAUD_RATE = 115200
MAX_QUEUE_SIZE = 3
USE_SMOOTHING = False
SERIAL_TIMEOUT = 0.001
VELOCITY_FILTER_ALPHA = 0.3
WINDOW_TRANSPARENCY = 0.92

# Link lengths in mm
a2 = 152.230   # L2
a3 = 160.418   # L3
a5 = 160.418   # L5
a6 = 60.222    # L6

# ================= CAMERA PRESETS =================
CAMERA_PRESETS = {
    'front': (2.5, 90, -35, 0, 0, 0),
    'back': (2.5, -90, -35, 0, 0, 0),
    'left': (2.5, 180, -35, 0, 0, 0),
    'right': (2.5, 0, -35, 0, 0, 0),
    'isometric': (3.0, 45, -30, 0, 0, 0),
    'default': (2.5, 50, -35, 0, 0, 0),
    'top': (2, 0, -80, 0, 0.5, 0)
}

# ================= SERIAL SETUP =================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
ser.flushInput()

# ================= PYBULLET SETUP =================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Optimize physics
p.setRealTimeSimulation(False)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/120)

p.setPhysicsEngineParameter(
    numSolverIterations=4,
    numSubSteps=1,
    contactBreakingThreshold=0.01,
    contactSlop=0.01
)

# Disable GUI for performance
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

# Load environment
plane = p.loadURDF("plane.urdf")

# Load robot
robot = p.loadURDF("robot.urdf", useFixedBase=True, flags=0)

# ================= FIND REVOLUTE JOINTS =================
num_joints = p.getNumJoints(robot)
revolute_joint_indices = []
joint_names = []

for i in range(num_joints):
    joint_info = p.getJointInfo(robot, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    
    if joint_type == p.JOINT_REVOLUTE and joint_name.startswith('J'):
        revolute_joint_indices.append(i)
        joint_names.append(joint_name)
        print(f"Found revolute joint: {joint_name} at index {i}")

NUM_JOINTS = len(revolute_joint_indices)
print(f"Total controllable joints: {NUM_JOINTS}")

# ================= END EFFECTOR =================
end_effector_index = revolute_joint_indices[-1] if revolute_joint_indices else None
print(f"End effector link index: {end_effector_index}")

# ================= JOINT LIMITS =================
JOINT_LIMITS = []
for idx in revolute_joint_indices:
    joint_info = p.getJointInfo(robot, idx)
    lower_limit = joint_info[8]
    upper_limit = joint_info[9]
    JOINT_LIMITS.append((lower_limit, upper_limit))

# ================= INITIAL POSE =================
INITIAL_POSE_DEG = [0, -90, 90, 0, 90, 0]
INITIAL_POSE_RAD = [math.radians(a) for a in INITIAL_POSE_DEG]

# ================= MOTOR PARAMETERS =================
MAX_FORCE = [320, 320, 176, 176, 110, 40]
POSITION_GAIN = [0.6, 0.6, 0.55, 0.5, 0.45, 0.4]
VELOCITY_GAIN = [1.0, 1.0, 0.9, 1.0, 0.8, 0.7]
MAX_VELOCITY = [3.0, 3.0, 3.5, 4.0, 4.5, 5.0]

# ================= DIFFERENTIAL KINEMATICS FUNCTIONS =================

def compute_jacobian_J123(theta):
    """
    Compute the 6x3 Jacobian matrix for joints 1, 2, 3.
    theta: list of 6 joint angles in radians (uses theta1, theta2, theta3)
    Returns: 6x3 numpy array
    """
    th1, th2, th3 = theta[0], theta[1], theta[2]
    
    # Convert link lengths to meters
    a2_m = a2 / 1000.0
    a3_m = a3 / 1000.0
    a2_plus_a3 = a2_m + a3_m
    
    s1 = math.sin(th1)
    c1 = math.cos(th1)
    s2 = math.sin(th2)
    c2 = math.cos(th2)
    s3 = math.sin(th3)
    c3 = math.cos(th3)
    
    # J123 matrix
    J123 = np.array([
        [-(a2_plus_a3) * s1 * c2,  -(a2_plus_a3) * s2 * c1,  0],
        [ (a2_plus_a3) * c1 * c2,  -(a2_plus_a3) * s1 * s2,  0],
        [ 0,                        (a2_plus_a3) * c2,        0],
        [ 0,                        s1,                        c1 * c2],
        [ 0,                       -c1,                        s1 * c2],
        [ 1,                        0,                         s2]
    ])
    
    return J123

def compute_jacobian_J456(theta):
    """
    Compute the 6x3 Jacobian matrix for joints 4, 5, 6.
    theta: list of 6 joint angles in radians (uses theta4, theta5, theta6)
    Returns: 6x3 numpy array
    """
    th4, th5, th6 = theta[3], theta[4], theta[5]
    
    # Convert link lengths to meters
    a5_m = a5 / 1000.0
    a6_m = a6 / 1000.0
    
    s4 = math.sin(th4)
    c4 = math.cos(th4)
    s5 = math.sin(th5)
    c5 = math.cos(th5)
    s6 = math.sin(th6)
    c6 = math.cos(th6)
    
    # J456 matrix
    J456 = np.array([
        [-a6_m * c6,         0,                         a6_m * c5 * c6],
        [-a6_m * s6,         0,                         a6_m * c5 * s6],
        [ 0,                 a6_m * (s4 * c6 + c4 * s6), a5_m * (s4 * c6 - c4 * s6)],
        [ 0,                 s4,                        c4 * s5],
        [ 0,                -c4,                        s4 * s5],
        [ 1,                 0,                        -c5]
    ])
    
    return J456

def compute_full_jacobian(theta):
    """
    Compute the complete 6x6 geometric Jacobian matrix.
    theta: list of 6 joint angles in radians
    Returns: 6x6 numpy array (concatenation of J123 and J456)
    """
    J123 = compute_jacobian_J123(theta)
    J456 = compute_jacobian_J456(theta)
    
    # Concatenate to form 6x6 Jacobian
    J_full = np.hstack([J123, J456])
    
    return J_full

def compute_end_effector_velocity(joint_velocities, theta):
    """
    Compute end effector velocity using separate Jacobians for joints 1-3 and 4-6.
    joint_velocities: list of 6 joint velocities in rad/s
    theta: list of 6 joint angles in radians
    Returns: tuple (linear_velocity [vx,vy,vz], angular_velocity [wx,wy,wz])
    """
    # Split joint velocities
    vel_1_3 = np.array(joint_velocities[0:3])
    vel_4_6 = np.array(joint_velocities[3:6])
    
    # Compute contribution from each set of joints
    J123 = compute_jacobian_J123(theta)
    J456 = compute_jacobian_J456(theta)
    
    # End effector twist = J123 * q_dot_1_3 + J456 * q_dot_4_6
    twist_1_3 = J123 @ vel_1_3
    twist_4_6 = J456 @ vel_4_6
    end_effector_twist = twist_1_3 + twist_4_6
    
    linear_vel = end_effector_twist[0:3].tolist()
    angular_vel = end_effector_twist[3:6].tolist()
    
    return linear_vel, angular_vel

def compute_end_effector_velocity_full(joint_velocities, theta):
    """
    Alternative method using full 6x6 Jacobian (for verification).
    joint_velocities: list of 6 joint velocities in rad/s
    theta: list of 6 joint angles in radians
    Returns: tuple (linear_velocity [vx,vy,vz], angular_velocity [wx,wy,wz])
    """
    J_full = compute_full_jacobian(theta)
    vel = np.array(joint_velocities)
    
    end_effector_twist = J_full @ vel
    
    linear_vel = end_effector_twist[0:3].tolist()
    angular_vel = end_effector_twist[3:6].tolist()
    
    return linear_vel, angular_vel

def integrate_joint_angles(joint_angles, joint_velocities, dt):
    """
    Integrate joint velocities to get new joint angles.
    joint_angles: current joint angles in radians
    joint_velocities: joint velocities in rad/s
    dt: time step in seconds
    Returns: updated joint angles in radians
    """
    new_angles = []
    for i in range(6):
        new_angle = joint_angles[i] + joint_velocities[i] * dt
        new_angles.append(new_angle)
    return new_angles

# ================= TKINTER GUI =================

class AngleDisplayWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Joint Angles")
        self.root.geometry("300x380")
        self.root.configure(bg='#0a0e27')
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', WINDOW_TRANSPARENCY)
        
        self.title_font = font.Font(family="Segoe UI", size=11, weight="bold")
        self.joint_font = font.Font(family="Consolas", size=10, weight="bold")
        self.value_font = font.Font(family="Consolas", size=13, weight="bold")
        self.status_font = font.Font(family="Segoe UI", size=8)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        main_container = tk.Frame(self.root, bg='#0a0e27')
        main_container.pack(fill='both', expand=True, padx=12, pady=12)
        
        header_frame = tk.Frame(main_container, bg='#0a0e27')
        header_frame.pack(fill='x', pady=(0, 10))
        
        title_label = tk.Label(header_frame, text="JOINT ANGLES", font=self.title_font, 
                               fg='#00d4ff', bg='#0a0e27')
        title_label.pack()
        
        separator = tk.Frame(main_container, height=1, bg='#00d4ff')
        separator.pack(fill='x', pady=(0, 10))
        
        self.joint_frames = []
        self.joint_labels = []
        self.value_labels = []
        
        colors = ['#1a237e', '#1c2a7a', '#1e2d76', '#203072', '#22336e', '#24366a']
        
        for i in range(6):
            frame = tk.Frame(main_container, bg=colors[i], relief='flat')
            frame.pack(fill='x', pady=3)
            
            content_frame = tk.Frame(frame, bg=colors[i])
            content_frame.pack(fill='x', padx=12, pady=8)
            
            name_label = tk.Label(content_frame, text=f" J{i+1}:", font=self.joint_font, 
                                  fg='#4caf50', bg=colors[i])
            name_label.pack(side='left')
            
            value_label = tk.Label(content_frame, text="0.0°", font=self.value_font, 
                                   fg='#ffd700', bg=colors[i])
            value_label.pack(side='right')
            
            self.value_labels.append(value_label)
            self.joint_frames.append(frame)
        
        status_container = tk.Frame(main_container, bg='#0a0e27')
        status_container.pack(fill='x', pady=(12, 0))
        
        separator2 = tk.Frame(status_container, height=1, bg='#00d4ff')
        separator2.pack(fill='x', pady=(0, 8))
        
        self.status_frame = tk.Frame(status_container, bg='#0a0e27')
        self.status_frame.pack()
        
        self.status_dot = tk.Label(self.status_frame, text="●", font=('Segoe UI', 9), 
                                   fg='#ff4444', bg='#0a0e27')
        self.status_dot.pack(side='left', padx=(0, 4))
        
        self.status_label = tk.Label(self.status_frame, text="Initializing...", 
                                     font=self.status_font, fg='#888888', bg='#0a0e27')
        self.status_label.pack(side='left')
        
        controls_frame = tk.Frame(main_container, bg='#0a0e27')
        controls_frame.pack(fill='x', side='bottom', pady=(10, 0))
        
        separator3 = tk.Frame(controls_frame, height=1, bg='#00d4ff')
        separator3.pack(fill='x', pady=(0, 6))
        
        controls_text = "↑↓←→ | Z/X | R | 1-6 Views | 0 Reset"
        controls_label = tk.Label(controls_frame, text=controls_text, 
                                  font=self.status_font, fg='#6c7a89', bg='#0a0e27')
        controls_label.pack()
        
        self.root.resizable(False, False)
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (300 // 2)
        y = (self.root.winfo_screenheight() // 2) - (380 // 2) - 100
        self.root.geometry(f'+{x}+{y}')
        
        self.running = True
        
    def update_angles(self, angles_deg):
        for i, (angle, label) in enumerate(zip(angles_deg, self.value_labels)):
            if abs(angle) > 150:
                color = '#ff4444'
            elif abs(angle) > 90:
                color = '#ffaa44'
            else:
                color = '#ffd700'
            
            label.config(text=f"{angle:6.1f}°", fg=color)
        
        if any(abs(a) > 0.01 for a in angles_deg):
            self.status_label.config(text="Receiving data", fg='#4caf50')
            self.status_dot.config(fg='#4caf50')
        else:
            self.status_label.config(text="Waiting...", fg='#ffaa44')
            self.status_dot.config(fg='#ffaa44')
        
        self.root.update_idletasks()
    
    def is_running(self):
        return self.running
    
    def close(self):
        self.running = False
        self.root.quit()
        self.root.destroy()

class VelocityDisplayWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Joint Velocities")
        self.root.geometry("350x620")
        self.root.configure(bg='#0a0e27')
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', WINDOW_TRANSPARENCY)
        
        self.title_font = font.Font(family="Segoe UI", size=11, weight="bold")
        self.section_font = font.Font(family="Segoe UI", size=9, weight="bold")
        self.value_font = font.Font(family="Consolas", size=10)
        self.big_value_font = font.Font(family="Consolas", size=11, weight="bold")
        
        main_container = tk.Frame(self.root, bg='#0a0e27')
        main_container.pack(fill='both', expand=True, padx=12, pady=12)
        
        header_frame = tk.Frame(main_container, bg='#0a0e27')
        header_frame.pack(fill='x', pady=(0, 10))
        
        title_label = tk.Label(header_frame, text="JOINT VELOCITIES (deg/s)", 
                               font=self.title_font, fg='#00d4ff', bg='#0a0e27')
        title_label.pack()
        
        separator = tk.Frame(main_container, height=1, bg='#00d4ff')
        separator.pack(fill='x', pady=(0, 10))
        
        # Joint Velocities Section
        joint_section = tk.Frame(main_container, bg='#0a0e27')
        joint_section.pack(fill='x', pady=(0, 10))
        
        self.joint_vel_labels = []
        vel_colors = ['#1a237e', '#1c2a7a', '#1e2d76', '#203072', '#22336e', '#24366a']
        
        for i in range(6):
            frame = tk.Frame(joint_section, bg=vel_colors[i], relief='flat')
            frame.pack(fill='x', pady=2)
            
            content_frame = tk.Frame(frame, bg=vel_colors[i])
            content_frame.pack(fill='x', padx=12, pady=6)
            
            name_label = tk.Label(content_frame, text=f"J{i+1}", font=self.value_font, 
                                  fg='#4caf50', bg=vel_colors[i], width=4, anchor='w')
            name_label.pack(side='left')
            
            value_label = tk.Label(content_frame, text="0.00", font=self.big_value_font, 
                                   fg='#ffd700', bg=vel_colors[i])
            value_label.pack(side='right')
            
            units_label = tk.Label(content_frame, text="deg/s", font=('Consolas', 8), 
                                   fg='#6c7a89', bg=vel_colors[i])
            units_label.pack(side='right', padx=(4, 0))
            
            self.joint_vel_labels.append(value_label)
        
        separator2 = tk.Frame(main_container, height=1, bg='#00d4ff')
        separator2.pack(fill='x', pady=8)
        
        # End Effector Section
        ee_section = tk.Frame(main_container, bg='#0a0e27')
        ee_section.pack(fill='x', pady=(6, 0))
        
        ee_header = tk.Label(ee_section, text="END EFFECTOR VELOCITY", 
                             font=self.section_font, fg='#00d4ff', bg='#0a0e27')
        ee_header.pack(anchor='center', fill='x', pady=(0, 6))
        
        ee_frame = tk.Frame(ee_section, bg='#161c3a', relief='flat')
        ee_frame.pack(fill='x', pady=3)
        
        border_top = tk.Frame(ee_frame, bg='#00d4ff', height=1)
        border_top.pack(fill='x')
        
        content_ee = tk.Frame(ee_frame, bg='#161c3a')
        content_ee.pack(fill='x', padx=15, pady=10)
        
        self.ee_linear_label = tk.Label(content_ee, text="Linear Vel (m/s)", 
                                        font=self.section_font, fg='#00d4ff', bg='#161c3a')
        self.ee_linear_label.pack(anchor='w', pady=(0, 5))
        
        self.ee_vx_label = tk.Label(content_ee, text="  → Vx: 0.000", font=self.value_font, 
                                    fg='#00d4ff', bg='#161c3a')
        self.ee_vx_label.pack(anchor='w', pady=2)
        
        self.ee_vy_label = tk.Label(content_ee, text="  ↑ Vy: 0.000", font=self.value_font, 
                                    fg='#00d4ff', bg='#161c3a')
        self.ee_vy_label.pack(anchor='w', pady=2)
        
        self.ee_vz_label = tk.Label(content_ee, text="  ↗ Vz: 0.000", font=self.value_font, 
                                    fg='#00d4ff', bg='#161c3a')
        self.ee_vz_label.pack(anchor='w', pady=2)
        
        separator3 = tk.Frame(content_ee, height=1, bg='#00d4ff')
        separator3.pack(fill='x', pady=(6, 5))
        
        self.ee_vmag_label = tk.Label(content_ee, text="Magnitude: 0.000 m/s", 
                                      font=self.big_value_font, fg='#ff88ff', bg='#161c3a')
        self.ee_vmag_label.pack(anchor='center', pady=(5, 0))
        
        # Angular velocity section
        separator4 = tk.Frame(content_ee, height=1, bg='#00d4ff')
        separator4.pack(fill='x', pady=(8, 5))
        
        self.ee_angular_label = tk.Label(content_ee, text="Angular Vel (rad/s)", 
                                         font=self.section_font, fg='#00d4ff', bg='#161c3a')
        self.ee_angular_label.pack(anchor='w', pady=(5, 5))
        
        self.ee_wx_label = tk.Label(content_ee, text="  ↻ Wx: 0.000", font=self.value_font, 
                                    fg='#ffaa44', bg='#161c3a')
        self.ee_wx_label.pack(anchor='w', pady=2)
        
        self.ee_wy_label = tk.Label(content_ee, text="  ↺ Wy: 0.000", font=self.value_font, 
                                    fg='#ffaa44', bg='#161c3a')
        self.ee_wy_label.pack(anchor='w', pady=2)
        
        self.ee_wz_label = tk.Label(content_ee, text="  ↻ Wz: 0.000", font=self.value_font, 
                                    fg='#ffaa44', bg='#161c3a')
        self.ee_wz_label.pack(anchor='w', pady=2)
        
        # Jacobian info section
        separator5 = tk.Frame(content_ee, height=1, bg='#00d4ff')
        separator5.pack(fill='x', pady=(8, 5))
        
        self.jacobian_info = tk.Label(content_ee, text="Using Provided Jacobian Matrices", 
                                      font=self.value_font, fg='#00ff88', bg='#161c3a')
        self.jacobian_info.pack(anchor='center', pady=(5, 0))
        
        # Reset status indicator
        self.reset_status_label = tk.Label(content_ee, text="", font=self.value_font, 
                                           fg='#ff4444', bg='#161c3a')
        self.reset_status_label.pack(anchor='center', pady=(10, 0))
        
        self.root.resizable(False, False)
        
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) + (300 // 2) + 20
        y = (self.root.winfo_screenheight() // 2) - (480 // 2) - 80
        self.root.geometry(f'+{x}+{y}')
        
        self.running = True
        self.reset_timer = None
        
    def update_velocities(self, joint_velocities_deg, ee_linear_vel, ee_angular_vel):
        # Update joint velocities
        for i, (vel, label) in enumerate(zip(joint_velocities_deg, self.joint_vel_labels)):
            abs_vel = abs(vel)
            if abs_vel > 100:
                color = '#ff4444'
            elif abs_vel > 50:
                color = '#ffaa44'
            else:
                color = '#ffd700'
            
            label.config(text=f"{vel:7.2f}", fg=color)
        
        # Update end effector linear velocity
        vx, vy, vz = ee_linear_vel
        vx_color = '#00ff88' if vx >= 0 else '#ff4444'
        vy_color = '#00ff88' if vy >= 0 else '#ff4444'
        vz_color = '#00ff88' if vz >= 0 else '#ff4444'
        
        self.ee_vx_label.config(text=f"  → Vx: {vx:7.3f}", fg=vx_color)
        self.ee_vy_label.config(text=f"  ↑ Vy: {vy:7.3f}", fg=vy_color)
        self.ee_vz_label.config(text=f"  ↗ Vz: {vz:7.3f}", fg=vz_color)
        
        magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
        if magnitude > 2.0:
            mag_color = '#ff4444'
        elif magnitude > 1.0:
            mag_color = '#ffaa44'
        else:
            mag_color = '#ff88ff'
        
        self.ee_vmag_label.config(text=f"Magnitude: {magnitude:7.3f} m/s", fg=mag_color)
        
        # Update end effector angular velocity
        wx, wy, wz = ee_angular_vel
        self.ee_wx_label.config(text=f"  ↻ Wx: {wx:7.3f}", fg='#ffaa44')
        self.ee_wy_label.config(text=f"  ↺ Wy: {wy:7.3f}", fg='#ffaa44')
        self.ee_wz_label.config(text=f"  ↻ Wz: {wz:7.3f}", fg='#ffaa44')
        
        self.root.update_idletasks()
    
    def show_reset_notification(self):
        """Show a temporary reset notification"""
        self.reset_status_label.config(text=">>> RESET COMMAND RECEIVED! <<<", fg='#00ff88')
        self.root.update_idletasks()
        if self.reset_timer:
            self.root.after_cancel(self.reset_timer)
        self.reset_timer = self.root.after(1000, self.clear_reset_notification)
    
    def clear_reset_notification(self):
        self.reset_status_label.config(text="", fg='#ff4444')
        self.root.update_idletasks()
    
    def is_running(self):
        return self.running
    
    def close(self):
        self.running = False
        if self.reset_timer:
            self.root.after_cancel(self.reset_timer)
        self.root.quit()
        self.root.destroy()

# ================= SERIAL THREAD WITH RESET DETECTION =================
class SerialReader(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.latest_velocities = [0.0]*6
        self.reset_detected = False
        self.running = True
        self.daemon = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode().strip()
                    if line:
                        # Check for reset command
                        if line == "RESET" or line.startswith("RESET"):
                            print("\n[Serial] RESET command received from Arduino!")
                            with self.lock:
                                self.reset_detected = True
                                self.latest_velocities = [0.0]*6
                        else:
                            # Parse velocity data
                            parts = line.split(',')
                            if len(parts) >= 6:
                                velocities = [float(p) for p in parts[:6]]
                                if all(-500 <= v <= 500 for v in velocities):
                                    with self.lock:
                                        self.latest_velocities = velocities
            except Exception as e:
                print(f"Serial read error: {e}")
            time.sleep(0.0001)

    def get_velocities(self):
        with self.lock:
            return self.latest_velocities.copy()
    
    def check_and_clear_reset(self):
        with self.lock:
            if self.reset_detected:
                self.reset_detected = False
                return True
            return False

    def stop(self):
        self.running = False

# ================= DISABLE COLLISIONS =================
link_names_to_indices = {}
for i in range(p.getNumJoints(robot)):
    link_info = p.getJointInfo(robot, i)
    link_names_to_indices[link_info[12].decode('utf-8')] = i
link_names_to_indices['pedestal'] = -1

collision_pairs = [
    ('pedestal', 'j1'), ('pedestal', 'j2'), ('pedestal', 'j3_base'),
    ('j1', 'j2'), ('j1', 'j3_base'), ('j2', 'j3_base'),
    ('j3_base', 'j4'), ('j3_base', 'j5_base'), ('j4', 'j5_base'),
    ('j5_base', 'j6'),
]

for link1, link2 in collision_pairs:
    if link1 in link_names_to_indices and link2 in link_names_to_indices:
        idx1 = link_names_to_indices[link1]
        idx2 = link_names_to_indices[link2]
        p.setCollisionFilterPair(robot, robot, idx1, idx2, 0)

# ================= CAMERA FUNCTIONS =================
cam_distance = 2.5
cam_yaw = 50
cam_pitch = -35
cam_target = [0, 0, 0]

def update_camera():
    p.resetDebugVisualizerCamera(cam_distance, cam_yaw, cam_pitch, cam_target)

def set_camera_view(preset_name):
    global cam_distance, cam_yaw, cam_pitch, cam_target
    if preset_name in CAMERA_PRESETS:
        cam_distance, cam_yaw, cam_pitch, cam_target_x, cam_target_y, cam_target_z = CAMERA_PRESETS[preset_name]
        cam_target = [cam_target_x, cam_target_y, cam_target_z]
        update_camera()
        return True
    return False

def reset_to_initial_pose():
    """Reset robot to initial pose and reset angle integration"""
    global current_angles_rad, current_angles_deg, last_vel_time
    
    print("\n" + "="*50)
    print("RESET: Moving robot to initial pose")
    print(f"Target angles: {INITIAL_POSE_DEG}")
    print("="*50 + "\n")
    
    current_angles_rad = INITIAL_POSE_RAD.copy()
    current_angles_deg = INITIAL_POSE_DEG.copy()
    last_vel_time = time.time()
    
    for i, joint_idx in enumerate(revolute_joint_indices):
        p.setJointMotorControl2(robot, joint_idx, p.POSITION_CONTROL, 
                                current_angles_rad[i],
                                force=MAX_FORCE[i], 
                                positionGain=POSITION_GAIN[i],
                                velocityGain=VELOCITY_GAIN[i], 
                                maxVelocity=MAX_VELOCITY[i])
    
    for _ in range(10):
        p.stepSimulation()

# ================= MAIN LOOP =================
angle_window = AngleDisplayWindow()
velocity_window = VelocityDisplayWindow()
reader = SerialReader()
reader.start()

current_angles_rad = INITIAL_POSE_RAD.copy()
current_angles_deg = INITIAL_POSE_DEG.copy()

last_vel_time = time.time()
frame_count = 0
last_time = time.time()
simulation_step = 0
last_gui_update = time.time()
gui_update_interval = 0.05

try:
    while angle_window.is_running() and velocity_window.is_running():
        # ===== CAMERA CONTROL =====
        keys = p.getKeyboardEvents()
        camera_moved = False
        
        if p.B3G_UP_ARROW in keys: cam_pitch -= 0.5; camera_moved = True
        if p.B3G_DOWN_ARROW in keys: cam_pitch += 0.5; camera_moved = True
        if p.B3G_LEFT_ARROW in keys: cam_yaw -= 0.5; camera_moved = True
        if p.B3G_RIGHT_ARROW in keys: cam_yaw += 0.5; camera_moved = True
        if ord('z') in keys: cam_distance += 0.01; camera_moved = True
        if ord('x') in keys: cam_distance -= 0.01; camera_moved = True
        
        if ord('1') in keys: set_camera_view('front')
        elif ord('2') in keys: set_camera_view('back')
        elif ord('3') in keys: set_camera_view('left')
        elif ord('4') in keys: set_camera_view('right')
        elif ord('5') in keys: set_camera_view('isometric')
        elif ord('6') in keys: set_camera_view('top')
        elif ord('0') in keys: set_camera_view('default')
        
        if ord('r') in keys:
            reset_to_initial_pose()
        
        if camera_moved:
            update_camera()

        # ===== CHECK FOR RESET FROM ARDUINO =====
        if reader.check_and_clear_reset():
            velocity_window.show_reset_notification()
            reset_to_initial_pose()
            continue
        
        # ===== GET JOINT VELOCITIES FROM ARDUINO =====
        joint_velocities_deg = reader.get_velocities()
        joint_velocities_rad = [math.radians(v) for v in joint_velocities_deg]
        
        # ===== INTEGRATE VELOCITIES TO GET ANGLES =====
        current_time = time.time()
        dt = current_time - last_vel_time
        if dt > 0.01: dt = 0.01
        if dt > 0 and dt < 0.1:
            current_angles_rad = integrate_joint_angles(current_angles_rad, joint_velocities_rad, dt)
            
            for i in range(6):
                lower, upper = JOINT_LIMITS[i]
                if current_angles_rad[i] < lower:
                    current_angles_rad[i] = lower
                elif current_angles_rad[i] > upper:
                    current_angles_rad[i] = upper
            
            current_angles_deg = [math.degrees(a) for a in current_angles_rad]
        
        last_vel_time = current_time
        
        # ===== CONTROL ROBOT USING INTEGRATED ANGLES =====
        for i, joint_idx in enumerate(revolute_joint_indices):
            p.setJointMotorControl2(robot, joint_idx, p.POSITION_CONTROL, 
                                    current_angles_rad[i],
                                    force=MAX_FORCE[i], 
                                    positionGain=POSITION_GAIN[i],
                                    velocityGain=VELOCITY_GAIN[i], 
                                    maxVelocity=MAX_VELOCITY[i])
        
        # ===== COMPUTE END EFFECTOR VELOCITIES USING PROVIDED JACOBIAN MATRICES =====
        ee_linear_vel, ee_angular_vel = compute_end_effector_velocity(joint_velocities_rad, current_angles_rad)
        
        # ===== STEP PHYSICS =====
        for _ in range(2):
            p.stepSimulation()
            simulation_step += 1
        
        # ===== UPDATE GUI =====
        if current_time - last_gui_update >= gui_update_interval:
            angle_window.update_angles(current_angles_deg)
            velocity_window.update_velocities(joint_velocities_deg, ee_linear_vel, ee_angular_vel)
            last_gui_update = current_time
        
        # ===== PERFORMANCE MONITORING =====
        frame_count += 1
        if frame_count >= 100:
            elapsed = time.time() - last_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            if frame_count % 300 == 0:
                print(f"FPS: {fps:.1f} | Angles: J1={current_angles_deg[0]:.1f}, J2={current_angles_deg[1]:.1f}, J3={current_angles_deg[2]:.1f}")
            frame_count = 0
            last_time = time.time()
        
        # Process GUI events
        angle_window.root.update_idletasks()
        angle_window.root.update()
        velocity_window.root.update_idletasks()
        velocity_window.root.update()
        
        time.sleep(0.0005)

except KeyboardInterrupt:
    print("\nStopping simulation...")
    reader.stop()
    ser.close()
    angle_window.close()
    velocity_window.close()
    p.disconnect()
except Exception as e:
    print(f"Error: {e}")
    reader.stop()
    ser.close()
    angle_window.close()
    velocity_window.close()
    p.disconnect()