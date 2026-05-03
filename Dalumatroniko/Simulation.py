import serial
import pybullet as p
import pybullet_data
import time
import math
import threading
from collections import deque
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, font

# ================= CONFIGURATION =================
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
MAX_QUEUE_SIZE = 3  # Reduced for faster response
USE_SMOOTHING = False  # Disabled for lower latency
SERIAL_TIMEOUT = 0.001  # Reduced timeout
VELOCITY_FILTER_ALPHA = 0.3  # Low-pass filter for velocity smoothing (0-1, lower = smoother)
WINDOW_TRANSPARENCY = 0.92  # 0.0 = fully transparent, 1.0 = fully opaque (92% opaque / 8% transparent)

# ================= CAMERA PRESETS =================
# Format: (distance, yaw, pitch, target_x, target_y, target_z)
CAMERA_PRESETS = {
    'front': (2.5, 90, -35, 0, 0, 0),      # Front view
    'back': (2.5, -90, -35, 0, 0, 0),      # Back view
    'left': (2.5, 180, -35, 0, 0, 0),      # Left side view
    'right': (2.5, 0, -35, 0, 0, 0),       # Right side view
    'isometric': (3.0, 45, -30, 0, 0, 0),  # Isometric view
    'default': (2.5, 50, -35, 0, 0, 0),    # Default view (matches initial)
    'top': (2, 0, -80, 0, 0.5, 0)          # Top view (looking straight down)
}

# ================= SERIAL =================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
ser.flushInput()

# ================= PYBULLET SETUP =================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Asset path - UPDATE THIS
ASSETS_PATH = "C:/path/to/your/assets/folder"
if os.path.exists(ASSETS_PATH):
    p.setAdditionalSearchPath(ASSETS_PATH)

# Optimized for speed
p.setRealTimeSimulation(False)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/120)  # Faster timestep (120Hz instead of 240Hz)

# Faster physics parameters
p.setPhysicsEngineParameter(
    numSolverIterations=4,  # Reduced from 10 for speed
    numSubSteps=1,          # Reduced from 2
    contactBreakingThreshold=0.01,  # Larger threshold for speed
    contactSlop=0.01
)

# Disable shadows and other visuals for performance
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI for speed
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

# Load environment
plane = p.loadURDF("plane.urdf")

# Load robot WITHOUT self-collision for speed
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

# ================= FIND END EFFECTOR LINK =================
# Assuming J6 is the end effector (last joint)
end_effector_index = revolute_joint_indices[-1] if revolute_joint_indices else None
print(f"End effector link index: {end_effector_index}")

# ================= JOINT LIMITS =================
JOINT_LIMITS = []
for idx in revolute_joint_indices:
    joint_info = p.getJointInfo(robot, idx)
    lower_limit = joint_info[8]
    upper_limit = joint_info[9]
    JOINT_LIMITS.append((lower_limit, upper_limit))

# ================= JOINT SETTINGS =================
INITIAL_POSE_DEG = [0, -90, 90, 0, 90, 0]
INITIAL_POSE_RAD = [math.radians(a) for a in INITIAL_POSE_DEG]

# Minimal damping for faster response
for idx in revolute_joint_indices:
    p.changeDynamics(robot, idx,
        linearDamping=0.001,
        angularDamping=0.001,
        jointDamping=0.005
    )

# ================= MOTOR PARAMETERS (HIGHER GAINS FOR FASTER RESPONSE) =================
MAX_FORCE = [320, 320, 176, 176, 110, 40]  # Back to original
POSITION_GAIN = [0.6, 0.6, 0.55, 0.5, 0.45, 0.4]  # Higher gains for faster following
VELOCITY_GAIN = [1.0, 1.0, 0.9, 0.8, 0.8, 0.7]  # Higher velocity gains
MAX_VELOCITY = [3.0, 3.0, 3.5, 4.0, 4.5, 5.0]  # Higher max velocities

# ================= ENHANCED TKINTER GUI - JOINT ANGLES WINDOW =================
class AngleDisplayWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Joint Angles")
        self.root.geometry("300x380")  # Reduced from 380x500
        self.root.configure(bg='#0a0e27')  # Deep dark blue background
        
        # Make window always on top
        self.root.attributes('-topmost', True)
        
        # Apply semi-transparency (alpha blending)
        self.root.attributes('-alpha', WINDOW_TRANSPARENCY)
        
        # Custom fonts (slightly smaller)
        self.title_font = font.Font(family="Segoe UI", size=11, weight="bold")
        self.joint_font = font.Font(family="Consolas", size=10, weight="bold")
        self.value_font = font.Font(family="Consolas", size=13, weight="bold")
        self.status_font = font.Font(family="Segoe UI", size=8)
        
        # Create custom style for ttk widgets
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TFrame', background='#0a0e27')
        style.configure('Card.TFrame', background='#161c3a', relief='flat', borderwidth=0)
        style.configure('Title.TLabel', font=('Segoe UI', 11, 'bold'), foreground='#00d4ff', background='#0a0e27')
        style.configure('Joint.TLabel', font=('Consolas', 10, 'bold'), foreground='#4caf50', background='#161c3a')
        style.configure('Value.TLabel', font=('Consolas', 13, 'bold'), foreground='#ffd700', background='#161c3a')
        style.configure('Status.TLabel', font=('Segoe UI', 8), background='#0a0e27')
        
        # Main container with padding
        main_container = tk.Frame(self.root, bg='#0a0e27')
        main_container.pack(fill='both', expand=True, padx=12, pady=12)
        
        # Header with gradient effect (simulated)
        header_frame = tk.Frame(main_container, bg='#0a0e27')
        header_frame.pack(fill='x', pady=(0, 10))
        
        # Title with icon
        title_label = tk.Label(header_frame, text="JOINT ANGLES", font=self.title_font, 
                               fg='#00d4ff', bg='#0a0e27')
        title_label.pack()
        
        # Separator line
        separator = tk.Frame(main_container, height=1, bg='#00d4ff')
        separator.pack(fill='x', pady=(0, 10))
        
        # Create frames and labels for each joint
        self.joint_frames = []
        self.joint_labels = []
        self.value_labels = []
        
        # Color gradient for joint frames (from cyan to blue)
        colors = ['#1a237e', '#1c2a7a', '#1e2d76', '#203072', '#22336e', '#24366a']
        
        for i in range(6):
            # Frame for each joint with card-like appearance
            frame = tk.Frame(main_container, bg=colors[i], relief='flat', bd=0, highlightthickness=0)
            frame.pack(fill='x', pady=3)
            
            # Add subtle border effect
            border_frame = tk.Frame(frame, bg='#00d4ff', height=1)
            border_frame.pack(fill='x', side='top')
            
            content_frame = tk.Frame(frame, bg=colors[i])
            content_frame.pack(fill='x', padx=12, pady=8)
            
            # Joint name with icon
            name_label = tk.Label(content_frame, text=f" J{i+1}:", font=self.joint_font, 
                                  fg='#4caf50', bg=colors[i])
            name_label.pack(side='left')
            
            # Angle value with dynamic styling
            value_label = tk.Label(content_frame, text="0.0°", font=self.value_font, 
                                   fg='#ffd700', bg=colors[i])
            value_label.pack(side='right')
            
            self.value_labels.append(value_label)
            self.joint_frames.append(frame)
        
        # Status section
        status_container = tk.Frame(main_container, bg='#0a0e27')
        status_container.pack(fill='x', pady=(12, 0))
        
        separator2 = tk.Frame(status_container, height=1, bg='#00d4ff')
        separator2.pack(fill='x', pady=(0, 8))
        
        # Status with animated dot effect
        self.status_frame = tk.Frame(status_container, bg='#0a0e27')
        self.status_frame.pack()
        
        self.status_dot = tk.Label(self.status_frame, text="●", font=('Segoe UI', 9), 
                                   fg='#ff4444', bg='#0a0e27')
        self.status_dot.pack(side='left', padx=(0, 4))
        
        self.status_label = tk.Label(self.status_frame, text="Initializing...", 
                                     font=self.status_font, fg='#888888', bg='#0a0e27')
        self.status_label.pack(side='left')
        
        # Controls info at bottom (expanded to show new camera shortcuts)
        controls_frame = tk.Frame(main_container, bg='#0a0e27')
        controls_frame.pack(fill='x', side='bottom', pady=(10, 0))
        
        separator3 = tk.Frame(controls_frame, height=1, bg='#00d4ff')
        separator3.pack(fill='x', pady=(0, 6))
        
        controls_text = "↑↓←→ | Z/X | R | 1-6 Views | 0 Reset"
        controls_label = tk.Label(controls_frame, text=controls_text, 
                                  font=self.status_font, fg='#6c7a89', bg='#0a0e27')
        controls_label.pack()
        
        # Make window non-resizable
        self.root.resizable(False, False)
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (300 // 2)
        y = (self.root.winfo_screenheight() // 2) - (380 // 2) - 100
        self.root.geometry(f'+{x}+{y}')
        
        # Flag for running
        self.running = True
        
    def update_angles(self, angles_deg):
        """Update the display with new angle values"""
        for i, (angle, label) in enumerate(zip(angles_deg, self.value_labels)):
            # Dynamic color coding based on angle value
            if abs(angle) > 150:
                color = '#ff4444'  # Red for extreme angles
            elif abs(angle) > 90:
                color = '#ffaa44'  # Orange for large angles
            else:
                color = '#ffd700'  # Gold for normal angles
            
            label.config(text=f"{angle:6.1f}°", fg=color)
        
        # Update status indicator
        if any(abs(a) > 0.01 for a in angles_deg):
            self.status_label.config(text="Receiving data", fg='#4caf50')
            self.status_dot.config(fg='#4caf50')
        else:
            self.status_label.config(text="Waiting...", fg='#ffaa44')
            self.status_dot.config(fg='#ffaa44')
        
        self.root.update_idletasks()
    
    def update_status(self, text, color='#888888'):
        """Update status message"""
        self.status_label.config(text=text, fg=color)
        self.root.update_idletasks()
    
    def is_running(self):
        return self.running
    
    def close(self):
        self.running = False
        self.root.quit()
        self.root.destroy()

# ================= ENHANCED TKINTER GUI - VELOCITIES & JACOBIAN WINDOW =================
class VelocityDisplayWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("⚡ Joint Velocities")
        self.root.geometry("350x600")  # Reduced from 450x620
        self.root.configure(bg='#0a0e27')  # Deep dark blue background
        
        # Make window always on top
        self.root.attributes('-topmost', True)
        
        # Apply semi-transparency (alpha blending)
        self.root.attributes('-alpha', WINDOW_TRANSPARENCY)
        
        # Custom fonts (smaller)
        self.title_font = font.Font(family="Segoe UI", size=11, weight="bold")
        self.section_font = font.Font(family="Segoe UI", size=9, weight="bold")
        self.value_font = font.Font(family="Consolas", size=10)
        self.big_value_font = font.Font(family="Consolas", size=11, weight="bold")
        
        # Main container with padding
        main_container = tk.Frame(self.root, bg='#0a0e27')
        main_container.pack(fill='both', expand=True, padx=12, pady=12)
        
        # Header
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
        
    
        # Create frames and labels for joint velocities
        self.joint_vel_frames = []
        self.joint_vel_labels = []
        
        # Colors for velocity display
        vel_colors = ['#1a237e', '#1c2a7a', '#1e2d76', '#203072', '#22336e', '#24366a']
        
        for i in range(6):
            # Frame for each joint velocity
            frame = tk.Frame(joint_section, bg=vel_colors[i], relief='flat')
            frame.pack(fill='x', pady=2)
            
            content_frame = tk.Frame(frame, bg=vel_colors[i])
            content_frame.pack(fill='x', padx=12, pady=6)
            
            # Joint name
            name_label = tk.Label(content_frame, text=f"J{i+1}", font=self.value_font, 
                                  fg='#4caf50', bg=vel_colors[i], width=4, anchor='w')
            name_label.pack(side='left')
            
            # Velocity value
            value_label = tk.Label(content_frame, text="0.00", font=self.big_value_font, 
                                   fg='#ffd700', bg=vel_colors[i])
            value_label.pack(side='right')
            
            # Units
            units_label = tk.Label(content_frame, text="deg/s", font=('Consolas', 8), 
                                   fg='#6c7a89', bg=vel_colors[i])
            units_label.pack(side='right', padx=(4, 0))
            
            self.joint_vel_labels.append(value_label)
            self.joint_vel_frames.append(frame)
        
        # Separator
        separator2 = tk.Frame(main_container, height=1, bg='#00d4ff')
        separator2.pack(fill='x', pady=8)
        
        # End Effector Velocity Section
        ee_section = tk.Frame(main_container, bg='#0a0e27')
        ee_section.pack(fill='x', pady=(6, 0))
        
        ee_header = tk.Label(ee_section, text="END EFFECTOR", 
                             font=self.section_font, fg='#00d4ff', bg='#0a0e27')
        ee_header.pack(anchor='center', fill='x', pady=(0, 6))
        
        # Frame for EE velocity with gradient background
        ee_frame = tk.Frame(ee_section, bg='#161c3a', relief='flat', bd=1, highlightthickness=0)
        ee_frame.pack(fill='x', pady=3)
        
        # Add border effect
        border_top = tk.Frame(ee_frame, bg='#00d4ff', height=1)
        border_top.pack(fill='x')
        
        content_ee = tk.Frame(ee_frame, bg='#161c3a')
        content_ee.pack(fill='x', padx=15, pady=10)
        
        # Velocity components with icons (condensed)
        self.ee_vx_label = tk.Label(content_ee, text="→ Vx: 0.000", font=self.value_font, 
                                    fg='#00d4ff', bg='#161c3a')
        self.ee_vx_label.pack(anchor='w', pady=2)
        
        self.ee_vy_label = tk.Label(content_ee, text="↑ Vy: 0.000", font=self.value_font, 
                                    fg='#00d4ff', bg='#161c3a')
        self.ee_vy_label.pack(anchor='w', pady=2)
        
        self.ee_vz_label = tk.Label(content_ee, text="↗ Vz: 0.000", font=self.value_font, 
                                    fg='#00d4ff', bg='#161c3a')
        self.ee_vz_label.pack(anchor='w', pady=2)
        
        # Magnitude with special styling
        magnitude_frame = tk.Frame(content_ee, bg='#161c3a')
        magnitude_frame.pack(fill='x', pady=(6, 0))
        
        separator_mag = tk.Frame(magnitude_frame, height=1, bg='#00d4ff')
        separator_mag.pack(fill='x', pady=(4, 5))
        
        self.ee_vmag_label = tk.Label(magnitude_frame, text="Magnitude: 0.000 m/s", 
                                      font=self.big_value_font, fg='#ff88ff', bg='#161c3a')
        self.ee_vmag_label.pack()
        
        # Make window non-resizable
        self.root.resizable(False, False)
        
        # Center window on screen (to the right of angles window)
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) + (300 // 2) + 20
        y = (self.root.winfo_screenheight() // 2) - (480 // 2) - 80
        self.root.geometry(f'+{x}+{y}')
        
        # Flag for running
        self.running = True
        
    def update_velocities(self, joint_velocities_deg, ee_linear_velocity):
        """Update the display with new velocity values"""
        # Update joint velocities with color coding
        for i, (vel, label) in enumerate(zip(joint_velocities_deg, self.joint_vel_labels)):
            # Color coding based on velocity magnitude
            abs_vel = abs(vel)
            if abs_vel > 100:
                color = '#ff4444'  # Red for high velocity
            elif abs_vel > 50:
                color = '#ffaa44'  # Orange for medium velocity
            else:
                color = '#ffd700'  # Gold for normal velocity
            
            label.config(text=f"{vel:7.2f}", fg=color)
        
        # Update end effector linear velocity components with direction arrows
        vx, vy, vz = ee_linear_velocity
        
        # Color coding for direction (positive vs negative)
        vx_color = '#00ff88' if vx >= 0 else '#ff4444'
        vy_color = '#00ff88' if vy >= 0 else '#ff4444'
        vz_color = '#00ff88' if vz >= 0 else '#ff4444'
        
        self.ee_vx_label.config(text=f"→ Vx: {vx:7.3f} m/s", fg=vx_color)
        self.ee_vy_label.config(text=f"↑ Vy: {vy:7.3f} m/s", fg=vy_color)
        self.ee_vz_label.config(text=f"↗ Vz: {vz:7.3f} m/s", fg=vz_color)
        
        # Calculate and update magnitude
        magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
        
        # Color code magnitude
        if magnitude > 2.0:
            mag_color = '#ff4444'
        elif magnitude > 1.0:
            mag_color = '#ffaa44'
        else:
            mag_color = '#ff88ff'
        
        self.ee_vmag_label.config(text=f" Magnitude: {magnitude:7.3f} m/s", fg=mag_color)
        
        self.root.update_idletasks()
    
    def update_status(self, text, color='#888888'):
        """Update status message"""
        self.jacobian_status.config(text=text, fg=color)
        self.root.update_idletasks()
    
    def is_running(self):
        return self.running
    
    def close(self):
        self.running = False
        self.root.quit()
        self.root.destroy()

# ================= OPTIMIZED SERIAL THREAD =================
class SerialReader(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.latest_angles = [0.0]*6
        self.running = True
        self.daemon = True  # Thread dies with main program
        self.lock = threading.Lock()  # Thread-safe access
        self.last_angles = [0.0]*6
        self.angle_changed = False

    def run(self):
        while self.running:
            try:
                # Read all available data (non-blocking)
                while ser.in_waiting > 0:
                    line = ser.readline().decode().strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 6:
                            angles = [float(p) for p in parts[:6]]
                            
                            # Quick validation (skip unrealistic values)
                            if all(-180 <= a <= 180 for a in angles):
                                with self.lock:
                                    self.latest_angles = angles
                                    self.angle_changed = True
                                
                                # No smoothing for lower latency
                                if not USE_SMOOTHING:
                                    self.last_angles = angles
            except:
                pass
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.0001)

    def get_angles(self):
        with self.lock:
            return self.latest_angles.copy()
    
    def has_new_data(self):
        with self.lock:
            changed = self.angle_changed
            self.angle_changed = False
            return changed

    def stop(self):
        self.running = False

# ================= VELOCITY CALCULATION CLASS =================
class VelocityCalculator:
    def __init__(self, num_joints=6, filter_alpha=0.3):
        self.num_joints = num_joints
        self.filter_alpha = filter_alpha
        self.prev_angles = None
        self.prev_time = None
        self.filtered_velocities = [0.0] * num_joints
        
    def calculate_velocities(self, current_angles_deg, current_time):
        """Calculate joint velocities in deg/s from angle changes over time"""
        if self.prev_angles is None or self.prev_time is None:
            self.prev_angles = current_angles_deg.copy()
            self.prev_time = current_time
            return [0.0] * self.num_joints
        
        # Calculate time difference
        dt = current_time - self.prev_time
        if dt < 0.001:  # Too small time difference
            return self.filtered_velocities.copy()
        
        # Calculate raw velocities (deg/s)
        raw_velocities = []
        for i in range(self.num_joints):
            delta_angle = current_angles_deg[i] - self.prev_angles[i]
            # Handle angle wrap-around (if passing through 0)
            if delta_angle > 180:
                delta_angle -= 360
            elif delta_angle < -180:
                delta_angle += 360
            raw_vel = delta_angle / dt
            raw_velocities.append(raw_vel)
        
        # Apply low-pass filter for smoothing
        for i in range(self.num_joints):
            self.filtered_velocities[i] = (self.filter_alpha * raw_velocities[i] + 
                                          (1 - self.filter_alpha) * self.filtered_velocities[i])
        
        # Update previous values
        self.prev_angles = current_angles_deg.copy()
        self.prev_time = current_time
        
        return self.filtered_velocities.copy()

# ================= JACOBIAN AND EE VELOCITY CALCULATION =================
class JacobianCalculator:
    def __init__(self, robot_id, end_effector_link_index, joint_indices):
        self.robot_id = robot_id
        self.end_effector_link = end_effector_link_index
        self.joint_indices = joint_indices
        self.num_joints = len(joint_indices)
        
    def get_end_effector_linear_velocity(self, joint_velocities_rad_s):
        """
        Calculate end effector linear velocity using PyBullet's Jacobian
        joint_velocities_rad_s: list of joint velocities in rad/s
        Returns: [vx, vy, vz] in m/s
        """
        if len(joint_velocities_rad_s) != self.num_joints:
            return [0, 0, 0]
        
        # Get current joint states (positions)
        joint_positions = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_positions.append(joint_state[0])  # position in radians
        
        # Compute Jacobian for the end effector
        # v, _ = p.calculateJacobian(robot, linkIndex, localPosition, jointPositions, jointVelocities, acceleration)
        # localPosition at the end effector (0,0,0 relative to link frame)
        local_pos = [0, 0, 0]
        
        # Calculate Jacobian (returns two 3xN matrices: linear Jacobian and angular Jacobian)
        linear_jacobian, angular_jacobian = p.calculateJacobian(
            self.robot_id, 
            self.end_effector_link, 
            local_pos, 
            joint_positions, 
            joint_velocities_rad_s, 
            [0.0] * self.num_joints
        )
        
        # End effector linear velocity = linear_jacobian * joint_velocities
        # Since we already have joint_velocities, we can compute the dot product
        ee_linear_velocity = [
            sum(linear_jacobian[0][i] * joint_velocities_rad_s[i] for i in range(self.num_joints)),
            sum(linear_jacobian[1][i] * joint_velocities_rad_s[i] for i in range(self.num_joints)),
            sum(linear_jacobian[2][i] * joint_velocities_rad_s[i] for i in range(self.num_joints))
        ]
        
        return ee_linear_velocity

# ================= DISABLE COLLISIONS FOR SPEED =================
# Get link indices
link_names_to_indices = {}
for i in range(p.getNumJoints(robot)):
    link_info = p.getJointInfo(robot, i)
    link_names_to_indices[link_info[12].decode('utf-8')] = i
link_names_to_indices['pedestal'] = -1

# Disable all collisions between robot links for maximum speed
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

# ================= HELPERS =================
def is_all_zero(angles, tol=1e-2):
    return all(abs(a) < tol for a in angles)

def reset_to_initial_pose():
    for i, joint_idx in enumerate(revolute_joint_indices):
        p.resetJointState(robot, joint_idx, INITIAL_POSE_RAD[i])
    # Only a few steps to stabilize
    for _ in range(5):
        p.stepSimulation()

# ================= CAMERA FUNCTIONS =================
cam_distance = 2.5
cam_yaw = 50
cam_pitch = -35
cam_target = [0, 0, 0]

def update_camera():
    p.resetDebugVisualizerCamera(cam_distance, cam_yaw, cam_pitch, cam_target)

def set_camera_view(preset_name):
    """Set camera to a predefined view"""
    global cam_distance, cam_yaw, cam_pitch, cam_target
    
    if preset_name in CAMERA_PRESETS:
        cam_distance, cam_yaw, cam_pitch, cam_target_x, cam_target_y, cam_target_z = CAMERA_PRESETS[preset_name]
        cam_target = [cam_target_x, cam_target_y, cam_target_z]
        update_camera()
        print(f"Camera switched to {preset_name} view")
        return True
    return False

# ================= MAIN LOOP (OPTIMIZED) =================
# Initialize GUI windows
angle_window = AngleDisplayWindow()
velocity_window = VelocityDisplayWindow()

# Start serial reader
reader = SerialReader()
reader.start()

# Initialize velocity calculator
velocity_calc = VelocityCalculator(num_joints=6, filter_alpha=VELOCITY_FILTER_ALPHA)

# Initialize Jacobian calculator
jacobian_calc = JacobianCalculator(robot, end_effector_index, revolute_joint_indices)

zero_triggered = False
last_time = time.time()
frame_count = 0
simulation_step = 0
last_vel_calc_time = time.time()
last_gui_update = time.time()
gui_update_interval = 0.05  # Update GUI every 50ms (20Hz)

# Pre-allocate arrays for speed
angles_rad_arr = [0.0]*6
target_rad_arr = [0.0]*6

try:
    while angle_window.is_running() and velocity_window.is_running():
        # ===== CAMERA CONTROL (ONLY WHEN NEEDED) =====
        keys = p.getKeyboardEvents()
        camera_moved = False
        
        # Existing camera controls
        if p.B3G_UP_ARROW in keys: cam_pitch -= 1; camera_moved = True
        if p.B3G_DOWN_ARROW in keys: cam_pitch += 1; camera_moved = True
        if p.B3G_LEFT_ARROW in keys: cam_yaw -= 1; camera_moved = True
        if p.B3G_RIGHT_ARROW in keys: cam_yaw += 1; camera_moved = True
        if ord('z') in keys: cam_distance += 0.05; camera_moved = True
        if ord('x') in keys: cam_distance -= 0.05; camera_moved = True
        
        # NEW: Camera preset shortcuts
        if ord('1') in keys:
            set_camera_view('front')
            camera_moved = False  # Don't trigger additional update (already handled)
        elif ord('2') in keys:
            set_camera_view('back')
            camera_moved = False
        elif ord('3') in keys:
            set_camera_view('left')
            camera_moved = False
        elif ord('4') in keys:
            set_camera_view('right')
            camera_moved = False
        elif ord('5') in keys:
            set_camera_view('isometric')
            camera_moved = False
        elif ord('6') in keys:
            set_camera_view('top')
            camera_moved = False
        elif ord('0') in keys:
            set_camera_view('default')
            camera_moved = False
        
        # Reset robot (existing functionality)
        if ord('r') in keys:
            reset_to_initial_pose()
            print("Reset to initial pose")
            # Reset velocity calculator on reset
            velocity_calc.prev_angles = None
            velocity_calc.prev_time = None
        
        if camera_moved:
            update_camera()

        # ===== GET ANGLES (NON-BLOCKING) =====
        angles_deg = reader.get_angles()

        # Zero detection
        if is_all_zero(angles_deg) and not zero_triggered:
            reset_to_initial_pose()
            zero_triggered = True
            continue
        elif not is_all_zero(angles_deg):
            zero_triggered = False

        # Convert to radians once
        for i in range(6):
            angles_rad_arr[i] = math.radians(angles_deg[i])

        # ===== CONTROL JOINTS (BATCHED) =====
        for i, joint_idx in enumerate(revolute_joint_indices):
            lower, upper = JOINT_LIMITS[i]
            # Clamp target
            target_rad = angles_rad_arr[i]
            if target_rad < lower:
                target_rad = lower
            elif target_rad > upper:
                target_rad = upper
            
            target_rad_arr[i] = target_rad
            
            p.setJointMotorControl2(robot, joint_idx, p.POSITION_CONTROL, 
                                    target_rad,
                                    force=MAX_FORCE[i], 
                                    positionGain=POSITION_GAIN[i],
                                    velocityGain=VELOCITY_GAIN[i], 
                                    maxVelocity=MAX_VELOCITY[i])

        # ===== STEP PHYSICS (MULTIPLE STEPS FOR SMOOTHER MOTION) =====
        for _ in range(2):  # 2 steps per control update
            p.stepSimulation()
            simulation_step += 1

        # ===== UPDATE VELOCITIES AND JACOBIAN =====
        current_time = time.time()
        
        # Calculate joint velocities (deg/s) from angle changes
        joint_velocities_deg = velocity_calc.calculate_velocities(angles_deg, current_time)
        
        # Convert joint velocities to rad/s for Jacobian
        joint_velocities_rad = [math.radians(v) for v in joint_velocities_deg]
        
        # Calculate end effector linear velocity using Jacobian
        ee_linear_velocity = jacobian_calc.get_end_effector_linear_velocity(joint_velocities_rad)
        
        # ===== UPDATE GUI WINDOWS (THROTTLED) =====
        if current_time - last_gui_update >= gui_update_interval:
            # Update angles window
            angle_window.update_angles(angles_deg)
            
            # Update velocities window
            velocity_window.update_velocities(joint_velocities_deg, ee_linear_velocity)
            
            last_gui_update = current_time

        # ===== PERFORMANCE MONITORING =====
        frame_count += 1
        if frame_count >= 100:
            current_time = time.time()
            elapsed = current_time - last_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            control_hz = frame_count / elapsed if elapsed > 0 else 0
            physics_hz = simulation_step / elapsed if elapsed > 0 else 0
            
            print(f"Control: {control_hz:.1f} Hz | Physics: {physics_hz:.1f} Hz | FPS: {fps:.1f}")
            
            frame_count = 0
            simulation_step = 0
            last_time = current_time

        # Process Tkinter events to keep GUI responsive
        angle_window.root.update_idletasks()
        angle_window.root.update()
        velocity_window.root.update_idletasks()
        velocity_window.root.update()
        
        # Minimal sleep to prevent CPU overuse
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