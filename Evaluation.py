import serial
import pybullet as p
import pybullet_data
import time
import math
import threading
from collections import deque
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import RadioButtons, Button
from datetime import datetime
import csv

# ================= CONFIGURATION =================
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
MAX_QUEUE_SIZE = 3
USE_SMOOTHING = False
SERIAL_TIMEOUT = 0.001

# ================= GRAPH CONFIGURATION =================
MAX_DATA_POINTS = 500
UPDATE_INTERVAL_MS = 50

# ================= SERIAL =================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
ser.flushInput()

# ================= PYBULLET SETUP =================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

ASSETS_PATH = "C:/path/to/your/assets/folder"
if os.path.exists(ASSETS_PATH):
    p.setAdditionalSearchPath(ASSETS_PATH)

p.setRealTimeSimulation(False)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/120)

p.setPhysicsEngineParameter(
    numSolverIterations=4,
    numSubSteps=1,
    contactBreakingThreshold=0.01,
    contactSlop=0.01
)

p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

plane = p.loadURDF("plane.urdf")
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

for idx in revolute_joint_indices:
    p.changeDynamics(robot, idx,
        linearDamping=0.001,
        angularDamping=0.001,
        jointDamping=0.005
    )

MAX_FORCE = [320, 320, 176, 176, 110, 40]
POSITION_GAIN = [0.6, 0.6, 0.55, 0.5, 0.45, 0.4]
VELOCITY_GAIN = [1.0, 1.0, 0.9, 0.8, 0.8, 0.7]
MAX_VELOCITY = [3.0, 3.0, 3.5, 4.0, 4.5, 5.0]

# ================= DATA STORAGE FOR GRAPHING =================
class DataLogger:
    def __init__(self, max_points=MAX_DATA_POINTS):
        self.timestamps = deque(maxlen=max_points)
        self.arduino_timestamps = deque(maxlen=max_points)
        self.angles = [deque(maxlen=max_points) for _ in range(6)]
        self.latencies = deque(maxlen=max_points)
        self.max_points = max_points
        self.is_paused = False
        
    def add_data(self, arduino_time, angles_deg, receive_time):
        if not self.is_paused:
            self.arduino_timestamps.append(arduino_time)
            self.timestamps.append(receive_time)
            for i in range(6):
                self.angles[i].append(angles_deg[i])
            
            latency = (receive_time - arduino_time) * 1000
            self.latencies.append(latency)
            return True
        return False
    
    def pause(self):
        self.is_paused = True
        print("Data logging PAUSED - Graph frozen")
        
    def resume(self):
        self.is_paused = False
        print("Data logging RESUMED - Graph updating")
        
    def get_time_range(self):
        if len(self.timestamps) > 0:
            return self.timestamps[0], self.timestamps[-1]
        return 0, 0

# ================= OPTIMIZED SERIAL THREAD =================
class SerialReader(threading.Thread):
    def __init__(self, data_logger):
        threading.Thread.__init__(self)
        self.latest_angles = [0.0]*6
        self.latest_arduino_time = 0
        self.running = True
        self.daemon = True
        self.lock = threading.Lock()
        self.last_angles = [0.0]*6
        self.angle_changed = False
        self.data_logger = data_logger

    def run(self):
        while self.running:
            try:
                while ser.in_waiting > 0:
                    line = ser.readline().decode().strip()
                    if line:
                        if line.startswith("PAUSE_GRAPH"):
                            self.data_logger.pause()
                            continue
                        elif line.startswith("RESUME_GRAPH"):
                            self.data_logger.resume()
                            continue
                        
                        parts = line.split(',')
                        if len(parts) >= 7:
                            try:
                                arduino_time_ms = float(parts[0])
                                angles = [float(p) for p in parts[1:7]]
                                receive_time = time.time()
                                arduino_time_sec = arduino_time_ms / 1000.0
                                
                                if all(-180 <= a <= 180 for a in angles):
                                    with self.lock:
                                        self.latest_angles = angles
                                        self.latest_arduino_time = arduino_time_sec
                                        self.angle_changed = True
                                    
                                    self.data_logger.add_data(arduino_time_sec, angles, receive_time)
                                    
                                    if not USE_SMOOTHING:
                                        self.last_angles = angles
                            except ValueError:
                                pass
            except:
                pass
            time.sleep(0.0001)

    def get_angles(self):
        with self.lock:
            return self.latest_angles.copy()
    
    def get_arduino_time(self):
        with self.lock:
            return self.latest_arduino_time
    
    def has_new_data(self):
        with self.lock:
            changed = self.angle_changed
            self.angle_changed = False
            return changed

    def stop(self):
        self.running = False

# ================= GRAPHING CLASS =================
class RealTimeGraph:
    def __init__(self, data_logger):
        self.data_logger = data_logger
        self.current_joint = 0
        self.show_latency = True
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle('Robot Joint Control - Real Time Monitoring', fontsize=16, fontweight='bold')
        
        self.ax1 = plt.subplot(2, 1, 1)
        self.ax2 = plt.subplot(2, 1, 2)
        
        self.angle_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Joint Angle')
        #self.angle_target_line, = self.ax1.plot([], [], 'r--', linewidth=1, label='Target (if available)')
        self.latency_line, = self.ax2.plot([], [], 'g-', linewidth=2, label='Latency')
        
        self.latency_threshold = self.ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Warning Threshold (50ms)')
        
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Angle (degrees)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right')
        
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Latency (ms)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper right')
        
        self.stats_text = self.fig.text(0.02, 0.98, '', transform=self.fig.transFigure, 
                                        fontsize=10, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax_radio = plt.axes([0.85, 0.5, 0.12, 0.3])
        joint_labels = [f'Joint {i+1}' for i in range(6)]
        self.radio = RadioButtons(ax_radio, joint_labels, active=self.current_joint)
        self.radio.on_clicked(self.on_joint_selected)
        
        ax_latency_toggle = plt.axes([0.85, 0.4, 0.12, 0.05])
        self.latency_button = Button(ax_latency_toggle, 'Toggle Latency View')
        self.latency_button.on_clicked(self.toggle_latency)
        
        ax_record = plt.axes([0.85, 0.34, 0.12, 0.05])
        self.record_button = Button(ax_record, 'Start Recording')
        self.record_button.on_clicked(self.toggle_recording)
        
        ax_export = plt.axes([0.85, 0.28, 0.12, 0.05])
        self.export_button = Button(ax_export, 'Export Data')
        self.export_button.on_clicked(self.export_data)
        
        ax_clear = plt.axes([0.85, 0.22, 0.12, 0.05])
        self.clear_button = Button(ax_clear, 'Clear Graph')
        self.clear_button.on_clicked(self.clear_graph)
        
        self.last_update = time.time()
        self.update_count = 0
        
    def on_joint_selected(self, label):
        self.current_joint = int(label.split()[-1]) - 1
        print(f"Displaying {label}")
        
    def toggle_latency(self, event):
        self.show_latency = not self.show_latency
        if self.show_latency:
            self.latency_line.set_visible(True)
            self.latency_threshold.set_visible(True)
            self.ax2.set_visible(True)
            self.latency_button.label.set_text('Hide Latency')
        else:
            self.latency_line.set_visible(False)
            self.latency_threshold.set_visible(False)
            self.ax2.set_visible(False)
            self.latency_button.label.set_text('Show Latency')
        plt.draw()
        
    def toggle_recording(self, event):
        if not self.is_recording:
            filename = f"robot_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['Time_s', 'Arduino_Time_s', 'Latency_ms', 
                                     'J1_deg', 'J2_deg', 'J3_deg', 'J4_deg', 'J5_deg', 'J6_deg', 'Paused'])
            self.is_recording = True
            self.record_button.label.set_text('Stop Recording')
            print(f"Recording data to {filename}")
        else:
            if self.csv_file:
                self.csv_file.close()
            self.is_recording = False
            self.record_button.label.set_text('Start Recording')
            print("Recording stopped")
            
    def export_data(self, event):
        filename = f"robot_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_s', 'Arduino_Time_s', 'Latency_ms', 
                           'J1_deg', 'J2_deg', 'J3_deg', 'J4_deg', 'J5_deg', 'J6_deg'])
            
            timestamps = list(self.data_logger.timestamps)
            arduino_times = list(self.data_logger.arduino_timestamps)
            latencies = list(self.data_logger.latencies)
            
            for i in range(len(timestamps)):
                row = [timestamps[i], arduino_times[i], latencies[i]]
                for j in range(6):
                    if i < len(self.data_logger.angles[j]):
                        row.append(self.data_logger.angles[j][i])
                    else:
                        row.append(0)
                writer.writerow(row)
        
        print(f"Data exported to {filename}")
        
    def clear_graph(self, event):
        self.data_logger.timestamps.clear()
        self.data_logger.arduino_timestamps.clear()
        self.data_logger.latencies.clear()
        for i in range(6):
            self.data_logger.angles[i].clear()
        print("Graph cleared")
        
    def update_statistics(self):
        if len(self.data_logger.latencies) > 0:
            avg_latency = np.mean(list(self.data_logger.latencies))
            max_latency = max(list(self.data_logger.latencies))
            min_latency = min(list(self.data_logger.latencies))
            
            current_angles = [list(self.data_logger.angles[i])[-1] if len(self.data_logger.angles[i]) > 0 else 0 for i in range(6)]
            
            stats = f"Joint {self.current_joint + 1}\n"
            stats += f"Current Angle: {current_angles[self.current_joint]:.1f}°\n"
            stats += f"Latency - Avg: {avg_latency:.1f}ms Max: {max_latency:.1f}ms Min: {min_latency:.1f}ms\n"
            stats += f"Data Points: {len(self.data_logger.timestamps)}"
            
            if self.data_logger.is_paused:
                stats += "\n\n>>> PAUSED"
            
            if avg_latency < 20:
                stats += "\nLatency: EXCELLENT"
            elif avg_latency < 50:
                stats += "\nLatency: GOOD"
            elif avg_latency < 100:
                stats += "\nLatency: ACCEPTABLE"
            else:
                stats += "\nLatency: HIGH - Check connection"
                
            self.stats_text.set_text(stats)
                    
    def update(self, frame):
        if len(self.data_logger.timestamps) > 1:
            times = list(self.data_logger.timestamps)
            angles = list(self.data_logger.angles[self.current_joint])
            
            if len(times) == len(angles):
                self.angle_line.set_data(times, angles)
                
                if len(times) > 0:
                    self.ax1.set_xlim(times[0], times[-1])
                    if len(angles) > 0:
                        angle_min = min(angles)
                        angle_max = max(angles)
                        padding = max(5, (angle_max - angle_min) * 0.1)
                        self.ax1.set_ylim(angle_min - padding, angle_max + padding)
            
            if self.show_latency and len(self.data_logger.latencies) > 0:
                latencies = list(self.data_logger.latencies)
                if len(times) == len(latencies):
                    self.latency_line.set_data(times, latencies)
                    if len(latencies) > 0:
                        lat_max = max(latencies)
                        self.ax2.set_ylim(0, max(100, lat_max * 1.1))
                        self.ax2.set_xlim(times[0], times[-1])
            
            self.update_count += 1
            if self.update_count % 10 == 0:
                self.update_statistics()
            
            if self.is_recording and len(self.data_logger.timestamps) > 0 and not self.data_logger.is_paused:
                latest_idx = -1
                row = [
                    self.data_logger.timestamps[latest_idx],
                    self.data_logger.arduino_timestamps[latest_idx],
                    self.data_logger.latencies[latest_idx],
                    self.data_logger.angles[0][latest_idx],
                    self.data_logger.angles[1][latest_idx],
                    self.data_logger.angles[2][latest_idx],
                    self.data_logger.angles[3][latest_idx],
                    self.data_logger.angles[4][latest_idx],
                    self.data_logger.angles[5][latest_idx],
                    self.data_logger.is_paused
                ]
                self.csv_writer.writerow(row)
                self.csv_file.flush()
        
        return self.angle_line, self.latency_line

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

# ================= HELPERS =================
def is_all_zero(angles, tol=1e-2):
    return all(abs(a) < tol for a in angles)

def reset_to_initial_pose():
    for i, joint_idx in enumerate(revolute_joint_indices):
        p.resetJointState(robot, joint_idx, INITIAL_POSE_RAD[i])
    for _ in range(5):
        p.stepSimulation()

# ================= HUD =================
hud_text = ["JOINT ANGLES"] + [f"J{i+1}: 0.0" for i in range(6)]
hud_ids = [p.addUserDebugText("", [0,0,0], textSize=0.8, textColorRGB=[0,0,0]) for _ in range(7)]

# ================= CAMERA =================
cam_distance = 2.5
cam_yaw = 50
cam_pitch = -35
cam_target = [0, 0, 0]

def update_camera():
    p.resetDebugVisualizerCamera(cam_distance, cam_yaw, cam_pitch, cam_target)

# ================= MAIN LOOP =================
def main():
    data_logger = DataLogger()
    reader = SerialReader(data_logger)
    reader.start()
    
    graph = RealTimeGraph(data_logger)
    ani = animation.FuncAnimation(graph.fig, graph.update, interval=UPDATE_INTERVAL_MS, 
                                  blit=False, cache_frame_data=False)
    
    plt.ion()
    plt.show(block=False)
    
    zero_triggered = False
    last_time = time.time()
    frame_count = 0
    simulation_step = 0
    
    angles_rad_arr = [0.0]*6
    target_rad_arr = [0.0]*6
    
    print("\n=== SYSTEM READY ===")
    print("Controls:")
    print("  - Graph: Select joint using radio buttons")
    print("  - Graph: Toggle latency view")
    print("  - Graph: Start/stop recording to CSV")
    print("  - Graph: Export current data")
    print("  - Graph: Clear graph data")
    print("  - PyBullet: Arrow keys - Camera rotation")
    print("  - PyBullet: Z/X - Camera zoom")
    print("  - PyBullet: R - Reset to initial pose")
    print("  - Hardware: SHORT PRESS (<2 sec) - Reset robot to initial position")
    print("  - Hardware: LONG PRESS (≥2 sec) - PAUSE/UNPAUSE graph and MPU readings")
    print("  - Hardware: Hold 3+ seconds - Recalibrate gyros")
    print("\n=== TIP: Long press to freeze graph for clean screenshots ===\n")
    
    try:
        while True:
            keys = p.getKeyboardEvents()
            camera_moved = False
            if p.B3G_UP_ARROW in keys: cam_pitch -= 1; camera_moved = True
            if p.B3G_DOWN_ARROW in keys: cam_pitch += 1; camera_moved = True
            if p.B3G_LEFT_ARROW in keys: cam_yaw -= 1; camera_moved = True
            if p.B3G_RIGHT_ARROW in keys: cam_yaw += 1; camera_moved = True
            if ord('z') in keys: cam_distance += 0.05; camera_moved = True
            if ord('x') in keys: cam_distance -= 0.05; camera_moved = True
            if ord('r') in keys:
                reset_to_initial_pose()
                print("Reset to initial pose")
            
            if camera_moved:
                update_camera()
            
            angles_deg = reader.get_angles()
            
            if is_all_zero(angles_deg) and not zero_triggered:
                reset_to_initial_pose()
                zero_triggered = True
                continue
            elif not is_all_zero(angles_deg):
                zero_triggered = False
            
            for i in range(6):
                angles_rad_arr[i] = math.radians(angles_deg[i])
            
            for i, joint_idx in enumerate(revolute_joint_indices):
                lower, upper = JOINT_LIMITS[i]
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
            
            for _ in range(2):
                p.stepSimulation()
                simulation_step += 1
            
            if frame_count % 10 == 0:
                cam_info = p.getDebugVisualizerCamera()
                cam_target_pos = cam_info[11]
                cam_dist = cam_info[10]
                text_size = max(0.6, 1.2 / cam_dist)
                hud_y = cam_target_pos[2] + 0.8
                
                for i, line in enumerate(hud_text):
                    if i == 0:
                        text = "JOINT ANGLES"
                    else:
                        text = f"J{i}: {angles_deg[i-1]:6.1f}"
                    
                    p.addUserDebugText(
                        text,
                        [cam_target_pos[0] + 0.4, cam_target_pos[1] + 0.4, hud_y - i*0.12],
                        textSize=text_size,
                        textColorRGB=[0, 0, 0],
                        replaceItemUniqueId=hud_ids[i]
                    )
            
            frame_count += 1
            if frame_count >= 100:
                current_time = time.time()
                elapsed = current_time - last_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                control_hz = frame_count / elapsed if elapsed > 0 else 0
                physics_hz = simulation_step / elapsed if elapsed > 0 else 0
                
                avg_latency = np.mean(list(data_logger.latencies)) if len(data_logger.latencies) > 0 else 0
                status = "PAUSED" if data_logger.is_paused else "RUNNING"
                
                print(f"[{status}] Control: {control_hz:.1f} Hz | Physics: {physics_hz:.1f} Hz | FPS: {fps:.1f} | Avg Latency: {avg_latency:.1f}ms")
                
                frame_count = 0
                simulation_step = 0
                last_time = current_time
            
            plt.pause(0.001)
            time.sleep(0.0005)
            
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        reader.stop()
        ser.close()
        p.disconnect()
        if graph.is_recording and graph.csv_file:
            graph.csv_file.close()
        plt.close()
        print("Simulation stopped.")

if __name__ == "__main__":
    main()