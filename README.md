# Design and Development of Arm Wearable Device for 6-DOF Augmentation
## 🤖 KUKA LBR iiwa 14 R820 - IMU-Based Motion Control

Real-time control of a 6-DOF robot arm using dual MPU9250 sensors with sensor fusion, PyBullet simulation, and live GUI visualization.

## 🎯 Overview

This project enables intuitive robot arm control through natural human arm movements using two MPU9250 sensors mounted on the upper arm and forearm. The system performs real-time sensing to eliminate drift in the yaw axis, providing accurate joint angle estimation without positional drift over time.

## ✨ Features

- **Dual Sensor Setup**: Two MPU9250 sensors track upper arm and forearm/wrist movements independently
- **Sensor Fusion**: Complementary filter (98% gyro + 2% magnetometer) for drift-free yaw angle (Joint 1)
- **6-DOF Control**: Complete control of all 6 robot joints:
  - J1: Yaw (Z-axis) - Gyro + Mag Fusion
  - J2: Roll (X-axis) - Arm raise/lower
  - J3: Pitch (Y-axis) - Bicep rotation
  - J4: Elbow (Z-axis) - Relative to J1
  - J5: Wrist pitch (Y-axis)
  - J6: Wrist flexion (X-axis) - Relative to J2
- **PyBullet Simulation**: Real-time 3D visualization with physics engine
- **Live GUI**: Two-window display showing:
  - Current joint angles (degrees)
  - Joint velocities (deg/s)
  - End effector linear velocity (m/s) via Jacobian calculation
- **Automatic Calibration**: Gyro bias and magnetometer hard-iron offset calibration
- **Configurable Gains**: Adjustable velocity gains and motion thresholds per joint
- **Joint Limits**: Enforced safety limits for all 6 axes
- **Camera Controls**: Multiple preset views (front, back, isometric, top) with keyboard shortcuts

## 🛠️ Hardware Requirements

- **Microcontroller**: ESP32-S3
- **Sensors**: 2x MPU9250 
- **Robot**: KUKA LBR iiwa 14 R820
- **Computer**: Windows/Linux with Python 3.7+

## 🎮 How It Works

1. **Arduino Sketch (Arm Motion Capture.ino)**: Reads raw IMU data, applies calibration, performs sensor fusion, outputs joint angles over serial
2. **Python Script (Simulation.py)**: Receives angles, calculates velocities, computes Jacobian, drives PyBullet simulation
3. **GUI**: Real-time visualization of angles, velocities, and end effector motion
