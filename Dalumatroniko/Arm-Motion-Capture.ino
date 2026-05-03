#include <Wire.h>
#include <math.h>

// ================= PIN DEFINITIONS =================
#define BUTTON_PIN 4

// ================= MPU9250 DEFINITIONS =================
#define MPU_ADDR_123 0x68    // MPU near end of L2 (upper arm)
#define MPU_ADDR_J5 0x69     // MPU near end of L3 (forearm/wrist)

// MPU9250 Registers
#define ACCEL_XOUT_H 0x3B
#define PWR_MGMT_1 0x6B
#define GYRO_CONFIG 0x1B
#define ACCEL_CONFIG 0x1C
#define INT_PIN_CFG 0x37
#define WHO_AM_I 0x75
#define USER_CTRL 0x6A
#define SIGNAL_PATH_RESET 0x68

// Magnetometer registers
#define MAG_ADDR 0x0C
#define AK8963_CNTL1 0x0A
#define AK8963_XOUT_L 0x03

// ================= ROBOT PARAMETERS (Link Lengths) =================
const float L1 = 266.495;  // Joint1 to Joint2
const float L2 = 312.648;  // Joint2 to Joint4
const float L3 = 297.760;  // Joint4 to Joint6

// ================= DIRECTION CONTROL =================
const float JOINT1_DIRECTION = -1;
const float JOINT2_DIRECTION = 1;
const float JOINT3_DIRECTION = -1;
const float JOINT4_DIRECTION = -1;
const float JOINT5_DIRECTION = -1;
const float JOINT6_DIRECTION = 1;

// ================= JOINT LIMITS =================
const float JOINT1_MIN = -170.0;
const float JOINT1_MAX = 170.0;
const float JOINT2_MIN = -120.0;
const float JOINT2_MAX = 120.0;
const float JOINT3_MIN = -170.0;
const float JOINT3_MAX = 170.0;
const float JOINT4_MIN = -120.0;
const float JOINT4_MAX = 120.0;
const float JOINT5_MIN = -170.0;
const float JOINT5_MAX = 170.0;
const float JOINT6_MIN = -120.0;
const float JOINT6_MAX = 120.0;

// ================= DEFAULT HOME POSITION (Initial angles) =================
const float INITIAL_JOINT1 = 0.0;
const float INITIAL_JOINT2 = -90.0;
const float INITIAL_JOINT3 = 90.0;
const float INITIAL_JOINT4 = 0.0;
const float INITIAL_JOINT5 = 90.0;
const float INITIAL_JOINT6 = 0.0;

// ================= VELOCITY CONTROL GAINS =================
const float VELOCITY_GAIN_JOINT1 = 1.5;   // Shoulder rotation (Z-axis)
const float VELOCITY_GAIN_JOINT2 = 1.75;  // Arm raise/lower (X-axis)
const float VELOCITY_GAIN_JOINT3 = 1.0;   // Bicep rotation (Y-axis)
const float VELOCITY_GAIN_JOINT4 = 2.0;   // Elbow (relative Z-axis)
const float VELOCITY_GAIN_JOINT5 = 2.0;   // Wrist pitch (Y-axis)
const float VELOCITY_GAIN_JOINT6 = 2.0;   // Wrist flexion (relative X-axis)

const float MOTION_THRESHOLD = 0.05;  // rad/s threshold for motion detection

// ================= COMPLEMENTARY FILTER FOR YAW (J1) =================
// Higher alpha = more trust in gyro (smoother but more drift)
// Lower alpha = more trust in mag (less drift but more noise)
const float YAW_FILTER_ALPHA = 0.98;  // 98% gyro, 2% mag - good balance

// Magnetometer calibration offsets (adjust these based on your environment)
float mag_offset_x = 0.0;
float mag_offset_y = 0.0;
float mag_offset_z = 0.0;
float mag_scale_x = 1.0;
float mag_scale_y = 1.0;
float mag_scale_z = 1.0;
bool mag_calibrated = false;

// ================= DATA STRUCTURES =================
float ax, ay, az;
float gx, gy, gz;
float mx, my, mz;

float ax5, ay5, az5;
float gx5, gy5, gz5;

float ax_f = 0, ay_f = 0, az_f = 0;
float gx_f = 0, gy_f = 0, gz_f = 0;
float ax5_f = 0, ay5_f = 0, az5_f = 0;
float gx5_f = 0, gy5_f = 0, gz5_f = 0;

// Filtered magnetometer values
float mx_f = 0, my_f = 0, mz_f = 0;

// ================= TIME =================
unsigned long lastTime;
float dt;

// ================= JOINT ANGLES (Current absolute angles in radians) =================
float theta1 = 0, theta2 = 0, theta3 = 0;
float theta4 = 0, theta5 = 0, theta6 = 0;

// For gyro-only yaw integration (used in complementary filter)
float theta1_gyro_only = 0;

// ================= INITIAL REFERENCE ANGLES (Zero position at reset) =================
float theta1_initial = 0;
float theta2_initial = 0;
float theta3_initial = 0;
float theta4_initial = 0;
float theta5_initial = 0;
float theta6_initial = 0;

// ================= TARGET ANGLES =================
float theta1_target = 0, theta2_target = 0, theta3_target = 0;
float theta4_target = 0, theta5_target = 0, theta6_target = 0;
float theta1_s = 0, theta2_s = 0, theta3_s = 0, theta4_s = 0, theta5_s = 0, theta6_s = 0;

// ================= JOINT VELOCITIES =================
float theta1_velocity = 0;
float theta2_velocity = 0;
float theta3_velocity = 0;
float theta4_velocity = 0;
float theta5_velocity = 0;
float theta6_velocity = 0;

float motion_threshold = MOTION_THRESHOLD;

// ================= SMOOTH =================
const float SMOOTH_ALPHA = 0.2;

// ================= KALMAN FILTER =================
struct Kalman {
  float angle = 0;
  float bias = 0;
  float P[2][2] = {{0,0},{0,0}};
};

Kalman kPitch, kRoll, kYaw;
Kalman kPitch5, kRoll5;

const float Q_angle = 0.001;
const float Q_bias  = 0.003;
const float R_measure = 0.03;

// ================= FILTER PARAMETERS =================
const float LPF_ALPHA = 0.2;
const float MAG_LPF_ALPHA = 0.1;  // Slower filter for magnetometer

// ================= BUTTON STATE =================
bool lastButtonState = HIGH;
bool calibrating = false;
unsigned long calibrationStartTime = 0;
const unsigned long CALIBRATION_DURATION = 2000;

// ================= RESET MODE =================
bool resetMode = false;
unsigned long resetStartTime = 0;
const unsigned long RESET_DELAY = 1000;  // 1 second delay during reset

// ================= MPU STATUS =================
bool mpu123_initialized = false;
bool mpuJ5_initialized = false;
bool mag_initialized = false;
unsigned long lastMPU123Read = 0;
unsigned long lastMPUJ5Read = 0;
int mpu123_read_failures = 0;
int mpuJ5_read_failures = 0;

// ================= GYRO BIAS CALIBRATION =================
float gyro_bias_base_x = 0, gyro_bias_base_y = 0, gyro_bias_base_z = 0;
float gyro_bias_wrist_x = 0, gyro_bias_wrist_y = 0, gyro_bias_wrist_z = 0;
bool gyro_bias_calibrated = false;

// ================= FUNCTION PROTOTYPES =================
bool checkI2CDevice(byte addr);
void writeReg(byte addr, byte reg, byte val);
bool readBytes(byte addr, byte reg, byte count, byte *dest);
void resetMPU(byte addr);
bool initMPU123();
bool initMPUJ5();
bool readMPU123();
bool readMPUJ5();
bool readMag();
void filterData();
void filterDataJ5();
void calibrateGyroBias();
void calibrateMagnetometer();
float kalmanUpdate(Kalman &k, float newAngle, float newRate, float dt);
float smooth(float target, float current);
void updateJointsFromMPU68(float dt);
void updateRelativeJointsFromMPU69(float dt);
void resetToInitialPosition();
void calibrateRobot();
void sendJointAngles();
void printMPUData();

// ================= I2C FUNCTIONS =================
bool checkI2CDevice(byte addr) {
  Wire.beginTransmission(addr);
  byte error = Wire.endTransmission();
  return (error == 0);
}

void writeReg(byte addr, byte reg, byte val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
  delay(1);
}

bool readBytes(byte addr, byte reg, byte count, byte *dest) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  byte error = Wire.endTransmission(false);
  if (error != 0) return false;
  
  Wire.requestFrom(addr, count);
  byte bytes_received = Wire.available();
  
  if (bytes_received >= count) {
    for(byte i = 0; i < count; i++) {
      dest[i] = Wire.read();
    }
    return true;
  }
  return false;
}

void resetMPU(byte addr) {
  Serial.printf("Resetting MPU at 0x%02X...\n", addr);
  writeReg(addr, PWR_MGMT_1, 0x80);
  delay(100);
  writeReg(addr, PWR_MGMT_1, 0x00);
  delay(100);
}

// ================= GYRO BIAS CALIBRATION =================
void calibrateGyroBias() {
  Serial.println("\n=== Calibrating Gyro Bias for ALL axes ===");
  Serial.println("Keep MPUs completely still...");
  
  float sum_gx_base = 0, sum_gy_base = 0, sum_gz_base = 0;
  float sum_gx_wrist = 0, sum_gy_wrist = 0, sum_gz_wrist = 0;
  int samples = 500;
  
  for (int i = 0; i < samples; i++) {
    if (readMPU123()) {
      sum_gx_base += gx;
      sum_gy_base += gy;
      sum_gz_base += gz;
    }
    if (readMPUJ5()) {
      sum_gx_wrist += gx5;
      sum_gy_wrist += gy5;
      sum_gz_wrist += gz5;
    }
    delay(5);
  }
  
  gyro_bias_base_x = sum_gx_base / samples;
  gyro_bias_base_y = sum_gy_base / samples;
  gyro_bias_base_z = sum_gz_base / samples;
  gyro_bias_wrist_x = sum_gx_wrist / samples;
  gyro_bias_wrist_y = sum_gy_wrist / samples;
  gyro_bias_wrist_z = sum_gz_wrist / samples;
  gyro_bias_calibrated = true;
  
  Serial.println("MPU 0x68 (Upper Arm) biases:");
  Serial.printf("  X (Roll/J2): %.3f, Y (Pitch/J3): %.3f, Z (Yaw/J1): %.3f deg/s\n", 
                gyro_bias_base_x, gyro_bias_base_y, gyro_bias_base_z);
  Serial.println("MPU 0x69 (Forearm/Wrist) biases:");
  Serial.printf("  X (Roll/J6): %.3f, Y (Pitch/J5): %.3f, Z (Yaw/J4): %.3f deg/s\n", 
                gyro_bias_wrist_x, gyro_bias_wrist_y, gyro_bias_wrist_z);
  Serial.println("Gyro bias calibration complete!\n");
}

// ================= MAGNETOMETER CALIBRATION =================
void calibrateMagnetometer() {
  Serial.println("\n=== Calibrating Magnetometer ===");
  Serial.println("Rotate the sensor in all directions for 10 seconds...");
  
  float min_x = 0, min_y = 0, min_z = 0;
  float max_x = 0, max_y = 0, max_z = 0;
  bool first = true;
  
  unsigned long startTime = millis();
  int samples = 0;
  
  while (millis() - startTime < 10000) {
    if (readMag()) {
      if (first) {
        min_x = max_x = mx;
        min_y = max_y = my;
        min_z = max_z = mz;
        first = false;
      } else {
        min_x = min(min_x, mx);
        max_x = max(max_x, mx);
        min_y = min(min_y, my);
        max_y = max(max_y, my);
        min_z = min(min_z, mz);
        max_z = max(max_z, mz);
      }
      samples++;
    }
    delay(10);
  }
  
  if (samples > 0) {
    mag_offset_x = (max_x + min_x) / 2.0;
    mag_offset_y = (max_y + min_y) / 2.0;
    mag_offset_z = (max_z + min_z) / 2.0;
    
    float avg_scale = ((max_x - min_x) + (max_y - min_y) + (max_z - min_z)) / 3.0;
    mag_scale_x = avg_scale / (max_x - min_x);
    mag_scale_y = avg_scale / (max_y - min_y);
    mag_scale_z = avg_scale / (max_z - min_z);
    
    mag_calibrated = true;
    
    Serial.println("Magnetometer calibration complete:");
    Serial.printf("  Offsets: X=%.1f, Y=%.1f, Z=%.1f\n", mag_offset_x, mag_offset_y, mag_offset_z);
    Serial.printf("  Scales: X=%.3f, Y=%.3f, Z=%.3f\n", mag_scale_x, mag_scale_y, mag_scale_z);
  } else {
    Serial.println("Magnetometer calibration failed - no data collected");
  }
}

// ================= MPU INITIALIZATION =================
bool initMPU123() {
  Serial.println("\n=== Initializing MPU9250 at 0x68 (Upper Arm - near end of L2) ===");
  
  if (!checkI2CDevice(MPU_ADDR_123)) {
    Serial.println("ERROR: MPU9250 at 0x68 not found!");
    return false;
  }
  Serial.println("Device found on I2C bus");
  
  resetMPU(MPU_ADDR_123);
  
  byte whoami;
  if (readBytes(MPU_ADDR_123, WHO_AM_I, 1, &whoami)) {
    Serial.printf("WHO_AM_I register: 0x%02X (expected 0x71)\n", whoami);
    if (whoami != 0x71) {
      Serial.println("WARNING: Unexpected WHO_AM_I value!");
    }
  } else {
    Serial.println("ERROR: Failed to read WHO_AM_I register!");
    return false;
  }
  
  writeReg(MPU_ADDR_123, PWR_MGMT_1, 0x00);
  delay(10);
  writeReg(MPU_ADDR_123, USER_CTRL, 0x00);
  delay(10);
  writeReg(MPU_ADDR_123, SIGNAL_PATH_RESET, 0x07);
  delay(10);
  writeReg(MPU_ADDR_123, GYRO_CONFIG, 0x00);
  delay(10);
  writeReg(MPU_ADDR_123, ACCEL_CONFIG, 0x00);
  delay(10);
  writeReg(MPU_ADDR_123, INT_PIN_CFG, 0x02);
  delay(10);
  
  byte test_data[14];
  if (readBytes(MPU_ADDR_123, ACCEL_XOUT_H, 14, test_data)) {
    Serial.println("MPU9250 at 0x68 configured successfully!");
    return true;
  } else {
    Serial.println("ERROR: MPU9250 at 0x68 configuration failed!");
    return false;
  }
}

bool initMPUJ5() {
  Serial.println("\n=== Initializing MPU9250 at 0x69 (Forearm/Wrist - near end of L3) ===");
  
  if (!checkI2CDevice(MPU_ADDR_J5)) {
    Serial.println("ERROR: MPU9250 at 0x69 not found!");
    return false;
  }
  Serial.println("Device found on I2C bus");
  
  resetMPU(MPU_ADDR_J5);
  
  byte whoami;
  if (readBytes(MPU_ADDR_J5, WHO_AM_I, 1, &whoami)) {
    Serial.printf("WHO_AM_I register: 0x%02X (expected 0x71)\n", whoami);
    if (whoami != 0x71) {
      Serial.println("WARNING: Unexpected WHO_AM_I value!");
    }
  } else {
    Serial.println("ERROR: Failed to read WHO_AM_I register!");
    return false;
  }
  
  writeReg(MPU_ADDR_J5, PWR_MGMT_1, 0x00);
  delay(10);
  writeReg(MPU_ADDR_J5, USER_CTRL, 0x00);
  delay(10);
  writeReg(MPU_ADDR_J5, SIGNAL_PATH_RESET, 0x07);
  delay(10);
  writeReg(MPU_ADDR_J5, GYRO_CONFIG, 0x00);
  delay(10);
  writeReg(MPU_ADDR_J5, ACCEL_CONFIG, 0x00);
  delay(10);
  
  byte test_data[14];
  if (readBytes(MPU_ADDR_J5, ACCEL_XOUT_H, 14, test_data)) {
    Serial.println("MPU9250 at 0x69 configured successfully!");
    return true;
  } else {
    Serial.println("ERROR: MPU9250 at 0x69 configuration failed!");
    return false;
  }
}

bool initMag() {
  Serial.println("\n=== Initializing Magnetometer at 0x0C ===");
  
  if (!checkI2CDevice(MAG_ADDR)) {
    Serial.println("Magnetometer not found (optional)");
    return false;
  }
  
  byte whoami;
  if (readBytes(MAG_ADDR, 0x00, 1, &whoami)) {
    Serial.printf("Magnetometer WHO_AM_I: 0x%02X (expected 0x48)\n", whoami);
    if (whoami != 0x48) {
      Serial.println("WARNING: Unexpected magnetometer WHO_AM_I value!");
    }
  }
  
  writeReg(MAG_ADDR, AK8963_CNTL1, 0x16);
  delay(10);
  
  Serial.println("Magnetometer initialized!");
  return true;
}

// ================= MPU DATA READING =================
bool readMPU123() {
  if (!mpu123_initialized) return false;
  
  byte data[14];
  if (!readBytes(MPU_ADDR_123, ACCEL_XOUT_H, 14, data)) {
    mpu123_read_failures++;
    if (mpu123_read_failures % 100 == 0) {
      Serial.printf("MPU123 read failure count: %d\n", mpu123_read_failures);
    }
    return false;
  }
  
  mpu123_read_failures = 0;
  lastMPU123Read = millis();
  
  int16_t ra = (data[0]<<8)|data[1];
  int16_t rb = (data[2]<<8)|data[3];
  int16_t rc = (data[4]<<8)|data[5];
  int16_t rgx = (data[8]<<8)|data[9];
  int16_t rgy = (data[10]<<8)|data[11];
  int16_t rgz = (data[12]<<8)|data[13];
  
  ax = ra / 16384.0;
  ay = rb / 16384.0;
  az = rc / 16384.0;
  
  gx = rgx / 131.0;
  gy = rgy / 131.0;
  gz = rgz / 131.0;
  
  return true;
}

bool readMPUJ5() {
  if (!mpuJ5_initialized) return false;
  
  byte data[14];
  if (!readBytes(MPU_ADDR_J5, ACCEL_XOUT_H, 14, data)) {
    mpuJ5_read_failures++;
    if (mpuJ5_read_failures % 100 == 0) {
      Serial.printf("MPUJ5 read failure count: %d\n", mpuJ5_read_failures);
    }
    return false;
  }
  
  mpuJ5_read_failures = 0;
  lastMPUJ5Read = millis();
  
  int16_t ra = (data[0]<<8)|data[1];
  int16_t rb = (data[2]<<8)|data[3];
  int16_t rc = (data[4]<<8)|data[5];
  int16_t rgx = (data[8]<<8)|data[9];
  int16_t rgy = (data[10]<<8)|data[11];
  int16_t rgz = (data[12]<<8)|data[13];
  
  ax5 = ra / 16384.0;
  ay5 = rb / 16384.0;
  az5 = rc / 16384.0;
  
  gx5 = rgx / 131.0;
  gy5 = rgy / 131.0;
  gz5 = rgz / 131.0;
  
  return true;
}

bool readMag() {
  if (!mag_initialized) return false;
  
  byte data[7];
  if (!readBytes(MAG_ADDR, AK8963_XOUT_L, 7, data)) return false;
  
  int16_t rmx = (data[1]<<8)|data[0];
  int16_t rmy = (data[3]<<8)|data[2];
  int16_t rmz = (data[5]<<8)|data[4];
  
  mx = rmx;
  my = rmy;
  mz = rmz;
  
  // Apply calibration if available
  if (mag_calibrated) {
    mx = (mx - mag_offset_x) * mag_scale_x;
    my = (my - mag_offset_y) * mag_scale_y;
    mz = (mz - mag_offset_z) * mag_scale_z;
  }
  
  return true;
}

void printMPUData() {
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 2000) {
    lastPrint = millis();
    
    // Compute change in angles from initial position
    float delta1 = (theta1 - theta1_initial) * 180.0 / PI;
    float delta2 = (theta2 - theta2_initial) * 180.0 / PI;
    float delta3 = (theta3 - theta3_initial) * 180.0 / PI;
    float delta4 = (theta4 - theta4_initial) * 180.0 / PI;
    float delta5 = (theta5 - theta5_initial) * 180.0 / PI;
    float delta6 = (theta6 - theta6_initial) * 180.0 / PI;
    
    // Also print mag heading for debugging
    if (mag_initialized && mag_calibrated) {
      float heading = atan2(my, mx) * 180.0 / PI;
      Serial.printf("Mag Heading: %.1f deg\n", heading);
    }
  }
}

// ================= FILTERS =================
void filterData() {
  ax_f = LPF_ALPHA * ax + (1 - LPF_ALPHA) * ax_f;
  ay_f = LPF_ALPHA * ay + (1 - LPF_ALPHA) * ay_f;
  az_f = LPF_ALPHA * az + (1 - LPF_ALPHA) * az_f;
  
  gx_f = LPF_ALPHA * gx + (1 - LPF_ALPHA) * gx_f;
  gy_f = LPF_ALPHA * gy + (1 - LPF_ALPHA) * gy_f;
  gz_f = LPF_ALPHA * gz + (1 - LPF_ALPHA) * gz_f;
  
  // Low-pass filter for magnetometer
  mx_f = MAG_LPF_ALPHA * mx + (1 - MAG_LPF_ALPHA) * mx_f;
  my_f = MAG_LPF_ALPHA * my + (1 - MAG_LPF_ALPHA) * my_f;
  mz_f = MAG_LPF_ALPHA * mz + (1 - MAG_LPF_ALPHA) * mz_f;
}

void filterDataJ5() {
  ax5_f = LPF_ALPHA * ax5 + (1 - LPF_ALPHA) * ax5_f;
  ay5_f = LPF_ALPHA * ay5 + (1 - LPF_ALPHA) * ay5_f;
  az5_f = LPF_ALPHA * az5 + (1 - LPF_ALPHA) * az5_f;
  
  gx5_f = LPF_ALPHA * gx5 + (1 - LPF_ALPHA) * gx5_f;
  gy5_f = LPF_ALPHA * gy5 + (1 - LPF_ALPHA) * gy5_f;
  gz5_f = LPF_ALPHA * gz5 + (1 - LPF_ALPHA) * gz5_f;
}

// ================= KALMAN FILTER =================
float kalmanUpdate(Kalman &k, float newAngle, float newRate, float dt) {
  float rate = newRate - k.bias;
  k.angle += dt * rate;
  
  k.P[0][0] += dt*(dt*k.P[1][1] - k.P[0][1] - k.P[1][0] + Q_angle);
  k.P[0][1] -= dt*k.P[1][1];
  k.P[1][0] -= dt*k.P[1][1];
  k.P[1][1] += Q_bias*dt;
  
  float S = k.P[0][0] + R_measure;
  float K0 = k.P[0][0]/S;
  float K1 = k.P[1][0]/S;
  
  float y = newAngle - k.angle;
  k.angle += K0*y;
  k.bias  += K1*y;
  
  float P00 = k.P[0][0];
  float P01 = k.P[0][1];
  
  k.P[0][0] -= K0*P00;
  k.P[0][1] -= K0*P01;
  k.P[1][0] -= K1*P00;
  k.P[1][1] -= K1*P01;
  
  return k.angle;
}

float smooth(float target, float current) {
  return current + SMOOTH_ALPHA * (target - current);
}

// ================= RESET TO INITIAL POSITION =================
void resetToInitialPosition() {
  Serial.println("\n*** RESETTING TO INITIAL POSITION ***");
  Serial.printf("Target: (%.0f, %.0f, %.0f, %.0f, %.0f, %.0f) degrees\n", 
                INITIAL_JOINT1, INITIAL_JOINT2, INITIAL_JOINT3, 
                INITIAL_JOINT4, INITIAL_JOINT5, INITIAL_JOINT6);
  
  // Store the current gyro angles as the new zero reference
  theta1_initial = theta1;
  theta2_initial = theta2;
  theta3_initial = theta3;
  theta4_initial = theta4;
  theta5_initial = theta5;
  theta6_initial = theta6;
  
  // Also reset the gyro-only yaw reference
  theta1_gyro_only = theta1;
  
  Serial.println("*** INITIAL POSITION SET ***");
  Serial.println("All joint changes will be relative to this position");
}

// ================= JOINT CONTROL FUNCTIONS =================
// MPU 0x68 controls Joints 1, 2, 3 (absolute)
void updateJointsFromMPU68(float dt) {
  if (!mpu123_initialized) return;
  if (!gyro_bias_calibrated) return;
  
  // Get gyro readings with bias removed (convert to rad/s)
  float gyro_x = (gx_f - gyro_bias_base_x) * PI / 180.0;
  float gyro_y = (gy_f - gyro_bias_base_y) * PI / 180.0;
  float gyro_z = (gz_f - gyro_bias_base_z) * PI / 180.0;
  
  // ===== JOINT 1: Yaw (Z-axis) with Gyro + Magnetometer Fusion =====
  bool moving_z = fabs(gyro_z) > motion_threshold;
  
  // Step 1: Integrate gyro to get relative yaw change
  if(moving_z) {
    theta1_velocity = gyro_z * VELOCITY_GAIN_JOINT1 * JOINT1_DIRECTION;
    theta1_velocity = constrain(theta1_velocity, -3.0, 3.0);
    theta1_gyro_only += theta1_velocity * dt;
  }
  
  // Step 2: Get magnetometer heading (absolute yaw)
  float mag_heading = theta1;  // default to current value if mag not available
  
  if (mag_initialized && mag_calibrated && (mx_f != 0 || my_f != 0)) {
    // Calculate heading from magnetometer (in radians)
    // atan2(y, x) gives angle from X-axis
    mag_heading = atan2(my_f, mx_f);
    
    // The magnetometer heading needs to be aligned with the robot's coordinate system
    // Adjust this offset based on how you mounted the MPU
    const float MAG_DECLINATION_OFFSET = 0.0;  // Adjust for your location and mounting
    mag_heading += MAG_DECLINATION_OFFSET;
  }
  
  // Step 3: Apply complementary filter
  // theta1 = 98% from gyro integration + 2% from magnetometer
  // The magnetometer slowly corrects gyro drift
  theta1 = YAW_FILTER_ALPHA * theta1_gyro_only + (1.0 - YAW_FILTER_ALPHA) * mag_heading;
  
  // Apply joint limits
  theta1 = constrain(theta1, JOINT1_MIN * PI/180.0, JOINT1_MAX * PI/180.0);
  
  // Also keep gyro-only value within limits to prevent divergence
  theta1_gyro_only = constrain(theta1_gyro_only, JOINT1_MIN * PI/180.0, JOINT1_MAX * PI/180.0);
  
  // ===== JOINT 2: Roll (X-axis) - Arm raise/lower =====
  bool moving_x = fabs(gyro_x) > motion_threshold;
  if(moving_x) {
    theta2_velocity = gyro_x * VELOCITY_GAIN_JOINT2 * JOINT2_DIRECTION;
    theta2_velocity = constrain(theta2_velocity, -3.0, 3.0);
    theta2 += theta2_velocity * dt;
  }
  theta2 = constrain(theta2, JOINT2_MIN * PI/180.0, JOINT2_MAX * PI/180.0);
  
  // ===== JOINT 3: Pitch (Y-axis) - Bicep rotation =====
  bool moving_y = fabs(gyro_y) > motion_threshold;
  if(moving_y) {
    theta3_velocity = gyro_y * VELOCITY_GAIN_JOINT3 * JOINT3_DIRECTION;
    theta3_velocity = constrain(theta3_velocity, -3.0, 3.0);
    theta3 += theta3_velocity * dt;
  }
  theta3 = constrain(theta3, JOINT3_MIN * PI/180.0, JOINT3_MAX * PI/180.0);
}

// MPU 0x69 controls Joints 4, 5, 6 (RELATIVE to Joint 1 and Joint 2)
void updateRelativeJointsFromMPU69(float dt) {
  if (!mpuJ5_initialized) return;
  if (!gyro_bias_calibrated) return;
  
  // Get wrist gyro readings (rad/s)
  float wrist_gyro_y = (gy5_f - gyro_bias_wrist_y) * PI / 180.0;
  float wrist_gyro_z = (gz5_f - gyro_bias_wrist_z) * PI / 180.0;
  float wrist_gyro_x = (gx5_f - gyro_bias_wrist_x) * PI / 180.0;
  
  // ===== JOINT 5: Wrist Pitch (Y-axis) - ABSOLUTE =====
  bool moving_y = fabs(wrist_gyro_y) > motion_threshold;
  if(moving_y) {
    theta5_velocity = wrist_gyro_y * VELOCITY_GAIN_JOINT5 * JOINT5_DIRECTION;
    theta5_velocity = constrain(theta5_velocity, -2.0, 2.0);
    theta5 += theta5_velocity * dt;
  }
  theta5 = constrain(theta5, JOINT5_MIN * PI/180.0, JOINT5_MAX * PI/180.0);
  
  // ===== JOINT 4: Elbow (Z-axis) - RELATIVE to Joint 1 =====
  float base_gyro_z = (gz_f - gyro_bias_base_z) * PI / 180.0;
  float relative_gyro_z = wrist_gyro_z - base_gyro_z;
  
  bool moving_z = fabs(relative_gyro_z) > motion_threshold;
  if(moving_z) {
    theta4_velocity = relative_gyro_z * VELOCITY_GAIN_JOINT4 * JOINT4_DIRECTION;
    theta4_velocity = constrain(theta4_velocity, -3.0, 3.0);
    theta4 += theta4_velocity * dt;
  }
  theta4 = constrain(theta4, JOINT4_MIN * PI/180.0, JOINT4_MAX * PI/180.0);
  
  // ===== JOINT 6: Wrist Flexion (X-axis) - RELATIVE to Joint 2 =====
  float base_gyro_x = (gx_f - gyro_bias_base_x) * PI / 180.0;
  float relative_gyro_x = wrist_gyro_x - base_gyro_x;
  
  bool moving_x = fabs(relative_gyro_x) > motion_threshold;
  if(moving_x) {
    theta6_velocity = relative_gyro_x * VELOCITY_GAIN_JOINT6 * JOINT6_DIRECTION;
    theta6_velocity = constrain(theta6_velocity, -3.0, 3.0);
    theta6 += theta6_velocity * dt;
  }
  theta6 = constrain(theta6, JOINT6_MIN * PI/180.0, JOINT6_MAX * PI/180.0);
}

// ================= CALIBRATION FUNCTION =================
void calibrateRobot() {
  Serial.println("\n*** CALIBRATION - Re-calibrating gyros and resetting position ***");
  
  // Re-calibrate gyro bias
  if (mpu123_initialized && mpuJ5_initialized) {
    calibrateGyroBias();
  }
  
  // Calibrate magnetometer if available
  if (mag_initialized) {
    calibrateMagnetometer();
  }
  
  // Reset to initial position
  resetToInitialPosition();
  
  Serial.println("*** CALIBRATION COMPLETE ***");
}

// ================= SEND JOINT ANGLES =================
void sendJointAngles() {
  static bool lastButtonState = HIGH;
  
  bool currentButtonState = digitalRead(BUTTON_PIN);
  
  // Button pressed - enter reset mode
  if (lastButtonState == HIGH && currentButtonState == LOW) {
    resetMode = true;
    resetStartTime = millis();
    Serial.println("\n*** RESET MODE ACTIVE - Output will show initial position for 1 second ***");
  }
  
  lastButtonState = currentButtonState;
  
  // Handle reset mode
  if (resetMode) {
    unsigned long elapsed = millis() - resetStartTime;
    
    if (elapsed < RESET_DELAY) {
      // During the 1 second reset period, output the initial position
      Serial.printf("%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n", 
                    INITIAL_JOINT1, INITIAL_JOINT2, INITIAL_JOINT3,
                    INITIAL_JOINT4, INITIAL_JOINT5, INITIAL_JOINT6);
      return;
    } else {
      // After 1 second, perform the actual reset and exit reset mode
      resetToInitialPosition();
      resetMode = false;
      Serial.println("*** RESET MODE ENDED - Now tracking changes from initial position ***\n");
    }
  }
  
  // Calculate OUTPUT angles = INITIAL_ANGLE + CHANGE_IN_ANGLE
  float change1 = (theta1 - theta1_initial) * 180.0 / PI;
  float change2 = (theta2 - theta2_initial) * 180.0 / PI;
  float change3 = (theta3 - theta3_initial) * 180.0 / PI;
  float change4 = (theta4 - theta4_initial) * 180.0 / PI;
  float change5 = (theta5 - theta5_initial) * 180.0 / PI;
  float change6 = (theta6 - theta6_initial) * 180.0 / PI;
  
  // Output angles = initial position + change
  float d1 = INITIAL_JOINT1 + change1;
  float d2 = INITIAL_JOINT2 + change2;
  float d3 = INITIAL_JOINT3 + change3;
  float d4 = INITIAL_JOINT4 + change4;
  float d5 = INITIAL_JOINT5 + change5;
  float d6 = INITIAL_JOINT6 + change6;
  
  // Apply joint limits
  d1 = constrain(d1, JOINT1_MIN, JOINT1_MAX);
  d2 = constrain(d2, JOINT2_MIN, JOINT2_MAX);
  d3 = constrain(d3, JOINT3_MIN, JOINT3_MAX);
  d4 = constrain(d4, JOINT4_MIN, JOINT4_MAX);
  d5 = constrain(d5, JOINT5_MIN, JOINT5_MAX);
  d6 = constrain(d6, JOINT6_MIN, JOINT6_MAX);
  
  // Output the angles
  Serial.printf("%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n", d1, d2, d3, d4, d5, d6);
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n\n========================================");
  Serial.println("KUKA LBR iiwa Robot Control System");
  Serial.println("Joint 1: Gyro + Magnetometer Fusion (Drift-Free Yaw)");
  Serial.println("Initial Position: (0, -90, 90, 0, 90, 0)");
  Serial.println("========================================\n");
  
  Serial.printf("Yaw Filter: %.0f%% Gyro, %.0f%% Magnetometer\n", 
                YAW_FILTER_ALPHA * 100, (1.0 - YAW_FILTER_ALPHA) * 100);
  Serial.println();
  
  Serial.printf("Link Lengths:\n");
  Serial.printf("  L1 (J1 to J2): %.3f mm\n", L1);
  Serial.printf("  L2 (J2 to J4): %.3f mm\n", L2);
  Serial.printf("  L3 (J4 to J6): %.3f mm\n", L3);
  Serial.println();
  
  // Initialize I2C
  Wire.begin(8, 9);
  Wire.setClock(100000);
  delay(100);
  
  // Scan I2C bus
  Serial.println("Scanning I2C bus...");
  for (byte addr = 1; addr < 127; addr++) {
    if (checkI2CDevice(addr)) {
      Serial.printf("  Device found at 0x%02X\n", addr);
    }
  }
  
  // Initialize MPUs
  mpu123_initialized = initMPU123();
  mpuJ5_initialized = initMPUJ5();
  mag_initialized = initMag();
  
  if (!mpu123_initialized) {
    Serial.println("\n!!! WARNING: MPU at 0x68 not working !!!");
  }
  
  if (!mpuJ5_initialized) {
    Serial.println("\n!!! WARNING: MPU at 0x69 not working !!!");
  }
  
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Calibrate gyro bias for all axes
  if (mpu123_initialized && mpuJ5_initialized) {
    calibrateGyroBias();
  }
  
  // Calibrate magnetometer if available
  if (mag_initialized) {
    calibrateMagnetometer();
  }
  
  // Reset to initial position at system start
  resetToInitialPosition();
  
  lastTime = micros();
  
  Serial.println("\nSystem Ready!");
  Serial.println("Press button to reset to initial position (0,-90,90,0,90,0)");
  Serial.println("Hold button for 2 seconds to recalibrate gyros AND magnetometer");
  Serial.println("");
  Serial.println("CONTROL MAPPING:");
  Serial.println("  MPU 0x68 (Upper Arm):");
  Serial.println("    Joint 1 = Z-axis (Yaw) - Gyro + Magnetometer Fusion (Drift-free)");
  Serial.println("    Joint 2 = X-axis (Roll)");
  Serial.println("    Joint 3 = Y-axis (Pitch)");
  Serial.println("");
  Serial.println("  MPU 0x69 (Forearm/Wrist):");
  Serial.println("    Joint 4 = Z-axis (Yaw) - RELATIVE to J1");
  Serial.println("    Joint 5 = Y-axis (Pitch)");
  Serial.println("    Joint 6 = X-axis (Roll) - RELATIVE to J2");
  Serial.println("========================================\n");
  
  delay(1000);
}

// ================= MAIN LOOP =================
void loop() {
  // Button handling for long press calibration (2 seconds)
  bool currentButtonState = digitalRead(BUTTON_PIN);
  
  if (currentButtonState == LOW) {
    if (!calibrating) {
      calibrating = true;
      calibrationStartTime = millis();
    }
    
    // Long press (2 seconds) triggers full calibration
    if (millis() - calibrationStartTime >= CALIBRATION_DURATION) {
      calibrateRobot();
      calibrating = false;
    }
  } else {
    calibrating = false;
  }
  
  // Time management
  unsigned long now = micros();
  dt = (now - lastTime) / 1000000.0;
  lastTime = now;
  
  if(dt > 0.01) dt = 0.01;
  if(dt <= 0) dt = 0.001;
  
  // Read MPU 0x68 (Upper Arm) - Controls Joints 1, 2, 3
  if(mpu123_initialized && readMPU123()) {
    filterData();
    
    if (mag_initialized) readMag();
    
    // Update joints from MPU 0x68
    updateJointsFromMPU68(dt);
  }
  
  // Read MPU 0x69 (Forearm/Wrist) - Controls Joints 4, 5, 6
  if(mpuJ5_initialized && readMPUJ5()) {
    filterDataJ5();
    
    // Update relative joints from MPU 0x69
    updateRelativeJointsFromMPU69(dt);
  }
  
  // Apply smoothing
  theta1_s = smooth(theta1, theta1_s);
  theta2_s = smooth(theta2, theta2_s);
  theta3_s = smooth(theta3, theta3_s);
  theta4_s = smooth(theta4, theta4_s);
  theta5_s = smooth(theta5, theta5_s);
  theta6_s = smooth(theta6, theta6_s);
  
  // Send angles
  sendJointAngles();
  
  // Print debug info
  printMPUData();
  
  delay(10);
}