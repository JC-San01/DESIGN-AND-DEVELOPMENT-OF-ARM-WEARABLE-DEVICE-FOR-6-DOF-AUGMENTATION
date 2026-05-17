#include <Wire.h>
#include <math.h>

// ================= PIN DEFINITIONS =================
#define BUTTON_PIN 4

// ================= MPU9250 DEFINITIONS =================
#define MPU_ADDR_BASE 0x68    // MPU on upper arm
#define MPU_ADDR_WRIST 0x69   // MPU on wrist

// MPU9250 Registers
#define ACCEL_XOUT_H 0x3B
#define PWR_MGMT_1 0x6B
#define GYRO_CONFIG 0x1B
#define ACCEL_CONFIG 0x1C
#define WHO_AM_I 0x75

// ================= DIRECTION CONTROL =================
const float JOINT1_DIRECTION = -1;
const float JOINT2_DIRECTION = 1;
const float JOINT3_DIRECTION = -1;
const float JOINT4_DIRECTION = -1;
const float JOINT5_DIRECTION = -1;
const float JOINT6_DIRECTION = 1;

// ================= VELOCITY CONTROL GAINS =================
const float VELOCITY_GAIN_JOINT1 = 1.0;
const float VELOCITY_GAIN_JOINT2 = 1.0;
const float VELOCITY_GAIN_JOINT3 = 0.9;
const float VELOCITY_GAIN_JOINT4 = 1.0;
const float VELOCITY_GAIN_JOINT5 = 0.8;
const float VELOCITY_GAIN_JOINT6 = 0.7;

const float MOTION_THRESHOLD = 0.05;  // rad/s

// ================= KALMAN FILTER FOR VELOCITY =================
// This Kalman filter smooths the gyro velocity signal
struct KalmanVelocityFilter {
  float Q;        // Process noise covariance (how much we trust the model)
  float R;        // Measurement noise covariance (how much we trust the sensor)
  float P;        // Estimation error covariance
  float X;        // Filtered velocity state
  float K;        // Kalman gain
};

// Initialize Kalman filters for each axis (base and wrist)
KalmanVelocityFilter kf_base_x, kf_base_y, kf_base_z;
KalmanVelocityFilter kf_wrist_x, kf_wrist_y, kf_wrist_z;

// ================= DATA STRUCTURES =================
// Base MPU (Upper arm) - raw and filtered
float gx_base_raw = 0, gy_base_raw = 0, gz_base_raw = 0;
float gx_base_filt = 0, gy_base_filt = 0, gz_base_filt = 0;

// Wrist MPU - raw and filtered
float gx_wrist_raw = 0, gy_wrist_raw = 0, gz_wrist_raw = 0;
float gx_wrist_filt = 0, gy_wrist_filt = 0, gz_wrist_filt = 0;

// ================= TIME =================
unsigned long lastTime;
float dt;

// ================= JOINT ANGLES (for reference reset only) =================
float theta1 = 0, theta2 = 0, theta3 = 0;
float theta4 = 0, theta5 = 0, theta6 = 0;

// ================= INITIAL REFERENCE =================
float theta1_initial = 0, theta2_initial = 0, theta3_initial = 0;
float theta4_initial = 0, theta5_initial = 0, theta6_initial = 0;

// ================= JOINT VELOCITIES (rad/s) =================
float theta1_velocity = 0, theta2_velocity = 0, theta3_velocity = 0;
float theta4_velocity = 0, theta5_velocity = 0, theta6_velocity = 0;

// ================= BUTTON STATE =================
bool lastButtonState = HIGH;
bool calibrating = false;
unsigned long calibrationStartTime = 0;
const unsigned long CALIBRATION_DURATION = 2000;

bool resetRequested = false;
unsigned long resetStartTime = 0;
const unsigned long RESET_SIGNAL_DURATION = 100;

// ================= MPU STATUS =================
bool mpu_base_initialized = false;
bool mpu_wrist_initialized = false;

// ================= GYRO BIAS =================
float gyro_bias_base_x = 0, gyro_bias_base_y = 0, gyro_bias_base_z = 0;
float gyro_bias_wrist_x = 0, gyro_bias_wrist_y = 0, gyro_bias_wrist_z = 0;
bool gyro_bias_calibrated = false;

// ================= FUNCTION PROTOTYPES =================
bool checkI2CDevice(byte addr);
void writeReg(byte addr, byte reg, byte val);
bool readBytes(byte addr, byte reg, byte count, byte *dest);
bool initMPU(byte addr, const char* name);
bool readMPU(byte addr, float* gx, float* gy, float* gz);
void scanI2CBus();
void calibrateGyroBias();
void kalmanVelocityInit(KalmanVelocityFilter* kf, float Q, float R, float initial_X);
float kalmanVelocityUpdate(KalmanVelocityFilter* kf, float measurement, float dt);
void updateJointVelocitiesFromSensors(float dt);
void resetReferencePosition();
void calibrateRobot();
void sendDataToPython();

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
  if (Wire.available() >= count) {
    for(byte i = 0; i < count; i++) dest[i] = Wire.read();
    return true;
  }
  return false;
}

void resetMPU(byte addr) {
  writeReg(addr, PWR_MGMT_1, 0x80);
  delay(100);
  writeReg(addr, PWR_MGMT_1, 0x00);
  delay(100);
}

// Scan I2C bus
void scanI2CBus() {
  Serial.println("\nScanning I2C bus...");
  byte count = 0;
  
  for(byte addr = 1; addr < 127; addr++) {
    if(checkI2CDevice(addr)) {
      Serial.printf("  Device at 0x%02X", addr);
      count++;
      
      byte whoami;
      if(readBytes(addr, WHO_AM_I, 1, &whoami)) {
        if(whoami == 0x71) Serial.printf(" -> MPU9250\n");
        else Serial.printf(" -> WHO_AM_I: 0x%02X\n", whoami);
      } else {
        Serial.println();
      }
    }
  }
  
  if(count == 0) Serial.println("  No devices found!");
  Serial.printf("Total: %d devices\n\n", count);
}

// ================= MPU INITIALIZATION =================
bool initMPU(byte addr, const char* name) {
  Serial.printf("Initializing MPU at 0x%02X (%s)...\n", addr, name);
  
  if (!checkI2CDevice(addr)) {
    Serial.printf("  ERROR: Device not found!\n");
    return false;
  }
  
  resetMPU(addr);
  writeReg(addr, PWR_MGMT_1, 0x00);
  delay(10);
  writeReg(addr, GYRO_CONFIG, 0x00);  // ±250 deg/s
  delay(10);
  writeReg(addr, ACCEL_CONFIG, 0x00);
  delay(10);
  
  Serial.printf("  SUCCESS! MPU at 0x%02X initialized\n", addr);
  return true;
}

// Read gyro data
bool readMPU(byte addr, float* gx, float* gy, float* gz) {
  byte data[14];
  if (!readBytes(addr, ACCEL_XOUT_H, 14, data)) return false;
  
  int16_t raw_gx = (int16_t)(data[8] << 8 | data[9]);
  int16_t raw_gy = (int16_t)(data[10] << 8 | data[11]);
  int16_t raw_gz = (int16_t)(data[12] << 8 | data[13]);
  
  // Conversion to DPS
  *gx = raw_gx / 131.0;
  *gy = raw_gy / 131.0;
  *gz = raw_gz / 131.0;
  
  return true;
}

// ================= KALMAN FILTER FOR VELOCITY SMOOTHING =================
// This is a 1D Kalman filter that smooths noisy velocity measurements
// It assumes constant velocity between measurements (random walk model)

void kalmanVelocityInit(KalmanVelocityFilter* kf, float Q, float R, float initial_X) {
  kf->Q = Q;           // Process noise - lower = smoother, higher = more responsive
  kf->R = R;           // Measurement noise - higher = trust sensor less
  kf->P = 1.0;         // Initial uncertainty
  kf->X = initial_X;   // Initial state
  kf->K = 0;
}

float kalmanVelocityUpdate(KalmanVelocityFilter* kf, float measurement, float dt) {
  // Prediction step (assume constant velocity)
  // x_pred = x_prev (no change since we're filtering velocity directly)
  
  // Update error covariance prediction
  // P_pred = P_prev + Q * dt (process noise accumulates over time)
  kf->P = kf->P + kf->Q * dt;
  
  // Kalman gain
  // K = P_pred / (P_pred + R)
  kf->K = kf->P / (kf->P + kf->R);
  
  // Update step - correct prediction with measurement
  // x_new = x_pred + K * (measurement - x_pred)
  kf->X = kf->X + kf->K * (measurement - kf->X);
  
  // Update error covariance
  // P_new = (1 - K) * P_pred
  kf->P = (1.0 - kf->K) * kf->P;
  
  return kf->X;
}

// ================= GYRO BIAS CALIBRATION =================
void calibrateGyroBias() {
  Serial.println("\n=== Calibrating Gyro Bias ===");
  Serial.println("Keep MPUs completely still...");
  
  float sum_gx_base = 0, sum_gy_base = 0, sum_gz_base = 0;
  float sum_gx_wrist = 0, sum_gy_wrist = 0, sum_gz_wrist = 0;
  int samples = 0;
  unsigned long startTime = millis();
  
  while (millis() - startTime < 2000) {
    float gx_t, gy_t, gz_t;
    
    if (readMPU(MPU_ADDR_BASE, &gx_t, &gy_t, &gz_t)) {
      sum_gx_base += gx_t;
      sum_gy_base += gy_t;
      sum_gz_base += gz_t;
    }
    
    if (readMPU(MPU_ADDR_WRIST, &gx_t, &gy_t, &gz_t)) {
      sum_gx_wrist += gx_t;
      sum_gy_wrist += gy_t;
      sum_gz_wrist += gz_t;
    }
    
    samples++;
    delay(5);
  }
  
  if (samples > 0) {
    gyro_bias_base_x = sum_gx_base / samples;
    gyro_bias_base_y = sum_gy_base / samples;
    gyro_bias_base_z = sum_gz_base / samples;
    gyro_bias_wrist_x = sum_gx_wrist / samples;
    gyro_bias_wrist_y = sum_gy_wrist / samples;
    gyro_bias_wrist_z = sum_gz_wrist / samples;
    gyro_bias_calibrated = true;
    
    Serial.println("Gyro bias calibration complete!");
    Serial.printf("  Base  MPU: X=%6.3f, Y=%6.3f, Z=%6.3f deg/s\n", 
                  gyro_bias_base_x, gyro_bias_base_y, gyro_bias_base_z);
    Serial.printf("  Wrist MPU: X=%6.3f, Y=%6.3f, Z=%6.3f deg/s\n", 
                  gyro_bias_wrist_x, gyro_bias_wrist_y, gyro_bias_wrist_z);
  }
}

// ================= JOINT VELOCITY CALCULATION WITH KALMAN FILTER =================
void updateJointVelocitiesFromSensors(float dt) {
  if (!gyro_bias_calibrated) return;
  
  // Read raw gyro data
  if (mpu_base_initialized) {
    readMPU(MPU_ADDR_BASE, &gx_base_raw, &gy_base_raw, &gz_base_raw);
  }
  
  if (mpu_wrist_initialized) {
    readMPU(MPU_ADDR_WRIST, &gx_wrist_raw, &gy_wrist_raw, &gz_wrist_raw);
  }
  
  // Remove bias from raw measurements
  float gyro_z_raw_bias = (gz_base_raw - gyro_bias_base_z) * PI / 180.0;
  float gyro_x_raw_bias = (gx_base_raw - gyro_bias_base_x) * PI / 180.0;
  float gyro_y_raw_bias = (gy_base_raw - gyro_bias_base_y) * PI / 180.0;
  
  float wrist_gyro_z_raw_bias = (gz_wrist_raw - gyro_bias_wrist_z) * PI / 180.0;
  float wrist_gyro_y_raw_bias = (gy_wrist_raw - gyro_bias_wrist_y) * PI / 180.0;
  float wrist_gyro_x_raw_bias = (gx_wrist_raw - gyro_bias_wrist_x) * PI / 180.0;
  
  // Apply Kalman filter to each axis independently
  // This smooths the velocity signal and reduces noise
  gx_base_filt = kalmanVelocityUpdate(&kf_base_x, gyro_x_raw_bias, dt);
  gy_base_filt = kalmanVelocityUpdate(&kf_base_y, gyro_y_raw_bias, dt);
  gz_base_filt = kalmanVelocityUpdate(&kf_base_z, gyro_z_raw_bias, dt);
  
  gx_wrist_filt = kalmanVelocityUpdate(&kf_wrist_x, wrist_gyro_x_raw_bias, dt);
  gy_wrist_filt = kalmanVelocityUpdate(&kf_wrist_y, wrist_gyro_y_raw_bias, dt);
  gz_wrist_filt = kalmanVelocityUpdate(&kf_wrist_z, wrist_gyro_z_raw_bias, dt);
  
  // Calculate joint velocities using filtered gyro data
  theta1_velocity = gz_base_filt * VELOCITY_GAIN_JOINT1 * JOINT1_DIRECTION;
  theta2_velocity = gx_base_filt * VELOCITY_GAIN_JOINT2 * JOINT2_DIRECTION;
  theta3_velocity = gy_base_filt * VELOCITY_GAIN_JOINT3 * JOINT3_DIRECTION;
  theta4_velocity = (gz_wrist_filt - gz_base_filt) * VELOCITY_GAIN_JOINT4 * JOINT4_DIRECTION;
  theta5_velocity = gy_wrist_filt * VELOCITY_GAIN_JOINT5 * JOINT5_DIRECTION;
  theta6_velocity = (gx_wrist_filt - gx_base_filt) * VELOCITY_GAIN_JOINT6 * JOINT6_DIRECTION;
  
  // Apply velocity limits
  theta1_velocity = constrain(theta1_velocity, -3.0, 3.0);
  theta2_velocity = constrain(theta2_velocity, -3.0, 3.0);
  theta3_velocity = constrain(theta3_velocity, -3.0, 3.0);
  theta4_velocity = constrain(theta4_velocity, -3.0, 3.0);
  theta5_velocity = constrain(theta5_velocity, -3.0, 3.0);
  theta6_velocity = constrain(theta6_velocity, -3.0, 3.0);
  
  // Apply motion threshold
  if (fabs(theta1_velocity) < MOTION_THRESHOLD) theta1_velocity = 0;
  if (fabs(theta2_velocity) < MOTION_THRESHOLD) theta2_velocity = 0;
  if (fabs(theta3_velocity) < MOTION_THRESHOLD) theta3_velocity = 0;
  if (fabs(theta4_velocity) < MOTION_THRESHOLD) theta4_velocity = 0;
  if (fabs(theta5_velocity) < MOTION_THRESHOLD) theta5_velocity = 0;
  if (fabs(theta6_velocity) < MOTION_THRESHOLD) theta6_velocity = 0;
}

// ================= RESET FUNCTIONS =================
void resetReferencePosition() {
  Serial.println("\n*** RESETTING REFERENCE POSITION ***");
  
  theta1_initial = theta1;
  theta2_initial = theta2;
  theta3_initial = theta3;
  theta4_initial = theta4;
  theta5_initial = theta5;
  theta6_initial = theta6;
  
  theta1_velocity = 0;
  theta2_velocity = 0;
  theta3_velocity = 0;
  theta4_velocity = 0;
  theta5_velocity = 0;
  theta6_velocity = 0;
  
  Serial.println("*** REFERENCE POSITION RESET ***");
}

void calibrateRobot() {
  Serial.println("\n*** CALIBRATION - Re-calibrating sensors ***");
  
  if (mpu_base_initialized && mpu_wrist_initialized) {
    calibrateGyroBias();
  }
  
  resetReferencePosition();
  Serial.println("*** CALIBRATION COMPLETE ***");
}

// ================= SEND DATA TO PYTHON =================
void sendDataToPython() {
  static bool lastButtonState = HIGH;
  
  bool currentButtonState = digitalRead(BUTTON_PIN);
  
  if (lastButtonState == HIGH && currentButtonState == LOW) {
    resetRequested = true;
    resetStartTime = millis();
    Serial.println("\n>>> RESET REQUESTED <<<\n");
  }
  
  lastButtonState = currentButtonState;
  
  if (resetRequested) {
    if (millis() - resetStartTime < RESET_SIGNAL_DURATION) {
      Serial.println("RESET");
    } else {
      resetRequested = false;
      resetReferencePosition();
      Serial.println(">>> RESET COMPLETE - Resuming velocity data <<<\n");
      delay(100);
    }
    return;
  }
  
  // Send filtered joint velocities in degrees per second
  Serial.printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                theta1_velocity * 180.0 / PI,
                theta2_velocity * 180.0 / PI,
                theta3_velocity * 180.0 / PI,
                theta4_velocity * 180.0 / PI,
                theta5_velocity * 180.0 / PI,
                theta6_velocity * 180.0 / PI);
}

// ================= DEBUG PRINT =================
void printDebugInfo() {
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 2000) {
    lastPrint = millis();
    
   // Serial.println("\n--- Kalman Filter Debug ---");
   // Serial.printf("Base MPU - Raw: X=%6.2f, Y=%6.2f, Z=%6.2f deg/s\n", 
   //               (gx_base_raw - gyro_bias_base_x), 
   //               (gy_base_raw - gyro_bias_base_y), 
   //               (gz_base_raw - gyro_bias_base_z));
   // Serial.printf("Base MPU - Kalman: X=%6.2f, Y=%6.2f, Z=%6.2f deg/s\n",
   //               gx_base_filt * 180.0/PI,
   //               gy_base_filt * 180.0/PI,
   //               gz_base_filt * 180.0/PI);
   // Serial.printf("Joint Velocities: J1=%5.1f, J2=%5.1f, J3=%5.1f, J4=%5.1f, J5=%5.1f, J6=%5.1f deg/s\n",
   //               theta1_velocity * 180.0/PI, theta2_velocity * 180.0/PI,
   //               theta3_velocity * 180.0/PI, theta4_velocity * 180.0/PI,
   //               theta5_velocity * 180.0/PI, theta6_velocity * 180.0/PI);
   // Serial.println();
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n\n========================================");
  Serial.println("ARM MOTION CAPTURE - KALMAN VELOCITY FILTER");
  Serial.println("========================================\n");
  
  // Initialize I2C (ESP32 pins)
  Wire.begin(8, 9);
  Wire.setClock(100000);
  delay(100);
  
  // Scan I2C bus
  scanI2CBus();
  
  // Initialize MPUs
  mpu_base_initialized = initMPU(MPU_ADDR_BASE, "Base MPU");
  mpu_wrist_initialized = initMPU(MPU_ADDR_WRIST, "Wrist MPU");
  
  if (!mpu_base_initialized || !mpu_wrist_initialized) {
    Serial.println("\n!!! WARNING: Not all MPUs initialized !!!");
  }
  
  // Initialize Kalman filters for each axis
  // Tuning parameters:
  //   Q (process noise): lower = smoother output, higher = more responsive
  //   R (measurement noise): higher = trust sensor less, lower = trust sensor more
  // Start with Q=0.1, R=1.0 for good smoothing without too much lag
  
  kalmanVelocityInit(&kf_base_x, 0.1, 1.0, 0);
  kalmanVelocityInit(&kf_base_y, 0.1, 1.0, 0);
  kalmanVelocityInit(&kf_base_z, 0.1, 1.0, 0);
  kalmanVelocityInit(&kf_wrist_x, 0.1, 1.0, 0);
  kalmanVelocityInit(&kf_wrist_y, 0.1, 1.0, 0);
  kalmanVelocityInit(&kf_wrist_z, 0.1, 1.0, 0);
  
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Calibrate gyros
  if (mpu_base_initialized && mpu_wrist_initialized) {
    calibrateGyroBias();
  }
  
  resetReferencePosition();
  lastTime = micros();
  
  Serial.println("\nSystem Ready!");
  Serial.println("Kalman filter active - smoothing velocity output");
  Serial.println("Press button to send RESET command to Python");
  Serial.println("Hold button for 2 seconds to recalibrate sensors");
  Serial.println("\nOUTPUT: vel1,vel2,vel3,vel4,vel5,vel6 (deg/s)\n");
  
  delay(1000);
}

// ================= MAIN LOOP =================
void loop() {
  // Handle long press calibration
  bool currentButtonState = digitalRead(BUTTON_PIN);
  
  if (currentButtonState == LOW) {
    if (!calibrating) {
      calibrating = true;
      calibrationStartTime = millis();
    }
    
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
  
  // Update joint velocities with Kalman filtering
  updateJointVelocitiesFromSensors(dt);
  
  // Integrate velocities to track absolute angles (for reference reset)
  theta1 += theta1_velocity * dt;
  theta2 += theta2_velocity * dt;
  theta3 += theta3_velocity * dt;
  theta4 += theta4_velocity * dt;
  theta5 += theta5_velocity * dt;
  theta6 += theta6_velocity * dt;
  
  // Send data to Python
  sendDataToPython();
  
  // Debug output every 2 seconds
  printDebugInfo();
  
  delay(10);
}