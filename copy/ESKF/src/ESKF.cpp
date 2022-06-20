#ifndef NDEBUG
#define NDEBUG
#endif

#include <ESKF.hpp>
#include <cmath>

using namespace Eigen;

namespace eskf {

  ESKF::ESKF() {
    // zeros state_
    state_.quat_nominal = quat(1, 0, 0, 0);
    state_.vel = vec3(0, 0, 0);
    state_.pos = vec3(0, 0, 0);
    state_.gyro_bias = vec3(0, 0, 0);
    state_.accel_bias = vec3(0, 0, 0);
    state_.mag_I.setZero();
    state_.mag_B.setZero();

    //  zeros P_
    for (unsigned i = 0; i < k_num_states_; i++) {
      for (unsigned j = 0; j < k_num_states_; j++) {
        P_[i][j] = 0.0f;
      }
    }

    imu_down_sampled_.delta_ang.setZero();
    imu_down_sampled_.delta_vel.setZero();
    imu_down_sampled_.delta_ang_dt = 0.0f;
    imu_down_sampled_.delta_vel_dt = 0.0f;

    q_down_sampled_.w() = 1.0f;
    q_down_sampled_.x() = 0.0f;
    q_down_sampled_.y() = 0.0f;
    q_down_sampled_.z() = 0.0f;

    imu_buffer_.allocate(imu_buffer_length_);
    for (int index = 0; index < imu_buffer_length_; index++) {
      imuSample imu_sample_init = {};
      imu_buffer_.push(imu_sample_init);
    }

    ext_vision_buffer_.allocate(obs_buffer_length_);
    for (int index = 0; index < obs_buffer_length_; index++) {
      extVisionSample ext_vision_sample_init = {};
      ext_vision_buffer_.push(ext_vision_sample_init);
    }

    gps_buffer_.allocate(obs_buffer_length_);
    for (int index = 0; index < obs_buffer_length_; index++) {
      gpsSample gps_sample_init = {};
      gps_buffer_.push(gps_sample_init);
    }

    opt_flow_buffer_.allocate(obs_buffer_length_);
    for (int index = 0; index < obs_buffer_length_; index++) {
      optFlowSample opt_flow_sample_init = {};
      opt_flow_buffer_.push(opt_flow_sample_init);
    }

    range_buffer_.allocate(obs_buffer_length_);
    for (int index = 0; index < obs_buffer_length_; index++) {
      rangeSample range_sample_init = {};
      range_buffer_.push(range_sample_init);
    }

    mag_buffer_.allocate(obs_buffer_length_);
    for (int index = 0; index < obs_buffer_length_; index++) {
      magSample mag_sample_init = {};
      mag_buffer_.push(mag_sample_init);
    }

    dt_ekf_avg_ = 0.001f * (scalar_t)(FILTER_UPDATE_PERIOD_MS);

    ///< filter initialisation
    NED_origin_initialised_ = false;
    filter_initialised_ = false;
    terrain_initialised_ = false;

    imu_updated_ = false;
    memset(vel_pos_innov_, 0, 6*sizeof(scalar_t));
    last_known_posNED_ = vec3(0, 0, 0);
  }

  void ESKF::initialiseCovariance() {
    // define the initial angle uncertainty as variances for a rotation vector

    for (unsigned i = 0; i < k_num_states_; i++) {
      for (unsigned j = 0; j < k_num_states_; j++) {
	P_[i][j] = 0.0f;
      }
    }

    // calculate average prediction time step in sec
    float dt = 0.001f * (float)FILTER_UPDATE_PERIOD_MS;

    vec3 rot_vec_var;
    rot_vec_var(2) = rot_vec_var(1) = rot_vec_var(0) = sq(initial_tilt_err_);

    // update the quaternion state covariances
    initialiseQuatCovariances(rot_vec_var);

    // velocity
    P_[4][4] = sq(fmaxf(vel_noise_, 0.01f));
    P_[5][5] = P_[4][4];
    P_[6][6] = sq(1.5f) * P_[4][4];

    // position
    P_[7][7] = sq(fmaxf(pos_noise_, 0.01f));
    P_[8][8] = P_[7][7];
    P_[9][9] = sq(fmaxf(range_noise_, 0.01f));

    // gyro bias
    P_[10][10] = sq(switch_on_gyro_bias_ * dt);
    P_[11][11] = P_[10][10];
    P_[12][12] = P_[10][10];

    P_[13][13] = sq(switch_on_accel_bias_ * dt);
    P_[14][14] = P_[13][13];
    P_[15][15] = P_[13][13];
    // variances for optional states

    // earth frame and body frame magnetic field
    // set to observation variance
    for (uint8_t index = 16; index <= 21; index ++) {
      P_[index][index] = sq(mag_noise_);
    }
  }
  
  bool ESKF::initializeFilter() {
    scalar_t pitch = 0.0;
    scalar_t roll = 0.0;
    scalar_t yaw = 0.0;
    imuSample imu_init = imu_buffer_.get_newest();
    static vec3 delVel_sum(0, 0, 0); ///< summed delta velocity (m/sec)
    delVel_sum += imu_init.delta_vel;
    if (delVel_sum.norm() > 0.001) {
      delVel_sum.normalize();
      pitch = asin(delVel_sum(0));
      roll = atan2(-delVel_sum(1), -delVel_sum(2));
    } else {
      return false;
    }
    // calculate initial tilt alignment
    state_.quat_nominal = AngleAxis<scalar_t>(yaw, vec3::UnitZ()) * AngleAxis<scalar_t>(pitch, vec3::UnitY()) * AngleAxis<scalar_t>(roll, vec3::UnitX());
    // update transformation matrix from body to world frame
    R_to_earth_ = quat_to_invrotmat(state_.quat_nominal);
    initialiseCovariance();
    return true;    
  }
  
  bool ESKF::collect_imu(imuSample &imu) {
    // accumulate and downsample IMU data across a period FILTER_UPDATE_PERIOD_MS long

    // copy imu data to local variables
    imu_sample_new_.delta_ang	= imu.delta_ang;
    imu_sample_new_.delta_vel	= imu.delta_vel;
    imu_sample_new_.delta_ang_dt = imu.delta_ang_dt;
    imu_sample_new_.delta_vel_dt = imu.delta_vel_dt;
    imu_sample_new_.time_us	= imu.time_us;

    // accumulate the time deltas
    imu_down_sampled_.delta_ang_dt += imu.delta_ang_dt;
    imu_down_sampled_.delta_vel_dt += imu.delta_vel_dt;

    // use a quaternion to accumulate delta angle data
    // this quaternion represents the rotation from the start to end of the accumulation period
    quat delta_q(1, 0, 0, 0);
    quat res = from_axis_angle(imu.delta_ang);
    delta_q = delta_q * res;
    q_down_sampled_ = q_down_sampled_ * delta_q;
    q_down_sampled_.normalize();

    // rotate the accumulated delta velocity data forward each time so it is always in the updated rotation frame
    mat3 delta_R = quat2dcm(delta_q.inverse());
    imu_down_sampled_.delta_vel = delta_R * imu_down_sampled_.delta_vel;

    // accumulate the most recent delta velocity data at the updated rotation frame
    // assume effective sample time is halfway between the previous and current rotation frame
    imu_down_sampled_.delta_vel += (imu_sample_new_.delta_vel + delta_R * imu_sample_new_.delta_vel) * 0.5f;

    // if the target time delta between filter prediction steps has been exceeded
    // write the accumulated IMU data to the ring buffer
    scalar_t target_dt = (scalar_t)(FILTER_UPDATE_PERIOD_MS) / 1000;

    if (imu_down_sampled_.delta_ang_dt >= target_dt - imu_collection_time_adj_) {

      // accumulate the amount of time to advance the IMU collection time so that we meet the
      // average EKF update rate requirement
      imu_collection_time_adj_ += 0.01f * (imu_down_sampled_.delta_ang_dt - target_dt);
      imu_collection_time_adj_ = constrain(imu_collection_time_adj_, -0.5f * target_dt, 0.5f * target_dt);

      imu.delta_ang     = to_axis_angle(q_down_sampled_);
      imu.delta_vel     = imu_down_sampled_.delta_vel;
      imu.delta_ang_dt  = imu_down_sampled_.delta_ang_dt;
      imu.delta_vel_dt  = imu_down_sampled_.delta_vel_dt;

      imu_down_sampled_.delta_ang.setZero();
      imu_down_sampled_.delta_vel.setZero();
      imu_down_sampled_.delta_ang_dt = 0.0f;
      imu_down_sampled_.delta_vel_dt = 0.0f;
      q_down_sampled_.w() = 1.0f;
      q_down_sampled_.x() = q_down_sampled_.y() = q_down_sampled_.z() = 0.0f;

      return true;
    }

    min_obs_interval_us_ = (imu_sample_new_.time_us - imu_sample_delayed_.time_us) / (obs_buffer_length_ - 1);

    return false;
  }
  
  void ESKF::run(const vec3 &w, const vec3 &a, uint64_t time_us, scalar_t dt) {
    // convert FLU to FRD body frame IMU data
    vec3 gyro_b = q_FLU2FRD.toRotationMatrix() * w;
    vec3 accel_b = q_FLU2FRD.toRotationMatrix() * a;

    vec3 delta_ang = vec3(gyro_b.x(), gyro_b.y(), gyro_b.z()) * dt; // current delta angle  (rad)
    vec3 delta_vel = vec3(accel_b.x(), accel_b.y(), accel_b.z()) * dt; //current delta velocity (m/s)

    // copy data
    imuSample imu_sample_new = {};
    imu_sample_new.delta_ang = delta_ang;
    imu_sample_new.delta_vel = delta_vel;
    imu_sample_new.delta_ang_dt = dt;
    imu_sample_new.delta_vel_dt = dt;
    imu_sample_new.time_us = time_us;
    
    time_last_imu_ = time_us;
        
    if(collect_imu(imu_sample_new)) {
      imu_buffer_.push(imu_sample_new);
      imu_updated_ = true;
      // get the oldest data from the buffer
      imu_sample_delayed_ = imu_buffer_.get_oldest();
    } else {
      imu_updated_ = false;
      return;
    }
    
    if (!filter_initialised_) {
      filter_initialised_ = initializeFilter();

      if (!filter_initialised_) {
        return;
      }
    }
    
    if(!imu_updated_) return;
    
    // apply imu bias corrections
    vec3 corrected_delta_ang = imu_sample_delayed_.delta_ang - state_.gyro_bias;
    vec3 corrected_delta_vel = imu_sample_delayed_.delta_vel - state_.accel_bias; 
    
    // convert the delta angle to a delta quaternion
    quat dq;
    dq = from_axis_angle(corrected_delta_ang);
    // rotate the previous quaternion by the delta quaternion using a quaternion multiplication
    state_.quat_nominal = state_.quat_nominal * dq;
    // quaternions must be normalised whenever they are modified
    state_.quat_nominal.normalize();
    
    // save the previous value of velocity so we can use trapezoidal integration
    vec3 vel_last = state_.vel;
    
    // update transformation matrix from body to world frame
    R_to_earth_ = quat_to_invrotmat(state_.quat_nominal);
    
    // Calculate an earth frame delta velocity
    vec3 corrected_delta_vel_ef = R_to_earth_ * corrected_delta_vel;
        
    // calculate the increment in velocity using the current orientation
    state_.vel += corrected_delta_vel_ef;

    // compensate for acceleration due to gravity
    state_.vel(2) += kOneG * imu_sample_delayed_.delta_vel_dt;
        
    // predict position states via trapezoidal integration of velocity
    state_.pos += (vel_last + state_.vel) * imu_sample_delayed_.delta_vel_dt * 0.5f;
        
    constrainStates();
        
    // calculate an average filter update time
    scalar_t input = 0.5f * (imu_sample_delayed_.delta_vel_dt + imu_sample_delayed_.delta_ang_dt);

    // filter and limit input between -50% and +100% of nominal value
    input = constrain(input, 0.0005f * (scalar_t)(FILTER_UPDATE_PERIOD_MS), 0.002f * (scalar_t)(FILTER_UPDATE_PERIOD_MS));
    dt_ekf_avg_ = 0.99f * dt_ekf_avg_ + 0.01f * input;
    
    predictCovariance();
    controlFusionModes();
  }
  
  void ESKF::predictCovariance() {
    // error-state jacobian
    // assign intermediate state variables
    scalar_t q0 = state_.quat_nominal.w();
    scalar_t q1 = state_.quat_nominal.x();
    scalar_t q2 = state_.quat_nominal.y();
    scalar_t q3 = state_.quat_nominal.z();

    scalar_t dax = imu_sample_delayed_.delta_ang(0);
    scalar_t day = imu_sample_delayed_.delta_ang(1);
    scalar_t daz = imu_sample_delayed_.delta_ang(2);

    scalar_t dvx = imu_sample_delayed_.delta_vel(0);
    scalar_t dvy = imu_sample_delayed_.delta_vel(1);
    scalar_t dvz = imu_sample_delayed_.delta_vel(2);

    scalar_t dax_b = state_.gyro_bias(0);
    scalar_t day_b = state_.gyro_bias(1);
    scalar_t daz_b = state_.gyro_bias(2);

    scalar_t dvx_b = state_.accel_bias(0);
    scalar_t dvy_b = state_.accel_bias(1);
    scalar_t dvz_b = state_.accel_bias(2);
	  
    // compute noise variance for stationary processes
    scalar_t process_noise[k_num_states_] = {};
    
    scalar_t dt = constrain(imu_sample_delayed_.delta_ang_dt, 0.0005f * (scalar_t)(FILTER_UPDATE_PERIOD_MS), 0.002f * (scalar_t)(FILTER_UPDATE_PERIOD_MS));
    
    // convert rate of change of rate gyro bias (rad/s**2) as specified by the parameter to an expected change in delta angle (rad) since the last update
    scalar_t d_ang_bias_sig = dt * dt * constrain(gyro_bias_p_noise_, 0.0f, 1.0f);

    // convert rate of change of accelerometer bias (m/s**3) as specified by the parameter to an expected change in delta velocity (m/s) since the last update
    scalar_t d_vel_bias_sig = dt * dt * constrain(accel_bias_p_noise_, 0.0f, 1.0f);

    // Don't continue to grow the earth field variances if they are becoming too large or we are not doing 3-axis fusion as this can make the covariance matrix badly conditioned
    scalar_t mag_I_sig;

    if (mag_3D_ && (P_[16][16] + P_[17][17] + P_[18][18]) < 0.1f) {
      mag_I_sig = dt * constrain(mage_p_noise_, 0.0f, 1.0f);
    } else {
      mag_I_sig = 0.0f;
    }

    // Don't continue to grow the body field variances if they is becoming too large or we are not doing 3-axis fusion as this can make the covariance matrix badly conditioned
    scalar_t mag_B_sig;

    if (mag_3D_ && (P_[19][19] + P_[20][20] + P_[21][21]) < 0.1f) {
      mag_B_sig = dt * constrain(magb_p_noise_, 0.0f, 1.0f);
    } else {
      mag_B_sig = 0.0f;
    }

    // Construct the process noise variance diagonal for those states with a stationary process model
    // These are kinematic states and their error growth is controlled separately by the IMU noise variances
    for (unsigned i = 0; i <= 9; i++) {
      process_noise[i] = 0.0;
    }

    // delta angle bias states
    process_noise[12] = process_noise[11] = process_noise[10] = sq(d_ang_bias_sig);
    // delta_velocity bias states
    process_noise[15] = process_noise[14] = process_noise[13] = sq(d_vel_bias_sig);
    // earth frame magnetic field states
    process_noise[18] = process_noise[17] = process_noise[16] = sq(mag_I_sig);
    // body frame magnetic field states
    process_noise[21] = process_noise[20] = process_noise[19] = sq(mag_B_sig);

    // assign IMU noise variances
    // inputs to the system are 3 delta angles and 3 delta velocities
    scalar_t daxVar, dayVar, dazVar;
    scalar_t dvxVar, dvyVar, dvzVar;
    daxVar = dayVar = dazVar = sq(dt * gyro_noise_); // gyro prediction variance TODO get variance from sensor
    dvxVar = dvyVar = dvzVar = sq(dt * accel_noise_); //accel prediction variance TODO get variance from sensor

    // intermediate calculations
    scalar_t SF[21];
    SF[0] = dvz - dvz_b;
    SF[1] = dvy - dvy_b;
    SF[2] = dvx - dvx_b;
    SF[3] = 2*q1*SF[2] + 2*q2*SF[1] + 2*q3*SF[0];
    SF[4] = 2*q0*SF[1] - 2*q1*SF[0] + 2*q3*SF[2];
    SF[5] = 2*q0*SF[2] + 2*q2*SF[0] - 2*q3*SF[1];
    SF[6] = day/2 - day_b/2;
    SF[7] = daz/2 - daz_b/2;
    SF[8] = dax/2 - dax_b/2;
    SF[9] = dax_b/2 - dax/2;
    SF[10] = daz_b/2 - daz/2;
    SF[11] = day_b/2 - day/2;
    SF[12] = 2*q1*SF[1];
    SF[13] = 2*q0*SF[0];
    SF[14] = q1/2;
    SF[15] = q2/2;
    SF[16] = q3/2;
    SF[17] = sq(q3);
    SF[18] = sq(q2);
    SF[19] = sq(q1);
    SF[20] = sq(q0);
    
    scalar_t SG[8];
    SG[0] = q0/2;
    SG[1] = sq(q3);
    SG[2] = sq(q2);
    SG[3] = sq(q1);
    SG[4] = sq(q0);
    SG[5] = 2*q2*q3;
    SG[6] = 2*q1*q3;
    SG[7] = 2*q1*q2;
    
    scalar_t SQ[11];
    SQ[0] = dvzVar*(SG[5] - 2*q0*q1)*(SG[1] - SG[2] - SG[3] + SG[4]) - dvyVar*(SG[5] + 2*q0*q1)*(SG[1] - SG[2] + SG[3] - SG[4]) + dvxVar*(SG[6] - 2*q0*q2)*(SG[7] + 2*q0*q3);
    SQ[1] = dvzVar*(SG[6] + 2*q0*q2)*(SG[1] - SG[2] - SG[3] + SG[4]) - dvxVar*(SG[6] - 2*q0*q2)*(SG[1] + SG[2] - SG[3] - SG[4]) + dvyVar*(SG[5] + 2*q0*q1)*(SG[7] - 2*q0*q3);
    SQ[2] = dvzVar*(SG[5] - 2*q0*q1)*(SG[6] + 2*q0*q2) - dvyVar*(SG[7] - 2*q0*q3)*(SG[1] - SG[2] + SG[3] - SG[4]) - dvxVar*(SG[7] + 2*q0*q3)*(SG[1] + SG[2] - SG[3] - SG[4]);
    SQ[3] = (dayVar*q1*SG[0])/2 - (dazVar*q1*SG[0])/2 - (daxVar*q2*q3)/4;
    SQ[4] = (dazVar*q2*SG[0])/2 - (daxVar*q2*SG[0])/2 - (dayVar*q1*q3)/4;
    SQ[5] = (daxVar*q3*SG[0])/2 - (dayVar*q3*SG[0])/2 - (dazVar*q1*q2)/4;
    SQ[6] = (daxVar*q1*q2)/4 - (dazVar*q3*SG[0])/2 - (dayVar*q1*q2)/4;
    SQ[7] = (dazVar*q1*q3)/4 - (daxVar*q1*q3)/4 - (dayVar*q2*SG[0])/2;
    SQ[8] = (dayVar*q2*q3)/4 - (daxVar*q1*SG[0])/2 - (dazVar*q2*q3)/4;
    SQ[9] = sq(SG[0]);
    SQ[10] = sq(q1);
    
    scalar_t SPP[11];
    SPP[0] = SF[12] + SF[13] - 2*q2*SF[2];
    SPP[1] = SF[17] - SF[18] - SF[19] + SF[20];
    SPP[2] = SF[17] - SF[18] + SF[19] - SF[20];
    SPP[3] = SF[17] + SF[18] - SF[19] - SF[20];
    SPP[4] = 2*q0*q2 - 2*q1*q3;
    SPP[5] = 2*q0*q1 - 2*q2*q3;
    SPP[6] = 2*q0*q3 - 2*q1*q2;
    SPP[7] = 2*q0*q1 + 2*q2*q3;
    SPP[8] = 2*q0*q3 + 2*q1*q2;
    SPP[9] = 2*q0*q2 + 2*q1*q3;
    SPP[10] = SF[16];
    
    // covariance update
    // calculate variances and upper diagonal covariances for quaternion, velocity, position and gyro bias states
    scalar_t nextP[k_num_states_][k_num_states_];
    nextP[0][0] = P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10] + (daxVar*SQ[10])/4 + SF[9]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) + SF[11]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SF[10]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) + SF[14]*(P_[0][10] + P_[1][10]*SF[9] + P_[2][10]*SF[11] + P_[3][10]*SF[10] + P_[10][10]*SF[14] + P_[11][10]*SF[15] + P_[12][10]*SPP[10]) + SF[15]*(P_[0][11] + P_[1][11]*SF[9] + P_[2][11]*SF[11] + P_[3][11]*SF[10] + P_[10][11]*SF[14] + P_[11][11]*SF[15] + P_[12][11]*SPP[10]) + SPP[10]*(P_[0][12] + P_[1][12]*SF[9] + P_[2][12]*SF[11] + P_[3][12]*SF[10] + P_[10][12]*SF[14] + P_[11][12]*SF[15] + P_[12][12]*SPP[10]) + (dayVar*sq(q2))/4 + (dazVar*sq(q3))/4;
    nextP[0][1] = P_[0][1] + SQ[8] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10] + SF[8]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SF[7]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SF[11]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) - SF[15]*(P_[0][12] + P_[1][12]*SF[9] + P_[2][12]*SF[11] + P_[3][12]*SF[10] + P_[10][12]*SF[14] + P_[11][12]*SF[15] + P_[12][12]*SPP[10]) + SPP[10]*(P_[0][11] + P_[1][11]*SF[9] + P_[2][11]*SF[11] + P_[3][11]*SF[10] + P_[10][11]*SF[14] + P_[11][11]*SF[15] + P_[12][11]*SPP[10]) - (q0*(P_[0][10] + P_[1][10]*SF[9] + P_[2][10]*SF[11] + P_[3][10]*SF[10] + P_[10][10]*SF[14] + P_[11][10]*SF[15] + P_[12][10]*SPP[10]))/2;
    nextP[1][1] = P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] + daxVar*SQ[9] - (P_[10][1]*q0)/2 + SF[8]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SF[7]*(P_[1][2] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2) + SF[11]*(P_[1][3] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2) - SF[15]*(P_[1][12] + P_[0][12]*SF[8] + P_[2][12]*SF[7] + P_[3][12]*SF[11] - P_[12][12]*SF[15] + P_[11][12]*SPP[10] - (P_[10][12]*q0)/2) + SPP[10]*(P_[1][11] + P_[0][11]*SF[8] + P_[2][11]*SF[7] + P_[3][11]*SF[11] - P_[12][11]*SF[15] + P_[11][11]*SPP[10] - (P_[10][11]*q0)/2) + (dayVar*sq(q3))/4 + (dazVar*sq(q2))/4 - (q0*(P_[1][10] + P_[0][10]*SF[8] + P_[2][10]*SF[7] + P_[3][10]*SF[11] - P_[12][10]*SF[15] + P_[11][10]*SPP[10] - (P_[10][10]*q0)/2))/2;
    nextP[0][2] = P_[0][2] + SQ[7] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10] + SF[6]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SF[10]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) + SF[8]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) + SF[14]*(P_[0][12] + P_[1][12]*SF[9] + P_[2][12]*SF[11] + P_[3][12]*SF[10] + P_[10][12]*SF[14] + P_[11][12]*SF[15] + P_[12][12]*SPP[10]) - SPP[10]*(P_[0][10] + P_[1][10]*SF[9] + P_[2][10]*SF[11] + P_[3][10]*SF[10] + P_[10][10]*SF[14] + P_[11][10]*SF[15] + P_[12][10]*SPP[10]) - (q0*(P_[0][11] + P_[1][11]*SF[9] + P_[2][11]*SF[11] + P_[3][11]*SF[10] + P_[10][11]*SF[14] + P_[11][11]*SF[15] + P_[12][11]*SPP[10]))/2;
    nextP[1][2] = P_[1][2] + SQ[5] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2 + SF[6]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SF[10]*(P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] - (P_[10][1]*q0)/2) + SF[8]*(P_[1][3] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2) + SF[14]*(P_[1][12] + P_[0][12]*SF[8] + P_[2][12]*SF[7] + P_[3][12]*SF[11] - P_[12][12]*SF[15] + P_[11][12]*SPP[10] - (P_[10][12]*q0)/2) - SPP[10]*(P_[1][10] + P_[0][10]*SF[8] + P_[2][10]*SF[7] + P_[3][10]*SF[11] - P_[12][10]*SF[15] + P_[11][10]*SPP[10] - (P_[10][10]*q0)/2) - (q0*(P_[1][11] + P_[0][11]*SF[8] + P_[2][11]*SF[7] + P_[3][11]*SF[11] - P_[12][11]*SF[15] + P_[11][11]*SPP[10] - (P_[10][11]*q0)/2))/2;
    nextP[2][2] = P_[2][2] + P_[0][2]*SF[6] + P_[1][2]*SF[10] + P_[3][2]*SF[8] + P_[12][2]*SF[14] - P_[10][2]*SPP[10] + dayVar*SQ[9] + (dazVar*SQ[10])/4 - (P_[11][2]*q0)/2 + SF[6]*(P_[2][0] + P_[0][0]*SF[6] + P_[1][0]*SF[10] + P_[3][0]*SF[8] + P_[12][0]*SF[14] - P_[10][0]*SPP[10] - (P_[11][0]*q0)/2) + SF[10]*(P_[2][1] + P_[0][1]*SF[6] + P_[1][1]*SF[10] + P_[3][1]*SF[8] + P_[12][1]*SF[14] - P_[10][1]*SPP[10] - (P_[11][1]*q0)/2) + SF[8]*(P_[2][3] + P_[0][3]*SF[6] + P_[1][3]*SF[10] + P_[3][3]*SF[8] + P_[12][3]*SF[14] - P_[10][3]*SPP[10] - (P_[11][3]*q0)/2) + SF[14]*(P_[2][12] + P_[0][12]*SF[6] + P_[1][12]*SF[10] + P_[3][12]*SF[8] + P_[12][12]*SF[14] - P_[10][12]*SPP[10] - (P_[11][12]*q0)/2) - SPP[10]*(P_[2][10] + P_[0][10]*SF[6] + P_[1][10]*SF[10] + P_[3][10]*SF[8] + P_[12][10]*SF[14] - P_[10][10]*SPP[10] - (P_[11][10]*q0)/2) + (daxVar*sq(q3))/4 - (q0*(P_[2][11] + P_[0][11]*SF[6] + P_[1][11]*SF[10] + P_[3][11]*SF[8] + P_[12][11]*SF[14] - P_[10][11]*SPP[10] - (P_[11][11]*q0)/2))/2;
    nextP[0][3] = P_[0][3] + SQ[6] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10] + SF[7]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SF[6]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) + SF[9]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SF[15]*(P_[0][10] + P_[1][10]*SF[9] + P_[2][10]*SF[11] + P_[3][10]*SF[10] + P_[10][10]*SF[14] + P_[11][10]*SF[15] + P_[12][10]*SPP[10]) - SF[14]*(P_[0][11] + P_[1][11]*SF[9] + P_[2][11]*SF[11] + P_[3][11]*SF[10] + P_[10][11]*SF[14] + P_[11][11]*SF[15] + P_[12][11]*SPP[10]) - (q0*(P_[0][12] + P_[1][12]*SF[9] + P_[2][12]*SF[11] + P_[3][12]*SF[10] + P_[10][12]*SF[14] + P_[11][12]*SF[15] + P_[12][12]*SPP[10]))/2;
    nextP[1][3] = P_[1][3] + SQ[4] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2 + SF[7]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SF[6]*(P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] - (P_[10][1]*q0)/2) + SF[9]*(P_[1][2] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2) + SF[15]*(P_[1][10] + P_[0][10]*SF[8] + P_[2][10]*SF[7] + P_[3][10]*SF[11] - P_[12][10]*SF[15] + P_[11][10]*SPP[10] - (P_[10][10]*q0)/2) - SF[14]*(P_[1][11] + P_[0][11]*SF[8] + P_[2][11]*SF[7] + P_[3][11]*SF[11] - P_[12][11]*SF[15] + P_[11][11]*SPP[10] - (P_[10][11]*q0)/2) - (q0*(P_[1][12] + P_[0][12]*SF[8] + P_[2][12]*SF[7] + P_[3][12]*SF[11] - P_[12][12]*SF[15] + P_[11][12]*SPP[10] - (P_[10][12]*q0)/2))/2;
    nextP[2][3] = P_[2][3] + SQ[3] + P_[0][3]*SF[6] + P_[1][3]*SF[10] + P_[3][3]*SF[8] + P_[12][3]*SF[14] - P_[10][3]*SPP[10] - (P_[11][3]*q0)/2 + SF[7]*(P_[2][0] + P_[0][0]*SF[6] + P_[1][0]*SF[10] + P_[3][0]*SF[8] + P_[12][0]*SF[14] - P_[10][0]*SPP[10] - (P_[11][0]*q0)/2) + SF[6]*(P_[2][1] + P_[0][1]*SF[6] + P_[1][1]*SF[10] + P_[3][1]*SF[8] + P_[12][1]*SF[14] - P_[10][1]*SPP[10] - (P_[11][1]*q0)/2) + SF[9]*(P_[2][2] + P_[0][2]*SF[6] + P_[1][2]*SF[10] + P_[3][2]*SF[8] + P_[12][2]*SF[14] - P_[10][2]*SPP[10] - (P_[11][2]*q0)/2) + SF[15]*(P_[2][10] + P_[0][10]*SF[6] + P_[1][10]*SF[10] + P_[3][10]*SF[8] + P_[12][10]*SF[14] - P_[10][10]*SPP[10] - (P_[11][10]*q0)/2) - SF[14]*(P_[2][11] + P_[0][11]*SF[6] + P_[1][11]*SF[10] + P_[3][11]*SF[8] + P_[12][11]*SF[14] - P_[10][11]*SPP[10] - (P_[11][11]*q0)/2) - (q0*(P_[2][12] + P_[0][12]*SF[6] + P_[1][12]*SF[10] + P_[3][12]*SF[8] + P_[12][12]*SF[14] - P_[10][12]*SPP[10] - (P_[11][12]*q0)/2))/2;
    nextP[3][3] = P_[3][3] + P_[0][3]*SF[7] + P_[1][3]*SF[6] + P_[2][3]*SF[9] + P_[10][3]*SF[15] - P_[11][3]*SF[14] + (dayVar*SQ[10])/4 + dazVar*SQ[9] - (P_[12][3]*q0)/2 + SF[7]*(P_[3][0] + P_[0][0]*SF[7] + P_[1][0]*SF[6] + P_[2][0]*SF[9] + P_[10][0]*SF[15] - P_[11][0]*SF[14] - (P_[12][0]*q0)/2) + SF[6]*(P_[3][1] + P_[0][1]*SF[7] + P_[1][1]*SF[6] + P_[2][1]*SF[9] + P_[10][1]*SF[15] - P_[11][1]*SF[14] - (P_[12][1]*q0)/2) + SF[9]*(P_[3][2] + P_[0][2]*SF[7] + P_[1][2]*SF[6] + P_[2][2]*SF[9] + P_[10][2]*SF[15] - P_[11][2]*SF[14] - (P_[12][2]*q0)/2) + SF[15]*(P_[3][10] + P_[0][10]*SF[7] + P_[1][10]*SF[6] + P_[2][10]*SF[9] + P_[10][10]*SF[15] - P_[11][10]*SF[14] - (P_[12][10]*q0)/2) - SF[14]*(P_[3][11] + P_[0][11]*SF[7] + P_[1][11]*SF[6] + P_[2][11]*SF[9] + P_[10][11]*SF[15] - P_[11][11]*SF[14] - (P_[12][11]*q0)/2) + (daxVar*sq(q2))/4 - (q0*(P_[3][12] + P_[0][12]*SF[7] + P_[1][12]*SF[6] + P_[2][12]*SF[9] + P_[10][12]*SF[15] - P_[11][12]*SF[14] - (P_[12][12]*q0)/2))/2;
    nextP[0][4] = P_[0][4] + P_[1][4]*SF[9] + P_[2][4]*SF[11] + P_[3][4]*SF[10] + P_[10][4]*SF[14] + P_[11][4]*SF[15] + P_[12][4]*SPP[10] + SF[5]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SF[3]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) - SF[4]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) + SPP[0]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SPP[3]*(P_[0][13] + P_[1][13]*SF[9] + P_[2][13]*SF[11] + P_[3][13]*SF[10] + P_[10][13]*SF[14] + P_[11][13]*SF[15] + P_[12][13]*SPP[10]) + SPP[6]*(P_[0][14] + P_[1][14]*SF[9] + P_[2][14]*SF[11] + P_[3][14]*SF[10] + P_[10][14]*SF[14] + P_[11][14]*SF[15] + P_[12][14]*SPP[10]) - SPP[9]*(P_[0][15] + P_[1][15]*SF[9] + P_[2][15]*SF[11] + P_[3][15]*SF[10] + P_[10][15]*SF[14] + P_[11][15]*SF[15] + P_[12][15]*SPP[10]);
    nextP[1][4] = P_[1][4] + P_[0][4]*SF[8] + P_[2][4]*SF[7] + P_[3][4]*SF[11] - P_[12][4]*SF[15] + P_[11][4]*SPP[10] - (P_[10][4]*q0)/2 + SF[5]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SF[3]*(P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] - (P_[10][1]*q0)/2) - SF[4]*(P_[1][3] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2) + SPP[0]*(P_[1][2] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2) + SPP[3]*(P_[1][13] + P_[0][13]*SF[8] + P_[2][13]*SF[7] + P_[3][13]*SF[11] - P_[12][13]*SF[15] + P_[11][13]*SPP[10] - (P_[10][13]*q0)/2) + SPP[6]*(P_[1][14] + P_[0][14]*SF[8] + P_[2][14]*SF[7] + P_[3][14]*SF[11] - P_[12][14]*SF[15] + P_[11][14]*SPP[10] - (P_[10][14]*q0)/2) - SPP[9]*(P_[1][15] + P_[0][15]*SF[8] + P_[2][15]*SF[7] + P_[3][15]*SF[11] - P_[12][15]*SF[15] + P_[11][15]*SPP[10] - (P_[10][15]*q0)/2);
    nextP[2][4] = P_[2][4] + P_[0][4]*SF[6] + P_[1][4]*SF[10] + P_[3][4]*SF[8] + P_[12][4]*SF[14] - P_[10][4]*SPP[10] - (P_[11][4]*q0)/2 + SF[5]*(P_[2][0] + P_[0][0]*SF[6] + P_[1][0]*SF[10] + P_[3][0]*SF[8] + P_[12][0]*SF[14] - P_[10][0]*SPP[10] - (P_[11][0]*q0)/2) + SF[3]*(P_[2][1] + P_[0][1]*SF[6] + P_[1][1]*SF[10] + P_[3][1]*SF[8] + P_[12][1]*SF[14] - P_[10][1]*SPP[10] - (P_[11][1]*q0)/2) - SF[4]*(P_[2][3] + P_[0][3]*SF[6] + P_[1][3]*SF[10] + P_[3][3]*SF[8] + P_[12][3]*SF[14] - P_[10][3]*SPP[10] - (P_[11][3]*q0)/2) + SPP[0]*(P_[2][2] + P_[0][2]*SF[6] + P_[1][2]*SF[10] + P_[3][2]*SF[8] + P_[12][2]*SF[14] - P_[10][2]*SPP[10] - (P_[11][2]*q0)/2) + SPP[3]*(P_[2][13] + P_[0][13]*SF[6] + P_[1][13]*SF[10] + P_[3][13]*SF[8] + P_[12][13]*SF[14] - P_[10][13]*SPP[10] - (P_[11][13]*q0)/2) + SPP[6]*(P_[2][14] + P_[0][14]*SF[6] + P_[1][14]*SF[10] + P_[3][14]*SF[8] + P_[12][14]*SF[14] - P_[10][14]*SPP[10] - (P_[11][14]*q0)/2) - SPP[9]*(P_[2][15] + P_[0][15]*SF[6] + P_[1][15]*SF[10] + P_[3][15]*SF[8] + P_[12][15]*SF[14] - P_[10][15]*SPP[10] - (P_[11][15]*q0)/2);
    nextP[3][4] = P_[3][4] + P_[0][4]*SF[7] + P_[1][4]*SF[6] + P_[2][4]*SF[9] + P_[10][4]*SF[15] - P_[11][4]*SF[14] - (P_[12][4]*q0)/2 + SF[5]*(P_[3][0] + P_[0][0]*SF[7] + P_[1][0]*SF[6] + P_[2][0]*SF[9] + P_[10][0]*SF[15] - P_[11][0]*SF[14] - (P_[12][0]*q0)/2) + SF[3]*(P_[3][1] + P_[0][1]*SF[7] + P_[1][1]*SF[6] + P_[2][1]*SF[9] + P_[10][1]*SF[15] - P_[11][1]*SF[14] - (P_[12][1]*q0)/2) - SF[4]*(P_[3][3] + P_[0][3]*SF[7] + P_[1][3]*SF[6] + P_[2][3]*SF[9] + P_[10][3]*SF[15] - P_[11][3]*SF[14] - (P_[12][3]*q0)/2) + SPP[0]*(P_[3][2] + P_[0][2]*SF[7] + P_[1][2]*SF[6] + P_[2][2]*SF[9] + P_[10][2]*SF[15] - P_[11][2]*SF[14] - (P_[12][2]*q0)/2) + SPP[3]*(P_[3][13] + P_[0][13]*SF[7] + P_[1][13]*SF[6] + P_[2][13]*SF[9] + P_[10][13]*SF[15] - P_[11][13]*SF[14] - (P_[12][13]*q0)/2) + SPP[6]*(P_[3][14] + P_[0][14]*SF[7] + P_[1][14]*SF[6] + P_[2][14]*SF[9] + P_[10][14]*SF[15] - P_[11][14]*SF[14] - (P_[12][14]*q0)/2) - SPP[9]*(P_[3][15] + P_[0][15]*SF[7] + P_[1][15]*SF[6] + P_[2][15]*SF[9] + P_[10][15]*SF[15] - P_[11][15]*SF[14] - (P_[12][15]*q0)/2);
    nextP[4][4] = P_[4][4] + P_[0][4]*SF[5] + P_[1][4]*SF[3] - P_[3][4]*SF[4] + P_[2][4]*SPP[0] + P_[13][4]*SPP[3] + P_[14][4]*SPP[6] - P_[15][4]*SPP[9] + dvyVar*sq(SG[7] - 2*q0*q3) + dvzVar*sq(SG[6] + 2*q0*q2) + SF[5]*(P_[4][0] + P_[0][0]*SF[5] + P_[1][0]*SF[3] - P_[3][0]*SF[4] + P_[2][0]*SPP[0] + P_[13][0]*SPP[3] + P_[14][0]*SPP[6] - P_[15][0]*SPP[9]) + SF[3]*(P_[4][1] + P_[0][1]*SF[5] + P_[1][1]*SF[3] - P_[3][1]*SF[4] + P_[2][1]*SPP[0] + P_[13][1]*SPP[3] + P_[14][1]*SPP[6] - P_[15][1]*SPP[9]) - SF[4]*(P_[4][3] + P_[0][3]*SF[5] + P_[1][3]*SF[3] - P_[3][3]*SF[4] + P_[2][3]*SPP[0] + P_[13][3]*SPP[3] + P_[14][3]*SPP[6] - P_[15][3]*SPP[9]) + SPP[0]*(P_[4][2] + P_[0][2]*SF[5] + P_[1][2]*SF[3] - P_[3][2]*SF[4] + P_[2][2]*SPP[0] + P_[13][2]*SPP[3] + P_[14][2]*SPP[6] - P_[15][2]*SPP[9]) + SPP[3]*(P_[4][13] + P_[0][13]*SF[5] + P_[1][13]*SF[3] - P_[3][13]*SF[4] + P_[2][13]*SPP[0] + P_[13][13]*SPP[3] + P_[14][13]*SPP[6] - P_[15][13]*SPP[9]) + SPP[6]*(P_[4][14] + P_[0][14]*SF[5] + P_[1][14]*SF[3] - P_[3][14]*SF[4] + P_[2][14]*SPP[0] + P_[13][14]*SPP[3] + P_[14][14]*SPP[6] - P_[15][14]*SPP[9]) - SPP[9]*(P_[4][15] + P_[0][15]*SF[5] + P_[1][15]*SF[3] - P_[3][15]*SF[4] + P_[2][15]*SPP[0] + P_[13][15]*SPP[3] + P_[14][15]*SPP[6] - P_[15][15]*SPP[9]) + dvxVar*sq(SG[1] + SG[2] - SG[3] - SG[4]);
    nextP[0][5] = P_[0][5] + P_[1][5]*SF[9] + P_[2][5]*SF[11] + P_[3][5]*SF[10] + P_[10][5]*SF[14] + P_[11][5]*SF[15] + P_[12][5]*SPP[10] + SF[4]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SF[3]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SF[5]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) - SPP[0]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) - SPP[8]*(P_[0][13] + P_[1][13]*SF[9] + P_[2][13]*SF[11] + P_[3][13]*SF[10] + P_[10][13]*SF[14] + P_[11][13]*SF[15] + P_[12][13]*SPP[10]) + SPP[2]*(P_[0][14] + P_[1][14]*SF[9] + P_[2][14]*SF[11] + P_[3][14]*SF[10] + P_[10][14]*SF[14] + P_[11][14]*SF[15] + P_[12][14]*SPP[10]) + SPP[5]*(P_[0][15] + P_[1][15]*SF[9] + P_[2][15]*SF[11] + P_[3][15]*SF[10] + P_[10][15]*SF[14] + P_[11][15]*SF[15] + P_[12][15]*SPP[10]);
    nextP[1][5] = P_[1][5] + P_[0][5]*SF[8] + P_[2][5]*SF[7] + P_[3][5]*SF[11] - P_[12][5]*SF[15] + P_[11][5]*SPP[10] - (P_[10][5]*q0)/2 + SF[4]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SF[3]*(P_[1][2] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2) + SF[5]*(P_[1][3] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2) - SPP[0]*(P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] - (P_[10][1]*q0)/2) - SPP[8]*(P_[1][13] + P_[0][13]*SF[8] + P_[2][13]*SF[7] + P_[3][13]*SF[11] - P_[12][13]*SF[15] + P_[11][13]*SPP[10] - (P_[10][13]*q0)/2) + SPP[2]*(P_[1][14] + P_[0][14]*SF[8] + P_[2][14]*SF[7] + P_[3][14]*SF[11] - P_[12][14]*SF[15] + P_[11][14]*SPP[10] - (P_[10][14]*q0)/2) + SPP[5]*(P_[1][15] + P_[0][15]*SF[8] + P_[2][15]*SF[7] + P_[3][15]*SF[11] - P_[12][15]*SF[15] + P_[11][15]*SPP[10] - (P_[10][15]*q0)/2);
    nextP[2][5] = P_[2][5] + P_[0][5]*SF[6] + P_[1][5]*SF[10] + P_[3][5]*SF[8] + P_[12][5]*SF[14] - P_[10][5]*SPP[10] - (P_[11][5]*q0)/2 + SF[4]*(P_[2][0] + P_[0][0]*SF[6] + P_[1][0]*SF[10] + P_[3][0]*SF[8] + P_[12][0]*SF[14] - P_[10][0]*SPP[10] - (P_[11][0]*q0)/2) + SF[3]*(P_[2][2] + P_[0][2]*SF[6] + P_[1][2]*SF[10] + P_[3][2]*SF[8] + P_[12][2]*SF[14] - P_[10][2]*SPP[10] - (P_[11][2]*q0)/2) + SF[5]*(P_[2][3] + P_[0][3]*SF[6] + P_[1][3]*SF[10] + P_[3][3]*SF[8] + P_[12][3]*SF[14] - P_[10][3]*SPP[10] - (P_[11][3]*q0)/2) - SPP[0]*(P_[2][1] + P_[0][1]*SF[6] + P_[1][1]*SF[10] + P_[3][1]*SF[8] + P_[12][1]*SF[14] - P_[10][1]*SPP[10] - (P_[11][1]*q0)/2) - SPP[8]*(P_[2][13] + P_[0][13]*SF[6] + P_[1][13]*SF[10] + P_[3][13]*SF[8] + P_[12][13]*SF[14] - P_[10][13]*SPP[10] - (P_[11][13]*q0)/2) + SPP[2]*(P_[2][14] + P_[0][14]*SF[6] + P_[1][14]*SF[10] + P_[3][14]*SF[8] + P_[12][14]*SF[14] - P_[10][14]*SPP[10] - (P_[11][14]*q0)/2) + SPP[5]*(P_[2][15] + P_[0][15]*SF[6] + P_[1][15]*SF[10] + P_[3][15]*SF[8] + P_[12][15]*SF[14] - P_[10][15]*SPP[10] - (P_[11][15]*q0)/2);
    nextP[3][5] = P_[3][5] + P_[0][5]*SF[7] + P_[1][5]*SF[6] + P_[2][5]*SF[9] + P_[10][5]*SF[15] - P_[11][5]*SF[14] - (P_[12][5]*q0)/2 + SF[4]*(P_[3][0] + P_[0][0]*SF[7] + P_[1][0]*SF[6] + P_[2][0]*SF[9] + P_[10][0]*SF[15] - P_[11][0]*SF[14] - (P_[12][0]*q0)/2) + SF[3]*(P_[3][2] + P_[0][2]*SF[7] + P_[1][2]*SF[6] + P_[2][2]*SF[9] + P_[10][2]*SF[15] - P_[11][2]*SF[14] - (P_[12][2]*q0)/2) + SF[5]*(P_[3][3] + P_[0][3]*SF[7] + P_[1][3]*SF[6] + P_[2][3]*SF[9] + P_[10][3]*SF[15] - P_[11][3]*SF[14] - (P_[12][3]*q0)/2) - SPP[0]*(P_[3][1] + P_[0][1]*SF[7] + P_[1][1]*SF[6] + P_[2][1]*SF[9] + P_[10][1]*SF[15] - P_[11][1]*SF[14] - (P_[12][1]*q0)/2) - SPP[8]*(P_[3][13] + P_[0][13]*SF[7] + P_[1][13]*SF[6] + P_[2][13]*SF[9] + P_[10][13]*SF[15] - P_[11][13]*SF[14] - (P_[12][13]*q0)/2) + SPP[2]*(P_[3][14] + P_[0][14]*SF[7] + P_[1][14]*SF[6] + P_[2][14]*SF[9] + P_[10][14]*SF[15] - P_[11][14]*SF[14] - (P_[12][14]*q0)/2) + SPP[5]*(P_[3][15] + P_[0][15]*SF[7] + P_[1][15]*SF[6] + P_[2][15]*SF[9] + P_[10][15]*SF[15] - P_[11][15]*SF[14] - (P_[12][15]*q0)/2);
    nextP[4][5] = P_[4][5] + SQ[2] + P_[0][5]*SF[5] + P_[1][5]*SF[3] - P_[3][5]*SF[4] + P_[2][5]*SPP[0] + P_[13][5]*SPP[3] + P_[14][5]*SPP[6] - P_[15][5]*SPP[9] + SF[4]*(P_[4][0] + P_[0][0]*SF[5] + P_[1][0]*SF[3] - P_[3][0]*SF[4] + P_[2][0]*SPP[0] + P_[13][0]*SPP[3] + P_[14][0]*SPP[6] - P_[15][0]*SPP[9]) + SF[3]*(P_[4][2] + P_[0][2]*SF[5] + P_[1][2]*SF[3] - P_[3][2]*SF[4] + P_[2][2]*SPP[0] + P_[13][2]*SPP[3] + P_[14][2]*SPP[6] - P_[15][2]*SPP[9]) + SF[5]*(P_[4][3] + P_[0][3]*SF[5] + P_[1][3]*SF[3] - P_[3][3]*SF[4] + P_[2][3]*SPP[0] + P_[13][3]*SPP[3] + P_[14][3]*SPP[6] - P_[15][3]*SPP[9]) - SPP[0]*(P_[4][1] + P_[0][1]*SF[5] + P_[1][1]*SF[3] - P_[3][1]*SF[4] + P_[2][1]*SPP[0] + P_[13][1]*SPP[3] + P_[14][1]*SPP[6] - P_[15][1]*SPP[9]) - SPP[8]*(P_[4][13] + P_[0][13]*SF[5] + P_[1][13]*SF[3] - P_[3][13]*SF[4] + P_[2][13]*SPP[0] + P_[13][13]*SPP[3] + P_[14][13]*SPP[6] - P_[15][13]*SPP[9]) + SPP[2]*(P_[4][14] + P_[0][14]*SF[5] + P_[1][14]*SF[3] - P_[3][14]*SF[4] + P_[2][14]*SPP[0] + P_[13][14]*SPP[3] + P_[14][14]*SPP[6] - P_[15][14]*SPP[9]) + SPP[5]*(P_[4][15] + P_[0][15]*SF[5] + P_[1][15]*SF[3] - P_[3][15]*SF[4] + P_[2][15]*SPP[0] + P_[13][15]*SPP[3] + P_[14][15]*SPP[6] - P_[15][15]*SPP[9]);
    nextP[5][5] = P_[5][5] + P_[0][5]*SF[4] + P_[2][5]*SF[3] + P_[3][5]*SF[5] - P_[1][5]*SPP[0] - P_[13][5]*SPP[8] + P_[14][5]*SPP[2] + P_[15][5]*SPP[5] + dvxVar*sq(SG[7] + 2*q0*q3) + dvzVar*sq(SG[5] - 2*q0*q1) + SF[4]*(P_[5][0] + P_[0][0]*SF[4] + P_[2][0]*SF[3] + P_[3][0]*SF[5] - P_[1][0]*SPP[0] - P_[13][0]*SPP[8] + P_[14][0]*SPP[2] + P_[15][0]*SPP[5]) + SF[3]*(P_[5][2] + P_[0][2]*SF[4] + P_[2][2]*SF[3] + P_[3][2]*SF[5] - P_[1][2]*SPP[0] - P_[13][2]*SPP[8] + P_[14][2]*SPP[2] + P_[15][2]*SPP[5]) + SF[5]*(P_[5][3] + P_[0][3]*SF[4] + P_[2][3]*SF[3] + P_[3][3]*SF[5] - P_[1][3]*SPP[0] - P_[13][3]*SPP[8] + P_[14][3]*SPP[2] + P_[15][3]*SPP[5]) - SPP[0]*(P_[5][1] + P_[0][1]*SF[4] + P_[2][1]*SF[3] + P_[3][1]*SF[5] - P_[1][1]*SPP[0] - P_[13][1]*SPP[8] + P_[14][1]*SPP[2] + P_[15][1]*SPP[5]) - SPP[8]*(P_[5][13] + P_[0][13]*SF[4] + P_[2][13]*SF[3] + P_[3][13]*SF[5] - P_[1][13]*SPP[0] - P_[13][13]*SPP[8] + P_[14][13]*SPP[2] + P_[15][13]*SPP[5]) + SPP[2]*(P_[5][14] + P_[0][14]*SF[4] + P_[2][14]*SF[3] + P_[3][14]*SF[5] - P_[1][14]*SPP[0] - P_[13][14]*SPP[8] + P_[14][14]*SPP[2] + P_[15][14]*SPP[5]) + SPP[5]*(P_[5][15] + P_[0][15]*SF[4] + P_[2][15]*SF[3] + P_[3][15]*SF[5] - P_[1][15]*SPP[0] - P_[13][15]*SPP[8] + P_[14][15]*SPP[2] + P_[15][15]*SPP[5]) + dvyVar*sq(SG[1] - SG[2] + SG[3] - SG[4]);
    nextP[0][6] = P_[0][6] + P_[1][6]*SF[9] + P_[2][6]*SF[11] + P_[3][6]*SF[10] + P_[10][6]*SF[14] + P_[11][6]*SF[15] + P_[12][6]*SPP[10] + SF[4]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) - SF[5]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SF[3]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) + SPP[0]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SPP[4]*(P_[0][13] + P_[1][13]*SF[9] + P_[2][13]*SF[11] + P_[3][13]*SF[10] + P_[10][13]*SF[14] + P_[11][13]*SF[15] + P_[12][13]*SPP[10]) - SPP[7]*(P_[0][14] + P_[1][14]*SF[9] + P_[2][14]*SF[11] + P_[3][14]*SF[10] + P_[10][14]*SF[14] + P_[11][14]*SF[15] + P_[12][14]*SPP[10]) - SPP[1]*(P_[0][15] + P_[1][15]*SF[9] + P_[2][15]*SF[11] + P_[3][15]*SF[10] + P_[10][15]*SF[14] + P_[11][15]*SF[15] + P_[12][15]*SPP[10]);
    nextP[1][6] = P_[1][6] + P_[0][6]*SF[8] + P_[2][6]*SF[7] + P_[3][6]*SF[11] - P_[12][6]*SF[15] + P_[11][6]*SPP[10] - (P_[10][6]*q0)/2 + SF[4]*(P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] - (P_[10][1]*q0)/2) - SF[5]*(P_[1][2] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2) + SF[3]*(P_[1][3] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2) + SPP[0]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SPP[4]*(P_[1][13] + P_[0][13]*SF[8] + P_[2][13]*SF[7] + P_[3][13]*SF[11] - P_[12][13]*SF[15] + P_[11][13]*SPP[10] - (P_[10][13]*q0)/2) - SPP[7]*(P_[1][14] + P_[0][14]*SF[8] + P_[2][14]*SF[7] + P_[3][14]*SF[11] - P_[12][14]*SF[15] + P_[11][14]*SPP[10] - (P_[10][14]*q0)/2) - SPP[1]*(P_[1][15] + P_[0][15]*SF[8] + P_[2][15]*SF[7] + P_[3][15]*SF[11] - P_[12][15]*SF[15] + P_[11][15]*SPP[10] - (P_[10][15]*q0)/2);
    nextP[2][6] = P_[2][6] + P_[0][6]*SF[6] + P_[1][6]*SF[10] + P_[3][6]*SF[8] + P_[12][6]*SF[14] - P_[10][6]*SPP[10] - (P_[11][6]*q0)/2 + SF[4]*(P_[2][1] + P_[0][1]*SF[6] + P_[1][1]*SF[10] + P_[3][1]*SF[8] + P_[12][1]*SF[14] - P_[10][1]*SPP[10] - (P_[11][1]*q0)/2) - SF[5]*(P_[2][2] + P_[0][2]*SF[6] + P_[1][2]*SF[10] + P_[3][2]*SF[8] + P_[12][2]*SF[14] - P_[10][2]*SPP[10] - (P_[11][2]*q0)/2) + SF[3]*(P_[2][3] + P_[0][3]*SF[6] + P_[1][3]*SF[10] + P_[3][3]*SF[8] + P_[12][3]*SF[14] - P_[10][3]*SPP[10] - (P_[11][3]*q0)/2) + SPP[0]*(P_[2][0] + P_[0][0]*SF[6] + P_[1][0]*SF[10] + P_[3][0]*SF[8] + P_[12][0]*SF[14] - P_[10][0]*SPP[10] - (P_[11][0]*q0)/2) + SPP[4]*(P_[2][13] + P_[0][13]*SF[6] + P_[1][13]*SF[10] + P_[3][13]*SF[8] + P_[12][13]*SF[14] - P_[10][13]*SPP[10] - (P_[11][13]*q0)/2) - SPP[7]*(P_[2][14] + P_[0][14]*SF[6] + P_[1][14]*SF[10] + P_[3][14]*SF[8] + P_[12][14]*SF[14] - P_[10][14]*SPP[10] - (P_[11][14]*q0)/2) - SPP[1]*(P_[2][15] + P_[0][15]*SF[6] + P_[1][15]*SF[10] + P_[3][15]*SF[8] + P_[12][15]*SF[14] - P_[10][15]*SPP[10] - (P_[11][15]*q0)/2);
    nextP[3][6] = P_[3][6] + P_[0][6]*SF[7] + P_[1][6]*SF[6] + P_[2][6]*SF[9] + P_[10][6]*SF[15] - P_[11][6]*SF[14] - (P_[12][6]*q0)/2 + SF[4]*(P_[3][1] + P_[0][1]*SF[7] + P_[1][1]*SF[6] + P_[2][1]*SF[9] + P_[10][1]*SF[15] - P_[11][1]*SF[14] - (P_[12][1]*q0)/2) - SF[5]*(P_[3][2] + P_[0][2]*SF[7] + P_[1][2]*SF[6] + P_[2][2]*SF[9] + P_[10][2]*SF[15] - P_[11][2]*SF[14] - (P_[12][2]*q0)/2) + SF[3]*(P_[3][3] + P_[0][3]*SF[7] + P_[1][3]*SF[6] + P_[2][3]*SF[9] + P_[10][3]*SF[15] - P_[11][3]*SF[14] - (P_[12][3]*q0)/2) + SPP[0]*(P_[3][0] + P_[0][0]*SF[7] + P_[1][0]*SF[6] + P_[2][0]*SF[9] + P_[10][0]*SF[15] - P_[11][0]*SF[14] - (P_[12][0]*q0)/2) + SPP[4]*(P_[3][13] + P_[0][13]*SF[7] + P_[1][13]*SF[6] + P_[2][13]*SF[9] + P_[10][13]*SF[15] - P_[11][13]*SF[14] - (P_[12][13]*q0)/2) - SPP[7]*(P_[3][14] + P_[0][14]*SF[7] + P_[1][14]*SF[6] + P_[2][14]*SF[9] + P_[10][14]*SF[15] - P_[11][14]*SF[14] - (P_[12][14]*q0)/2) - SPP[1]*(P_[3][15] + P_[0][15]*SF[7] + P_[1][15]*SF[6] + P_[2][15]*SF[9] + P_[10][15]*SF[15] - P_[11][15]*SF[14] - (P_[12][15]*q0)/2);
    nextP[4][6] = P_[4][6] + SQ[1] + P_[0][6]*SF[5] + P_[1][6]*SF[3] - P_[3][6]*SF[4] + P_[2][6]*SPP[0] + P_[13][6]*SPP[3] + P_[14][6]*SPP[6] - P_[15][6]*SPP[9] + SF[4]*(P_[4][1] + P_[0][1]*SF[5] + P_[1][1]*SF[3] - P_[3][1]*SF[4] + P_[2][1]*SPP[0] + P_[13][1]*SPP[3] + P_[14][1]*SPP[6] - P_[15][1]*SPP[9]) - SF[5]*(P_[4][2] + P_[0][2]*SF[5] + P_[1][2]*SF[3] - P_[3][2]*SF[4] + P_[2][2]*SPP[0] + P_[13][2]*SPP[3] + P_[14][2]*SPP[6] - P_[15][2]*SPP[9]) + SF[3]*(P_[4][3] + P_[0][3]*SF[5] + P_[1][3]*SF[3] - P_[3][3]*SF[4] + P_[2][3]*SPP[0] + P_[13][3]*SPP[3] + P_[14][3]*SPP[6] - P_[15][3]*SPP[9]) + SPP[0]*(P_[4][0] + P_[0][0]*SF[5] + P_[1][0]*SF[3] - P_[3][0]*SF[4] + P_[2][0]*SPP[0] + P_[13][0]*SPP[3] + P_[14][0]*SPP[6] - P_[15][0]*SPP[9]) + SPP[4]*(P_[4][13] + P_[0][13]*SF[5] + P_[1][13]*SF[3] - P_[3][13]*SF[4] + P_[2][13]*SPP[0] + P_[13][13]*SPP[3] + P_[14][13]*SPP[6] - P_[15][13]*SPP[9]) - SPP[7]*(P_[4][14] + P_[0][14]*SF[5] + P_[1][14]*SF[3] - P_[3][14]*SF[4] + P_[2][14]*SPP[0] + P_[13][14]*SPP[3] + P_[14][14]*SPP[6] - P_[15][14]*SPP[9]) - SPP[1]*(P_[4][15] + P_[0][15]*SF[5] + P_[1][15]*SF[3] - P_[3][15]*SF[4] + P_[2][15]*SPP[0] + P_[13][15]*SPP[3] + P_[14][15]*SPP[6] - P_[15][15]*SPP[9]);
    nextP[5][6] = P_[5][6] + SQ[0] + P_[0][6]*SF[4] + P_[2][6]*SF[3] + P_[3][6]*SF[5] - P_[1][6]*SPP[0] - P_[13][6]*SPP[8] + P_[14][6]*SPP[2] + P_[15][6]*SPP[5] + SF[4]*(P_[5][1] + P_[0][1]*SF[4] + P_[2][1]*SF[3] + P_[3][1]*SF[5] - P_[1][1]*SPP[0] - P_[13][1]*SPP[8] + P_[14][1]*SPP[2] + P_[15][1]*SPP[5]) - SF[5]*(P_[5][2] + P_[0][2]*SF[4] + P_[2][2]*SF[3] + P_[3][2]*SF[5] - P_[1][2]*SPP[0] - P_[13][2]*SPP[8] + P_[14][2]*SPP[2] + P_[15][2]*SPP[5]) + SF[3]*(P_[5][3] + P_[0][3]*SF[4] + P_[2][3]*SF[3] + P_[3][3]*SF[5] - P_[1][3]*SPP[0] - P_[13][3]*SPP[8] + P_[14][3]*SPP[2] + P_[15][3]*SPP[5]) + SPP[0]*(P_[5][0] + P_[0][0]*SF[4] + P_[2][0]*SF[3] + P_[3][0]*SF[5] - P_[1][0]*SPP[0] - P_[13][0]*SPP[8] + P_[14][0]*SPP[2] + P_[15][0]*SPP[5]) + SPP[4]*(P_[5][13] + P_[0][13]*SF[4] + P_[2][13]*SF[3] + P_[3][13]*SF[5] - P_[1][13]*SPP[0] - P_[13][13]*SPP[8] + P_[14][13]*SPP[2] + P_[15][13]*SPP[5]) - SPP[7]*(P_[5][14] + P_[0][14]*SF[4] + P_[2][14]*SF[3] + P_[3][14]*SF[5] - P_[1][14]*SPP[0] - P_[13][14]*SPP[8] + P_[14][14]*SPP[2] + P_[15][14]*SPP[5]) - SPP[1]*(P_[5][15] + P_[0][15]*SF[4] + P_[2][15]*SF[3] + P_[3][15]*SF[5] - P_[1][15]*SPP[0] - P_[13][15]*SPP[8] + P_[14][15]*SPP[2] + P_[15][15]*SPP[5]);
    nextP[6][6] = P_[6][6] + P_[1][6]*SF[4] - P_[2][6]*SF[5] + P_[3][6]*SF[3] + P_[0][6]*SPP[0] + P_[13][6]*SPP[4] - P_[14][6]*SPP[7] - P_[15][6]*SPP[1] + dvxVar*sq(SG[6] - 2*q0*q2) + dvyVar*sq(SG[5] + 2*q0*q1) + SF[4]*(P_[6][1] + P_[1][1]*SF[4] - P_[2][1]*SF[5] + P_[3][1]*SF[3] + P_[0][1]*SPP[0] + P_[13][1]*SPP[4] - P_[14][1]*SPP[7] - P_[15][1]*SPP[1]) - SF[5]*(P_[6][2] + P_[1][2]*SF[4] - P_[2][2]*SF[5] + P_[3][2]*SF[3] + P_[0][2]*SPP[0] + P_[13][2]*SPP[4] - P_[14][2]*SPP[7] - P_[15][2]*SPP[1]) + SF[3]*(P_[6][3] + P_[1][3]*SF[4] - P_[2][3]*SF[5] + P_[3][3]*SF[3] + P_[0][3]*SPP[0] + P_[13][3]*SPP[4] - P_[14][3]*SPP[7] - P_[15][3]*SPP[1]) + SPP[0]*(P_[6][0] + P_[1][0]*SF[4] - P_[2][0]*SF[5] + P_[3][0]*SF[3] + P_[0][0]*SPP[0] + P_[13][0]*SPP[4] - P_[14][0]*SPP[7] - P_[15][0]*SPP[1]) + SPP[4]*(P_[6][13] + P_[1][13]*SF[4] - P_[2][13]*SF[5] + P_[3][13]*SF[3] + P_[0][13]*SPP[0] + P_[13][13]*SPP[4] - P_[14][13]*SPP[7] - P_[15][13]*SPP[1]) - SPP[7]*(P_[6][14] + P_[1][14]*SF[4] - P_[2][14]*SF[5] + P_[3][14]*SF[3] + P_[0][14]*SPP[0] + P_[13][14]*SPP[4] - P_[14][14]*SPP[7] - P_[15][14]*SPP[1]) - SPP[1]*(P_[6][15] + P_[1][15]*SF[4] - P_[2][15]*SF[5] + P_[3][15]*SF[3] + P_[0][15]*SPP[0] + P_[13][15]*SPP[4] - P_[14][15]*SPP[7] - P_[15][15]*SPP[1]) + dvzVar*sq(SG[1] - SG[2] - SG[3] + SG[4]);
    nextP[0][7] = P_[0][7] + P_[1][7]*SF[9] + P_[2][7]*SF[11] + P_[3][7]*SF[10] + P_[10][7]*SF[14] + P_[11][7]*SF[15] + P_[12][7]*SPP[10] + dt*(P_[0][4] + P_[1][4]*SF[9] + P_[2][4]*SF[11] + P_[3][4]*SF[10] + P_[10][4]*SF[14] + P_[11][4]*SF[15] + P_[12][4]*SPP[10]);
    nextP[1][7] = P_[1][7] + P_[0][7]*SF[8] + P_[2][7]*SF[7] + P_[3][7]*SF[11] - P_[12][7]*SF[15] + P_[11][7]*SPP[10] - (P_[10][7]*q0)/2 + dt*(P_[1][4] + P_[0][4]*SF[8] + P_[2][4]*SF[7] + P_[3][4]*SF[11] - P_[12][4]*SF[15] + P_[11][4]*SPP[10] - (P_[10][4]*q0)/2);
    nextP[2][7] = P_[2][7] + P_[0][7]*SF[6] + P_[1][7]*SF[10] + P_[3][7]*SF[8] + P_[12][7]*SF[14] - P_[10][7]*SPP[10] - (P_[11][7]*q0)/2 + dt*(P_[2][4] + P_[0][4]*SF[6] + P_[1][4]*SF[10] + P_[3][4]*SF[8] + P_[12][4]*SF[14] - P_[10][4]*SPP[10] - (P_[11][4]*q0)/2);
    nextP[3][7] = P_[3][7] + P_[0][7]*SF[7] + P_[1][7]*SF[6] + P_[2][7]*SF[9] + P_[10][7]*SF[15] - P_[11][7]*SF[14] - (P_[12][7]*q0)/2 + dt*(P_[3][4] + P_[0][4]*SF[7] + P_[1][4]*SF[6] + P_[2][4]*SF[9] + P_[10][4]*SF[15] - P_[11][4]*SF[14] - (P_[12][4]*q0)/2);
    nextP[4][7] = P_[4][7] + P_[0][7]*SF[5] + P_[1][7]*SF[3] - P_[3][7]*SF[4] + P_[2][7]*SPP[0] + P_[13][7]*SPP[3] + P_[14][7]*SPP[6] - P_[15][7]*SPP[9] + dt*(P_[4][4] + P_[0][4]*SF[5] + P_[1][4]*SF[3] - P_[3][4]*SF[4] + P_[2][4]*SPP[0] + P_[13][4]*SPP[3] + P_[14][4]*SPP[6] - P_[15][4]*SPP[9]);
    nextP[5][7] = P_[5][7] + P_[0][7]*SF[4] + P_[2][7]*SF[3] + P_[3][7]*SF[5] - P_[1][7]*SPP[0] - P_[13][7]*SPP[8] + P_[14][7]*SPP[2] + P_[15][7]*SPP[5] + dt*(P_[5][4] + P_[0][4]*SF[4] + P_[2][4]*SF[3] + P_[3][4]*SF[5] - P_[1][4]*SPP[0] - P_[13][4]*SPP[8] + P_[14][4]*SPP[2] + P_[15][4]*SPP[5]);
    nextP[6][7] = P_[6][7] + P_[1][7]*SF[4] - P_[2][7]*SF[5] + P_[3][7]*SF[3] + P_[0][7]*SPP[0] + P_[13][7]*SPP[4] - P_[14][7]*SPP[7] - P_[15][7]*SPP[1] + dt*(P_[6][4] + P_[1][4]*SF[4] - P_[2][4]*SF[5] + P_[3][4]*SF[3] + P_[0][4]*SPP[0] + P_[13][4]*SPP[4] - P_[14][4]*SPP[7] - P_[15][4]*SPP[1]);
    nextP[7][7] = P_[7][7] + P_[4][7]*dt + dt*(P_[7][4] + P_[4][4]*dt);
    nextP[0][8] = P_[0][8] + P_[1][8]*SF[9] + P_[2][8]*SF[11] + P_[3][8]*SF[10] + P_[10][8]*SF[14] + P_[11][8]*SF[15] + P_[12][8]*SPP[10] + dt*(P_[0][5] + P_[1][5]*SF[9] + P_[2][5]*SF[11] + P_[3][5]*SF[10] + P_[10][5]*SF[14] + P_[11][5]*SF[15] + P_[12][5]*SPP[10]);
    nextP[1][8] = P_[1][8] + P_[0][8]*SF[8] + P_[2][8]*SF[7] + P_[3][8]*SF[11] - P_[12][8]*SF[15] + P_[11][8]*SPP[10] - (P_[10][8]*q0)/2 + dt*(P_[1][5] + P_[0][5]*SF[8] + P_[2][5]*SF[7] + P_[3][5]*SF[11] - P_[12][5]*SF[15] + P_[11][5]*SPP[10] - (P_[10][5]*q0)/2);
    nextP[2][8] = P_[2][8] + P_[0][8]*SF[6] + P_[1][8]*SF[10] + P_[3][8]*SF[8] + P_[12][8]*SF[14] - P_[10][8]*SPP[10] - (P_[11][8]*q0)/2 + dt*(P_[2][5] + P_[0][5]*SF[6] + P_[1][5]*SF[10] + P_[3][5]*SF[8] + P_[12][5]*SF[14] - P_[10][5]*SPP[10] - (P_[11][5]*q0)/2);
    nextP[3][8] = P_[3][8] + P_[0][8]*SF[7] + P_[1][8]*SF[6] + P_[2][8]*SF[9] + P_[10][8]*SF[15] - P_[11][8]*SF[14] - (P_[12][8]*q0)/2 + dt*(P_[3][5] + P_[0][5]*SF[7] + P_[1][5]*SF[6] + P_[2][5]*SF[9] + P_[10][5]*SF[15] - P_[11][5]*SF[14] - (P_[12][5]*q0)/2);
    nextP[4][8] = P_[4][8] + P_[0][8]*SF[5] + P_[1][8]*SF[3] - P_[3][8]*SF[4] + P_[2][8]*SPP[0] + P_[13][8]*SPP[3] + P_[14][8]*SPP[6] - P_[15][8]*SPP[9] + dt*(P_[4][5] + P_[0][5]*SF[5] + P_[1][5]*SF[3] - P_[3][5]*SF[4] + P_[2][5]*SPP[0] + P_[13][5]*SPP[3] + P_[14][5]*SPP[6] - P_[15][5]*SPP[9]);
    nextP[5][8] = P_[5][8] + P_[0][8]*SF[4] + P_[2][8]*SF[3] + P_[3][8]*SF[5] - P_[1][8]*SPP[0] - P_[13][8]*SPP[8] + P_[14][8]*SPP[2] + P_[15][8]*SPP[5] + dt*(P_[5][5] + P_[0][5]*SF[4] + P_[2][5]*SF[3] + P_[3][5]*SF[5] - P_[1][5]*SPP[0] - P_[13][5]*SPP[8] + P_[14][5]*SPP[2] + P_[15][5]*SPP[5]);
    nextP[6][8] = P_[6][8] + P_[1][8]*SF[4] - P_[2][8]*SF[5] + P_[3][8]*SF[3] + P_[0][8]*SPP[0] + P_[13][8]*SPP[4] - P_[14][8]*SPP[7] - P_[15][8]*SPP[1] + dt*(P_[6][5] + P_[1][5]*SF[4] - P_[2][5]*SF[5] + P_[3][5]*SF[3] + P_[0][5]*SPP[0] + P_[13][5]*SPP[4] - P_[14][5]*SPP[7] - P_[15][5]*SPP[1]);
    nextP[7][8] = P_[7][8] + P_[4][8]*dt + dt*(P_[7][5] + P_[4][5]*dt);
    nextP[8][8] = P_[8][8] + P_[5][8]*dt + dt*(P_[8][5] + P_[5][5]*dt);
    nextP[0][9] = P_[0][9] + P_[1][9]*SF[9] + P_[2][9]*SF[11] + P_[3][9]*SF[10] + P_[10][9]*SF[14] + P_[11][9]*SF[15] + P_[12][9]*SPP[10] + dt*(P_[0][6] + P_[1][6]*SF[9] + P_[2][6]*SF[11] + P_[3][6]*SF[10] + P_[10][6]*SF[14] + P_[11][6]*SF[15] + P_[12][6]*SPP[10]);
    nextP[1][9] = P_[1][9] + P_[0][9]*SF[8] + P_[2][9]*SF[7] + P_[3][9]*SF[11] - P_[12][9]*SF[15] + P_[11][9]*SPP[10] - (P_[10][9]*q0)/2 + dt*(P_[1][6] + P_[0][6]*SF[8] + P_[2][6]*SF[7] + P_[3][6]*SF[11] - P_[12][6]*SF[15] + P_[11][6]*SPP[10] - (P_[10][6]*q0)/2);
    nextP[2][9] = P_[2][9] + P_[0][9]*SF[6] + P_[1][9]*SF[10] + P_[3][9]*SF[8] + P_[12][9]*SF[14] - P_[10][9]*SPP[10] - (P_[11][9]*q0)/2 + dt*(P_[2][6] + P_[0][6]*SF[6] + P_[1][6]*SF[10] + P_[3][6]*SF[8] + P_[12][6]*SF[14] - P_[10][6]*SPP[10] - (P_[11][6]*q0)/2);
    nextP[3][9] = P_[3][9] + P_[0][9]*SF[7] + P_[1][9]*SF[6] + P_[2][9]*SF[9] + P_[10][9]*SF[15] - P_[11][9]*SF[14] - (P_[12][9]*q0)/2 + dt*(P_[3][6] + P_[0][6]*SF[7] + P_[1][6]*SF[6] + P_[2][6]*SF[9] + P_[10][6]*SF[15] - P_[11][6]*SF[14] - (P_[12][6]*q0)/2);
    nextP[4][9] = P_[4][9] + P_[0][9]*SF[5] + P_[1][9]*SF[3] - P_[3][9]*SF[4] + P_[2][9]*SPP[0] + P_[13][9]*SPP[3] + P_[14][9]*SPP[6] - P_[15][9]*SPP[9] + dt*(P_[4][6] + P_[0][6]*SF[5] + P_[1][6]*SF[3] - P_[3][6]*SF[4] + P_[2][6]*SPP[0] + P_[13][6]*SPP[3] + P_[14][6]*SPP[6] - P_[15][6]*SPP[9]);
    nextP[5][9] = P_[5][9] + P_[0][9]*SF[4] + P_[2][9]*SF[3] + P_[3][9]*SF[5] - P_[1][9]*SPP[0] - P_[13][9]*SPP[8] + P_[14][9]*SPP[2] + P_[15][9]*SPP[5] + dt*(P_[5][6] + P_[0][6]*SF[4] + P_[2][6]*SF[3] + P_[3][6]*SF[5] - P_[1][6]*SPP[0] - P_[13][6]*SPP[8] + P_[14][6]*SPP[2] + P_[15][6]*SPP[5]);
    nextP[6][9] = P_[6][9] + P_[1][9]*SF[4] - P_[2][9]*SF[5] + P_[3][9]*SF[3] + P_[0][9]*SPP[0] + P_[13][9]*SPP[4] - P_[14][9]*SPP[7] - P_[15][9]*SPP[1] + dt*(P_[6][6] + P_[1][6]*SF[4] - P_[2][6]*SF[5] + P_[3][6]*SF[3] + P_[0][6]*SPP[0] + P_[13][6]*SPP[4] - P_[14][6]*SPP[7] - P_[15][6]*SPP[1]);
    nextP[7][9] = P_[7][9] + P_[4][9]*dt + dt*(P_[7][6] + P_[4][6]*dt);
    nextP[8][9] = P_[8][9] + P_[5][9]*dt + dt*(P_[8][6] + P_[5][6]*dt);
    nextP[9][9] = P_[9][9] + P_[6][9]*dt + dt*(P_[9][6] + P_[6][6]*dt);
    nextP[0][10] = P_[0][10] + P_[1][10]*SF[9] + P_[2][10]*SF[11] + P_[3][10]*SF[10] + P_[10][10]*SF[14] + P_[11][10]*SF[15] + P_[12][10]*SPP[10];
    nextP[1][10] = P_[1][10] + P_[0][10]*SF[8] + P_[2][10]*SF[7] + P_[3][10]*SF[11] - P_[12][10]*SF[15] + P_[11][10]*SPP[10] - (P_[10][10]*q0)/2;
    nextP[2][10] = P_[2][10] + P_[0][10]*SF[6] + P_[1][10]*SF[10] + P_[3][10]*SF[8] + P_[12][10]*SF[14] - P_[10][10]*SPP[10] - (P_[11][10]*q0)/2;
    nextP[3][10] = P_[3][10] + P_[0][10]*SF[7] + P_[1][10]*SF[6] + P_[2][10]*SF[9] + P_[10][10]*SF[15] - P_[11][10]*SF[14] - (P_[12][10]*q0)/2;
    nextP[4][10] = P_[4][10] + P_[0][10]*SF[5] + P_[1][10]*SF[3] - P_[3][10]*SF[4] + P_[2][10]*SPP[0] + P_[13][10]*SPP[3] + P_[14][10]*SPP[6] - P_[15][10]*SPP[9];
    nextP[5][10] = P_[5][10] + P_[0][10]*SF[4] + P_[2][10]*SF[3] + P_[3][10]*SF[5] - P_[1][10]*SPP[0] - P_[13][10]*SPP[8] + P_[14][10]*SPP[2] + P_[15][10]*SPP[5];
    nextP[6][10] = P_[6][10] + P_[1][10]*SF[4] - P_[2][10]*SF[5] + P_[3][10]*SF[3] + P_[0][10]*SPP[0] + P_[13][10]*SPP[4] - P_[14][10]*SPP[7] - P_[15][10]*SPP[1];
    nextP[7][10] = P_[7][10] + P_[4][10]*dt;
    nextP[8][10] = P_[8][10] + P_[5][10]*dt;
    nextP[9][10] = P_[9][10] + P_[6][10]*dt;
    nextP[10][10] = P_[10][10];
    nextP[0][11] = P_[0][11] + P_[1][11]*SF[9] + P_[2][11]*SF[11] + P_[3][11]*SF[10] + P_[10][11]*SF[14] + P_[11][11]*SF[15] + P_[12][11]*SPP[10];
    nextP[1][11] = P_[1][11] + P_[0][11]*SF[8] + P_[2][11]*SF[7] + P_[3][11]*SF[11] - P_[12][11]*SF[15] + P_[11][11]*SPP[10] - (P_[10][11]*q0)/2;
    nextP[2][11] = P_[2][11] + P_[0][11]*SF[6] + P_[1][11]*SF[10] + P_[3][11]*SF[8] + P_[12][11]*SF[14] - P_[10][11]*SPP[10] - (P_[11][11]*q0)/2;
    nextP[3][11] = P_[3][11] + P_[0][11]*SF[7] + P_[1][11]*SF[6] + P_[2][11]*SF[9] + P_[10][11]*SF[15] - P_[11][11]*SF[14] - (P_[12][11]*q0)/2;
    nextP[4][11] = P_[4][11] + P_[0][11]*SF[5] + P_[1][11]*SF[3] - P_[3][11]*SF[4] + P_[2][11]*SPP[0] + P_[13][11]*SPP[3] + P_[14][11]*SPP[6] - P_[15][11]*SPP[9];
    nextP[5][11] = P_[5][11] + P_[0][11]*SF[4] + P_[2][11]*SF[3] + P_[3][11]*SF[5] - P_[1][11]*SPP[0] - P_[13][11]*SPP[8] + P_[14][11]*SPP[2] + P_[15][11]*SPP[5];
    nextP[6][11] = P_[6][11] + P_[1][11]*SF[4] - P_[2][11]*SF[5] + P_[3][11]*SF[3] + P_[0][11]*SPP[0] + P_[13][11]*SPP[4] - P_[14][11]*SPP[7] - P_[15][11]*SPP[1];
    nextP[7][11] = P_[7][11] + P_[4][11]*dt;
    nextP[8][11] = P_[8][11] + P_[5][11]*dt;
    nextP[9][11] = P_[9][11] + P_[6][11]*dt;
    nextP[10][11] = P_[10][11];
    nextP[11][11] = P_[11][11];
    nextP[0][12] = P_[0][12] + P_[1][12]*SF[9] + P_[2][12]*SF[11] + P_[3][12]*SF[10] + P_[10][12]*SF[14] + P_[11][12]*SF[15] + P_[12][12]*SPP[10];
    nextP[1][12] = P_[1][12] + P_[0][12]*SF[8] + P_[2][12]*SF[7] + P_[3][12]*SF[11] - P_[12][12]*SF[15] + P_[11][12]*SPP[10] - (P_[10][12]*q0)/2;
    nextP[2][12] = P_[2][12] + P_[0][12]*SF[6] + P_[1][12]*SF[10] + P_[3][12]*SF[8] + P_[12][12]*SF[14] - P_[10][12]*SPP[10] - (P_[11][12]*q0)/2;
    nextP[3][12] = P_[3][12] + P_[0][12]*SF[7] + P_[1][12]*SF[6] + P_[2][12]*SF[9] + P_[10][12]*SF[15] - P_[11][12]*SF[14] - (P_[12][12]*q0)/2;
    nextP[4][12] = P_[4][12] + P_[0][12]*SF[5] + P_[1][12]*SF[3] - P_[3][12]*SF[4] + P_[2][12]*SPP[0] + P_[13][12]*SPP[3] + P_[14][12]*SPP[6] - P_[15][12]*SPP[9];
    nextP[5][12] = P_[5][12] + P_[0][12]*SF[4] + P_[2][12]*SF[3] + P_[3][12]*SF[5] - P_[1][12]*SPP[0] - P_[13][12]*SPP[8] + P_[14][12]*SPP[2] + P_[15][12]*SPP[5];
    nextP[6][12] = P_[6][12] + P_[1][12]*SF[4] - P_[2][12]*SF[5] + P_[3][12]*SF[3] + P_[0][12]*SPP[0] + P_[13][12]*SPP[4] - P_[14][12]*SPP[7] - P_[15][12]*SPP[1];
    nextP[7][12] = P_[7][12] + P_[4][12]*dt;
    nextP[8][12] = P_[8][12] + P_[5][12]*dt;
    nextP[9][12] = P_[9][12] + P_[6][12]*dt;
    nextP[10][12] = P_[10][12];
    nextP[11][12] = P_[11][12];
    nextP[12][12] = P_[12][12];
    
    // add process noise that is not from the IMU
    for (unsigned i = 0; i <= 12; i++) {
      nextP[i][i] += process_noise[i];
    }

    // calculate variances and upper diagonal covariances for IMU delta velocity bias states
    nextP[0][13] = P_[0][13] + P_[1][13]*SF[9] + P_[2][13]*SF[11] + P_[3][13]*SF[10] + P_[10][13]*SF[14] + P_[11][13]*SF[15] + P_[12][13]*SPP[10];
    nextP[1][13] = P_[1][13] + P_[0][13]*SF[8] + P_[2][13]*SF[7] + P_[3][13]*SF[11] - P_[12][13]*SF[15] + P_[11][13]*SPP[10] - (P_[10][13]*q0)/2;
    nextP[2][13] = P_[2][13] + P_[0][13]*SF[6] + P_[1][13]*SF[10] + P_[3][13]*SF[8] + P_[12][13]*SF[14] - P_[10][13]*SPP[10] - (P_[11][13]*q0)/2;
    nextP[3][13] = P_[3][13] + P_[0][13]*SF[7] + P_[1][13]*SF[6] + P_[2][13]*SF[9] + P_[10][13]*SF[15] - P_[11][13]*SF[14] - (P_[12][13]*q0)/2;
    nextP[4][13] = P_[4][13] + P_[0][13]*SF[5] + P_[1][13]*SF[3] - P_[3][13]*SF[4] + P_[2][13]*SPP[0] + P_[13][13]*SPP[3] + P_[14][13]*SPP[6] - P_[15][13]*SPP[9];
    nextP[5][13] = P_[5][13] + P_[0][13]*SF[4] + P_[2][13]*SF[3] + P_[3][13]*SF[5] - P_[1][13]*SPP[0] - P_[13][13]*SPP[8] + P_[14][13]*SPP[2] + P_[15][13]*SPP[5];
    nextP[6][13] = P_[6][13] + P_[1][13]*SF[4] - P_[2][13]*SF[5] + P_[3][13]*SF[3] + P_[0][13]*SPP[0] + P_[13][13]*SPP[4] - P_[14][13]*SPP[7] - P_[15][13]*SPP[1];
    nextP[7][13] = P_[7][13] + P_[4][13]*dt;
    nextP[8][13] = P_[8][13] + P_[5][13]*dt;
    nextP[9][13] = P_[9][13] + P_[6][13]*dt;
    nextP[10][13] = P_[10][13];
    nextP[11][13] = P_[11][13];
    nextP[12][13] = P_[12][13];
    nextP[13][13] = P_[13][13];
    nextP[0][14] = P_[0][14] + P_[1][14]*SF[9] + P_[2][14]*SF[11] + P_[3][14]*SF[10] + P_[10][14]*SF[14] + P_[11][14]*SF[15] + P_[12][14]*SPP[10];
    nextP[1][14] = P_[1][14] + P_[0][14]*SF[8] + P_[2][14]*SF[7] + P_[3][14]*SF[11] - P_[12][14]*SF[15] + P_[11][14]*SPP[10] - (P_[10][14]*q0)/2;
    nextP[2][14] = P_[2][14] + P_[0][14]*SF[6] + P_[1][14]*SF[10] + P_[3][14]*SF[8] + P_[12][14]*SF[14] - P_[10][14]*SPP[10] - (P_[11][14]*q0)/2;
    nextP[3][14] = P_[3][14] + P_[0][14]*SF[7] + P_[1][14]*SF[6] + P_[2][14]*SF[9] + P_[10][14]*SF[15] - P_[11][14]*SF[14] - (P_[12][14]*q0)/2;
    nextP[4][14] = P_[4][14] + P_[0][14]*SF[5] + P_[1][14]*SF[3] - P_[3][14]*SF[4] + P_[2][14]*SPP[0] + P_[13][14]*SPP[3] + P_[14][14]*SPP[6] - P_[15][14]*SPP[9];
    nextP[5][14] = P_[5][14] + P_[0][14]*SF[4] + P_[2][14]*SF[3] + P_[3][14]*SF[5] - P_[1][14]*SPP[0] - P_[13][14]*SPP[8] + P_[14][14]*SPP[2] + P_[15][14]*SPP[5];
    nextP[6][14] = P_[6][14] + P_[1][14]*SF[4] - P_[2][14]*SF[5] + P_[3][14]*SF[3] + P_[0][14]*SPP[0] + P_[13][14]*SPP[4] - P_[14][14]*SPP[7] - P_[15][14]*SPP[1];
    nextP[7][14] = P_[7][14] + P_[4][14]*dt;
    nextP[8][14] = P_[8][14] + P_[5][14]*dt;
    nextP[9][14] = P_[9][14] + P_[6][14]*dt;
    nextP[10][14] = P_[10][14];
    nextP[11][14] = P_[11][14];
    nextP[12][14] = P_[12][14];
    nextP[13][14] = P_[13][14];
    nextP[14][14] = P_[14][14];
    nextP[0][15] = P_[0][15] + P_[1][15]*SF[9] + P_[2][15]*SF[11] + P_[3][15]*SF[10] + P_[10][15]*SF[14] + P_[11][15]*SF[15] + P_[12][15]*SPP[10];
    nextP[1][15] = P_[1][15] + P_[0][15]*SF[8] + P_[2][15]*SF[7] + P_[3][15]*SF[11] - P_[12][15]*SF[15] + P_[11][15]*SPP[10] - (P_[10][15]*q0)/2;
    nextP[2][15] = P_[2][15] + P_[0][15]*SF[6] + P_[1][15]*SF[10] + P_[3][15]*SF[8] + P_[12][15]*SF[14] - P_[10][15]*SPP[10] - (P_[11][15]*q0)/2;
    nextP[3][15] = P_[3][15] + P_[0][15]*SF[7] + P_[1][15]*SF[6] + P_[2][15]*SF[9] + P_[10][15]*SF[15] - P_[11][15]*SF[14] - (P_[12][15]*q0)/2;
    nextP[4][15] = P_[4][15] + P_[0][15]*SF[5] + P_[1][15]*SF[3] - P_[3][15]*SF[4] + P_[2][15]*SPP[0] + P_[13][15]*SPP[3] + P_[14][15]*SPP[6] - P_[15][15]*SPP[9];
    nextP[5][15] = P_[5][15] + P_[0][15]*SF[4] + P_[2][15]*SF[3] + P_[3][15]*SF[5] - P_[1][15]*SPP[0] - P_[13][15]*SPP[8] + P_[14][15]*SPP[2] + P_[15][15]*SPP[5];
    nextP[6][15] = P_[6][15] + P_[1][15]*SF[4] - P_[2][15]*SF[5] + P_[3][15]*SF[3] + P_[0][15]*SPP[0] + P_[13][15]*SPP[4] - P_[14][15]*SPP[7] - P_[15][15]*SPP[1];
    nextP[7][15] = P_[7][15] + P_[4][15]*dt;
    nextP[8][15] = P_[8][15] + P_[5][15]*dt;
    nextP[9][15] = P_[9][15] + P_[6][15]*dt;
    nextP[10][15] = P_[10][15];
    nextP[11][15] = P_[11][15];
    nextP[12][15] = P_[12][15];
    nextP[13][15] = P_[13][15];
    nextP[14][15] = P_[14][15];
    nextP[15][15] = P_[15][15];

    // add process noise that is not from the IMU
    for (unsigned i = 13; i <= 15; i++) {
      nextP[i][i] += process_noise[i];
    }

    // Don't do covariance prediction on magnetic field states unless we are using 3-axis fusion
    if (mag_3D_) {
      // calculate variances and upper diagonal covariances for earth and body magnetic field states
      nextP[0][16] = P_[0][16] + P_[1][16]*SF[9] + P_[2][16]*SF[11] + P_[3][16]*SF[10] + P_[10][16]*SF[14] + P_[11][16]*SF[15] + P_[12][16]*SPP[10];
      nextP[1][16] = P_[1][16] + P_[0][16]*SF[8] + P_[2][16]*SF[7] + P_[3][16]*SF[11] - P_[12][16]*SF[15] + P_[11][16]*SPP[10] - (P_[10][16]*q0)/2;
      nextP[2][16] = P_[2][16] + P_[0][16]*SF[6] + P_[1][16]*SF[10] + P_[3][16]*SF[8] + P_[12][16]*SF[14] - P_[10][16]*SPP[10] - (P_[11][16]*q0)/2;
      nextP[3][16] = P_[3][16] + P_[0][16]*SF[7] + P_[1][16]*SF[6] + P_[2][16]*SF[9] + P_[10][16]*SF[15] - P_[11][16]*SF[14] - (P_[12][16]*q0)/2;
      nextP[4][16] = P_[4][16] + P_[0][16]*SF[5] + P_[1][16]*SF[3] - P_[3][16]*SF[4] + P_[2][16]*SPP[0] + P_[13][16]*SPP[3] + P_[14][16]*SPP[6] - P_[15][16]*SPP[9];
      nextP[5][16] = P_[5][16] + P_[0][16]*SF[4] + P_[2][16]*SF[3] + P_[3][16]*SF[5] - P_[1][16]*SPP[0] - P_[13][16]*SPP[8] + P_[14][16]*SPP[2] + P_[15][16]*SPP[5];
      nextP[6][16] = P_[6][16] + P_[1][16]*SF[4] - P_[2][16]*SF[5] + P_[3][16]*SF[3] + P_[0][16]*SPP[0] + P_[13][16]*SPP[4] - P_[14][16]*SPP[7] - P_[15][16]*SPP[1];
      nextP[7][16] = P_[7][16] + P_[4][16]*dt;
      nextP[8][16] = P_[8][16] + P_[5][16]*dt;
      nextP[9][16] = P_[9][16] + P_[6][16]*dt;
      nextP[10][16] = P_[10][16];
      nextP[11][16] = P_[11][16];
      nextP[12][16] = P_[12][16];
      nextP[13][16] = P_[13][16];
      nextP[14][16] = P_[14][16];
      nextP[15][16] = P_[15][16];
      nextP[16][16] = P_[16][16];
      nextP[0][17] = P_[0][17] + P_[1][17]*SF[9] + P_[2][17]*SF[11] + P_[3][17]*SF[10] + P_[10][17]*SF[14] + P_[11][17]*SF[15] + P_[12][17]*SPP[10];
      nextP[1][17] = P_[1][17] + P_[0][17]*SF[8] + P_[2][17]*SF[7] + P_[3][17]*SF[11] - P_[12][17]*SF[15] + P_[11][17]*SPP[10] - (P_[10][17]*q0)/2;
      nextP[2][17] = P_[2][17] + P_[0][17]*SF[6] + P_[1][17]*SF[10] + P_[3][17]*SF[8] + P_[12][17]*SF[14] - P_[10][17]*SPP[10] - (P_[11][17]*q0)/2;
      nextP[3][17] = P_[3][17] + P_[0][17]*SF[7] + P_[1][17]*SF[6] + P_[2][17]*SF[9] + P_[10][17]*SF[15] - P_[11][17]*SF[14] - (P_[12][17]*q0)/2;
      nextP[4][17] = P_[4][17] + P_[0][17]*SF[5] + P_[1][17]*SF[3] - P_[3][17]*SF[4] + P_[2][17]*SPP[0] + P_[13][17]*SPP[3] + P_[14][17]*SPP[6] - P_[15][17]*SPP[9];
      nextP[5][17] = P_[5][17] + P_[0][17]*SF[4] + P_[2][17]*SF[3] + P_[3][17]*SF[5] - P_[1][17]*SPP[0] - P_[13][17]*SPP[8] + P_[14][17]*SPP[2] + P_[15][17]*SPP[5];
      nextP[6][17] = P_[6][17] + P_[1][17]*SF[4] - P_[2][17]*SF[5] + P_[3][17]*SF[3] + P_[0][17]*SPP[0] + P_[13][17]*SPP[4] - P_[14][17]*SPP[7] - P_[15][17]*SPP[1];
      nextP[7][17] = P_[7][17] + P_[4][17]*dt;
      nextP[8][17] = P_[8][17] + P_[5][17]*dt;
      nextP[9][17] = P_[9][17] + P_[6][17]*dt;
      nextP[10][17] = P_[10][17];
      nextP[11][17] = P_[11][17];
      nextP[12][17] = P_[12][17];
      nextP[13][17] = P_[13][17];
      nextP[14][17] = P_[14][17];
      nextP[15][17] = P_[15][17];
      nextP[16][17] = P_[16][17];
      nextP[17][17] = P_[17][17];
      nextP[0][18] = P_[0][18] + P_[1][18]*SF[9] + P_[2][18]*SF[11] + P_[3][18]*SF[10] + P_[10][18]*SF[14] + P_[11][18]*SF[15] + P_[12][18]*SPP[10];
      nextP[1][18] = P_[1][18] + P_[0][18]*SF[8] + P_[2][18]*SF[7] + P_[3][18]*SF[11] - P_[12][18]*SF[15] + P_[11][18]*SPP[10] - (P_[10][18]*q0)/2;
      nextP[2][18] = P_[2][18] + P_[0][18]*SF[6] + P_[1][18]*SF[10] + P_[3][18]*SF[8] + P_[12][18]*SF[14] - P_[10][18]*SPP[10] - (P_[11][18]*q0)/2;
      nextP[3][18] = P_[3][18] + P_[0][18]*SF[7] + P_[1][18]*SF[6] + P_[2][18]*SF[9] + P_[10][18]*SF[15] - P_[11][18]*SF[14] - (P_[12][18]*q0)/2;
      nextP[4][18] = P_[4][18] + P_[0][18]*SF[5] + P_[1][18]*SF[3] - P_[3][18]*SF[4] + P_[2][18]*SPP[0] + P_[13][18]*SPP[3] + P_[14][18]*SPP[6] - P_[15][18]*SPP[9];
      nextP[5][18] = P_[5][18] + P_[0][18]*SF[4] + P_[2][18]*SF[3] + P_[3][18]*SF[5] - P_[1][18]*SPP[0] - P_[13][18]*SPP[8] + P_[14][18]*SPP[2] + P_[15][18]*SPP[5];
      nextP[6][18] = P_[6][18] + P_[1][18]*SF[4] - P_[2][18]*SF[5] + P_[3][18]*SF[3] + P_[0][18]*SPP[0] + P_[13][18]*SPP[4] - P_[14][18]*SPP[7] - P_[15][18]*SPP[1];
      nextP[7][18] = P_[7][18] + P_[4][18]*dt;
      nextP[8][18] = P_[8][18] + P_[5][18]*dt;
      nextP[9][18] = P_[9][18] + P_[6][18]*dt;
      nextP[10][18] = P_[10][18];
      nextP[11][18] = P_[11][18];
      nextP[12][18] = P_[12][18];
      nextP[13][18] = P_[13][18];
      nextP[14][18] = P_[14][18];
      nextP[15][18] = P_[15][18];
      nextP[16][18] = P_[16][18];
      nextP[17][18] = P_[17][18];
      nextP[18][18] = P_[18][18];
      nextP[0][19] = P_[0][19] + P_[1][19]*SF[9] + P_[2][19]*SF[11] + P_[3][19]*SF[10] + P_[10][19]*SF[14] + P_[11][19]*SF[15] + P_[12][19]*SPP[10];
      nextP[1][19] = P_[1][19] + P_[0][19]*SF[8] + P_[2][19]*SF[7] + P_[3][19]*SF[11] - P_[12][19]*SF[15] + P_[11][19]*SPP[10] - (P_[10][19]*q0)/2;
      nextP[2][19] = P_[2][19] + P_[0][19]*SF[6] + P_[1][19]*SF[10] + P_[3][19]*SF[8] + P_[12][19]*SF[14] - P_[10][19]*SPP[10] - (P_[11][19]*q0)/2;
      nextP[3][19] = P_[3][19] + P_[0][19]*SF[7] + P_[1][19]*SF[6] + P_[2][19]*SF[9] + P_[10][19]*SF[15] - P_[11][19]*SF[14] - (P_[12][19]*q0)/2;
      nextP[4][19] = P_[4][19] + P_[0][19]*SF[5] + P_[1][19]*SF[3] - P_[3][19]*SF[4] + P_[2][19]*SPP[0] + P_[13][19]*SPP[3] + P_[14][19]*SPP[6] - P_[15][19]*SPP[9];
      nextP[5][19] = P_[5][19] + P_[0][19]*SF[4] + P_[2][19]*SF[3] + P_[3][19]*SF[5] - P_[1][19]*SPP[0] - P_[13][19]*SPP[8] + P_[14][19]*SPP[2] + P_[15][19]*SPP[5];
      nextP[6][19] = P_[6][19] + P_[1][19]*SF[4] - P_[2][19]*SF[5] + P_[3][19]*SF[3] + P_[0][19]*SPP[0] + P_[13][19]*SPP[4] - P_[14][19]*SPP[7] - P_[15][19]*SPP[1];
      nextP[7][19] = P_[7][19] + P_[4][19]*dt;
      nextP[8][19] = P_[8][19] + P_[5][19]*dt;
      nextP[9][19] = P_[9][19] + P_[6][19]*dt;
      nextP[10][19] = P_[10][19];
      nextP[11][19] = P_[11][19];
      nextP[12][19] = P_[12][19];
      nextP[13][19] = P_[13][19];
      nextP[14][19] = P_[14][19];
      nextP[15][19] = P_[15][19];
      nextP[16][19] = P_[16][19];
      nextP[17][19] = P_[17][19];
      nextP[18][19] = P_[18][19];
      nextP[19][19] = P_[19][19];
      nextP[0][20] = P_[0][20] + P_[1][20]*SF[9] + P_[2][20]*SF[11] + P_[3][20]*SF[10] + P_[10][20]*SF[14] + P_[11][20]*SF[15] + P_[12][20]*SPP[10];
      nextP[1][20] = P_[1][20] + P_[0][20]*SF[8] + P_[2][20]*SF[7] + P_[3][20]*SF[11] - P_[12][20]*SF[15] + P_[11][20]*SPP[10] - (P_[10][20]*q0)/2;
      nextP[2][20] = P_[2][20] + P_[0][20]*SF[6] + P_[1][20]*SF[10] + P_[3][20]*SF[8] + P_[12][20]*SF[14] - P_[10][20]*SPP[10] - (P_[11][20]*q0)/2;
      nextP[3][20] = P_[3][20] + P_[0][20]*SF[7] + P_[1][20]*SF[6] + P_[2][20]*SF[9] + P_[10][20]*SF[15] - P_[11][20]*SF[14] - (P_[12][20]*q0)/2;
      nextP[4][20] = P_[4][20] + P_[0][20]*SF[5] + P_[1][20]*SF[3] - P_[3][20]*SF[4] + P_[2][20]*SPP[0] + P_[13][20]*SPP[3] + P_[14][20]*SPP[6] - P_[15][20]*SPP[9];
      nextP[5][20] = P_[5][20] + P_[0][20]*SF[4] + P_[2][20]*SF[3] + P_[3][20]*SF[5] - P_[1][20]*SPP[0] - P_[13][20]*SPP[8] + P_[14][20]*SPP[2] + P_[15][20]*SPP[5];
      nextP[6][20] = P_[6][20] + P_[1][20]*SF[4] - P_[2][20]*SF[5] + P_[3][20]*SF[3] + P_[0][20]*SPP[0] + P_[13][20]*SPP[4] - P_[14][20]*SPP[7] - P_[15][20]*SPP[1];
      nextP[7][20] = P_[7][20] + P_[4][20]*dt;
      nextP[8][20] = P_[8][20] + P_[5][20]*dt;
      nextP[9][20] = P_[9][20] + P_[6][20]*dt;
      nextP[10][20] = P_[10][20];
      nextP[11][20] = P_[11][20];
      nextP[12][20] = P_[12][20];
      nextP[13][20] = P_[13][20];
      nextP[14][20] = P_[14][20];
      nextP[15][20] = P_[15][20];
      nextP[16][20] = P_[16][20];
      nextP[17][20] = P_[17][20];
      nextP[18][20] = P_[18][20];
      nextP[19][20] = P_[19][20];
      nextP[20][20] = P_[20][20];
      nextP[0][21] = P_[0][21] + P_[1][21]*SF[9] + P_[2][21]*SF[11] + P_[3][21]*SF[10] + P_[10][21]*SF[14] + P_[11][21]*SF[15] + P_[12][21]*SPP[10];
      nextP[1][21] = P_[1][21] + P_[0][21]*SF[8] + P_[2][21]*SF[7] + P_[3][21]*SF[11] - P_[12][21]*SF[15] + P_[11][21]*SPP[10] - (P_[10][21]*q0)/2;
      nextP[2][21] = P_[2][21] + P_[0][21]*SF[6] + P_[1][21]*SF[10] + P_[3][21]*SF[8] + P_[12][21]*SF[14] - P_[10][21]*SPP[10] - (P_[11][21]*q0)/2;
      nextP[3][21] = P_[3][21] + P_[0][21]*SF[7] + P_[1][21]*SF[6] + P_[2][21]*SF[9] + P_[10][21]*SF[15] - P_[11][21]*SF[14] - (P_[12][21]*q0)/2;
      nextP[4][21] = P_[4][21] + P_[0][21]*SF[5] + P_[1][21]*SF[3] - P_[3][21]*SF[4] + P_[2][21]*SPP[0] + P_[13][21]*SPP[3] + P_[14][21]*SPP[6] - P_[15][21]*SPP[9];
      nextP[5][21] = P_[5][21] + P_[0][21]*SF[4] + P_[2][21]*SF[3] + P_[3][21]*SF[5] - P_[1][21]*SPP[0] - P_[13][21]*SPP[8] + P_[14][21]*SPP[2] + P_[15][21]*SPP[5];
      nextP[6][21] = P_[6][21] + P_[1][21]*SF[4] - P_[2][21]*SF[5] + P_[3][21]*SF[3] + P_[0][21]*SPP[0] + P_[13][21]*SPP[4] - P_[14][21]*SPP[7] - P_[15][21]*SPP[1];
      nextP[7][21] = P_[7][21] + P_[4][21]*dt;
      nextP[8][21] = P_[8][21] + P_[5][21]*dt;
      nextP[9][21] = P_[9][21] + P_[6][21]*dt;
      nextP[10][21] = P_[10][21];
      nextP[11][21] = P_[11][21];
      nextP[12][21] = P_[12][21];
      nextP[13][21] = P_[13][21];
      nextP[14][21] = P_[14][21];
      nextP[15][21] = P_[15][21];
      nextP[16][21] = P_[16][21];
      nextP[17][21] = P_[17][21];
      nextP[18][21] = P_[18][21];
      nextP[19][21] = P_[19][21];
      nextP[20][21] = P_[20][21];
      nextP[21][21] = P_[21][21];

      // add process noise that is not from the IMU
      for (unsigned i = 16; i <= 21; i++) {
        nextP[i][i] += process_noise[i];
      }
    }

    // stop position covariance growth if our total position variance reaches 100m
    if ((P_[7][7] + P_[8][8]) > 1.0e4) {
      for (uint8_t i = 7; i <= 8; i++) {
        for (uint8_t j = 0; j < k_num_states_; j++) {
          nextP[i][j] = P_[i][j];
          nextP[j][i] = P_[j][i];
        }
      }
    }

    // covariance matrix is symmetrical, so copy upper half to lower half
    for (unsigned row = 1; row < k_num_states_; row++) {
      for (unsigned column = 0 ; column < row; column++) {
        P_[row][column] = P_[column][row] = nextP[column][row];
      }
    }

    // copy variances (diagonals)
    for (unsigned i = 0; i < k_num_states_; i++) {
      P_[i][i] = nextP[i][i];
    }
    
    fixCovarianceErrors();
  }
  
  void ESKF::controlFusionModes() {
    
    //gps_data_ready_ = gps_buffer_.pop_first_older_than(imu_sample_delayed_.time_us, &gps_sample_delayed_);
    vision_data_ready_ = ext_vision_buffer_.pop_first_older_than(imu_sample_delayed_.time_us, &ev_sample_delayed_);
    //flow_data_ready_ = opt_flow_buffer_.pop_first_older_than(imu_sample_delayed_.time_us, &opt_flow_sample_delayed_);
    //mag_data_ready_ = mag_buffer_.pop_first_older_than(imu_sample_delayed_.time_us, &mag_sample_delayed_);

    //R_rng_to_earth_2_2_ = R_to_earth_(2, 0) * sin_tilt_rng_ + R_to_earth_(2, 2) * cos_tilt_rng_;
    //range_data_ready_ = range_buffer_.pop_first_older_than(imu_sample_delayed_.time_us, &range_sample_delayed_) && (R_rng_to_earth_2_2_ > range_cos_max_tilt_);
    
    //controlHeightSensorTimeouts();

    // For efficiency, fusion of direct state observations for position and velocity is performed sequentially
    // in a single function using sensor data from multiple sources
    controlVelPosFusion();

    //controlExternalVisionFusion();
    //controlGpsFusion();
    //controlOpticalFlowFusion();
    //controlMagFusion();
    
    //runTerrainEstimator();
  }

  void ESKF::controlExternalVisionFusion() {
    if(vision_data_ready_) {
      // Fuse available NED position data into the main filter
      if ((fusion_mask_ & MASK_EV_POS) && (!ev_pos_)) {
        // check for an external vision measurement that has fallen behind the fusion time horizon
        if (time_last_imu_ - time_last_ext_vision_ < 2 * EV_MAX_INTERVAL) {
          ev_pos_ = true;
          printf("ESKF commencing external vision position fusion\n");
        }
        // reset the position if we are not already aiding using GPS, else use a relative position method for fusing the position data
        if (gps_pos_) {
	  //
        } else {
          resetPosition();
          resetVelocity();
        }
      }

      // determine if we should use the yaw observation
      if ((fusion_mask_ & MASK_EV_YAW) && (!ev_yaw_)) {
        if (time_last_imu_ - time_last_ext_vision_ < 2 * EV_MAX_INTERVAL) {
          // reset the yaw angle to the value from the observaton quaternion
          vec3 euler_init = dcm2vec(quat2dcm(state_.quat_nominal));

          // get initial yaw from the observation quaternion
          const extVisionSample &ev_newest = ext_vision_buffer_.get_newest();
          vec3 euler_obs = dcm2vec(quat2dcm(ev_newest.quatNED));
          euler_init(2) = euler_obs(2);

          // calculate initial quaternion states for the ekf
          state_.quat_nominal = from_axis_angle(euler_init);

          ev_yaw_ = true;
          printf("ESKF commencing external vision yaw fusion\n");
        }
      }
      
      // determine if we should use the hgt observation
      if ((fusion_mask_ & MASK_EV_HGT) && (!ev_hgt_)) {
        // don't start using EV data unless data is arriving frequently
        if (time_last_imu_ - time_last_ext_vision_ < 2 * EV_MAX_INTERVAL) {
          ev_hgt_ = true;
          printf("ESKF commencing external vision hgt fusion\n");
          if(rng_hgt_) {
            //
          } else {
            resetHeight();
          }
        }
      }
      
      if (ev_hgt_) {
        fuse_height_ = true;
      }
      
      if (ev_pos_) {
        fuse_pos_ = true;
      }
      
      if(fuse_height_ || fuse_pos_) {
        fuseVelPosHeight();
        fuse_pos_ = fuse_height_ = false;
      }

      if (ev_yaw_) {
        //fuseHeading();
      }
    }
  }

  void ESKF::controlGpsFusion() {
    if (gps_data_ready_) {
      if ((fusion_mask_ & MASK_GPS_POS) && (!gps_pos_)) {
        gps_pos_ = true;
        printf("ESKF commencing GPS pos fusion\n");
      }
      if(gps_pos_) {
        fuse_pos_ = true;
        fuse_vert_vel_ = true;
        fuse_hor_vel_ = true;
        time_last_gps_ = time_last_imu_;
      }
      if ((fusion_mask_ & MASK_GPS_HGT) && (!gps_hgt_)) {
        gps_hgt_ = true;
        printf("ESKF commencing GPS hgt fusion\n");
      }
      if(gps_pos_) {
        fuse_height_ = true;
        time_last_gps_ = time_last_imu_;
      }
    }
  }

  void ESKF::controlVelPosFusion() {
    if (!gps_pos_ && !opt_flow_ && !ev_pos_) {
      // Fuse synthetic position observations every 200msec
      if ((time_last_imu_ - time_last_fake_gps_ > (uint64_t)2e5) || fuse_height_) {
        // Reset position and velocity states if we re-commence this aiding method
        if ((time_last_imu_ - time_last_fake_gps_) > (uint64_t)4e5) {
          resetPosition();
          resetVelocity();

          if (time_last_fake_gps_ != 0) {
            printf("ESKF stopping navigation\n");
          }
        }

        fuse_pos_ = true;
        fuse_hor_vel_ = false;
        fuse_vert_vel_ = false;
        time_last_fake_gps_ = time_last_imu_;

        vel_pos_innov_[0] = 0.0f;
        vel_pos_innov_[1] = 0.0f;
        vel_pos_innov_[2] = 0.0f;
        vel_pos_innov_[3] = state_.pos(0) - last_known_posNED_(0);
        vel_pos_innov_[4] = state_.pos(1) - last_known_posNED_(1);

        // glitch protection is not required so set gate to a large value
        posInnovGateNE_ = 100.0f;
      }
    }
    
    // Fuse available NED velocity and position data into the main filter
    if (fuse_height_ || fuse_pos_ || fuse_hor_vel_ || fuse_vert_vel_) {
      fuseVelPosHeight();
    }
  }

  void ESKF::fuseVelPosHeight() {
    bool fuse_map[6] = {}; // map of booleans true when [VN,VE,VD,PN,PE,PD] observations are available
    bool innov_check_pass_map[6] = {}; // true when innovations consistency checks pass for [PN,PE,PD] observations
    scalar_t R[6] = {}; // observation variances for [VN,VE,VD,PN,PE,PD]
    scalar_t gate_size[6] = {}; // innovation consistency check gate sizes for [VN,VE,VD,PN,PE,PD] observations
    scalar_t Kfusion[k_num_states_] = {}; // Kalman gain vector for any single observation - sequential fusion is used//lym卡尔曼增益
    
    // calculate innovations, innovations gate sizes and observation variances
    if (fuse_hor_vel_) {
      // enable fusion for NE velocity axes
      fuse_map[0] = fuse_map[1] = true;
      velObsVarNE_(1) = velObsVarNE_(0) = sq(fmaxf(gps_sample_delayed_.sacc, gps_vel_noise_));
      
      // Set observation noise variance and innovation consistency check gate size for the NE position observations
      R[0] = velObsVarNE_(0);
      R[1] = velObsVarNE_(1);
      
      hvelInnovGate_ = fmaxf(vel_innov_gate_, 1.0f);

      gate_size[1] = gate_size[0] = hvelInnovGate_;
    }

    if (fuse_vert_vel_) {
      fuse_map[2] = true;
      // observation variance - use receiver reported accuracy with parameter setting the minimum value
      R[2] = fmaxf(gps_vel_noise_, 0.01f);
      // use scaled horizontal speed accuracy assuming typical ratio of VDOP/HDOP
      R[2] = 1.5f * fmaxf(R[2], gps_sample_delayed_.sacc);
      R[2] = R[2] * R[2];
      // innovation gate size
      gate_size[2] = fmaxf(vel_innov_gate_, 1.0f);
    }
    
    if (fuse_pos_) {
      fuse_map[3] = fuse_map[4] = true;
      
      if(gps_pos_) {
        // calculate observation process noise
        scalar_t lower_limit = fmaxf(gps_pos_noise_, 0.01f);
        scalar_t upper_limit = fmaxf(pos_noaid_noise_, lower_limit);
        posObsNoiseNE_ = constrain(gps_sample_delayed_.hacc, lower_limit, upper_limit);
        velObsVarNE_(1) = velObsVarNE_(0) = sq(fmaxf(gps_sample_delayed_.sacc, gps_vel_noise_));

        // calculate innovations
        vel_pos_innov_[0] = state_.vel(0) - gps_sample_delayed_.vel(0);
        vel_pos_innov_[1] = state_.vel(1) - gps_sample_delayed_.vel(1);
        vel_pos_innov_[2] = state_.vel(2) - gps_sample_delayed_.vel(2);
        vel_pos_innov_[3] = state_.pos(0) - gps_sample_delayed_.pos(0);
        vel_pos_innov_[4] = state_.pos(1) - gps_sample_delayed_.pos(1);

        // observation 1-STD error
        R[3] = sq(posObsNoiseNE_);

        // set innovation gate size
        gate_size[3] = fmaxf(5.0, 1.0f);

      } else if (ev_pos_) {
        // calculate innovations
        // use the absolute position
        vel_pos_innov_[3] = state_.pos(0) - ev_sample_delayed_.posNED(0);
        vel_pos_innov_[4] = state_.pos(1) - ev_sample_delayed_.posNED(1);

        // observation 1-STD error
        R[3] = fmaxf(0.05f, 0.01f);

        // innovation gate size
        gate_size[3] = fmaxf(5.0f, 1.0f);

      } else {
        // No observations - use a static position to constrain drift
        if (in_air_) {
          R[3] = fmaxf(10.0f, 0.5f);
        } else {
          R[3] = 0.5f;
        }
        vel_pos_innov_[3] = state_.pos(0) - last_known_posNED_(0);
        vel_pos_innov_[4] = state_.pos(1) - last_known_posNED_(1);

        // glitch protection is not required so set gate to a large value
        gate_size[3] = 100.0f;

        vel_pos_innov_[5] = state_.pos(2) - last_known_posNED_(2);
        fuse_map[5] = true;
        R[5] = 0.5f;
        R[5] = R[5] * R[5];
        // innovation gate size
        gate_size[5] = 100.0f;
      }

      // convert North position noise to variance
      R[3] = R[3] * R[3];

      // copy North axis values to East axis
      R[4] = R[3];
      gate_size[4] = gate_size[3];
    }

    if (fuse_height_) {
      if(ev_hgt_) {
        fuse_map[5] = true;
        // calculate the innovation assuming the external vision observaton is in local NED frame
        vel_pos_innov_[5] = state_.pos(2) - ev_sample_delayed_.posNED(2);
        // observation variance - defined externally
        R[5] = fmaxf(0.05f, 0.01f);
        R[5] = R[5] * R[5];
        // innovation gate size
        gate_size[5] = fmaxf(5.0f, 1.0f);
      } else if(gps_hgt_) {
        // vertical position innovation - gps measurement has opposite sign to earth z axis
        vel_pos_innov_[5] = state_.pos(2) - gps_sample_delayed_.hgt;
        // observation variance - receiver defined and parameter limited
        // use scaled horizontal position accuracy assuming typical ratio of VDOP/HDOP
        scalar_t lower_limit = fmaxf(gps_pos_noise_, 0.01f);
        scalar_t upper_limit = fmaxf(pos_noaid_noise_, lower_limit);
        R[5] = 1.5f * constrain(gps_sample_delayed_.vacc, lower_limit, upper_limit);
        R[5] = R[5] * R[5];
        // innovation gate size
        gate_size[5] = fmaxf(5.0, 1.0f);
      } else if ((rng_hgt_) && (R_rng_to_earth_2_2_ > range_cos_max_tilt_)) {
        fuse_map[5] = true;
        // use range finder with tilt correction
        vel_pos_innov_[5] = state_.pos(2) - (-max(range_sample_delayed_.rng * R_rng_to_earth_2_2_, rng_gnd_clearance_)) - 0.1f;
        // observation variance - user parameter defined
        R[5] = fmaxf((sq(range_noise_) + sq(range_noise_scaler_ * range_sample_delayed_.rng)) * sq(R_rng_to_earth_2_2_), 0.01f);
        // innovation gate size
        gate_size[5] = fmaxf(range_innov_gate_, 1.0f);
      }
    }

    // calculate innovation test ratios
    for (unsigned obs_index = 0; obs_index < 6; obs_index++) {
      if (fuse_map[obs_index]) {
        // compute the innovation variance SK = HPH + R
        unsigned state_index = obs_index + 4;	// we start with vx and this is the 4. state
        vel_pos_innov_var_[obs_index] = P_[state_index][state_index] + R[obs_index];
        // Compute the ratio of innovation to gate size
        vel_pos_test_ratio_[obs_index] = sq(vel_pos_innov_[obs_index]) / (sq(gate_size[obs_index]) * vel_pos_innov_var_[obs_index]);
      }
    }

    // check position, velocity and height innovations
    // treat 2D position and height as separate sensors
    bool pos_check_pass = ((vel_pos_test_ratio_[3] <= 1.0f) && (vel_pos_test_ratio_[4] <= 1.0f));
    innov_check_pass_map[3] = innov_check_pass_map[4] = pos_check_pass;
    innov_check_pass_map[5] = (vel_pos_test_ratio_[5] <= 1.0f);

    for (unsigned obs_index = 0; obs_index < 6; obs_index++) {
      // skip fusion if not requested or checks have failed
      if (!fuse_map[obs_index] || !innov_check_pass_map[obs_index]) {
        continue;
      }

      unsigned state_index = obs_index + 4;	// we start with vx and this is the 4. state

      // calculate kalman gain K = PHS, where S = 1/innovation variance
      for (int row = 0; row < k_num_states_; row++) {
        Kfusion[row] = P_[row][state_index] / vel_pos_innov_var_[obs_index];
      }

      // update covarinace matrix via Pnew = (I - KH)P
      float KHP[k_num_states_][k_num_states_];
      for (unsigned row = 0; row < k_num_states_; row++) {
        for (unsigned column = 0; column < k_num_states_; column++) {
          KHP[row][column] = Kfusion[row] * P_[state_index][column];
        }
      }

      // if the covariance correction will result in a negative variance, then
      // the covariance marix is unhealthy and must be corrected
      bool healthy = true;
      for (int i = 0; i < k_num_states_; i++) {
        if (P_[i][i] < KHP[i][i]) {
          // zero rows and columns
          zeroRows(P_,i,i);
          zeroCols(P_,i,i);

          //flag as unhealthy
          healthy = false;
        } 
      }

      // only apply covariance and state corrrections if healthy
      if (healthy) {
        // apply the covariance corrections
        for (unsigned row = 0; row < k_num_states_; row++) {
          for (unsigned column = 0; column < k_num_states_; column++) {
            P_[row][column] = P_[row][column] - KHP[row][column];
          }
        }

        // correct the covariance marix for gross errors
        fixCovarianceErrors();

        // apply the state corrections
        fuse(Kfusion, vel_pos_innov_[obs_index]);
      }
    }
  }

  //更新后验位姿 fuse measurement
  void ESKF::fuse(scalar_t *K, scalar_t innovation) {
    state_.quat_nominal.w() = state_.quat_nominal.w() - K[0] * innovation;
    state_.quat_nominal.x() = state_.quat_nominal.x() - K[1] * innovation;
    state_.quat_nominal.y() = state_.quat_nominal.y() - K[2] * innovation;
    state_.quat_nominal.z() = state_.quat_nominal.z() - K[3] * innovation;
    
    state_.quat_nominal.normalize();

    for (unsigned i = 0; i < 3; i++) {
      state_.vel(i) = state_.vel(i) - K[i + 4] * innovation;
    }

    for (unsigned i = 0; i < 3; i++) {
      state_.pos(i) = state_.pos(i) - K[i + 7] * innovation;
    }

    for (unsigned i = 0; i < 3; i++) {
      state_.gyro_bias(i) = state_.gyro_bias(i) - K[i + 10] * innovation;
    }

    for (unsigned i = 0; i < 3; i++) {
      state_.accel_bias(i) = state_.accel_bias(i) - K[i + 13] * innovation;
    }
    
    for (unsigned i = 0; i < 3; i++) {
      state_.mag_I(i) = state_.mag_I(i) - K[i + 16] * innovation;
    }

    for (unsigned i = 0; i < 3; i++) {
      state_.mag_B(i) = state_.mag_B(i) - K[i + 19] * innovation;
    }
  }

  void ESKF::updateVision(const quat& q, const vec3& p, uint64_t time_usec, scalar_t dt) {
    // transform orientation from (ENU2FLU) to (NED2FRD):
    quat q_nb = q_NED2ENU * q * q_FLU2FRD;

    // transform position from local ENU to local NED frame
    vec3 pos_nb = q_NED2ENU.inverse().toRotationMatrix() * p;

    // limit data rate to prevent data being lost
    if (time_usec - time_last_ext_vision_ > min_obs_interval_us_) {
      extVisionSample ev_sample_new;
      // calculate the system time-stamp for the mid point of the integration period
      // copy required data
      ev_sample_new.angErr = 0.05f;
      ev_sample_new.posErr = 0.05f;
      ev_sample_new.quatNED = q_nb;
      ev_sample_new.posNED = pos_nb;
      ev_sample_new.time_us = time_usec - ev_delay_ms_ * 1000;
      time_last_ext_vision_ = time_usec;
      // push to buffer
      ext_vision_buffer_.push(ev_sample_new);
    }
  }

  void ESKF::updateGps(const vec3& v, const vec3& p, uint64_t time_us, scalar_t dt) {
    // transform linear velocity from local ENU to body FRD frame
    vec3 vel_nb = q_NED2ENU.inverse().toRotationMatrix() * v;

    // transform position from local ENU to local NED frame
    vec3 pos_nb = q_NED2ENU.inverse().toRotationMatrix() * p;

    // check for arrival of new sensor data at the fusion time horizon
    if (time_us - time_last_gps_ > min_obs_interval_us_) {
      gpsSample gps_sample_new;
      gps_sample_new.time_us = time_us - gps_delay_ms_ * 1000;

      gps_sample_new.time_us -= FILTER_UPDATE_PERIOD_MS * 1000 / 2;
      time_last_gps_ = time_us;

      gps_sample_new.time_us = max(gps_sample_new.time_us, imu_sample_delayed_.time_us);
      gps_sample_new.vel = vel_nb;
      gps_sample_new.hacc = 1.0;
      gps_sample_new.vacc = 1.0;
      gps_sample_new.sacc = 0.0;
      
      gps_sample_new.pos(0) = pos_nb(0);
      gps_sample_new.pos(1) = pos_nb(1);
      gps_sample_new.hgt = pos_nb(2);
      gps_buffer_.push(gps_sample_new);
    }
  }
    
//修正协方差
  void ESKF::fixCovarianceErrors() {
    scalar_t P_lim[7] = {};
    P_lim[0] = 1.0f;		// quaternion max var
    P_lim[1] = 1e6f;		// velocity max var
    P_lim[2] = 1e6f;		// positiion max var
    P_lim[3] = 1.0f;		// gyro bias max var
    P_lim[4] = 1.0f;		// delta velocity z bias max var
    P_lim[5] = 1.0f;		// earth mag field max var
    P_lim[6] = 1.0f;		// body mag field max var
    
    for (int i = 0; i <= 3; i++) {
      // quaternion states
      P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[0]);
    }
    
    for (int i = 4; i <= 6; i++) {
      // NED velocity states
      P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[1]);
    }

    for (int i = 7; i <= 9; i++) {
      // NED position states
      P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[2]);
    }
    
    for (int i = 10; i <= 12; i++) {
      // gyro bias states
      P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[3]);
    }
    
    // force symmetry on the quaternion, velocity, positon and gyro bias state covariances
    makeSymmetrical(P_,0,12);

    // Find the maximum delta velocity bias state variance and request a covariance reset if any variance is below the safe minimum
    const scalar_t minSafeStateVar = 1e-9f;
    scalar_t maxStateVar = minSafeStateVar;
    bool resetRequired = false;

    for (uint8_t stateIndex = 13; stateIndex <= 15; stateIndex++) {
      if (P_[stateIndex][stateIndex] > maxStateVar) {
        maxStateVar = P_[stateIndex][stateIndex];
      } else if (P_[stateIndex][stateIndex] < minSafeStateVar) {
        resetRequired = true;
      }
    }

    // To ensure stability of the covariance matrix operations, the ratio of a max and min variance must
    // not exceed 100 and the minimum variance must not fall below the target minimum
    // Also limit variance to a maximum equivalent to a 0.1g uncertainty
    const scalar_t minStateVarTarget = 5E-8f;
    scalar_t minAllowedStateVar = fmaxf(0.01f * maxStateVar, minStateVarTarget);

    for (uint8_t stateIndex = 13; stateIndex <= 15; stateIndex++) {
      P_[stateIndex][stateIndex] = constrain(P_[stateIndex][stateIndex], minAllowedStateVar, sq(0.1f * CONSTANTS_ONE_G * dt_ekf_avg_));
    }

    // If any one axis has fallen below the safe minimum, all delta velocity covariance terms must be reset to zero
    if (resetRequired) {
      scalar_t delVelBiasVar[3];

      // store all delta velocity bias variances
      for (uint8_t stateIndex = 13; stateIndex <= 15; stateIndex++) {
        delVelBiasVar[stateIndex - 13] = P_[stateIndex][stateIndex];
      }

      // reset all delta velocity bias covariances
      zeroCols(P_, 13, 15);

      // restore all delta velocity bias variances
      for (uint8_t stateIndex = 13; stateIndex <= 15; stateIndex++) {
        P_[stateIndex][stateIndex] = delVelBiasVar[stateIndex - 13];
      }
    }

    // Run additional checks to see if the delta velocity bias has hit limits in a direction that is clearly wrong
    // calculate accel bias term aligned with the gravity vector
    //scalar_t dVel_bias_lim = 0.9f * acc_bias_lim * dt_ekf_avg_;
    scalar_t down_dvel_bias = 0.0f;

    for (uint8_t axis_index = 0; axis_index < 3; axis_index++) {
      down_dvel_bias += state_.accel_bias(axis_index) * R_to_earth_(2, axis_index);
    }

    // check that the vertical componenent of accel bias is consistent with both the vertical position and velocity innovation
    //bool bad_acc_bias = (fabsf(down_dvel_bias) > dVel_bias_lim && down_dvel_bias * vel_pos_innov_[2] < 0.0f && down_dvel_bias * vel_pos_innov_[5] < 0.0f);

    // if we have failed for 7 seconds continuously, reset the accel bias covariances to fix bad conditioning of
    // the covariance matrix but preserve the variances (diagonals) to allow bias learning to continue
    if (time_last_imu_ - time_acc_bias_check_ > (uint64_t)7e6) {
      scalar_t varX = P_[13][13];
      scalar_t varY = P_[14][14];
      scalar_t varZ = P_[15][15];
      zeroRows(P_, 13, 15);
      zeroCols(P_, 13, 15);
      P_[13][13] = varX;
      P_[14][14] = varY;
      P_[15][15] = varZ;
      //ECL_WARN("EKF invalid accel bias - resetting covariance");
    } else {
      // ensure the covariance values are symmetrical
      makeSymmetrical(P_, 13, 15);
    }

    // magnetic field states
    if (!mag_3D_) {
      zeroRows(P_, 16, 21);
      zeroCols(P_, 16, 21);
    } else {
      // constrain variances
      for (int i = 16; i <= 18; i++) {
        P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[5]);
      }

      for (int i = 19; i <= 21; i++) {
        P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[6]);
      }

      // force symmetry
      makeSymmetrical(P_, 16, 21);
    }
  }
  
  // 强制对齐This function forces the covariance matrix to be symmetric
  void ESKF::makeSymmetrical(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last) {
    for (unsigned row = first; row <= last; row++) {
      for (unsigned column = 0; column < row; column++) {
        float tmp = (cov_mat[row][column] + cov_mat[column][row]) / 2;
        cov_mat[row][column] = tmp;
        cov_mat[column][row] = tmp;
      }
    }
  }

void ESKF::resetPosition() {
    if (gps_pos_) {
      // this reset is only called if we have new gps data at the fusion time horizon
      state_.pos(0) = gps_sample_delayed_.pos(0);
      state_.pos(1) = gps_sample_delayed_.pos(1);

      // use GPS accuracy to reset variances
      setDiag(P_, 7, 8, sq(gps_sample_delayed_.hacc));

    } else if (ev_pos_) {
      // this reset is only called if we have new ev data at the fusion time horizon
      state_.pos(0) = ev_sample_delayed_.posNED(0);
      state_.pos(1) = ev_sample_delayed_.posNED(1);

      // use EV accuracy to reset variances
      setDiag(P_, 7, 8, sq(ev_sample_delayed_.posErr));

    } else if (opt_flow_) {
      if (!in_air_) {
        // we are likely starting OF for the first time so reset the horizontal position
        state_.pos(0) = 0.0f;
        state_.pos(1) = 0.0f;
      } else {
        // set to the last known position
        state_.pos(0) = last_known_posNED_(0);
        state_.pos(1) = last_known_posNED_(1);
      }
      // estimate is relative to initial positon in this mode, so we start with zero error.
      zeroCols(P_, 7, 8);
      zeroRows(P_, 7, 8);
    } else {
      // Used when falling back to non-aiding mode of operation
      state_.pos(0) = last_known_posNED_(0);
      state_.pos(1) = last_known_posNED_(1);
      setDiag(P_, 7, 8, sq(pos_noaid_noise_));
    }
  }

  void ESKF::resetVelocity() {
    
  }

  void ESKF::resetHeight() {
    // reset the vertical position
    if (ev_hgt_) {
      // initialize vertical position with newest measurement
      extVisionSample ev_newest = ext_vision_buffer_.get_newest();

      // use the most recent data if it's time offset from the fusion time horizon is smaller
      int32_t dt_newest = ev_newest.time_us - imu_sample_delayed_.time_us;
      int32_t dt_delayed = ev_sample_delayed_.time_us - imu_sample_delayed_.time_us;

      if (std::abs(dt_newest) < std::abs(dt_delayed)) {
        state_.pos(2) = ev_newest.posNED(2);
      } else {
        state_.pos(2) = ev_sample_delayed_.posNED(2);
      }
    } else if(gps_hgt_) {
      // Get the most recent GPS data
      const gpsSample &gps_newest = gps_buffer_.get_newest();
      if (time_last_imu_ - gps_newest.time_us < 2 * GPS_MAX_INTERVAL) {
        state_.pos(2) = gps_newest.hgt;

        // reset the associated covarince values
        zeroRows(P_, 9, 9);
        zeroCols(P_, 9, 9);

        // the state variance is the same as the observation
        P_[9][9] = sq(gps_newest.hacc);
      }
    }

    // reset the vertical velocity covariance values
    zeroRows(P_, 6, 6);
    zeroCols(P_, 6, 6);

    // we don't know what the vertical velocity is, so set it to zero
    state_.vel(2) = 0.0f;

    // Set the variance to a value large enough to allow the state to converge quickly
    // that does not destabilise the filter
    P_[6][6] = 10.0f;
  }

  quat ESKF::getQuat() {
    // transform orientation from (NED2FRD) to (ENU2FLU)
    return q_NED2ENU.conjugate() * state_.quat_nominal * q_FLU2FRD.conjugate(); 
  }

  vec3 ESKF::getPosition() {
    // transform position from local NED to local ENU frame
    return q_NED2ENU.toRotationMatrix() * state_.pos;
  }

  vec3 ESKF::getVelocity() {
    // transform velocity from local NED to local ENU frame
    return q_NED2ENU.toRotationMatrix() * state_.vel;
  }

  // initialise the quaternion covariances using rotation vector variances
  void ESKF::initialiseQuatCovariances(const vec3& rot_vec_var) {
    // calculate an equivalent rotation vector from the quaternion
    scalar_t q0,q1,q2,q3;
    if (state_.quat_nominal.w() >= 0.0f) {
      q0 = state_.quat_nominal.w();
      q1 = state_.quat_nominal.x();
      q2 = state_.quat_nominal.y();
      q3 = state_.quat_nominal.z();
    } else {
      q0 = -state_.quat_nominal.w();
      q1 = -state_.quat_nominal.x();
      q2 = -state_.quat_nominal.y();
      q3 = -state_.quat_nominal.z();
    }
    scalar_t delta = 2.0f*acosf(q0);
    scalar_t scaler = (delta/sinf(delta*0.5f));
    scalar_t rotX = scaler*q1;
    scalar_t rotY = scaler*q2;
    scalar_t rotZ = scaler*q3;

    // autocode generated using matlab symbolic toolbox
    scalar_t t2 = rotX*rotX;
    scalar_t t4 = rotY*rotY;
    scalar_t t5 = rotZ*rotZ;
    scalar_t t6 = t2+t4+t5;
    if (t6 > 1e-9f) {
      scalar_t t7 = sqrtf(t6);
      scalar_t t8 = t7*0.5f;
      scalar_t t3 = sinf(t8);
      scalar_t t9 = t3*t3;
      scalar_t t10 = 1.0f/t6;
      scalar_t t11 = 1.0f/sqrtf(t6);
      scalar_t t12 = cosf(t8);
      scalar_t t13 = 1.0f/powf(t6,1.5f);
      scalar_t t14 = t3*t11;
      scalar_t t15 = rotX*rotY*t3*t13;
      scalar_t t16 = rotX*rotZ*t3*t13;
      scalar_t t17 = rotY*rotZ*t3*t13;
      scalar_t t18 = t2*t10*t12*0.5f;
      scalar_t t27 = t2*t3*t13;
      scalar_t t19 = t14+t18-t27;
      scalar_t t23 = rotX*rotY*t10*t12*0.5f;
      scalar_t t28 = t15-t23;
      scalar_t t20 = rotY*rot_vec_var(1)*t3*t11*t28*0.5f;
      scalar_t t25 = rotX*rotZ*t10*t12*0.5f;
      scalar_t t31 = t16-t25;
      scalar_t t21 = rotZ*rot_vec_var(2)*t3*t11*t31*0.5f;
      scalar_t t22 = t20+t21-rotX*rot_vec_var(0)*t3*t11*t19*0.5f;
      scalar_t t24 = t15-t23;
      scalar_t t26 = t16-t25;
      scalar_t t29 = t4*t10*t12*0.5f;
      scalar_t t34 = t3*t4*t13;
      scalar_t t30 = t14+t29-t34;
      scalar_t t32 = t5*t10*t12*0.5f;
      scalar_t t40 = t3*t5*t13;
      scalar_t t33 = t14+t32-t40;
      scalar_t t36 = rotY*rotZ*t10*t12*0.5f;
      scalar_t t39 = t17-t36;
      scalar_t t35 = rotZ*rot_vec_var(2)*t3*t11*t39*0.5f;
      scalar_t t37 = t15-t23;
      scalar_t t38 = t17-t36;
      scalar_t t41 = rot_vec_var(0)*(t15-t23)*(t16-t25);
      scalar_t t42 = t41-rot_vec_var(1)*t30*t39-rot_vec_var(2)*t33*t39;
      scalar_t t43 = t16-t25;
      scalar_t t44 = t17-t36;

      // zero all the quaternion covariances
      zeroRows(P_,0,3);
      zeroCols(P_,0,3);

      // Update the quaternion internal covariances using auto-code generated using matlab symbolic toolbox
      P_[0][0] = rot_vec_var(0)*t2*t9*t10*0.25f+rot_vec_var(1)*t4*t9*t10*0.25f+rot_vec_var(2)*t5*t9*t10*0.25f;
      P_[0][1] = t22;
      P_[0][2] = t35+rotX*rot_vec_var(0)*t3*t11*(t15-rotX*rotY*t10*t12*0.5f)*0.5f-rotY*rot_vec_var(1)*t3*t11*t30*0.5f;
      P_[0][3] = rotX*rot_vec_var(0)*t3*t11*(t16-rotX*rotZ*t10*t12*0.5f)*0.5f+rotY*rot_vec_var(1)*t3*t11*(t17-rotY*rotZ*t10*t12*0.5f)*0.5f-rotZ*rot_vec_var(2)*t3*t11*t33*0.5f;
      P_[1][0] = t22;
      P_[1][1] = rot_vec_var(0)*(t19*t19)+rot_vec_var(1)*(t24*t24)+rot_vec_var(2)*(t26*t26);
      P_[1][2] = rot_vec_var(2)*(t16-t25)*(t17-rotY*rotZ*t10*t12*0.5f)-rot_vec_var(0)*t19*t28-rot_vec_var(1)*t28*t30;
      P_[1][3] = rot_vec_var(1)*(t15-t23)*(t17-rotY*rotZ*t10*t12*0.5f)-rot_vec_var(0)*t19*t31-rot_vec_var(2)*t31*t33;
      P_[2][0] = t35-rotY*rot_vec_var(1)*t3*t11*t30*0.5f+rotX*rot_vec_var(0)*t3*t11*(t15-t23)*0.5f;
      P_[2][1] = rot_vec_var(2)*(t16-t25)*(t17-t36)-rot_vec_var(0)*t19*t28-rot_vec_var(1)*t28*t30;
      P_[2][2] = rot_vec_var(1)*(t30*t30)+rot_vec_var(0)*(t37*t37)+rot_vec_var(2)*(t38*t38);
      P_[2][3] = t42;
      P_[3][0] = rotZ*rot_vec_var(2)*t3*t11*t33*(-0.5f)+rotX*rot_vec_var(0)*t3*t11*(t16-t25)*0.5f+rotY*rot_vec_var(1)*t3*t11*(t17-t36)*0.5f;
      P_[3][1] = rot_vec_var(1)*(t15-t23)*(t17-t36)-rot_vec_var(0)*t19*t31-rot_vec_var(2)*t31*t33;
      P_[3][2] = t42;
      P_[3][3] = rot_vec_var(2)*(t33*t33)+rot_vec_var(0)*(t43*t43)+rot_vec_var(1)*(t44*t44);
    } else {
      // the equations are badly conditioned so use a small angle approximation
      P_[0][0] = 0.0f;
      P_[0][1] = 0.0f;
      P_[0][2] = 0.0f;
      P_[0][3] = 0.0f;
      P_[1][0] = 0.0f;
      P_[1][1] = 0.25f * rot_vec_var(0);
      P_[1][2] = 0.0f;
      P_[1][3] = 0.0f;
      P_[2][0] = 0.0f;
      P_[2][1] = 0.0f;
      P_[2][2] = 0.25f * rot_vec_var(1);
      P_[2][3] = 0.0f;
      P_[3][0] = 0.0f;
      P_[3][1] = 0.0f;
      P_[3][2] = 0.0f;
      P_[3][3] = 0.25f * rot_vec_var(2);
    }
  }
  
  // zero specified range of rows in the state covariance matrix
  void ESKF::zeroRows(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last) {
    uint8_t row;
    for (row = first; row <= last; row++) {
      memset(&cov_mat[row][0], 0, sizeof(cov_mat[0][0]) * k_num_states_);
    }
  }

  // zero specified range of columns in the state covariance matrix
  void ESKF::zeroCols(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last) {
    uint8_t row;
    for (row = 0; row <= k_num_states_-1; row++) {
      memset(&cov_mat[row][first], 0, sizeof(cov_mat[0][0]) * (1 + last - first));
    }
  }

  void ESKF::setDiag(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last, scalar_t variance) {
    // zero rows and columns
    zeroRows(cov_mat, first, last);
    zeroCols(cov_mat, first, last);

    // set diagonals
    for (uint8_t row = first; row <= last; row++) {
      cov_mat[row][row] = variance;
    }
  }

  void ESKF::constrainStates() {
    state_.quat_nominal.w() = constrain(state_.quat_nominal.w(), -1.0f, 1.0f);
    state_.quat_nominal.x() = constrain(state_.quat_nominal.x(), -1.0f, 1.0f);
    state_.quat_nominal.y() = constrain(state_.quat_nominal.y(), -1.0f, 1.0f);
    state_.quat_nominal.z() = constrain(state_.quat_nominal.z(), -1.0f, 1.0f);
	  
    for (int i = 0; i < 3; i++) {
      state_.vel(i) = constrain(state_.vel(i), -1000.0f, 1000.0f);
    }

    for (int i = 0; i < 3; i++) {
      state_.pos(i) = constrain(state_.pos(i), -1.e6f, 1.e6f);
    }

    for (int i = 0; i < 3; i++) {
      state_.gyro_bias(i) = constrain(state_.gyro_bias(i), -0.349066f * dt_ekf_avg_, 0.349066f * dt_ekf_avg_);
    }

    for (int i = 0; i < 3; i++) {
      state_.accel_bias(i) = constrain(state_.accel_bias(i), -acc_bias_lim * dt_ekf_avg_, acc_bias_lim * dt_ekf_avg_);
    }
    
    for (int i = 0; i < 3; i++) {
      state_.mag_I(i) = constrain(state_.mag_I(i), -1.0f, 1.0f);
    }

    for (int i = 0; i < 3; i++) {
      state_.mag_B(i) = constrain(state_.mag_B(i), -0.5f, 0.5f);
    }
  }

  void ESKF::setFusionMask(int fusion_mask) {
    fusion_mask_ = fusion_mask;
  }
} //  namespace eskf
