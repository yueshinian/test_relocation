#include <mutex>
#include <memory>
#include <iostream>
#include <chrono>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <std_srvs/Empty.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseWithCovariance.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/ScanMatchingStatus.h>
#include <hdl_global_localization/SetGlobalMap.h>
#include <hdl_global_localization/QueryGlobalLocalization.h>
#include "eskf.hpp"
#include "global_location.hpp"

const int imuQueLength = 200;
const float scanPeriod = 0.1;

namespace hdl_localization {

class HdlLocalizationNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() : tf_buffer(), tf_listener(tf_buffer) {}
  virtual ~HdlLocalizationNodelet() {}

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    if (use_imu) {
      NODELET_INFO("enable imu-based prediction");
      imu_sub = mt_nh.subscribe("/imu/imu_data", 256, &HdlLocalizationNodelet::imu_callback, this);
    }
    points_sub = mt_nh.subscribe("/velodyne_points", 5, &HdlLocalizationNodelet::points_callback, this);
    globalmap_sub = nh.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmap_callback, this);
    initialpose_sub = nh.subscribe("/initialpose", 8, &HdlLocalizationNodelet::initialpose_callback, this);
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 5, false);

    correctOdomPub = nh.advertise<nav_msgs::Odometry>("/correctOdom",5);
    pose_pub = nh.advertise<nav_msgs::Odometry>("/state_estimation2", 5, false);            // lym
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/registered_scan2", 5, false);  // lym
    queryLaser = nh.advertise<sensor_msgs::PointCloud2>("/queryLaser", 10);
    queryPose = nh.subscribe<nav_msgs::Odometry>("/queryOdom", 10, &HdlLocalizationNodelet::queryPoseCallback, this);
    pubLocalMap = nh.advertise<sensor_msgs::PointCloud2>("/localMap", 10);
    mapPubTimer = nh.createTimer(ros::Duration(1), &HdlLocalizationNodelet::mpTimer, this);
    subImu = mt_nh.subscribe("/imu/data", 200, &HdlLocalizationNodelet::imuHandler, this);

    // global localization
    if (use_global_localization) {
      NODELET_INFO_STREAM("wait for global localization services");
      ros::service::waitForService("/hdl_global_localization/set_global_map");
      ros::service::waitForService("/hdl_global_localization/query");

      set_global_map_service = nh.serviceClient<hdl_global_localization::SetGlobalMap>("/hdl_global_localization/set_global_map");
      query_global_localization_service = nh.serviceClient<hdl_global_localization::QueryGlobalLocalization>("/hdl_global_localization/query");

      relocalize_server = nh.advertiseService("/relocalize", &HdlLocalizationNodelet::relocalize, this);
    }
  }

private:
  pcl::Registration<PointT, PointT>::Ptr create_registration() const {
    if (reg_method == "NDT_OMP") {
      NODELET_INFO("NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);  // lym误差允许值
      ndt->setResolution(ndt_resolution);   /// lym网格分辨率
      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          NODELET_INFO("search_method KDTREE is selected");
        } else {
          NODELET_WARN("invalid search method was given");
          NODELET_WARN("default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    } else if (reg_method.find("NDT_CUDA") != std::string::npos) {
      NODELET_INFO("NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if (reg_method.find("D2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      } else if (reg_method.find("P2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
        NODELET_INFO_STREAM("search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      } else {
        NODELET_WARN("invalid search method was given");
      }
      return ndt;
    }

    NODELET_ERROR_STREAM("unknown registration method:" << reg_method);
    return nullptr;
  }

  void initialize_params() {
    eskfPtr.reset(new ESKF::eskf());
    gLocatePtr.reset(new GLOBAL_LOCALION::globalLocation());
    isInitSystem = 0;
    getMeasure = false;
    scPose.pose.pose.orientation.w = -999;
    localMap.reset(new pcl::PointCloud<PointT>());
    globalMap.reset(new pcl::PointCloud<PointT>());
    velodyneCloud.reset(new pcl::PointCloud<PointT>());
    g_globalCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    g_velodyneCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    localMapOrigin.x = localMapOrigin.y = localMapOrigin.z = 0.0;
    localLength = private_nh.param<double>("localLength", 10);
    useScInit = private_nh.param<bool>("useScInit", false);

    lastOdom = Eigen::Matrix4f::Identity();
    curOdom = Eigen::Matrix4f::Identity();
    nextOdom = Eigen::Matrix4f::Identity();

    use_global_localization = private_nh.param<bool>("use_global_localization", true);
    isRelocalize = private_nh.param<bool>("isRelocalize", false);

    robot_odom_frame_id = private_nh.param<std::string>("robot_odom_frame_id", "robot_odom");
    odom_child_frame_id = private_nh.param<std::string>("odom_child_frame_id", "base_link");

    use_imu = private_nh.param<bool>("use_imu", true);
    invert_acc = private_nh.param<bool>("invert_acc", false);
    invert_gyro = private_nh.param<bool>("invert_gyro", false);

    reg_method = private_nh.param<std::string>("reg_method", "NDT_OMP");
    ndt_neighbor_search_method = private_nh.param<std::string>("ndt_neighbor_search_method", "DIRECT7");
    ndt_neighbor_search_radius = private_nh.param<double>("ndt_neighbor_search_radius", 2.0);
    ndt_resolution = private_nh.param<double>("ndt_resolution", 1.0);

    cool_time_duration = private_nh.param<double>("cool_time_duration", 0.5);

    downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    //滤波器
    downsample_filter = createFilter(downsample_resolution);
    // ndt匹配
    ROS_INFO("create registration [ %s ] method for localization", reg_method);
    registration = create_registration();
    if (use_imu) {
      imuPointerFront = 0;
      imuPointerLast = -1;
      imuPointerLastIteration = 0;

      imuRollStart = 0;
      imuPitchStart = 0;
      imuYawStart = 0;
      cosImuRollStart = 0;
      cosImuPitchStart = 0;
      cosImuYawStart = 0;
      sinImuRollStart = 0;
      sinImuPitchStart = 0;
      sinImuYawStart = 0;
      imuRollCur = 0;
      imuPitchCur = 0;
      imuYawCur = 0;

      imuVeloXStart = 0;
      imuVeloYStart = 0;
      imuVeloZStart = 0;
      imuShiftXStart = 0;
      imuShiftYStart = 0;
      imuShiftZStart = 0;

      imuVeloXCur = 0;
      imuVeloYCur = 0;
      imuVeloZCur = 0;
      imuShiftXCur = 0;
      imuShiftYCur = 0;
      imuShiftZCur = 0;

      imuShiftFromStartXCur = 0;
      imuShiftFromStartYCur = 0;
      imuShiftFromStartZCur = 0;
      imuVeloFromStartXCur = 0;
      imuVeloFromStartYCur = 0;
      imuVeloFromStartZCur = 0;

      imuAngularRotationXCur = 0;
      imuAngularRotationYCur = 0;
      imuAngularRotationZCur = 0;
      imuAngularRotationXLast = 0;
      imuAngularRotationYLast = 0;
      imuAngularRotationZLast = 0;
      imuAngularFromStartX = 0;
      imuAngularFromStartY = 0;
      imuAngularFromStartZ = 0;

      for (int i = 0; i < imuQueLength; ++i) {
        imuTime[i] = 0;
        imuRoll[i] = 0;
        imuPitch[i] = 0;
        imuYaw[i] = 0;
        imuAccX[i] = 0;
        imuAccY[i] = 0;
        imuAccZ[i] = 0;
        imuVeloX[i] = 0;
        imuVeloY[i] = 0;
        imuVeloZ[i] = 0;
        imuShiftX[i] = 0;
        imuShiftY[i] = 0;
        imuShiftZ[i] = 0;
        imuAngularVeloX[i] = 0;
        imuAngularVeloY[i] = 0;
        imuAngularVeloZ[i] = 0;
        imuAngularRotationX[i] = 0;
        imuAngularRotationY[i] = 0;
        imuAngularRotationZ[i] = 0;
      }

      imuRollLast = 0;
      imuPitchLast = 0;
      imuYawLast = 0;
      imuShiftFromStartX = 0;
      imuShiftFromStartY = 0;
      imuShiftFromStartZ = 0;
      imuVeloFromStartX = 0;
      imuVeloFromStartY = 0;
      imuVeloFromStartZ = 0;
    }
  }

private:
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex);
    imu_data.push_back(imu_msg);
  }

  void AccumulateIMUShiftAndRotation() {
    float roll = imuRoll[imuPointerLast];
    float pitch = imuPitch[imuPointerLast];
    float yaw = imuYaw[imuPointerLast];
    float accX = imuAccX[imuPointerLast];
    float accY = imuAccY[imuPointerLast];
    float accZ = imuAccZ[imuPointerLast];

    float x1 = cos(roll) * accX - sin(roll) * accY;
    float y1 = sin(roll) * accX + cos(roll) * accY;
    float z1 = accZ;

    float x2 = x1;
    float y2 = cos(pitch) * y1 - sin(pitch) * z1;
    float z2 = sin(pitch) * y1 + cos(pitch) * z1;

    accX = cos(yaw) * x2 + sin(yaw) * z2;
    accY = y2;
    accZ = -sin(yaw) * x2 + cos(yaw) * z2;

    int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
    double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
    if (timeDiff < scanPeriod) {
      imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff + accX * timeDiff * timeDiff / 2;
      imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff + accY * timeDiff * timeDiff / 2;
      imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff + accZ * timeDiff * timeDiff / 2;

      imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
      imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
      imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;

      imuAngularRotationX[imuPointerLast] = imuAngularRotationX[imuPointerBack] + imuAngularVeloX[imuPointerBack] * timeDiff;
      imuAngularRotationY[imuPointerLast] = imuAngularRotationY[imuPointerBack] + imuAngularVeloY[imuPointerBack] * timeDiff;
      imuAngularRotationZ[imuPointerLast] = imuAngularRotationZ[imuPointerBack] + imuAngularVeloZ[imuPointerBack] * timeDiff;
    }
  }

  void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn) {
    // double roll, pitch, yaw;
    // tf::Quaternion orientation;
    // tf::quaternionMsgToTF(imuIn->orientation, orientation);
    // tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    // //去重力
    // float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
    // float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
    // float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

    // imuPointerLast = (imuPointerLast + 1) % imuQueLength;

    // imuTime[imuPointerLast] = imuIn->header.stamp.toSec();

    // imuRoll[imuPointerLast] = roll;
    // imuPitch[imuPointerLast] = pitch;
    // imuYaw[imuPointerLast] = yaw;

    // imuAccX[imuPointerLast] = accX;
    // imuAccY[imuPointerLast] = accY;
    // imuAccZ[imuPointerLast] = accZ;

    // imuAngularVeloX[imuPointerLast] = imuIn->angular_velocity.x;
    // imuAngularVeloY[imuPointerLast] = imuIn->angular_velocity.y;
    // imuAngularVeloZ[imuPointerLast] = imuIn->angular_velocity.z;

    // AccumulateIMUShiftAndRotation();
    if (isInitSystem < initSystem) {
      isInitSystem++;
      return;
    }
    sensor_msgs::Imu imuOut = *imuIn;
    eskfPtr->predict(imuOut);
    if(getMeasure){
      nav_msgs::Odometry correctOdom;
      correctOdom.header.frame_id = "/map";
      correctOdom.pose.pose =eskfPtr->correct(observePose);
      correctOdomPub.publish(correctOdom);
      getMeasure = false;
    }
  }

  void points_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    if (isInitSystem < initSystem) {
      isInitSystem++;
      return;
    }
    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);
    pcl::fromROSMsg(*points_msg, *velodyneCloud);
    pcl::fromROSMsg(*points_msg, *g_velodyneCloud);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pcl_cloud, *pcl_cloud, indices);

    if (pcl_cloud->empty()) {
      NODELET_ERROR("laser cloud is empty!!");
      return;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    cloud = pcl_cloud;

    auto filtered = downsample(cloud);
    last_scan = filtered;  //之用作重定位

    if (!globalmap) {
      NODELET_ERROR("globalmap has not been received!!");
      return;
    }

    auto startTime = std::chrono::steady_clock::now();
    gLocatePtr->setSourceCloud(g_velodyneCloud);
    gLocatePtr->downSampleCompute(0.5);
    Eigen::Matrix4f alignTrans = gLocatePtr->aligned(2.0, 8.0);
    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    std::cout<<"********************************"<<std::endl;
    std::cout<<"align time is: "<<duration/1000.0<<" s"<<std::endl;
    std::cout<<alignTrans<<std::endl;
    return;
    //全局重定位
    if (!isRelocalize && use_global_localization) {
      isRelocalize = globalRelocalize();
      return;
    }

    if (!pose_estimator) {
      NODELET_ERROR("waiting for initial pose input!!");
      return;
    }

    Eigen::Matrix4f before = pose_estimator->matrix();

    // predict
    if (!use_imu) {
      pose_estimator->predict(stamp);
    } else {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      for (imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        if (stamp < (*imu_iter)->header.stamp) {
          break;
        }
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        double acc_sign = invert_acc ? -1.0 : 1.0;
        double gyro_sign = invert_gyro ? -1.0 : 1.0;
        pose_estimator->predict((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      imu_data.erase(imu_data.begin(), imu_iter);
    }

    // correct
    auto aligned = pose_estimator->correct(stamp, filtered);

    if (aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      aligned_pub.publish(aligned);
    }
    /* ymliu*/
    // predict
    nextOdom = curOdom * lastOdom.inverse() * curOdom;
    lastOdom = curOdom;
    // correct
    if(use_imu){
      Eigen::Isometry3d predictOdom;
      predictOdom.rotate(eskfPtr->getQuaternion().toRotationMatrix());
      predictOdom.pretranslate(eskfPtr->getPosition());
      Eigen::Matrix4d temp_matrix = predictOdom.matrix();
      for(int i=0;i<4;++i){
        for(int j=0;j<4;++j){
           nextOdom(i,j) = temp_matrix(i,j);
        }
      }
    }
    registration->setInputCloud(filtered);
    pcl::PointCloud<PointT>::Ptr alignPoints(new pcl::PointCloud<PointT>());
    registration->align(*alignPoints, nextOdom);
    curOdom = registration->getFinalTransformation();
    // pub tf
    geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(curOdom.cast<double>()));
    odom_trans.header.stamp = stamp;
    odom_trans.header.frame_id = "map";
    odom_trans.child_frame_id = "myOdom";
    tf_broadcaster.sendTransform(odom_trans);
    //fusion imu
    tf::poseEigenToMsg(Eigen::Isometry3d(pose_estimator->matrix().cast<double>()), observePose);
    getMeasure = true;
    /* ymliu */
    publish_odometry(points_msg->header.stamp, pose_estimator->matrix(), points_msg);
  }

  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    pcl::fromROSMsg(*points_msg, *globalMap);
    pcl::fromROSMsg(*points_msg, *g_globalCloud);
    globalmap = cloud;

    registration->setInputTarget(globalmap);

    gLocatePtr->setTargetCloud(g_globalCloud);

    if (use_global_localization) {
      NODELET_INFO("set globalmap for global localization!");
      hdl_global_localization::SetGlobalMap srv;
      pcl::toROSMsg(*globalmap, srv.request.global_map);

      if (!set_global_map_service.call(srv)) {
        NODELET_INFO("failed to set global map");
      } else {
        NODELET_INFO("done");
      }
    }

    ROS_WARN("waiting pointcloud...");
  }

  bool relocalize(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res) {
    if (last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;

    if (!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
      NODELET_INFO_STREAM("global localization failed");
      return false;
    }

    const auto& result = srv.response.poses[0];

    NODELET_INFO_STREAM("--- Global localization result ---");
    NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
    NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);

    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(new hdl_localization::PoseEstimator(registration, ros::Time::now(), pose.translation(), Eigen::Quaternionf(pose.linear()), cool_time_duration));

    return true;
  }

  bool globalRelocalize() {
    if (last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    if (useScInit) {
      pcl::PointCloud<PointT>::Ptr newGlobalMap(new pcl::PointCloud<PointT>());
      sensor_msgs::PointCloud2 laserPoint;
      pcl::toROSMsg(*velodyneCloud, laserPoint);
      queryLaser.publish(laserPoint);
      while (scPose.pose.pose.orientation.w == -999) {
        ROS_INFO("query scan context!");
      }
      double length = 30;
      cropFilter.setMin(Eigen::Vector4f(scPose.pose.pose.position.x - length, scPose.pose.pose.position.y - length, scPose.pose.pose.position.z - length, 1.0));
      cropFilter.setMax(Eigen::Vector4f(scPose.pose.pose.position.x + length, scPose.pose.pose.position.y + length, scPose.pose.pose.position.z + length, 1.0));
      cropFilter.setNegative(false);  // false，保留里面
      cropFilter.setInputCloud(globalMap);
      cropFilter.filter(*newGlobalMap);
      if (use_global_localization) {
        NODELET_INFO("set globalmap for global localization!");
        hdl_global_localization::SetGlobalMap globalmap_srv;
        pcl::toROSMsg(*newGlobalMap, globalmap_srv.request.global_map);
        if (!set_global_map_service.call(globalmap_srv)) {
          NODELET_INFO("failed to set global map");
        } else {
          NODELET_INFO("done");
        }
      }
    }

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;

    auto startTime = std::chrono::steady_clock::now();
    if (!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
      NODELET_INFO_STREAM("global localization failed");
      return false;
    } else {
      auto endTime = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();
      ROS_WARN("global localization succeeded! cost time is %f ms.", duration);
    }

    const auto& result = srv.response.poses[0];

    NODELET_INFO_STREAM("--- Global localization result ---");
    NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
    NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);

    // std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    //配准方式，时间戳，位置，姿态，冷却时间
    pose_estimator.reset(new hdl_localization::PoseEstimator(registration, ros::Time::now(), pose.translation(), Eigen::Quaternionf(pose.linear()), cool_time_duration));
    eskfPtr->setPosition(Eigen::Vector3d(result.position.x, result.position.y, result.position.z));
    eskfPtr->setOrientation(Eigen::Quaterniond(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z));
    updateLocalMap(result.position.x, result.position.y, result.position.z, localLength, localLength, localLength);
    registration->setInputTarget(localMap);
    return true;
  }

  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    NODELET_INFO("initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator.reset(
      new hdl_localization::PoseEstimator(registration, ros::Time::now(), Eigen::Vector3f(p.x, p.y, p.z), Eigen::Quaternionf(q.w, q.x, q.y, q.z), cool_time_duration));
    lastOdom.block<3, 1>(0, 3) = Eigen::Vector3f(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    lastOdom.block<3, 3>(0, 0) =
      Eigen::Quaternionf(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x, pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z)
        .normalized()
        .toRotationMatrix();
    curOdom = lastOdom;
    eskfPtr->setPosition(Eigen::Vector3d(p.x,p.y,p.z));
    eskfPtr->setOrientation(Eigen::Quaterniond(q.w, q.x, q.y, q.z));
    isRelocalize = true;
  }

  pcl::Filter<PointT>::Ptr createFilter(double downsample_resolution) {
    boost::shared_ptr<pcl::VoxelGrid<PointT>> chooseFilter(new pcl::VoxelGrid<PointT>());
    chooseFilter->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    return chooseFilter;
  }

  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if (!downsample_filter) {
      return cloud;
    }
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;
    return filtered;
  }

  void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose, const sensor_msgs::PointCloud2ConstPtr& laserCloud2) {
    // broadcast tf
    geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
    odom_trans.header.stamp = stamp;
    odom_trans.header.frame_id = "map";
    odom_trans.child_frame_id = odom_child_frame_id;
    tf_broadcaster.sendTransform(odom_trans);

    // publish odom
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";
    tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    odom.child_frame_id = "sensor";
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;
    pose_pub.publish(odom);

    // update local map
    if (
      ((localMapOrigin.x - odom.pose.pose.position.x) * (localMapOrigin.x - odom.pose.pose.position.x) +
       (localMapOrigin.y - odom.pose.pose.position.y) * (localMapOrigin.y - odom.pose.pose.position.y) +
       (localMapOrigin.z - odom.pose.pose.position.z) * (localMapOrigin.z - odom.pose.pose.position.z)) > 0.8 * localLength * 0.8 * localLength) {
      localMapOrigin = odom.pose.pose.position;
      updateLocalMap(localMapOrigin.x, localMapOrigin.y, localMapOrigin.z, localLength, localLength, localLength);
      registration->setInputTarget(localMap);
    }

    // lym,velodyne points to map frame
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCLoudInMapFrame(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*laserCloud2, *laserCloudIn);
    tf::StampedTransform transformToMap;
    transformToMap.setOrigin(tf::Vector3(odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z));
    transformToMap.setRotation(tf::Quaternion(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w));
    pcl::PointXYZI p1;
    tf::Vector3 vec;
    int laserCloudInNum = laserCloudIn->points.size();
    for (int i = 0; i < laserCloudInNum; i++) {
      p1 = laserCloudIn->points[i];
      vec.setX(p1.x);
      vec.setY(p1.y);
      vec.setZ(p1.z);
      vec = transformToMap * vec;
      p1.intensity = laserCloudIn->points[i].intensity;
      laserCLoudInMapFrame->points.emplace_back(p1);
    }
    sensor_msgs::PointCloud2 scan_data;
    pcl::toROSMsg(*laserCLoudInMapFrame, scan_data);
    scan_data.header.stamp = laserCloud2->header.stamp;
    scan_data.header.frame_id = "/map";
    pubLaserCloud.publish(scan_data);
  }

  void queryPoseCallback(const nav_msgs::Odometry::ConstPtr& odom_msg) {
    ROS_INFO("position is x=%f, y=%f, z=%f.", odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z);
    ROS_INFO(
      "sc orientation is w=%f, x=%f, y=%f, z=%f.",
      odom_msg->pose.pose.orientation.x,
      odom_msg->pose.pose.orientation.y,
      odom_msg->pose.pose.orientation.z,
      odom_msg->pose.pose.orientation.w);
    scPose.header = odom_msg->header;
    scPose.pose = odom_msg->pose;
  }

  void updateLocalMap(double originx, double originy, double originz, double lengthx, double lengthy, double lengthz) {
    cropFilter.setMin(Eigen::Vector4f(originx - lengthx, originy - lengthy, originz - lengthz, 1.0));
    cropFilter.setMax(Eigen::Vector4f(originx + lengthx, originy + lengthy, originz + lengthz, 1.0));
    cropFilter.setNegative(false);  // false，保留里面
    cropFilter.setInputCloud(globalMap);
    cropFilter.filter(*localMap);
    ROS_INFO("localMap update. size is %d.!", localMap->points.size());
  }

  void mpTimer(const ros::TimerEvent& e) {
    sensor_msgs::PointCloud2 mapClouds;
    pcl::toROSMsg(*localMap, mapClouds);
    mapClouds.header.frame_id = "/map";
    pubLocalMap.publish(mapClouds);
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  std::string robot_odom_frame_id;
  std::string odom_child_frame_id;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;
  ros::Subscriber imu_sub;
  ros::Subscriber points_sub;
  ros::Subscriber globalmap_sub;
  ros::Subscriber initialpose_sub;

  ros::Publisher pose_pub;
  ros::Publisher aligned_pub;
  ros::Publisher pubLaserCloud;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
  tf2_ros::TransformBroadcaster tf_broadcaster;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::ImuConstPtr> imu_data;

  // globalmap and registration method
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

  // global localization
  bool use_global_localization;
  // std::unique_ptr<DeltaEstimater> delta_estimater;

  pcl::PointCloud<PointT>::ConstPtr last_scan;
  ros::ServiceServer relocalize_server;
  ros::ServiceClient set_global_map_service;
  ros::ServiceClient query_global_localization_service;

  // lym
  std::unique_ptr<GLOBAL_LOCALION::globalLocation> gLocatePtr;
  std::unique_ptr<ESKF::eskf> eskfPtr;
  bool getMeasure;
  geometry_msgs::Pose observePose;
  ros::Publisher correctOdomPub;
  const int initSystem = 10;
  int isInitSystem;
  bool isRelocalize;
  ros::Publisher queryLaser;
  ros::Subscriber queryPose;

  double cool_time_duration;
  double ndt_neighbor_search_radius;
  double ndt_resolution;
  double downsample_resolution;
  std::string reg_method;
  std::string ndt_neighbor_search_method;

  pcl::CropBox<PointT> cropFilter;

  ros::Publisher pubLocalMap;
  ros::Subscriber subImu;
  pcl::PointCloud<PointT>::Ptr localMap;
  pcl::PointCloud<PointT>::Ptr globalMap;
  pcl::PointCloud<PointT>::Ptr velodyneCloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr g_globalCloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr g_velodyneCloud;
  geometry_msgs::Point localMapOrigin;
  double localLength;

  bool useScInit;
  nav_msgs::Odometry scPose;

  ros::Timer mapPubTimer;

  Eigen::Matrix4f lastOdom;
  Eigen::Matrix4f curOdom;
  Eigen::Matrix4f nextOdom;

  int imuPointerFront;
  int imuPointerLast;
  int imuPointerLastIteration;

  float imuRollStart, imuPitchStart, imuYawStart;
  float cosImuRollStart, cosImuPitchStart, cosImuYawStart, sinImuRollStart, sinImuPitchStart, sinImuYawStart;
  float imuRollCur, imuPitchCur, imuYawCur;

  float imuVeloXStart, imuVeloYStart, imuVeloZStart;
  float imuShiftXStart, imuShiftYStart, imuShiftZStart;

  float imuVeloXCur, imuVeloYCur, imuVeloZCur;
  float imuShiftXCur, imuShiftYCur, imuShiftZCur;

  float imuShiftFromStartXCur, imuShiftFromStartYCur, imuShiftFromStartZCur;
  float imuVeloFromStartXCur, imuVeloFromStartYCur, imuVeloFromStartZCur;

  float imuAngularRotationXCur, imuAngularRotationYCur, imuAngularRotationZCur;
  float imuAngularRotationXLast, imuAngularRotationYLast, imuAngularRotationZLast;
  float imuAngularFromStartX, imuAngularFromStartY, imuAngularFromStartZ;

  double imuTime[imuQueLength];
  float imuRoll[imuQueLength];
  float imuPitch[imuQueLength];
  float imuYaw[imuQueLength];

  float imuAccX[imuQueLength];
  float imuAccY[imuQueLength];
  float imuAccZ[imuQueLength];

  float imuVeloX[imuQueLength];
  float imuVeloY[imuQueLength];
  float imuVeloZ[imuQueLength];

  float imuShiftX[imuQueLength];
  float imuShiftY[imuQueLength];
  float imuShiftZ[imuQueLength];

  float imuAngularVeloX[imuQueLength];
  float imuAngularVeloY[imuQueLength];
  float imuAngularVeloZ[imuQueLength];

  float imuAngularRotationX[imuQueLength];
  float imuAngularRotationY[imuQueLength];
  float imuAngularRotationZ[imuQueLength];

  float imuRollLast, imuPitchLast, imuYawLast;
  float imuShiftFromStartX, imuShiftFromStartY, imuShiftFromStartZ;
  float imuVeloFromStartX, imuVeloFromStartY, imuVeloFromStartZ;
};
}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
