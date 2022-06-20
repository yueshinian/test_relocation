#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Point.h>
#include <atomic>
#include <vector>
#include <cmath>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include "Scancontext.h"

class samanger {
public:
  using PointT = pcl::PointCloud<pcl::PointXYZI>;
  samanger() : nh() {
    initParams();
    init();
    // ROS message filters
    boost::shared_ptr<Sync> sync_;
    subOdometry.subscribe(nh, "/integrated_to_init", 5);
    subLaserCloud.subscribe(nh, "/velodyne_points", 5);
    sync_.reset(new Sync(syncPolicy(10), subOdometry, subLaserCloud));
    sync_->registerCallback(boost::bind(&samanger::laserCloudAndOdometryHandler, this, _1, _2));
    subVelodyne = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points",10,&samanger::velodyneHandler,this);
    subSaveSingal = nh.subscribe<std_msgs::Bool>("/saveSingal", 10, &samanger::saveSingalHandler, this);
    subQueryLaser = nh.subscribe<sensor_msgs::PointCloud2>("/queryLaser", 10, &samanger::queryLaserHandler, this);
    subOdom = nh.subscribe<nav_msgs::Odometry>("/integrated_to_init",10,&samanger::odomHandler,this);

    pubOdom = nh.advertise<nav_msgs::Odometry>("/queryOdom", 1);
  }

  ~samanger() {}
  bool initParams() {
    voxelGrid.setLeafSize(0.5, 0.5, 0.5);
    return true;
  }
  bool init() {
    saveLaser = false;
    lastPoint.x = 999;
    lastPoint.y = 999;
    lastPoint.z = 999;
    curPoint.x = 999;
    curPoint.y = 999;
    curPoint.z = 999;
    return true;
  }
  void queryLaserHandler(const sensor_msgs::PointCloud2::ConstPtr& pcl_msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*pcl_msg, *laserCloudIn);
    voxelGrid.setInputCloud(laserCloudIn);
    voxelGrid.filter(*filtered);
    scManager.makeAndSaveScancontextAndKeys(*filtered);

    int SCclosestHistoryFrameID = -1;
    auto detectResult = scManager.detectLoopClosureID();  // first: nn index, second: yaw diff
    SCclosestHistoryFrameID = detectResult.first;
    double yawDiffRad = detectResult.second;  // not use for v1 (because pcl icp withi initial somthing wrong...)
    if (SCclosestHistoryFrameID == -1) {
      return;
    } else {
      pubOdom.publish(odomSaver[SCclosestHistoryFrameID]);
    }

    scManager.popFunc();
  }

  void saveSingalHandler(const std_msgs::Bool::ConstPtr& save_msg) { saveLaser = save_msg->data; }
  // void laserCloudHandle(const sensor_msgs::PointCloud2::ConstPtr &pcl_msg){}
  void laserCloudAndOdometryHandler(const nav_msgs::Odometry::ConstPtr& odometry, const sensor_msgs::PointCloud2ConstPtr& laserCloud) {
    // if (!saveLaser) {
    //   return;
    // }
    ROS_WARN("scan context");
    curPoint = odometry->pose.pose.position;
    if (
      std::sqrt(
        (lastPoint.x - curPoint.x) * (lastPoint.x - curPoint.x) + (lastPoint.y - curPoint.y) * (lastPoint.y - curPoint.y)+(lastPoint.z - curPoint.z) * (lastPoint.z - curPoint.z)) <
      0.3) {
      return;
    }
    lastPoint = curPoint;
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*laserCloud, *laserCloudIn);
    voxelGrid.setInputCloud(laserCloudIn);
    voxelGrid.filter(*filtered);
    scManager.makeAndSaveScancontextAndKeys(*filtered);
    nav_msgs::Odometry odom;
    odom.header = odometry->header;
    odom.pose = odometry->pose;
    odomSaver.emplace_back(odom);
    saveLaser = false;
    ROS_WARN("scan context size is: %d.", odomSaver.size());
  }

  void odomHandler(const nav_msgs::Odometry::ConstPtr &odometry)
  {
    ROS_WARN("odom received");
     curPoint = odometry->pose.pose.position;
    if (
      std::sqrt(
        (lastPoint.x - curPoint.x) * (lastPoint.x - curPoint.x) + (lastPoint.y - curPoint.y) * (lastPoint.y - curPoint.y)+(lastPoint.z - curPoint.z) * (lastPoint.z - curPoint.z)) <
      0.3) {
      return;
    }
    saveLaser = true;
    lastPoint = curPoint;
    nav_msgs::Odometry odom;
    odom.header = odometry->header;
    odom.pose = odometry->pose;
    odomSaver.emplace_back(odom);
    ROS_WARN("scan context size is: %d.", odomSaver.size());
  }

  void velodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &laserCloud)
  {
    if(!saveLaser){
      return;
    }else{
      saveLaser = false;
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*laserCloud, *laserCloudIn);
    voxelGrid.setInputCloud(laserCloudIn);
    voxelGrid.filter(*filtered);
    scManager.makeAndSaveScancontextAndKeys(*filtered);
    ROS_WARN("velodyne received");
  }
private:
  ros::NodeHandle nh;
  SCManager scManager;
  ros::Subscriber subVelodyne;
  ros::Subscriber subSaveSingal;
  ros::Subscriber subQueryLaser;
  ros::Subscriber subOdom;
  ros::Publisher pubOdom;
  std::atomic<bool> saveLaser;
  std::vector<nav_msgs::Odometry> odomSaver;
  pcl::VoxelGrid<pcl::PointXYZI> voxelGrid;
  message_filters::Subscriber<nav_msgs::Odometry> subOdometry;
  message_filters::Subscriber<sensor_msgs::PointCloud2> subLaserCloud;
  typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> syncPolicy;
  typedef message_filters::Synchronizer<syncPolicy> Sync;
  geometry_msgs::Point lastPoint;
  geometry_msgs::Point curPoint;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "scmanger");
  ROS_WARN("scmanger started!");
  samanger* scmanger = new samanger();
  ros::spin();
  return 0;
}