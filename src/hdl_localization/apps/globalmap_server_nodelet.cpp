#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>

#include <sensor_msgs/PointCloud2.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>

namespace hdl_localization {

class GlobalmapServerNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  GlobalmapServerNodelet() {
  }
  virtual ~GlobalmapServerNodelet() {
  }

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    // publish globalmap with "latched" publisher
    globalmap_pub = nh.advertise<sensor_msgs::PointCloud2>("/globalmap", 5, true);
    globalmap_pub_timer = nh.createWallTimer(ros::WallDuration(1.0), &GlobalmapServerNodelet::pub_once_cb, this, true, true);
  }

private:
  void initialize_params() {
    // read globalmap from a pcd file
    std::string globalmap_pcd = private_nh.param<std::string>("globalmap_pcd", "");
    globalmap.reset(new pcl::PointCloud<PointT>());
    while(  pcl::io::loadPCDFile(globalmap_pcd, *globalmap) != 0 ){
      ROS_INFO("waiting load the global map");
    }
    //pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
    globalmap->header.frame_id = "map";

    int mapCloudSize = globalmap->points.size();
    for (int i = 0; i < mapCloudSize; i++) {
      float temp = globalmap->points[i].x;
      globalmap->points[i].x = globalmap->points[i].z;
      globalmap->points[i].z = globalmap->points[i].y;
      globalmap->points[i].y = temp;
    }
    // downsample globalmap
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);

    globalmap = filtered;
     int mapCloudSize2 = filtered->points.size();
     ROS_WARN("map size is %d  filtered is %d",mapCloudSize,mapCloudSize2);
  }

  void pub_once_cb(const ros::WallTimerEvent& event) {
    globalmap_pub.publish(globalmap);
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  ros::Publisher globalmap_pub;

  ros::WallTimer globalmap_pub_timer;
  pcl::PointCloud<PointT>::Ptr globalmap;
};

}


PLUGINLIB_EXPORT_CLASS(hdl_localization::GlobalmapServerNodelet, nodelet::Nodelet)
