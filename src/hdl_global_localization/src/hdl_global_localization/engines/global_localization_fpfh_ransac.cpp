#include <hdl_global_localization/engines/global_localization_fpfh_ransac.hpp>

#include <ros/ros.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/impl/kdtree.hpp>

#include <hdl_global_localization/ransac/ransac_pose_estimation.hpp>

namespace hdl_global_localization {

GlobalLocalizationEngineFPFH_RANSAC::GlobalLocalizationEngineFPFH_RANSAC(ros::NodeHandle& private_nh) : private_nh(private_nh) {}

GlobalLocalizationEngineFPFH_RANSAC::~GlobalLocalizationEngineFPFH_RANSAC() {}

pcl::PointCloud<pcl::FPFHSignature33>::ConstPtr GlobalLocalizationEngineFPFH_RANSAC::extract_fpfh(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
  double normal_estimation_radius = private_nh.param<double>("fpfh/normal_estimation_radius", 2.0);//
  double search_radius = private_nh.param<double>("fpfh/search_radius", 8.0);
  //求解点云法线
  ROS_INFO_STREAM("Normal Estimation: Radius(" << normal_estimation_radius << ")");
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> nest;
  nest.setRadiusSearch(normal_estimation_radius);
  nest.setInputCloud(cloud);
  nest.compute(*normals);
  //计算fpfh特征
  ROS_INFO_STREAM("FPFH Extraction: Search Radius(" << search_radius << ")");
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>);
  pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fest;
  fest.setRadiusSearch(search_radius);//搜索半径
  fest.setInputCloud(cloud);//输入
  fest.setInputNormals(normals);//法线
  fest.compute(*features);//计算特征

  return features;
}

void GlobalLocalizationEngineFPFH_RANSAC::set_global_map(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
  global_map = cloud;
  global_map_features = extract_fpfh(cloud);

  ransac.reset(new RansacPoseEstimation<pcl::FPFHSignature33>(private_nh));
  ransac->set_target(global_map, global_map_features);
}

GlobalLocalizationResults GlobalLocalizationEngineFPFH_RANSAC::query(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, int max_num_candidates) {
  pcl::PointCloud<pcl::FPFHSignature33>::ConstPtr cloud_features = extract_fpfh(cloud);

  ransac->set_source(cloud, cloud_features);
  auto results = ransac->estimate();//lym根据map和velodyne的fpfh特征匹配

  return results.sort(max_num_candidates);
}

}  // namespace hdl_global_localization