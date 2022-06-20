//c++
#include <vector>
#include <algorithm>
#include <map>
//eigen
#include <Eigen/Core>
#include <Eigen/Dense>
//ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/flann_search.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/sample_consensus_prerejective.h>
//tf
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
//pcl
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

namespace GLOBAL_LOCALION{
class globalLocation{
using PointT = pcl::PointXYZ;
public:
    globalLocation()
    {
        sourceCloud.reset(new pcl::PointCloud<PointT>());
        targetCloud.reset(new pcl::PointCloud<PointT>());
        init();
    }

    void init()
    {
        sor.setMeanK(50);
        sor.setStddevMulThresh(2.0);
    }

    void setFilter(const double &resolution)
    {
        voxelgrid.setLeafSize(resolution, resolution, resolution);
    }

    void downsample(pcl::PointCloud<PointT>::Ptr cloudIn)
    {
        pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
        voxelgrid.setInputCloud(cloudIn);
        voxelgrid.filter(*filteredCloud);
        cloudIn = filtered;
    }

    void downsample(pcl::PointCloud<PointT>::Ptr cloudIn, pcl::PointCloud<PointT>::Ptr filteredCloud)
    {
        voxelgrid.setInputCloud(cloudIn);
        voxelgrid.filter(*filteredCloud);
    }


    pcl::PointCloud<pcl::FPFHSignature33>::Ptr extractFPFH(pcl::PointCloud<PointT>::Ptr cloudIn, const double &normalRadius, const double &fpfhRadius)
    {
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normalCompute;
        //pcl::search::FlannSearch<PointT>::Ptr searchTree(new pcl::search::FlannSearch<PointT>());
        pcl::PointCloud<pcl::Normal>::Ptr normalSet(new pcl::PointCloud<pcl::Normal>());
        normalCompute.setRadiusSearch(normalRadius);
        normalCompute.setInputCloud(cloudIn);
        //normalCompute.setSearchMethod(searchTree);
        normalCompute.compute(*normalSet);
        
        pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> fpfhCompute;
        //pcl::search::FlannSearch<PointT>::Ptr searchTree2(new pcl::search::FlannSearch<PointT>());
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhSet(new pcl::PointCloud<pcl::FPFHSignature33>());
        fpfhCompute.setRadiusSearch(fpfhRadius);
        fpfhCompute.setInputCloud(cloudIn);
        fpfhCompute.setInputNormals(normalSet);
        //fpfhCompute.setSearchMethod(searchTree2);
        fpfhCompute.compute(*fpfhSet);

        return fpfhSet;
    }
    //normal点云分辨率5倍以上，fpfh点云分辨率10倍以上
    Eigen::Matrix4f aligned(const double &normalRadius, const double &fpfhRadius)
    {
      double fitScore = 10;
      int fitCount = 0;
      std::map<double, Eigen::Matrix4f> fitMap;
      pcl::PointCloud<pcl::FPFHSignature33>::Ptr sourceFeature = extractFPFH(sourceCloud, normalRadius, fpfhRadius);
      pcl::PointCloud<pcl::FPFHSignature33>::Ptr targetFeature = extractFPFH(targetCloud, normalRadius, fpfhRadius);
      Eigen::Matrix4f alignedTrans = Eigen::Matrix4f::Zero();
      pcl::PointCloud<PointT>::Ptr alignedCloud(new pcl::PointCloud<PointT>());

    //   pcl::SampleConsensusPrerejective<PointT, PointT, pcl::FPFHSignature33> scpAlign;
    //   scpAlign.setInputSource(sourceCloud);
    //   scpAlign.setSourceFeatures(sourceFeature);
    //   scpAlign.setInputTarget(targetCloud);
    //   scpAlign.setTargetFeatures(targetFeature);
    //   scpAlign.setMaxCorrespondenceDistance(50000);
    //   scpAlign.setNumberOfSamples(3);
    //   scpAlign.setCorrespondenceRandomness(5);
    //   scpAlign.setSimilarityThreshold(0.9f);
    //   scpAlign.setMaxCorrespondenceDistance(1.0f);
    //   scpAlign.setInlierFraction(0.25f);
    //   scpAlign.align(*alignedCloud);
    //   if(scpAlign.hasConverged()){
    //       if((fitScore = scpAlign.getFitnessScore()) < 0.1){
    //           std::cout<<"scp fit score is: "<<fitScore<<std::endl;
    //           return alignedTrans = scpAlign.getFinalTransformation();
    //       }
    //   }

      pcl::SampleConsensusInitialAlignment<PointT, PointT, pcl::FPFHSignature33> alignedCompute;
      alignedCompute.setInputSource(sourceCloud);
      alignedCompute.setInputTarget(targetCloud);
      alignedCompute.setSourceFeatures(sourceFeature);
      alignedCompute.setTargetFeatures(targetFeature);

      pcl::IterativeClosestPoint<PointT, PointT> icpAlign;
      icpAlign.setInputSource(sourceCloud);
      icpAlign.setInputTarget(targetCloud);
      icpAlign.setMaxCorrespondenceDistance(100);
      icpAlign.setMaximumIterations(100);
      icpAlign.setTransformationEpsilon(1e-6);
      icpAlign.setEuclideanFitnessEpsilon(1e-6);
      icpAlign.setRANSACIterations(0);

      while (fitScore > 0.1 && fitCount < 10) {
        // alignedCompute.align(*alignedCloud);
        // alignedTrans = alignedCompute.getFinalTransformation();
        // if (!alignedCompute.hasConverged()) {
        //   continue;
        // }

        //fitScore = alignedCompute.getFitnessScore();
        //if (fitScore > 0.1) {
          icpAlign.align(*alignedCloud);
          if (!icpAlign.hasConverged()) {
            continue;
          }
          alignedTrans = icpAlign.getFinalTransformation();
          fitScore = icpAlign.getFitnessScore();
        //}

        fitMap[fitScore] = alignedTrans;
        std::cout << "[RS] ICP fit score: " << fitScore << std::endl;
      }

      return fitMap.begin()->second;
    } 

    void downSampleCompute(const double &resolution)
    {
        setFilter(resolution);
        downsample(sourceCloud);
        downsample(targetCloud);
    }

    void setSourceCloud(pcl::PointCloud<PointT>::Ptr cloudIn)
    {
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloudIn,*cloudIn,indices);
        sor.setInputCloud(cloudIn);
        sor.filter(*sourceCloud);
    }

    void setTargetCloud(pcl::PointCloud<PointT>::Ptr cloudIn)
    {
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloudIn, *cloudIn, indices);
        sor.setInputCloud(cloudIn);
        sor.filter(*targetCloud);
    }

    void setCropParam(double minx, double miny, double miny, double maxx, double maxy, double maxz, bool saveOut = false)
    {
      cropFilter.setMin(Eigen::Vector4f(minx, miny, minz, 1.0));
      cropFilter.setMax(Eigen::Vector4f(maxx, maxy, maxz, 1.0));
      cropFilter.setNegative(saveOut);  // false，保留里面
    }

    void cropPoint(pcl::PointCloud<PointT>::Ptr sourceCloud, pcl::PointCloud<PointT>::Ptr filteredCloud)
    {
      cropFilter.setInputCloud(sourceCloud);
      cropFilter.filter(*filteredCloud);
    }

    void pose2eigen(geometry_msgs::Pose pose)
    {

    }

    geometry_msgs::Pose eigen2pose(Eigen::Matrix4f matrix)
    {
      geometry_msgs::Pose pose;
      tf::poseEigenToMsg(Eigen::Isometry3d(matrix.cast<double>()), pose);
      return pose;
    }
    //void disFilter(pcl::PointCloud<PointT>::Pt)
    ~globalLocation(){}
private:
    pcl::PointCloud<PointT>::Ptr sourceCloud;
    pcl::PointCloud<PointT>::Ptr targetCloud;
    pcl::VoxelGrid<PointT> voxelgrid;
    pcl::StatisticalOutlierRemoval<PointT> sor;
    pcl::CropBox<PointT> cropFilter;
};
}