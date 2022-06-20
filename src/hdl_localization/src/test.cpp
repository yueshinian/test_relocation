#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_datatypes.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <vector>
#include "eskf.hpp"
#include <memory>
#include "global_location.hpp"

using std::vector;

std::shared_ptr<ESKF::eskf> eskfPtr;

sensor_msgs::Imu imuConverter(const sensor_msgs::Imu &imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;

        vector<double> extRotV = {1,0,0,
                                                                0,-1,0,
                                                                0,0,-1};
        vector<double> extRPYV = {1,0,0,
                                                                0,-1,0,
                                                                0,0,-1};
        Eigen::Matrix3d extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        Eigen::Matrix3d extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        Eigen::Quaterniond extQRPY = Eigen::Quaterniond(extRPY);
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = q_from * extQRPY;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        if (sqrt(q_final.x() * q_final.x() + q_final.y() * q_final.y() + q_final.z() * q_final.z() + q_final.w() * q_final.w()) < 0.1)
        {
            ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
            ros::shutdown();
        }
        
        double roll, pitch, yaw;
       geometry_msgs::Quaternion geoQuat = imu_out.orientation;
       tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);
       //std::cout<<roll<<' '<<pitch<<' '<<yaw<<' '<<std::endl;

        //pubConvertImu.publish(imu_out);

        return imu_out;
    }

int count=0;

void imuHandler(const sensor_msgs::Imu::ConstPtr &imu_msg)
{
    sensor_msgs::Imu imuIn = imuConverter(*imu_msg);
    double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        /*
        std::cout<<"***************"<<std::endl
                            <<roll<<std::endl
                            <<pitch<<std::endl
                            <<yaw<<std::endl
                            <<imuIn.linear_acceleration.x<<std::endl
                            <<imuIn.linear_acceleration.y<<std::endl
                            <<imuIn.linear_acceleration.z<<std::endl
                            <<imuIn.angular_velocity.x<<std::endl
                            <<imuIn.angular_velocity.y<<std::endl
                            <<imuIn.angular_velocity.z<<std::endl;
        */
       ++count;
       eskfPtr->predict(imuIn);
       if(count%20 == 0){
           std::cout<<"*********"<<std::endl;
            Eigen::Vector3d position = eskfPtr->getPosition();
            std::cout<<position<<std::endl;
       }
        
        // Eigen::Vector3d linear(imuIn.linear_acceleration.x,imuIn.linear_acceleration.y,
        //     imuIn.linear_acceleration.z);
        // Eigen::Quaterniond q = imuIn.orientation;
        // Eigen::Matrix3d rotationMatrix = 
}

int main(int argc, char**argv)
{
    ros::init(argc,argv,"test");
    ros::NodeHandle nh;
    eskfPtr.reset(new ESKF::eskf());
    /*
    Eigen::Vector3d v3;
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
     std::cout<<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w()<<std::endl;
    std::cout<<v3<<std::endl;
    std::cout<<rotation_matrix<<std::endl;
    rotation_matrix = q.toRotationMatrix();
    std::cout<<rotation_matrix<<std::endl;
    */
    ros::Subscriber subImu = nh.subscribe("/imu/data",200,imuHandler);
    ROS_WARN("test start");
    ros::spin();
}