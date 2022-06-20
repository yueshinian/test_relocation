#include <ros/ros.h>
#include "eskf.h"
#include "eskf.hpp"

int main(int argc, char** argv)
{
    ros::init(argc,argv,"fusion");
    ros::NodeHandle nh;
    ros::spin();
    return 0;
}