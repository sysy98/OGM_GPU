// headers in STL
#include <ros/ros.h>

// headers in local files
#include "roscuda_lidar_ogm/lidar_ogm_ros.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "sensor_setup_node");
    SensorProcessing sensor_setup(
            ros::NodeHandle(), ros::NodeHandle("~"));
    ros::spin();

    return 0;
}
