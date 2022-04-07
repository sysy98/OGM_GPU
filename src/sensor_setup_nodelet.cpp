/******************************************************************************
 *
 * Author: Shuai Yuan
 * Date: 01/07/2021
 *
 */

// headers in STL
#include <ros/ros.h>
#include <nodelet/nodelet.h>

// headers in local files
#include "roscuda_lidar_ogm/lidar_ogm_ros.h"

class SensorSetupNodelet : public nodelet::Nodelet {
public:
    SensorSetupNodelet() {}

    ~SensorSetupNodelet() {}

private:
    virtual void onInit() {
        sensor_fusion_.reset(
                new SensorProcessing(getNodeHandle(), getPrivateNodeHandle()));
    }

    std::shared_ptr<SensorProcessing> sensor_fusion_;
};
