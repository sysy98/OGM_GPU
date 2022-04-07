// Include guard
#ifndef LIDAR_OGM_ROS_H
#define LIDAR_OGM_ROS_H

// Includes
#include <vector>
#include <string>

#include <Eigen/Dense>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/PointStamped.h>
#include <message_filters/subscriber.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud2.h>

// headers in local files
#include "roscuda_lidar_ogm/common.h"
#include "roscuda_lidar_ogm/ogm_helper.h"
#include "roscuda_lidar_ogm/lidar_ogm_cuda.h"

class SensorProcessing {

public:
    // Default constructor
    SensorProcessing(ros::NodeHandle nh, const ros::NodeHandle &private_nh);

    // Destructor
    ~SensorProcessing();

    // Processes 3D Velodyne point cloud and publishes the output grid message
    void process(
            const sensor_msgs::PointCloud2::ConstPtr &cloud);

private:
    // Node handle
    ros::NodeHandle nh_, private_nh_;

    MapParams params_;
    GridInfo grid_info_;
    MapSize map_size_;

    // Class members
    std::string home_dir;
    std::string scenario_str;

    float lidar_height;
    float ransac_tolerance;
    int ransac_iterations;

    float width_gain_;  // 2
    float height_gain_; // 1.8
    int NUM_THREADS_;

    static int frame_count_;
    static ros::Time cloud_stamp_;
    // Transformation matrix of the vehicle position from the current frame to the first frame.
    static Eigen::Matrix4f transMat;

    float *tf_array;

    // pinned memory, will be copied asynchronously in the cuda stream
    float *h_global_cells_prob_array;
    int *h_ogm_data_array;

    VPointCloud::Ptr pcl_in_;
    VPointCloud::Ptr pcl_ground_plane_;
    VPointCloud::Ptr pcl_elevated_;

    tf2_ros::Buffer buffer_;
    tf2_ros::TransformListener tf_listener_;
    nav_msgs::OccupancyGrid::Ptr occ_grid_;
    geometry_msgs::PointStamped vehicle_pos_;
    std::shared_ptr<LidarOgmCuda> lidar_ogm_cuda_ptr_;

    // Publisher
    ros::Publisher grid_occupancy_pub_;
    ros::Publisher vehicle_pos_pub_;
    // ros::Publisher pcl_elevated_pub_;

    // Subscriber
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;

    // Class functions
    void processPointCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud);

    // tf transform
    void calculateTransMatrix();

    inline void from2dPolarIndexTo1d(int seg, int bin, int &cell_id);

    void fromVeloCoordsToPolarCell(float x, float y, int &cell_id);

    void fromPolarCellToVeloCoords(int seg, int bin,
                                   float &x, float &y);
};

#endif // LIDAR_OGM_ROS_H