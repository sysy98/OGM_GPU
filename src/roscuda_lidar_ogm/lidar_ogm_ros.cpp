/******************************************************************************
 *
 * Author: Shuai Yuan
 * Date: 01/07/2021
 *
 */
// headers in STL
#include <cmath>
#include <memory>
#include <chrono>

#include <geometry_msgs/TransformStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

// headers in local files
#include "roscuda_lidar_ogm/common.h"
#include "roscuda_lidar_ogm/lidar_ogm_ros.h"

using namespace Eigen;
using namespace geometry_msgs;
using namespace nav_msgs;

ros::Time SensorProcessing::cloud_stamp_{};                 // cloud stamp
int SensorProcessing::frame_count_{};                       // init counter for publishing
Matrix4f SensorProcessing::transMat = Matrix4f::Identity(); // init transformation matrix

// Initialize process duration
std::chrono::microseconds total_dur = std::chrono::microseconds(0);
std::chrono::microseconds cuda_part_dur = std::chrono::microseconds(0);

SensorProcessing::SensorProcessing(ros::NodeHandle nh,
                                   const ros::NodeHandle &private_nh) : nh_(nh),
                                                                        private_nh_(private_nh),
                                                                        pcl_in_(new VPointCloud),
                                                                        pcl_ground_plane_(new VPointCloud),
                                                                        pcl_elevated_(new VPointCloud),
                                                                        cloud_sub_(nh, "/kitti/velo/pointcloud", 10),
                                                                        tf_listener_(buffer_),
                                                                         /*Scenario 0*/
                                                                        // width_gain_(2),
                                                                        // height_gain_(1.8),
                                                                        /*Scenario 1 and 2*/
                                                                        width_gain_(2.8),
                                                                        height_gain_(3.8),
                                                                        NUM_THREADS_(256)
{
    // Get scenario parameter
    int scenario = 0;
    std::ostringstream scenario_stream;
    scenario_stream << std::setfill('0') << std::setw(4) << scenario;
    scenario_str = scenario_stream.str();
 
    // Define lidar parameters
    private_nh_.param<float>("lidar/height", lidar_height, -1.73);
    private_nh_.param<float>("lidar/z_min", params_.lidar_z_min, -2.4);

    // Define local grid map parametersI
    private_nh_.param<float>("grid/range/min", grid_info_.grid_range_min, 2.0);
    private_nh_.param<float>("grid/range/max", grid_info_.grid_range_max, 80.0);
    private_nh_.param<float>("grid/cell/size", grid_info_.grid_cell_size, 0.25);
    private_nh_.param<float>("grid/cell/height", params_.grid_cell_height, 0.25);
    private_nh_.param<int>("grid/segments", map_size_.grid_segments, 1040);

    // Define ransac ground plane parameters
    private_nh_.param<float>("ransac/tolerance", ransac_tolerance, 0.2);
    private_nh_.param<int>("ransac/iterations", ransac_iterations, 50);

    map_size_.grid_bins = (grid_info_.grid_range_max * std::sqrt(2)) / grid_info_.grid_cell_size + 1;
    map_size_.polar_grid_num = map_size_.grid_segments * map_size_.grid_bins;
    map_size_.height_grid = grid_info_.grid_range_max / grid_info_.grid_cell_size * 2;
    map_size_.width_grid = map_size_.height_grid;

    // Define global grid map parameters
    map_size_.occ_width_grid = width_gain_ * map_size_.width_grid;
    map_size_.occ_height_grid = height_gain_ * map_size_.height_grid;
    map_size_.occ_grid_num = map_size_.occ_width_grid * map_size_.occ_height_grid;

    // Define static conversion values
    grid_info_.inv_angular_res = map_size_.grid_segments / (2 * M_PI);
    grid_info_.inv_radial_res = 1.0f / grid_info_.grid_cell_size;

    params_.grid_info_ = grid_info_;
    params_.map_size_ = map_size_;

    // Print parameters
    ROS_INFO_STREAM("scenario " << scenario_str);
    ROS_INFO_STREAM("lidar_height " << lidar_height);
    ROS_INFO_STREAM("lidar_z_min " << params_.lidar_z_min);
    ROS_INFO_STREAM("grid_info_.grid_range_min " << grid_info_.grid_range_min);
    ROS_INFO_STREAM("grid_info_.grid_range_max " << grid_info_.grid_range_max);
    ROS_INFO_STREAM("height_grid " << map_size_.height_grid);
    ROS_INFO_STREAM("width_grid " << map_size_.width_grid);
    ROS_INFO_STREAM("grid_cell_size " << grid_info_.grid_cell_size);
    ROS_INFO_STREAM("grid_cell_height " << params_.grid_cell_height);
    ROS_INFO_STREAM("grid_bins " << map_size_.grid_bins);
    ROS_INFO_STREAM("grid_segments " << map_size_.grid_segments);
    ROS_INFO_STREAM("ransac_tolerance " << ransac_tolerance);
    ROS_INFO_STREAM("ransac_iterations " << ransac_iterations);
    ROS_INFO_STREAM("inv_angular_res " << grid_info_.inv_angular_res);
    ROS_INFO_STREAM("inv_radial_res " << grid_info_.inv_radial_res);

    occ_grid_ = boost::make_shared<OccupancyGrid>();
    occ_grid_->data.resize(map_size_.occ_width_grid * map_size_.occ_height_grid);
    occ_grid_->info.width = uint32_t(map_size_.occ_height_grid);
    occ_grid_->info.height = uint32_t(map_size_.occ_width_grid);
    occ_grid_->info.resolution = float(grid_info_.grid_cell_size);
    occ_grid_->info.origin.position.x = -map_size_.occ_height_grid / 2 * grid_info_.grid_cell_size;
    occ_grid_->info.origin.position.y = -map_size_.occ_width_grid / 2 * grid_info_.grid_cell_size;
    occ_grid_->info.origin.position.z = lidar_height;
    occ_grid_->info.origin.orientation.w = 1;
    occ_grid_->info.origin.orientation.x = 0;
    occ_grid_->info.origin.orientation.y = 0;
    occ_grid_->info.origin.orientation.z = 0;

    GPU_CHECK(cudaHostAlloc((void **)&h_global_cells_prob_array,
                            map_size_.occ_grid_num * sizeof(float),
                            cudaHostAllocMapped)); // 1

    GPU_CHECK(cudaHostAlloc((void **)&h_ogm_data_array,
                            map_size_.occ_grid_num * sizeof(int),
                            cudaHostAllocMapped)); // -1

    // Init cell probability in the global cartesian map
    for (int i = 0; i < map_size_.occ_grid_num; i++)
    {
        h_global_cells_prob_array[i] = 0;
        h_ogm_data_array[i] = -1;
    }

    lidar_ogm_cuda_ptr_ = std::make_shared<LidarOgmCuda>(NUM_THREADS_, params_);

    // Define Publisher
    grid_occupancy_pub_ = nh_.advertise<OccupancyGrid>(
        "/sensor/grid/occupancy", 2);
    vehicle_pos_pub_ = nh_.advertise<PointStamped>(
        "/vehicle_pose", 2);
    // pcl_elevated_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
	// 	"/pointcloud/elevated", 2);

    // Define Subscriber
    cloud_sub_.registerCallback(boost::bind(&SensorProcessing::process, this, _1));
}

SensorProcessing::~SensorProcessing()
{
    GPU_CHECK(cudaFreeHost(h_global_cells_prob_array));
    GPU_CHECK(cudaFreeHost(h_ogm_data_array));
}

void SensorProcessing::process(
    const sensor_msgs::PointCloud2::ConstPtr &cloud)
{   
    cloud_stamp_ = cloud->header.stamp;
    
    auto startPost = std::chrono::high_resolution_clock::now();

    calculateTransMatrix();

    // Preprocess point cloud
    processPointCloud(cloud);

    auto endPost = std::chrono::high_resolution_clock::now();

    // sum the total time
    total_dur += std::chrono::duration_cast<std::chrono::microseconds>(endPost - startPost);

    std::cout << "Average time (whole process): " << std::fixed << std::setprecision(2) << double(total_dur.count()) / (frame_count_ + 1) / 1000 << "ms"
              << "(cuda part): " << std::fixed << std::setprecision(2) << double(cuda_part_dur.count()) / (frame_count_ + 1) / 1000 << "ms\n";

    // Increment time frame
    frame_count_++;
}

void SensorProcessing::processPointCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud)
{
    /******************************************************************************
 * 1. Filter point cloud to only consider points in the front that can also be
 * found in image space.
 */
    // Convert input cloud
    pcl::fromROSMsg(*cloud, *pcl_in_);

    // Define point_cloud_inliers and indices
    pcl::PointIndices::Ptr pcl_inliers(new pcl::PointIndices());
    pcl::ExtractIndices<VPoint> pcl_extractor;

    // Define polar grid
    std::vector<PolarCell> polar_grid_(map_size_.polar_grid_num, PolarCell());

    // Loop through input point cloud
    for (int i = 0; i < pcl_in_->size(); ++i)
    {

        // Read current point
        VPoint &point = pcl_in_->at(i);

        // Determine range of lidar point and check
        float range = std::sqrt(point.x * point.x + point.y * point.y);
        
        if (range > grid_info_.grid_range_min &&
            range < grid_info_.grid_range_max * 1.50)
        {

            // Check height of lidar point
            if (point.z > params_.lidar_z_min)
            {

                // Add index for filtered point cloud
                pcl_inliers->indices.push_back(i);

                // Buffer variables
                int polar_id;

                // Get polar grid cell indices
                fromVeloCoordsToPolarCell(point.x, point.y, polar_id);

                // Grab cell
                PolarCell &cell = polar_grid_[polar_id];

                // Increase count
                cell.count++;

                // Update min max
                if (cell.count == 1)
                {
                    cell.x_min = point.x;
                    cell.y_min = point.y;
                    cell.z_min = point.z;
                    cell.z_max = point.z;
                }
                else
                {
                    if (point.z < cell.z_min)
                    {
                        cell.x_min = point.x;
                        cell.y_min = point.y;
                        cell.z_min = point.z;
                    }
                    if (point.z > cell.z_max)
                    {
                        cell.z_max = point.z;
                    }
                }
            }
        }
    }

    // Extract points from original point cloud
    pcl_extractor.setInputCloud(pcl_in_);
    pcl_extractor.setIndices(pcl_inliers);
    pcl_extractor.setNegative(false);
    pcl_extractor.filter(*pcl_in_);

    // Publish filtered cloud
    pcl_in_->header.frame_id = cloud->header.frame_id;
    pcl_in_->header.stamp = pcl_conversions::toPCL(cloud->header.stamp);

    /******************************************************************************
 * 2. Ground plane estimation and dividing point cloud in elevated and ground
 */
    // Clear ground plane points
    pcl_ground_plane_->points.clear();

    // Loop over cartesian grid map
    for (int i = 0; i < map_size_.grid_segments; ++i)
    {
        for (int j = 0; j < map_size_.grid_bins; ++j)
        {

            // Grab cell
            int polar_id;
            from2dPolarIndexTo1d(i, j, polar_id);
            PolarCell &cell = polar_grid_[polar_id];

            // Check if cell can be ground cell
            if (cell.count > 0 &&
                (cell.z_max - cell.z_min < params_.grid_cell_height))
            {

                // Push back cell attributes to ground plane cloud
                pcl_ground_plane_->points.emplace_back(
                    VPoint(cell.x_min, cell.y_min, cell.z_min));
            }
        }
    }

    // Estimate the ground plane using PCL and RANSAC
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Create the segment object
    pcl::SACSegmentation<VPoint> seg;
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(ransac_tolerance);
    seg.setMaxIterations(ransac_iterations);
    seg.setInputCloud(pcl_ground_plane_->makeShared());
    seg.segment(*inliers, *coefficients);

    // Sanity check
    if (inliers->indices.empty() || coefficients->values[3] > 2 ||
        coefficients->values[3] < 1.5)
    {
        ROS_WARN("Bad ground plane estimation! # Ransac Inliers [%d] # Lidar "
                 "height [%f]",
                 int(inliers->indices.size()),
                 coefficients->values[3]);
    }

    /******************************************************************************
 * 3. Divide filtered point cloud in elevated and ground
 */
    for (int s = 0; s < map_size_.grid_segments; ++s)
    {

        for (int b = 0; b < map_size_.grid_bins; ++b)
        {

            // Grab cell
            int polar_id;
            from2dPolarIndexTo1d(s, b, polar_id);
            PolarCell &cell = polar_grid_[polar_id];

            float x, y;
            fromPolarCellToVeloCoords(s, b, x, y);

            //Get ground height
            cell.ground = (-coefficients->values[0] * x -
                           coefficients->values[1] * y - coefficients->values[3]) /
                          coefficients->values[2];

            //if cell is not filled
            if (cell.count == 0)
                continue;

            //Calculate cell height
            else
                cell.height = cell.z_max - cell.ground;
        }
    }

    pcl_elevated_->points.clear();

    for (int i = 0; i < pcl_in_->size(); ++i)
    {

        // Read current point
        VPoint point = pcl_in_->at(i);

        //Buffer variables
        int polar_id;

        // Get polar grid cell indices
        fromVeloCoordsToPolarCell(point.x, point.y, polar_id);

        // Grab cell
        PolarCell &cell = polar_grid_[polar_id];

        if (point.z > cell.ground && cell.height > params_.grid_cell_height)
        {
            pcl_elevated_->points.push_back(point);
        }
    }

    // pcl_elevated_->header.frame_id = cloud->header.frame_id;
    // pcl_elevated_->header.stamp = pcl_conversions::toPCL(cloud->header.stamp);
    // pcl_elevated_pub_.publish(pcl_elevated_);

    // Print point cloud information
    ROS_INFO("Point Cloud [%d] # Total points [%d] # Elevated points [%d] ",
             frame_count_, int(pcl_in_->size()), int(pcl_elevated_->size()));

    /******************************************************************************
 * 4. Use elevated point cloud to calculate occupied probability of cells in polar grid
 */
    auto cudaStart = std::chrono::high_resolution_clock::now();

    lidar_ogm_cuda_ptr_->processPointsCuda(pcl_elevated_,
                                           transMat.data(),
                                           h_global_cells_prob_array,
                                           h_ogm_data_array);

    for (int i = 0; i < map_size_.occ_grid_num; i++)
    {
        occ_grid_->data[i] = h_ogm_data_array[i];
    }

    auto cudaEnd = std::chrono::high_resolution_clock::now();

    // Sum the processing time for cuda part
    cuda_part_dur += std::chrono::duration_cast<std::chrono::microseconds>(cudaEnd - cudaStart);

    // Publish occupancy grid
    occ_grid_->header.stamp = cloud->header.stamp;
    occ_grid_->header.frame_id = "world";
    occ_grid_->info.map_load_time = cloud->header.stamp;
    grid_occupancy_pub_.publish(occ_grid_);

    // Publish vehicle pose
    Vector4f vehicle_vec = transMat * Vector4f(0, 0, 0, 1);
    vehicle_pos_.header.frame_id = "world";
    vehicle_pos_.header.stamp = cloud_stamp_;
    vehicle_pos_.point.x = vehicle_vec[0];
    vehicle_pos_.point.y = vehicle_vec[1];
    vehicle_pos_.point.z = 0;
    vehicle_pos_pub_.publish(vehicle_pos_);
}

void SensorProcessing::calculateTransMatrix()
{

    geometry_msgs::TransformStamped tfStamped;
    try
    {
        tfStamped = buffer_.lookupTransform("world", "velo_link", cloud_stamp_, ros::Duration(1.0));
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("%s", ex.what());
        ros::Duration(1.0).sleep();
    }
    //translation vector T
    auto trans = tfStamped.transform.translation;
    transMat.block<3, 1>(0, 3) = Vector3f(trans.x, trans.y, trans.z);

    //rotation matrix R
    auto rot = tfStamped.transform.rotation;
    transMat.block<3, 3>(0, 0) = Quaternionf(rot.w, rot.x, rot.y, rot.z)
                                     .normalized()
                                     .toRotationMatrix();
}

inline void SensorProcessing::from2dPolarIndexTo1d(const int seg, const int bin,
                                                   int &polar_id)
{
    polar_id = seg * map_size_.grid_bins + bin;
}

void SensorProcessing::fromVeloCoordsToPolarCell(const float x, const float y,
                                                 int &polar_id)
{
    float mag = std::sqrt(x * x + y * y);
    float ang = std::atan2(x, y);
    int seg = int((ang + M_PI) * grid_info_.inv_angular_res);
    int bin = int(mag * grid_info_.inv_radial_res);
    from2dPolarIndexTo1d(seg, bin, polar_id);
}

void SensorProcessing::fromPolarCellToVeloCoords(int seg, int bin,
                                                 float &x, float &y)
{
    float mag = bin / grid_info_.inv_radial_res + grid_info_.grid_cell_size / 2;
    float ang = seg / grid_info_.inv_angular_res - M_PI;
    x = std::sin(ang) * mag;
    y = std::cos(ang) * mag;
}