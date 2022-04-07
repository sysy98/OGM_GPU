// Include guard
#ifndef LIDAR_OGM_CUDA_H
#define LIDAR_OGM_CUDA_H

#include <chrono>
#include <vector>
#include <pcl_ros/point_cloud.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "roscuda_lidar_ogm/ogm_helper.h"

// Types of point and cloud to work with
typedef pcl::PointXYZ VPoint;
typedef pcl::PointCloud<VPoint> VPointCloud;

__global__ void mark_elevated_cell_kernel(int points_num,
                                          const int max_num_points_per_seg,
                                          const float *dev_points,
                                          int *dev_points_in_seg_count,
                                          float *dev_point_distance_in_segs );

__global__ void update_cell_probs_kernel(int points_num,
                                         const int max_num_points_per_seg,   
                                         const int *dev_points_in_seg_count,
                                         float *dev_point_distance_in_segs,
                                         float *dev_bins_distance, 
                                         float *dev_p_logit);

__global__ void map_polar_to_cartesian_ogm_kernel(float *dev_tf_array,
                                                  float *dev_global_cells_prob_array,
                                                  int *dev_ogm_data_array,
                                                  const float *dev_p_logit);

class LidarOgmCuda
{
public:
    // Default constructor
    LidarOgmCuda(int NUM_THREADS,
                 MapParams &map_params);

    ~LidarOgmCuda();

    void processPointsCuda(VPointCloud::Ptr &pcl_in_,
                           float *tf_array,
                           float *h_global_cells_prob_array,
                           int *h_ogm_data_array);

private:
    const int NUM_THREADS_;
    const int grid_segments_;
    const int grid_bins_;
    const int polar_grid_num_;
    const int width_grid_;
    const int height_grid_;
    const int occ_grid_num_;
    const float grid_range_max_;
    const float grid_cell_size_;
    const float inv_radial_res_;
    const float inv_angular_res_;

    int points_num;
    int max_num_points_per_seg;

    thrust::device_vector<float> thrust_polar_cells_distance;

    void pclToHostVector(VPointCloud::Ptr &pcl_in_,
                         float *h_points_array) const;
};

#endif //LIDAR_OGM_CUDA_H