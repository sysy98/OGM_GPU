#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// headers in local files
#include "roscuda_lidar_ogm/common.h"
#include "roscuda_lidar_ogm/lidar_ogm_cuda.h"

__constant__ GridInfo dev_grid_info[1];
__constant__ MapSize dev_map_size[1];

float *h_bins_distance;
float *d_bins_distance;

template<typename value_type, typename matrix_type = value_type *, typename vector_type = value_type *>
struct MatVecMul {
    const matrix_type matrix;
    const size_t width;

    __device__
    MatVecMul(const matrix_type matrix, size_t width)
            : matrix(matrix), width(width) {}

    __device__
    void operator()(const vector_type vec, value_type &vec_out_x, value_type &vec_out_y) {
        vec_out_x = 0;
        vec_out_y = 0;

        for (size_t i = 0; i < width; i++) {
            // 'Eigen' matrix elements are stored in column-major order.
            vec_out_x += matrix[i * width + 0] * vec[i];
            vec_out_y += matrix[i * width + 1] * vec[i];
        }
    }
};


__device__ __forceinline__ int from2dPolarCellIndexTo1d(int seg, int bin) {

    return seg * dev_map_size[0].grid_bins + bin;
}

__device__ void fromVeloCoordsToPolarCell(const float x, const float y,
                                          const float point_dist,
                                          int &seg,
                                          int &bin,
                                          int &polar_id) {
    float ang = std::atan2(x, y);
    seg = int((ang + M_PI) * dev_grid_info[0].inv_angular_res);
    bin = int(point_dist * dev_grid_info[0].inv_radial_res);
    polar_id = from2dPolarCellIndexTo1d(seg, bin);
}

__device__ void fromLocalGridToFinalCartesian(const int grid_x,
                                              const int grid_y,
                                              float &x,
                                              float &y) {

    x = float(grid_x + 0.5 - dev_map_size[0].height_grid * 0.5) * dev_grid_info[0].grid_cell_size;
    y = float(grid_y + 0.5 - dev_map_size[0].width_grid * 0.5) * dev_grid_info[0].grid_cell_size;
}

__device__ int fromFinalCartesianToGridIndex(const float x, const float y) {

    int grid_x = x / dev_grid_info[0].grid_cell_size + dev_map_size[0].occ_height_grid / 2;
    int grid_y = y / dev_grid_info[0].grid_cell_size + dev_map_size[0].occ_width_grid / 2;

    if (grid_x >= 0 && grid_x < dev_map_size[0].occ_height_grid
        && grid_y >= 0 && grid_y < dev_map_size[0].occ_width_grid) {

        return grid_y * dev_map_size[0].occ_height_grid + grid_x;
    }

    printf("The measurement distance of the lidar has exceeded the range of the map. \
  Please enlarge the map size.");

    return -1;
}

__device__ int fromLocalOgmToFinalOgm(const int local_grid_x,
                                      const int local_grid_y,
                                      float *dev_tf_array) {
    
    float final_x, final_y;
    fromLocalGridToFinalCartesian(local_grid_x, local_grid_y, final_x, final_y);

    float vec_in[4]{final_x, final_y, 0, 1};

    MatVecMul<float> mat_mul(dev_tf_array, 4);
    mat_mul(vec_in, final_x, final_y);

    int final_grid_index = fromFinalCartesianToGridIndex(final_x, final_y);
    
    return final_grid_index;
}

LidarOgmCuda::LidarOgmCuda(int NUM_THREADS,
                           MapParams &map_params)

        : NUM_THREADS_(NUM_THREADS),
          grid_segments_(map_params.map_size_.grid_segments),
          grid_bins_(map_params.map_size_.grid_bins),
          polar_grid_num_(map_params.map_size_.polar_grid_num),
          width_grid_(map_params.map_size_.width_grid),
          height_grid_(map_params.map_size_.height_grid),
          occ_grid_num_(map_params.map_size_.occ_grid_num),
          grid_range_max_(map_params.grid_info_.grid_range_max),
          grid_cell_size_(map_params.grid_info_.grid_cell_size),
          inv_radial_res_(map_params.grid_info_.inv_radial_res),
          inv_angular_res_(map_params.grid_info_.inv_angular_res) {

    int max_num_points_per_seg = ceil(360.0 / grid_segments_ / 0.174) * 64;
    std::cout << "Max num points per seg: " << max_num_points_per_seg << "\n";

    cudaMemcpyToSymbol(dev_grid_info, &(map_params.grid_info_), sizeof(GridInfo));
    cudaMemcpyToSymbol(dev_map_size, &(map_params.map_size_), sizeof(MapSize));

    GPU_CHECK(cudaHostAlloc((void **) &h_bins_distance,
                            grid_bins_ * sizeof(float),
                            cudaHostAllocMapped));
    
    for(int b = 0; b < grid_bins_; b++){    
        // Calculate distance between each cell and the origin of lidar.
        h_bins_distance[b] = float(b) / inv_radial_res_ + grid_cell_size_ / 2;
    }
    
    GPU_CHECK(cudaHostGetDevicePointer((void**)&d_bins_distance, (void*)h_bins_distance, 0));
}

LidarOgmCuda::~LidarOgmCuda(){
    GPU_CHECK(cudaFreeHost(h_bins_distance));
}

void LidarOgmCuda::processPointsCuda(VPointCloud::Ptr &pcl_in_,
                                     float *tf_array,
                                     float *h_global_cells_prob_array,
                                     int *h_ogm_data_array) {

    points_num = (int) pcl_in_->size();
    float *h_points_array, *d_points_array;

    GPU_CHECK(cudaHostAlloc((void **) &h_points_array,
                            points_num * 3 * sizeof(float),
                            cudaHostAllocMapped));
    pclToHostVector(pcl_in_, h_points_array);

    // transfer points array to device
    cudaHostGetDevicePointer((void**)&d_points_array, (void*)h_points_array, 0);

    float *d_global_cells_probs_array;
    int *d_ogm_data_array;

    cudaHostGetDevicePointer((void**)&d_global_cells_probs_array, (void*)h_global_cells_prob_array, 0);
    cudaHostGetDevicePointer((void**)&d_ogm_data_array, (void*)h_ogm_data_array, 0);

    // transfer tf array to device
    thrust::device_vector<float> thrust_tf_array(tf_array, tf_array + 16);
    
    thrust::device_vector<int> thrust_points_in_seg_count(grid_segments_, 0);

    thrust::device_vector<float> thrust_point_distance_in_segs(grid_segments_ * max_num_points_per_seg , 0);

    thrust::device_vector<float> thrust_p_logit(polar_grid_num_, 0.0);

    
    int num_block = DIVUP(points_num, NUM_THREADS_);

    mark_elevated_cell_kernel<<<num_block, NUM_THREADS_>>>(
        points_num,
        max_num_points_per_seg,
        d_points_array,
        thrust::raw_pointer_cast(thrust_points_in_seg_count.data()),
        thrust::raw_pointer_cast(thrust_point_distance_in_segs.data()) );

    cudaDeviceSynchronize();

    update_cell_probs_kernel<<<grid_segments_, grid_bins_>>>(
        points_num,
        max_num_points_per_seg,
        thrust::raw_pointer_cast(thrust_points_in_seg_count.data()),
        thrust::raw_pointer_cast(thrust_point_distance_in_segs.data()),
        d_bins_distance,
        thrust::raw_pointer_cast(thrust_p_logit.data()) );
    
    cudaDeviceSynchronize();

    map_polar_to_cartesian_ogm_kernel<<<width_grid_, height_grid_>>>(
        thrust::raw_pointer_cast(thrust_tf_array.data()),
        d_global_cells_probs_array,
        d_ogm_data_array,
        thrust::raw_pointer_cast(thrust_p_logit.data()));

    cudaDeviceSynchronize();

    GPU_CHECK(cudaFreeHost(h_points_array));
}

void LidarOgmCuda::pclToHostVector(VPointCloud::Ptr &pcl_in_,
                                   float *h_points_array) const {
    // transform point cloud to 1d vector
    for (int i = 0; i < points_num; ++i) {

        VPoint point = pcl_in_->at(i);
        h_points_array[i * 3 + 0] = point.x;
        h_points_array[i * 3 + 1] = point.y;
        h_points_array[i * 3 + 2] = point.z;
    }
}

__global__ void mark_elevated_cell_kernel(int points_num,
                                          const int max_num_points_per_seg,
                                          const float *dev_points,
                                          int *dev_points_in_seg_count,
                                          float *dev_point_distance_in_segs) {

    // Mark all cells that contain points above the ground.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= points_num)      return;

    float point_x = dev_points[tid * 3 + 0];
    float point_y = dev_points[tid * 3 + 1];
    float point_dist = sqrtf(point_x * point_x + point_y * point_y);

    int seg, bin, polar_id;
    fromVeloCoordsToPolarCell(point_x, point_y, point_dist, seg, bin, polar_id);

    int count = atomicAdd(&dev_points_in_seg_count[seg], 1);

    if(count < max_num_points_per_seg){

        int ind = seg * max_num_points_per_seg + count;

        dev_point_distance_in_segs[ind] = point_dist;
    }
}


__global__ void update_cell_probs_kernel(int points_num,
                                         const int max_num_points_per_seg,   
                                         const int *dev_points_in_seg_count,
                                         float *dev_point_distance_in_segs,
                                         float *dev_bins_distance, 
                                         float *dev_p_logit) {
    int seg = blockIdx.x;
    int bin = threadIdx.x;

    if(seg >= dev_map_size[0].grid_segments || bin >= dev_map_size[0].grid_bins)    return;
    int polar_id = from2dPolarCellIndexTo1d(seg, bin);

    int count = dev_points_in_seg_count[seg];
    if(count > 0){
        float lo_up{log(0.96/ 0.04)};
        float lo_low{log(0.01 / 0.99)};
        const float bin_distance = dev_bins_distance[bin];
        
        float temp_init_free_p = 0.4;
        float temp_occ_p = 0.0;
        float temp_lo = 0.0;
        float temp_free_p;
        
        // free_p = 0.001 * dis + 0.35, dis < 50, free_p = 0.4, dis >= 50 m
        if (bin_distance < 50.0) {
            temp_init_free_p = float(0.35 + 0.001 * bin_distance);
        }

        for (int i = 0; i < count; i++) { 
    
            int point_id = seg * max_num_points_per_seg + i;
            
            temp_occ_p = 0.5 + 1.2 * (0.35 - abs(bin_distance - dev_point_distance_in_segs[point_id]));

            if (bin_distance > dev_point_distance_in_segs[point_id] + 0.125) {
                temp_free_p = 0.5; // bin after the measurement should be unknown
            } else{
                temp_free_p = temp_init_free_p;
            }

            float temp_p = fmax(temp_occ_p, temp_free_p);
            temp_lo += log(temp_p / (1 - temp_p));
            // 0.01 < p < 0.96
            temp_lo = fmax(lo_low, fmin(lo_up, temp_lo));
        }
        // temp_lo = fmax(lo_low, fmin(lo_up, temp_lo));
        // dev_p_final[polar_id] = 1 - 1 / (1 + exp(temp_lo));
        dev_p_logit[polar_id] = temp_lo;
    }else{
        dev_p_logit[polar_id] = log(0.3/0.7);
    }
}


__global__ void map_polar_to_cartesian_ogm_kernel(float *dev_tf_array, 
                                                  float *dev_cells_prob,
                                                  int *dev_ogm_data_array, 
                                                  const float *dev_p_logit) {

    unsigned int w = blockIdx.x;
    unsigned int h = threadIdx.x;

    if ( w < dev_map_size[0].width_grid && h < dev_map_size[0].height_grid){
        float lo_up{log(0.99 / 0.01)};
        float lo_low{log(0.04 / 0.96)};
            
        float velo_y = -dev_grid_info[0].grid_range_max + dev_grid_info[0].grid_cell_size * (0.5 + w);
        float velo_x = -dev_grid_info[0].grid_range_max + dev_grid_info[0].grid_cell_size * (0.5 + h);
        float velo_distance = sqrt(velo_x * velo_x + velo_y * velo_y);

        int seg, bin, polar_id;
        fromVeloCoordsToPolarCell(velo_x, velo_y, velo_distance, seg, bin, polar_id);

        int final_cartesian_id = fromLocalOgmToFinalOgm(int(h), int(w), dev_tf_array);

        float cell_lo_past = dev_cells_prob[final_cartesian_id];
        float final_lo = fmax(lo_low, fmin(lo_up, cell_lo_past + dev_p_logit[polar_id]));
        dev_cells_prob[final_cartesian_id] = final_lo;

        if (fabs(final_lo) <= 1e-5)
            dev_ogm_data_array[final_cartesian_id] = -1;

        else if (final_lo < 0.0)
            dev_ogm_data_array[final_cartesian_id] = 0;

        else
            dev_ogm_data_array[final_cartesian_id] = 100;
    }
}