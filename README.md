# roscuda_lidar_ogm
## Introduction
Occupancy grid mapping, as one of the most popular methods for constructing environment maps, 
is widely used for a variety of robotic tasks in the context of autonomous driving.

Therefore, it is necessary to implement this algorithm quickly and efficiently.

In this project, we exploit GPU parallelism to accelerate occupancy grid mapping with LiDAR point clouds in the CUDA framework.

The pipeline of our algorithm is as follows:
![image](https://github.com/sysy98/OGM_GPU/blob/master/pipeline.png)

The GPU implementation of occupancy grid mapping is illustrated by the following figure.
![image](https://github.com/sysy98/OGM_GPU/blob/master/GPU_implement.png)

## Project structure

[Download a peprocessed scenario](https://drive.google.com/drive/folders/1vHpkoC78fPXT64-VFL1H5Mm1bdukK5Qz) and unzip it into a separate `kitti_data` directory.

    The following folder structure is recommended:
```
     ~                                       # Home directory
    ├── catkin_ws                            # Catkin workspace
    │   ├── src                              # Source folder
    │       └── roscuda_lidar_ogm            # Package
    ├── kitti_data                           # Dataset
    │   ├── 0012                             # Demo scenario 0012
    │   │   └── synchronized_data.bag        # Synchronized ROSbag file
```

## Programs Information
### branches: 

**master**: CUDA data transfer method using Zero-copy memory with nearest neighbor mapping method.
    
**pinned_memory**: CUDA data transfer method using **pinned memory** with nearest neighbor mapping method.
    
**unified_memory**: CUDA data transfer method using **Unified Memory** with nearest neighbor mapping method.

**bilinear_interpolation**: mapping from polar grids to Cartesian grids with bilinear interpolation method. 
        The variables used in the bilinear interpolation are allocated with **Unified Memory**.

### Solution to compile errors
if there is an error occurred in the compile process: **error: namespace "boost" has no member "iequals"**,
please comment the line **#ifndef \_\_CUDACC__** and one line **#endif** in **pcl/io/boost.h**
