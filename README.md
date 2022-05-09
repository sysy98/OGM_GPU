# roscuda_lidar_ogm
## Introduction
Occupancy grid mapping, as one of the most popular methods for constructing environment maps, 
is widely used for a variety of robotic tasks in the context of autonomous driving.
Therefore, it is necessary to implement this algorithm quickly and efficiently.

This repository is a `ROS` package that leverages GPU parallelism to accelerate occupancy grid mapping with LiDAR point clouds in the `CUDA` framework.

## Visualize Results in Rviz
<p align="center">
  <img src="./videos/semantic.gif">
</p>

<p align="center">
  <img src="./videos/rviz.gif">
</p>

## Setup

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
1. Install [ROS](http://wiki.ros.org/Installation/Ubuntu) and create a [catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) in your home directory:  
```
mkdir -p ~/catkin_ws/src
```
2.  Clone this repository into the catkin workspace's source folder (src) and build it:  

```
cd ~/catkin_ws/src
git clone https://github.com/sysy98/OGM_GPU.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```
3. [Download a peprocessed scenario of KITTI dataset](https://drive.google.com/drive/folders/1vHpkoC78fPXT64-VFL1H5Mm1bdukK5Qz) and unzip it into a separate `kitti_data` directory.

## Usage
Modify the launch file, set the rosbag file to be processed ("scenario") and the replay speed ("speed").
```
source devel/setup.bash
roslaunch roscuda_lidar_ogm roscuda_lidar_ogm.launch
```

## Pipeline and branches

![image](https://github.com/sysy98/OGM_GPU/blob/master/pipeline.png)

The GPU implementation of occupancy grid mapping is illustrated by the following figure.
![image](https://github.com/sysy98/OGM_GPU/blob/master/GPU_implement.png)

### branches: 

1. `master`: CUDA data transfer method using Zero-copy memory with nearest neighbor mapping method.
    
2. `pinned_memory`: CUDA data transfer method using **pinned memory** with nearest neighbor mapping method.
    
3. `unified_memory`: CUDA data transfer method using **Unified Memory** with nearest neighbor mapping method.

4. `bilinear_interpolation`: mapping from polar grids to Cartesian grids with bilinear interpolation method. The variables used in the bilinear interpolation are allocated with **Unified Memory**.

### Solution to compile errors
if there is an error occurred in the compile process: **error: namespace "boost" has no member "iequals"**,
please comment the line **#ifndef \_\_CUDACC__** and one line **#endif** in **pcl/io/boost.h**
