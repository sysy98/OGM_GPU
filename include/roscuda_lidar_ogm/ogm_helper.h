// Include guard
#ifndef OGM_HELPER_H
#define OGM_HELPER_H

// Parameter handler
typedef struct {

    float grid_cell_size;
    float inv_angular_res;
    float inv_radial_res;
    float lidar_opening_angle;
    float grid_range_min;
    float grid_range_max;
} GridInfo;

typedef struct {

    int width_grid;
    int height_grid;
    int grid_segments;
    int grid_bins;
    int polar_grid_num;

    int occ_width_grid;
    int occ_height_grid;
    int occ_grid_num;
} MapSize;

typedef struct {
    GridInfo grid_info_;
    MapSize map_size_;

    float grid_cell_height;
    float lidar_z_min;
} MapParams;

// Attributes of cell from polar grid
struct PolarCell {

    float x_min, y_min, z_min;
    float z_max;
    float ground;
    float height;
    int count;

    // Default constructor
    PolarCell() : x_min(0), y_min(0), z_min(0), z_max(0), ground(0), height(0), count(0) {}
};

#endif // OGM_HELPER_H