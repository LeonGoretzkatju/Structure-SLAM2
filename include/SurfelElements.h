#ifndef ELEMENTS
#define ELEMENTS

struct SuperpixelSeed
{
    float x, y;
    float size;
    float norm_x, norm_y, norm_z;
    float posi_x, posi_y, posi_z;
    float view_cos;
    float mean_depth;
    float mean_intensity;
    int r, g, b;
    bool fused;
    bool stable;
    bool use = true;

    // for debug
    float min_eigen_value;
    float max_eigen_value;
};

struct SurfelElement
{
    float px, py, pz;
    float nx, ny, nz;
    float size;
    float color;
    int r, g, b;
    float weight;
    int update_times;
    int last_update;
};

#endif
