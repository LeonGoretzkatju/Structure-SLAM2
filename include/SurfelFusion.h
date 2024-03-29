#ifndef SURFEL_FUSION
#define SURFEL_FUSION

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <SurfelElements.h>

#define MIN_SURFEL_SIZE 0.02
#define ITERATION_NUM 3
#define THREAD_NUM 10
#define SP_SIZE 8
#define MAX_ANGLE_COS 0.1
// for drive
#define HUBER_RANGE 0.4
#define BASELINE 0.5
#define DISPARITY_ERROR 4.0
#define MIN_TOLERATE_DIFF 0.1
// for RGBD
// #define HUBER_RANGE 0.05
// #define BASELINE 0.08
// #define DISPARITY_ERROR 1.0
// #define MIN_TOLERATE_DIFF 0.05

class SurfelFusion
{
private:
    float fx, fy, cx, cy;
    int image_width, image_height;
    int sp_width, sp_height;
    float fuse_far, fuse_near;

    cv::Mat image;
    cv::Mat depth;
    cv::Mat planeMembershipImg;

    Eigen::Matrix4f pose;

    std::vector<double> space_map;
    std::vector<float> norm_map;
    std::vector<SuperpixelSeed> superpixel_seeds;
    std::vector<int> superpixel_index;

    std::vector<SurfelElement> *local_surfels_ptr;
    std::vector<SurfelElement> *new_surfels_ptr;

    // get the super pixels
    void generate_super_pixels();
    void back_project(
        const float &u, const float &v, const float &depth, double&x, double&y, double&z);
    bool calculate_cost(
        float &nodepth_cost, float &depth_cost,
        const float &pixel_intensity, const float &pixel_inverse_depth,
        const int &x, const int &y,
        const int &sp_x, const int &sp_y);
    void update_pixels_kernel(int thread_i, int thread_num);
    void update_pixels();
    void update_seeds_kernel(
        int thread_i, int thread_num);
    void update_seeds();
    void initialize_seeds_kernel(
        int thread_i, int thread_num);
    void get_huber_norm(
        float &nx, float &ny, float &nz, float &nb,
        std::vector<float> &points);
    void initialize_seeds();
    void calculate_spaces_kernel(int thread_i, int thread_num);
    void calculate_sp_norms_kernel(int thread_i, int thread_num);
    void calculate_sp_depth_norms_kernel(int thread_i, int thread_num);
    void calculate_pixels_norms_kernel(int thread_i, int thread_num);
    void calculate_norms();

    // for fusion
    void fuse_surfels_kernel(
        int thread_i, int thread_num,
        int reference_frame_index);
    void initialize_surfels(
        int reference_frame_index,
        const Eigen::Matrix4f& pose);
    void project(float &x, float &y, float &z, float &u, float &v);
    float get_weight(float &depth);

    // for debug
    void debug_show();

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void initialize(
        int _width, int _height,
        float _fx, float _fy, float _cx, float _cy,
        float _fuse_far, float _fuse_near);
    void fuse_initialize_map(
        int reference_frame_index,
        cv::Mat &input_image,
        cv::Mat &input_depth,
        cv::Mat &input_planeMembershipImg,
        const Eigen::Matrix4f &cam_pose,
        std::vector<SurfelElement> &local_surfels,
        std::vector<SurfelElement> &new_surfels);
};

#endif