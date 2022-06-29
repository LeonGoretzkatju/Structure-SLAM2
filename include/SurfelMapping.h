//
// Created by raza on 27.09.20.
//

#ifndef ORB_SLAM2_SURFELMAPPING_H
#define ORB_SLAM2_SURFELMAPPING_H

#include "System.h"
#include "Map.h"
#include "SurfelElements.h"
#include "SurfelFusion.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>


namespace ORB_SLAM2 {
    typedef pcl::PointXYZRGB PointType;
    typedef pcl::PointCloud<PointType> PointCloud;

    struct PoseElement{
        std::vector<SurfelElement> attached_surfels;
        std::vector<int> linked_pose_index;
        int points_begin_index;
        int points_pose_index;
        PoseElement() : points_begin_index(-1), points_pose_index(-1) {}
    };

    class SurfelMapping {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SurfelMapping(Map* map, const string &strSettingsFile);

        void Run();

        pcl::PointCloud<pcl::PointSurfel>::Ptr Stop();

        void InsertKeyFrame(const cv::Mat &imRGB, const cv::Mat &imDepth, const cv::Mat planeMembershipImg, const cv::Mat& pose, const int referenceIndex, const bool isNewKeyFrame);

    protected:

        bool CheckNewKeyFrames();
        void ProcessNewKeyFrame();

        void move_add_surfels(int reference_index);
        void get_add_remove_poses(int root_index, std::vector<int> &pose_to_add, std::vector<int> &pose_to_remove);
        void get_driftfree_poses(int root_index, std::vector<int> &driftfree_poses, int driftfree_range);
        void fuse_map(cv::Mat image, cv::Mat depth, cv::Mat planeMembershipImg, const Eigen::Matrix4f& pose_input, int reference_index);

        std::list<std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat, int, bool>> mlNewKeyFrames;

        std::mutex mMutexNewKFs;
        bool mbStop;
        std::mutex mMutexStop;

        Map* mMap;

        SurfelFusion mSurfelFusion;

        std::vector<PoseElement> poses_database;
        std::set<int> local_surfels_indexs;
        int drift_free_poses;

        std::vector<int> pointcloud_pose_index;
    };
}

#endif //ORB_SLAM2_SURFELMAPPING_H
