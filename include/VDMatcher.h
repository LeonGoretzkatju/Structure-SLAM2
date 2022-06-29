#ifndef ORB_SLAM2_VDMATCHER_H
#define ORB_SLAM2_VDMATCHER_H

#include "MapVanishingDirection.h"
#include "KeyFrame.h"
#include "Frame.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>

namespace ORB_SLAM2 {
    class VDMatcher {
    public:

        VDMatcher(float dTh = 0.1, float aTh = 0.86, float verTh = 0.08716, float parTh = 0.9962);

        int SearchMapByCoefficients(Frame &pF, const std::vector<MapVanishingDirection *> &vpMapVDs);


        int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPlane*> &vpMatches12, std::vector<MapPlane*> &vpVerticalMatches12,
                         std::vector<MapPlane*> &vpParallelMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12);

        int Fuse(KeyFrame *pKF, const std::vector<MapPlane *> &vpMapPlanes);

        int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPlane*> &vpPlanes, const std::vector<MapPlane*> &vpVerticalPlanes,
                 const std::vector<MapPlane*> &vpParallelPlanes, vector<MapPlane *> &vpReplacePlane,
                 vector<MapPlane *> &vpReplaceVerticalPlane, vector<MapPlane *> &vpReplaceParallelPlane);

    protected:
        float dTh, aTh, verTh, parTh;

        //double PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr pointCloud);
    };
}


#endif //ORB_SLAM2_VDMATCHER_H
