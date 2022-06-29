//
// Created by raza on 27.01.20.
//

#ifndef ORB_SLAM2_PLANEVIEWER_H
#define ORB_SLAM2_PLANEVIEWER_H

#include <pcl/visualization/cloud_viewer.h>
#include <thread>
#include <boost/make_shared.hpp>

namespace ORB_SLAM2 {
    class PlaneViewer {
    public:
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;

        PlaneViewer();

        std::shared_ptr<std::thread> viewerThread;
        static PointCloud::Ptr cloudPoints;
        bool shutdownFlag = false;

        void update(PointCloud::Ptr newPoints);

        void shutdown();

        void viewer();
    };
}

#endif //ORB_SLAM2_PLANEVIEWER_H
