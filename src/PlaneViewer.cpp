//
// Created by raza on 27.01.20.
//

#include "PlaneViewer.h"

namespace ORB_SLAM2 {

    PlaneViewer::PointCloud::Ptr PlaneViewer::cloudPoints = boost::make_shared<PointCloud>();

    PlaneViewer::PlaneViewer() {
//        cloudPoints = boost::make_shared<PointCloud>();

        viewerThread = std::make_shared<std::thread>(std::bind(&PlaneViewer::viewer, this));
//        std::thread(&PlaneViewer::viewer, this);
    }

//    void PlaneViewer::update(PointCloud::Ptr newPoints) {
//        PlaneViewer::cloudPoints = newPoints;
//    }

    void PlaneViewer::shutdown() {
        shutdownFlag = true;
    }

    void PlaneViewer::viewer() {
        pcl::visualization::CloudViewer viewer("Plane Viewer");
        while(1)
        {
            if (shutdownFlag) {
                break;
            }
            viewer.showCloud(cloudPoints);
        }
    }
}

