//
// Created by lan on 17-12-13.
//

#include "LSDextractor.h"

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {
    LineSegment::LineSegment() {}

    void LineSegment::ExtractLineSegment(const Mat &img, vector<KeyLine> &keylines, Mat &ldesc,
                                         vector<Vector3d> &keylineFunctions, float scale, int numOctaves) {
        Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
        Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();
        lsd->detect(img, keylines, scale, numOctaves);

        unsigned int lsdNFeatures = 40;

        // filter lines
        if (keylines.size() > lsdNFeatures) {
            sort(keylines.begin(), keylines.end(), sort_lines_by_response());
            keylines.resize(lsdNFeatures);
            for (unsigned int i = 0; i < lsdNFeatures; i++)
                keylines[i].class_id = i;
        }

        lbd->compute(img, keylines, ldesc);

        for (vector<KeyLine>::iterator it = keylines.begin(); it != keylines.end(); ++it) {
            Vector3d sp_l;
            sp_l << it->startPointX, it->startPointY, 1.0;
            Vector3d ep_l;
            ep_l << it->endPointX, it->endPointY, 1.0;
            Vector3d lineF;
            lineF << sp_l.cross(ep_l);
            lineF.normalize();
            keylineFunctions.push_back(lineF);
        }
    }
}