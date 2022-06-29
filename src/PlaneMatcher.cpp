#include "PlaneMatcher.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace ORB_SLAM2
{
    PlaneMatcher::PlaneMatcher(float dTh, float aTh, float verTh, float parTh):dTh(dTh), aTh(aTh), verTh(verTh), parTh(parTh) {}

    int PlaneMatcher::SearchMapByCoefficients(Frame &pF, const std::vector<MapPlane *> &vpMapPlanes) {
        pF.mbNewPlane = false;

        int nmatches = 0;

        for (int i = 0; i < pF.mnPlaneNum; ++i) {

            cv::Mat pM = pF.ComputePlaneWorldCoeff(i);


            float ldTh = dTh;
            float lverTh = verTh;
            float lparTh = parTh;

            bool found = false;
            // int j = 0;
            for (auto vpMapPlane : vpMapPlanes) {
                if (vpMapPlane->isBad()) {
                    // j++;
                    continue;
                }

                cv::Mat pW = vpMapPlane->GetWorldPos();

                float angle = pM.at<float>(0) * pW.at<float>(0) +
                              pM.at<float>(1) * pW.at<float>(1) +
                              pM.at<float>(2) * pW.at<float>(2);

//                cout << "Plane j: " << j << ", angle: " << angle << endl;
                // j++;

                // associate plane
//                if (angle > aTh || angle < -aTh) {
                if (angle > aTh) {
                    double dis = PointDistanceFromPlane(pM, vpMapPlane->mvPlanePoints);
                    if(dis < ldTh) {
                        ldTh = dis;
                        pF.mvpMapPlanes[i] = vpMapPlane;
                        found = true;
//                        cout << "Match! dis: " << dis << endl;
                        continue;
                    }
                }

                // vertical planes
                if (angle < lverTh && angle > -lverTh) {
                    lverTh = abs(angle);
                    pF.mvpVerticalPlanes[i] = vpMapPlane;
//                    cout << "Vertical Match!" << endl;
                    continue;
                }

                //parallel planes
                if (angle > lparTh || angle < -lparTh) {
                    lparTh = abs(angle);
                    pF.mvpParallelPlanes[i] = vpMapPlane;
//                    cout << "Parallel Match!" << endl;
                }
            }

            if (found) {
                nmatches++;
            }
        }

//        for(auto plane : pF.mvpMapPlanes){
//            if(!plane) {
//                pF.mbNewPlane = true;
//                break;
//            }
//        }

        return nmatches;
    }

    int PlaneMatcher::SearchByCoefficients(KeyFrame* pKF, KeyFrame *pKF2, std::vector<MapPlane*> &vpMapPlaneMatches, std::vector<MapPlane*> &vpMapVerticalPlaneMatches, std::vector<MapPlane*> &vpMapParallelPlaneMatches) {
        int nmatches = 0;

        vpMapPlaneMatches = vector<MapPlane*>(pKF->mvpMapPlanes.size(), static_cast<MapPlane*>(nullptr));
        vpMapVerticalPlaneMatches = vector<MapPlane*>(pKF->mvpMapPlanes.size(), static_cast<MapPlane*>(nullptr));
        vpMapParallelPlaneMatches = vector<MapPlane*>(pKF->mvpMapPlanes.size(), static_cast<MapPlane*>(nullptr));

        for (int i = 0; i < pKF->mvpMapPlanes.size(); ++i) {
            if (pKF->mvpMapPlanes[i] && !pKF->isBad()) {
//                cv::Mat p1 = pKF->mvpMapPlanes[i]->GetWorldPos();
//
//                float ldTh = dTh;
//                float lverTh = verTh;
//                float lparTh = parTh;
//
//                auto* bestPlaneMatch = static_cast<MapPlane*>(nullptr);
//                auto* bestVerticalPlaneMatch = static_cast<MapPlane*>(nullptr);
//                auto* bestParallelPlaneMatch = static_cast<MapPlane*>(nullptr);

                for (auto & mvpMapPlane : pKF2->mvpMapPlanes) {
                    if (mvpMapPlane && !mvpMapPlane->isBad()) {
                        if (pKF->mvpMapPlanes[i]->mnId == mvpMapPlane->mnId) {
                            vpMapPlaneMatches[i] = mvpMapPlane;
                            nmatches++;
                            continue;
                        }
//                        cv::Mat p2 = mvpMapPlane->GetWorldPos();
//
//                        float angle = p1.at<float>(0, 0) * p2.at<float>(0, 0) +
//                                      p1.at<float>(1, 0) * p2.at<float>(1, 0) +
//                                      p1.at<float>(2, 0) * p2.at<float>(2, 0);
//
//                        // associate plane
//                        if ((angle > aTh || angle < -aTh)) {
//                            double dis = PointDistanceFromPlane(p1, mvpMapPlane->mvPlanePoints);
//                            if (dis < ldTh) {
//                                ldTh = dis;
//                                bestPlaneMatch = mvpMapPlane;
//                                continue;
//                            }
//                        }
//
//                        // vertical planes
//                        if (angle < lverTh && angle > -lverTh) {
//                            lverTh = abs(angle);
//                            bestVerticalPlaneMatch = mvpMapPlane;
//                            continue;
//                        }
//
//                        //parallel planes
//                        if (angle > lparTh || angle < -lparTh) {
//                            lparTh = abs(angle);
//                            bestParallelPlaneMatch = mvpMapPlane;
//                        }
                    }
                }

//                if (bestPlaneMatch != static_cast<MapPlane*>(nullptr)) {
//                    vpMapPlaneMatches[i] = bestPlaneMatch;
//                    nmatches++;
//                }
//
//                if (bestVerticalPlaneMatch != static_cast<MapPlane*>(nullptr)) {
//                    vpMapVerticalPlaneMatches[i] = bestVerticalPlaneMatch;
//                }
//
//                if (bestParallelPlaneMatch != static_cast<MapPlane*>(nullptr)) {
//                    vpMapParallelPlaneMatches[i] = bestParallelPlaneMatch;
//                }
            }
        }

        return nmatches;
    }

    int PlaneMatcher::Fuse(KeyFrame *pKF, const std::vector<MapPlane *> &vpMapPlanes) {
        int nFused=0;

        cv::Mat Tcw = pKF->GetPose();

        const int nPlanes = vpMapPlanes.size();

        // For each candidate MapPlane project and match
        for(int iMP=0; iMP<nPlanes; iMP++)
        {
            MapPlane* pMP = vpMapPlanes[iMP];

            // Discard Bad MapPlanes and already found
            if(!pMP || pMP->isBad() || pMP->IsInKeyFrame(pKF))
                continue;

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = Tcw.t() * p3Dw;

            float ldTh = dTh;

            const int N = pKF->mnPlaneNum;

            int bestIdx = -1;
            for(int i=0; i<N; i++) {
                cv::Mat pKFc = pKF->mvPlaneCoefficients[i];

                float angle = p3Dc.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc.at<float>(2, 0) * pKFc.at<float>(2, 0);

                // associate plane
                if (angle > aTh || angle < -aTh)
                {
                    double dis = PointDistanceFromPlane(p3Dc, boost::make_shared<PointCloud>(pKF->mvPlanePoints[i]));
                    if(dis < ldTh) {
                        ldTh = dis;
                        bestIdx = i;
                    }
                }
            }

            if (bestIdx != -1) {
                MapPlane* pMPinKF = pKF->GetMapPlane(bestIdx);
                if(pMPinKF && !pMPinKF->isBad())
                {
                    if(pMPinKF->Observations()>pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
                else
                {
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPlane(pMP,bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

    double PlaneMatcher::PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr pointCloud) {
        double res = 100;
        for(auto p : pointCloud->points){
            // plane constraint in our system is probably
            // n.t() * p + d = 0
            double dis = abs(plane.at<float>(0, 0) * p.x +
                             plane.at<float>(1, 0) * p.y +
                             plane.at<float>(2, 0) * p.z +
                             plane.at<float>(3, 0));
            if(dis < res)
                res = dis;
        }
        return res;
    }

    int PlaneMatcher::SearchByCoefficients(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPlane *> &vpPlanes, const std::vector<MapPlane *> &vpVerticalPlanes, const std::vector<MapPlane *> &vpParallelPlanes,
                                           std::vector<MapPlane *> &vpMatched, std::vector<MapPlane *> &vpVerticalMatched, std::vector<MapPlane *> &vpParallelMatched) {

        // Set of MapPlanes already found in the KeyFrame
        set<MapPlane*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPlane*>(nullptr));
        set<MapPlane*> spVerticalAlreadyFound(vpVerticalMatched.begin(), vpVerticalMatched.end());
        spVerticalAlreadyFound.erase(static_cast<MapPlane*>(nullptr));
        set<MapPlane*> spParallelAlreadyFound(vpParallelMatched.begin(), vpParallelMatched.end());
        spParallelAlreadyFound.erase(static_cast<MapPlane*>(nullptr));

        cout << "Plane sim3 more Scw: " << Scw << endl;

        // Decompose Scw
        cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
        const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        cv::Mat Rcw = sRcw/scw;
        cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(auto pMP : vpPlanes)
        {
            // Discard already found
            if(!pMP || pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = cv::Mat::eye(4,1,CV_32F);
            p3Dc.rowRange(0,3).col(0) = Rcw * p3Dw.rowRange(0,3).col(0);
            p3Dc.at<float>(3,0) = p3Dw.at<float>(3,0) - tcw.dot(p3Dc.rowRange(0,3).col(0));
            if (p3Dc.at<float>(3,0) < 0.0)
                p3Dc = -p3Dc;

            cout << "Plane sim3 more p3Dc: " << p3Dc << endl;

            float ldTh = dTh;
            bool found = false;

            for (int i = 0; i < pKF->mnPlaneNum; ++i) {
                if (vpMatched[i]) {
                    continue;
                }
                cv::Mat pKFc = pKF->mvPlaneCoefficients[i];

                cout << "Plane sim3 more pKFc: i: " << i << " - " << pKFc << endl;

                float angle = p3Dc.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc.at<float>(2, 0) * pKFc.at<float>(2, 0);

                // associate plane
                if (angle > aTh || angle < -aTh)
                {
                    double dis = PointDistanceFromPlane(p3Dc, boost::make_shared<PointCloud>(pKF->mvPlanePoints[i]));
                    if(dis < ldTh) {
                        ldTh = dis;
                        cout << "Plane sim3 more match: i: " << i << ", Colour: " << pMP->mRed << ", " << pMP->mGreen << ", " << pMP->mBlue << endl;
                        vpMatched[i]=pMP;
                        if (pKF->GetMapPlane(i)) {
                            cout << "Plane sim3 more match: i: " << i << ", Colour: " << pKF->GetMapPlane(i)->mRed << ", " << pKF->GetMapPlane(i)->mGreen << ", " << pKF->GetMapPlane(i)->mBlue << endl;
                        }
                        found = true;
                    }
                }
            }

            if (found) {
                nmatches++;
            }
        }

        // For each Candidate MapPoint Project and Match
        for(auto pMP : vpVerticalPlanes)
        {
            // Discard already found
            if(!pMP || pMP->isBad() || spVerticalAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = cv::Mat::eye(4,1,CV_32F);
            p3Dc.rowRange(0,3).col(0) = Rcw * p3Dw.rowRange(0,3).col(0);
            p3Dc.at<float>(3,0) = p3Dw.at<float>(3,0) - tcw.dot(p3Dc.rowRange(0,3).col(0));
            if (p3Dc.at<float>(3,0) < 0.0)
                p3Dc = -p3Dc;

            float lverTh = verTh;
            bool found = false;

            for (int i = 0; i < pKF->mnPlaneNum; ++i) {
                if (vpVerticalMatched[i]) {
                    continue;
                }

                cv::Mat pKFc = pKF->mvPlaneCoefficients[i];

                float angle = p3Dc.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc.at<float>(2, 0) * pKFc.at<float>(2, 0);

                // vertical planes
                if (angle < lverTh && angle > -lverTh) {
                    lverTh = abs(angle);
                    vpVerticalMatched[i]=pMP;
                    found = true;
                }
            }

            if (found) {
                nmatches++;
            }
        }

        // For each Candidate MapPoint Project and Match
        for(auto pMP : vpParallelPlanes)
        {
            // Discard already found
            if(!pMP || pMP->isBad() || spParallelAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = cv::Mat::eye(4,1,CV_32F);
            p3Dc.rowRange(0,3).col(0) = Rcw * p3Dw.rowRange(0,3).col(0);
            p3Dc.at<float>(3,0) = p3Dw.at<float>(3,0) - tcw.dot(p3Dc.rowRange(0,3).col(0));
            if (p3Dc.at<float>(3,0) < 0.0)
                p3Dc = -p3Dc;

            float lparTh = parTh;
            bool found = false;

            for (int i = 0; i < pKF->mnPlaneNum; ++i) {
                if (vpParallelMatched[i]) {
                    continue;
                }

                cv::Mat pKFc = pKF->mvPlaneCoefficients[i];

                float angle = p3Dc.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc.at<float>(2, 0) * pKFc.at<float>(2, 0);

                //parallel planes
                if (angle > lparTh || angle < -lparTh) {
                    lparTh = abs(angle);
                    vpParallelMatched[i]=pMP;
                    found = true;
                }
            }

            if (found) {
                nmatches++;
            }
        }

        return nmatches;
    }

    int
    PlaneMatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPlane *> &vpMatches12,
                               std::vector<MapPlane*> &vpVerticalMatches12, std::vector<MapPlane*> &vpParallelMatches12,
                               const float &s12, const cv::Mat &R12, const cv::Mat &t12) {

        const vector<MapPlane*> vpMapPlanes1 = pKF1->GetMapPlaneMatches();
        const int N1 = vpMapPlanes1.size();

        const vector<MapPlane*> vpMapPlanes2 = pKF2->GetMapPlaneMatches();
        const int N2 = vpMapPlanes2.size();

        const vector<MapPlane*> vpMapVerticalPlanes1 = pKF1->GetMapVerticalPlaneMatches();
        const int NVertical1 = vpMapVerticalPlanes1.size();

        const vector<MapPlane*> vpMapVerticalPlanes2 = pKF2->GetMapVerticalPlaneMatches();
        const int NVertical2 = vpMapVerticalPlanes2.size();

        const vector<MapPlane*> vpMapParallelPlanes1 = pKF1->GetMapParallelPlaneMatches();
        const int NParallel1 = vpMapParallelPlanes1.size();

        const vector<MapPlane*> vpMapParallelPlanes2 = pKF2->GetMapParallelPlaneMatches();
        const int NParallel2 = vpMapParallelPlanes2.size();

        vpMatches12 = vector<MapPlane*>(N1, static_cast<MapPlane*>(nullptr));
        vpVerticalMatches12 = vector<MapPlane*>(NVertical1, static_cast<MapPlane*>(nullptr));
        vpParallelMatches12 = vector<MapPlane*>(NParallel1, static_cast<MapPlane*>(nullptr));

        // Camera 1 from world
        cv::Mat T1w = pKF1->GetPose();

        //Camera 2 from world
        cv::Mat T2w = pKF2->GetPose();

        //Transformation between cameras
        cv::Mat sR12 = s12*R12;
        cv::Mat sR21 = (1.0/s12)*R12.t();
        cv::Mat t21 = -sR21*t12;

        vector<int> vnMatch1(N1,-1);
        vector<int> vnMatch2(N2,-1);

        // Transform from KF1 to KF2 and search
        for(int i1=0; i1<N1; i1++)
        {
            if (!vpMapPlanes1[i1] || vpMapPlanes1[i1]->isBad()) {
                continue;
            }

            cv::Mat p3Dc1 = pKF1->mvPlaneCoefficients[i1];
            cv::Mat p3Dc2 = cv::Mat::eye(4,1,CV_32F);
            p3Dc2.rowRange(0,3).col(0) = sR21 * p3Dc1.rowRange(0,3).col(0);
            p3Dc2.at<float>(3,0) = p3Dc1.at<float>(3,0) -  t21.dot(p3Dc2.rowRange(0,3).col(0));
            if (p3Dc2.at<float>(3,0) < 0.0)
                p3Dc2 = -p3Dc2;

            float ldTh = dTh;

            for(int i2=0; i2<N2; i2++) {
                if (!vpMapPlanes2[i2] || vpMapPlanes2[i2]->isBad()) {
                    continue;
                }

                cv::Mat pKFc = pKF2->mvPlaneCoefficients[i2];

                float angle = p3Dc2.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc2.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc2.at<float>(2, 0) * pKFc.at<float>(2, 0);

                // associate plane
                if (angle > aTh || angle < -aTh)
                {
                    double dis = PointDistanceFromPlane(p3Dc2, boost::make_shared<PointCloud>(pKF2->mvPlanePoints[i2]));
                    if(dis < ldTh) {
                        ldTh = dis;
                        vnMatch1[i1]=i2;
                    }
                }
            }
        }

        // Transform from KF2 to KF1 and search
        for(int i2=0; i2<N2; i2++)
        {
            if (!vpMapPlanes2[i2] || vpMapPlanes2[i2]->isBad()) {
                continue;
            }
            cv::Mat p3Dc2 = pKF2->mvPlaneCoefficients[i2];
            cv::Mat p3Dc1 = cv::Mat::eye(4,1,CV_32F);
            p3Dc1.rowRange(0,3).col(0) = sR12 * p3Dc2.rowRange(0,3).col(0);
            p3Dc1.at<float>(3,0) = p3Dc2.at<float>(3,0) - t12.dot(p3Dc1.rowRange(0,3).col(0));
            if (p3Dc1.at<float>(3,0) < 0.0)
                p3Dc1 = -p3Dc1;

            float ldTh = dTh;

            for(int i1=0; i1<N1; i1++) {
                if (!vpMapPlanes1[i1] || vpMapPlanes1[i1]->isBad()) {
                    continue;
                }
                cv::Mat pKFc = pKF1->mvPlaneCoefficients[i1];

                float angle = p3Dc1.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc1.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc1.at<float>(2, 0) * pKFc.at<float>(2, 0);

                // associate plane
                if (angle > aTh || angle < -aTh)
                {
                    double dis = PointDistanceFromPlane(p3Dc1, boost::make_shared<PointCloud>(pKF1->mvPlanePoints[i1]));
                    if(dis < ldTh) {
                        ldTh = dis;
                        vnMatch2[i2]=i1;
                    }
                }
            }
        }

        // Check agreement
        int nFound = 0;

        for(int i1=0; i1<N1; i1++)
        {
            int idx2 = vnMatch1[i1];

            if(idx2>=0)
            {
                int idx1 = vnMatch2[idx2];
                if(idx1==i1)
                {
                    cout << "Plane sim3 match: " << vpMapPlanes1[idx1]->mRed << ", " << vpMapPlanes1[idx1]->mGreen << ", " << vpMapPlanes1[idx1]->mBlue
                         << " with: " << vpMapPlanes2[idx2]->mRed << ", " << vpMapPlanes2[idx2]->mGreen
                         << ", " << vpMapPlanes2[idx2]->mBlue << endl;
                    vpMatches12[i1] = vpMapPlanes2[idx2];
                    nFound++;
                }
            }
        }

        vector<int> vnVerticalMatch1(NVertical1,-1);
        vector<int> vnVerticalMatch2(NVertical2,-1);

        // Transform from KF1 to KF2 and search
        for(int i1=0; i1<NVertical1; i1++)
        {
            if (!vpMapVerticalPlanes1[i1] || vpMapVerticalPlanes1[i1]->isBad()) {
                continue;
            }

            cv::Mat p3Dc1 = pKF1->mvPlaneCoefficients[i1];
            cv::Mat p3Dc2 = cv::Mat::eye(4,1,CV_32F);
            p3Dc2.rowRange(0,3).col(0) = sR21 * p3Dc1.rowRange(0,3).col(0);
            p3Dc2.at<float>(3,0) = p3Dc1.at<float>(3,0) - t21.dot(p3Dc2.rowRange(0,3).col(0));
            if (p3Dc2.at<float>(3,0) < 0.0)
                p3Dc2 = -p3Dc2;

            float lverTh = verTh;

            for(int i2=0; i2<NVertical2; i2++) {
                if (!vpMapVerticalPlanes2[i2] || vpMapVerticalPlanes2[i2]->isBad()) {
                    continue;
                }
                cv::Mat pKFc = pKF2->mvPlaneCoefficients[i2];

                float angle = p3Dc2.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc2.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc2.at<float>(2, 0) * pKFc.at<float>(2, 0);

                if (angle < lverTh && angle > -lverTh) {
                    lverTh = abs(angle);
                    vnVerticalMatch1[i1]=i2;
                }
            }
        }

        // Transform from KF2 to KF1 and search
        for(int i2=0; i2<NVertical2; i2++)
        {
            if (!vpMapVerticalPlanes2[i2] || vpMapVerticalPlanes2[i2]->isBad()) {
                continue;
            }
            cv::Mat p3Dc2 = pKF2->mvPlaneCoefficients[i2];
            cv::Mat p3Dc1 = cv::Mat::eye(4,1,CV_32F);
            p3Dc1.rowRange(0,3).col(0) = sR12 * p3Dc2.rowRange(0,3).col(0);
            p3Dc1.at<float>(3,0) = p3Dc2.at<float>(3,0) - t12.dot(p3Dc1.rowRange(0,3).col(0));
            if (p3Dc1.at<float>(3,0) < 0.0)
                p3Dc1 = -p3Dc1;

            float lverTh = verTh;

            for(int i1=0; i1<NVertical1; i1++) {
                if (!vpMapVerticalPlanes1[i1] || vpMapVerticalPlanes1[i1]->isBad()) {
                    continue;
                }
                cv::Mat pKFc = pKF1->mvPlaneCoefficients[i1];

                float angle = p3Dc1.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc1.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc1.at<float>(2, 0) * pKFc.at<float>(2, 0);

                if (angle < lverTh && angle > -lverTh) {
                    lverTh = abs(angle);
                    vnVerticalMatch2[i2]=i1;
                }
            }
        }

        // Check agreement

        for(int i1=0; i1<NVertical1; i1++)
        {
            int idx2 = vnVerticalMatch1[i1];

            if(idx2>=0)
            {
                int idx1 = vnVerticalMatch2[idx2];
                if(idx1==i1)
                {
                    vpVerticalMatches12[i1] = vpMapVerticalPlanes2[idx2];
                    cout << "Plane sim3 vertical match: " << vpMapVerticalPlanes1[idx1]->mRed << ", " << vpMapVerticalPlanes1[idx1]->mGreen << ", " << vpMapVerticalPlanes1[idx1]->mBlue
                         << " with: " << vpMapVerticalPlanes2[idx2]->mRed << ", " << vpMapVerticalPlanes2[idx2]->mGreen
                         << ", " << vpMapVerticalPlanes2[idx2]->mBlue << endl;
                    nFound++;
                }
            }
        }

        vector<int> vnParallelMatch1(NParallel1,-1);
        vector<int> vnParallelMatch2(NParallel2,-1);

        // Transform from KF1 to KF2 and search
        for(int i1=0; i1<NParallel1; i1++)
        {
            if (!vpMapParallelPlanes1[i1] || vpMapParallelPlanes1[i1]->isBad()) {
                continue;
            }

            cv::Mat p3Dc1 = pKF1->mvPlaneCoefficients[i1];
            cv::Mat p3Dc2 = cv::Mat::eye(4,1,CV_32F);
            p3Dc2.rowRange(0,3).col(0) = sR21 * p3Dc1.rowRange(0,3).col(0);
            p3Dc2.at<float>(3,0) = p3Dc1.at<float>(3,0) - t21.dot(p3Dc2.rowRange(0,3).col(0));
            if (p3Dc2.at<float>(3,0) < 0.0)
                p3Dc2 = -p3Dc2;

            float lparTh = parTh;

            for(int i2=0; i2<NParallel2; i2++) {
                if (!vpMapParallelPlanes2[i2] || vpMapParallelPlanes2[i2]->isBad()) {
                    continue;
                }

                cv::Mat pKFc = pKF2->mvPlaneCoefficients[i2];

                float angle = p3Dc2.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc2.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc2.at<float>(2, 0) * pKFc.at<float>(2, 0);

                if (angle > lparTh || angle < -lparTh) {
                    lparTh = abs(angle);
                    vnParallelMatch1[i1]=i2;
                }
            }
        }

        // Transform from KF2 to KF1 and search
        for(int i2=0; i2<NParallel2; i2++)
        {
            if (!vpMapParallelPlanes2[i2] || vpMapParallelPlanes2[i2]->isBad()) {
                continue;
            }
            cv::Mat p3Dc2 = pKF2->mvPlaneCoefficients[i2];
            cv::Mat p3Dc1 = cv::Mat::eye(4,1,CV_32F);
            p3Dc1.rowRange(0,3).col(0) = sR12 * p3Dc2.rowRange(0,3).col(0);
            p3Dc1.at<float>(3,0) = p3Dc2.at<float>(3,0) - t12.dot(p3Dc1.rowRange(0,3).col(0));
            if (p3Dc1.at<float>(3,0) < 0.0)
                p3Dc1 = -p3Dc1;

            float lparTh = parTh;

            for(int i1=0; i1<NParallel1; i1++) {
                if (!vpMapParallelPlanes1[i1] || vpMapParallelPlanes1[i1]->isBad()) {
                    continue;
                }

                cv::Mat pKFc = pKF1->mvPlaneCoefficients[i1];

                float angle = p3Dc1.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc1.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc1.at<float>(2, 0) * pKFc.at<float>(2, 0);

                if (angle > lparTh || angle < -lparTh) {
                    lparTh = abs(angle);
                    vnParallelMatch2[i2]=i1;
                }
            }
        }

        // Check agreement

        for(int i1=0; i1<NParallel1; i1++)
        {
            int idx2 = vnParallelMatch1[i1];

            if(idx2>=0)
            {
                int idx1 = vnParallelMatch2[idx2];
                if(idx1==i1)
                {
                    vpParallelMatches12[i1] = vpMapParallelPlanes2[idx2];
                    cout << "Plane sim3 parallel match: " << vpMapParallelPlanes1[idx1]->mRed << ", " << vpMapParallelPlanes1[idx1]->mGreen << ", " << vpMapParallelPlanes1[idx1]->mBlue
                         << " with: " << vpMapParallelPlanes2[idx2]->mRed << ", " << vpMapParallelPlanes2[idx2]->mGreen
                         << ", " << vpMapParallelPlanes2[idx2]->mBlue << endl;
                    nFound++;
                }
            }
        }

        return nFound;
    }

    int PlaneMatcher::Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPlane*> &vpPlanes, const std::vector<MapPlane*> &vpVerticalPlanes,
                           const std::vector<MapPlane*> &vpParallelPlanes, vector<MapPlane *> &vpReplacePlane,
                           vector<MapPlane *> &vpReplaceVerticalPlane, vector<MapPlane *> &vpReplaceParallelPlane) {
        int nFused=0;

        cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
        cv::Mat tcw = Scw.rowRange(0,3).col(3);

        // Set of MapPlanes already found in the KeyFrame
        const set<MapPlane*> spAlreadyFound = pKF->GetMapPlanes();

        const int nPlanes = vpPlanes.size();

        // For each candidate MapPlane project and match
        for(int iMP=0; iMP<nPlanes; iMP++)
        {
            MapPlane* pMP = vpPlanes[iMP];

            // Discard Bad MapPlanes and already found
            if(!pMP || pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = cv::Mat::eye(4,1,CV_32F);
            p3Dc.rowRange(0,3).col(0) = sRcw * p3Dw.rowRange(0,3).col(0);
            p3Dc.at<float>(3,0) = p3Dw.at<float>(3,0) - tcw.dot(p3Dc.rowRange(0,3).col(0));
            if (p3Dc.at<float>(3,0) < 0.0)
                p3Dc = -p3Dc;

            float ldTh = dTh;

            const int N = pKF->mnPlaneNum;

            int bestIdx = -1;
            for(int i=0; i<N; i++) {
                cv::Mat pKFc = pKF->mvPlaneCoefficients[i];

                float angle = p3Dc.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                        p3Dc.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                        p3Dc.at<float>(2, 0) * pKFc.at<float>(2, 0);

                if (vpPlanes[iMP] && pKF->GetMapPlane(i)) {
                    cout << "Plane fuse plane1: " << vpPlanes[iMP]->mnId << " - " << p3Dw
                         << ", plane2: " << pKF->GetMapPlane(i)->mnId << " - " << pKF->mvPlaneCoefficients[i]
                         << ", p3Dc: " << p3Dc << endl;
                }

                // associate plane
                if (angle > aTh || angle < -aTh)
                {
                    double dis = PointDistanceFromPlane(p3Dc, boost::make_shared<PointCloud>(pKF->mvPlanePoints[i]));
                    if(dis < ldTh) {
                        ldTh = dis;
                        bestIdx = i;
                    }
                }
            }

            if (bestIdx != -1) {
                MapPlane* pMPinKF = pKF->GetMapPlane(bestIdx);
                if(pMPinKF && !pMPinKF->isBad())
                {
                    cout << "Plane fuse match: " << pMP->mRed << ", " << pMP->mGreen << ", " << pMP->mBlue
                         << " with: " << pMPinKF->mRed << ", " << pMPinKF->mGreen
                         << ", " << pMPinKF->mBlue << endl;
                    vpReplacePlane[iMP] = pMPinKF;
                }
                else
                {
                    cout << "Plane fuse match: " << pMP->mRed << ", " << pMP->mGreen << ", " << pMP->mBlue
                         << endl;
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPlane(pMP,bestIdx);
                }
                nFused++;
            }
        }

        const set<MapPlane*> spParallelAlreadyFound = pKF->GetMapParallelPlanes();

        const int nParallelPlanes = vpParallelPlanes.size();

        // For each candidate MapPlane project and match
        for(int iMP=0; iMP<nParallelPlanes; iMP++)
        {
            MapPlane* pMP = vpParallelPlanes[iMP];

            // Discard Bad MapPlanes and already found
            if(!pMP || pMP->isBad() || spParallelAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = cv::Mat::eye(4,1,CV_32F);
            p3Dc.rowRange(0,3).col(0) = sRcw * p3Dw.rowRange(0,3).col(0);
            p3Dc.at<float>(3,0) = p3Dw.at<float>(3,0) - tcw.dot(p3Dc.rowRange(0,3).col(0));
            if (p3Dc.at<float>(3,0) < 0.0)
                p3Dc = -p3Dc;

            float lparTh = parTh;

            const int N = pKF->mnPlaneNum;

            int bestIdx = -1;
            for(int i=0; i<N; i++) {
                cv::Mat pKFc = pKF->mvPlaneCoefficients[i];

                float angle = p3Dc.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc.at<float>(2, 0) * pKFc.at<float>(2, 0);

                if (angle > lparTh || angle < -lparTh) {
                    lparTh = abs(angle);
                    bestIdx = i;
                }
            }

            if (bestIdx != -1) {
                MapPlane* pMPinKF = pKF->GetMapParallelPlane(bestIdx);
                if(pMPinKF && !pMPinKF->isBad())
                {
                    vpReplaceParallelPlane[iMP] = pMPinKF;
                }
                else
                {
                    pMP->AddParObservation(pKF,bestIdx);
                    pKF->AddMapParallelPlane(pMP,bestIdx);
                }
                nFused++;
            }
        }

        const set<MapPlane*> spVerticalAlreadyFound = pKF->GetMapVerticalPlanes();

        const int nVerticalPlanes = vpVerticalPlanes.size();

        // For each candidate MapPlane project and match
        for(int iMP=0; iMP<nVerticalPlanes; iMP++)
        {
            MapPlane* pMP = vpVerticalPlanes[iMP];

            // Discard Bad MapPlanes and already found
            if(!pMP || pMP->isBad() || spVerticalAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = cv::Mat::eye(4,1,CV_32F);
            p3Dc.rowRange(0,3).col(0) = sRcw * p3Dw.rowRange(0,3).col(0);
            p3Dc.at<float>(3,0) = p3Dw.at<float>(3,0) - tcw.dot(p3Dc.rowRange(0,3).col(0));
            if (p3Dc.at<float>(3,0) < 0.0)
                p3Dc = -p3Dc;

            float lverTh = verTh;

            const int N = pKF->mnPlaneNum;

            int bestIdx = -1;
            for(int i=0; i<N; i++) {
                cv::Mat pKFc = pKF->mvPlaneCoefficients[i];

                float angle = p3Dc.at<float>(0, 0) * pKFc.at<float>(0, 0) +
                              p3Dc.at<float>(1, 0) * pKFc.at<float>(1, 0) +
                              p3Dc.at<float>(2, 0) * pKFc.at<float>(2, 0);

                // vertical planes
                if (angle < lverTh && angle > -lverTh) {
                    lverTh = abs(angle);
                    bestIdx = i;
                }
            }

            if (bestIdx != -1) {
                MapPlane* pMPinKF = pKF->GetMapVerticalPlane(bestIdx);
                if(pMPinKF && !pMPinKF->isBad())
                {
                    vpReplaceVerticalPlane[iMP] = pMPinKF;
                }
                else
                {
                    pMP->AddVerObservation(pKF,bestIdx);
                    pKF->AddMapVerticalPlane(pMP,bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }
}
