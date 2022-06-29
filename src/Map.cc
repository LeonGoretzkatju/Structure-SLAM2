/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Map.h"

#include <mutex>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {

    size_t PartialManhattanMapHash::operator() (const std::pair<MapPlane*, MapPlane*>& key) const {
        int id1, id2;
        if (key.first->mnId > key.second->mnId) {
            id1 = key.second->mnId;
            id2 = key.first->mnId;
        } else {
            id1 = key.first->mnId;
            id2 = key.second->mnId;
        }

        size_t hash = 0;
        hash += (71*hash + id1) % 5;
        hash += (71*hash + id2) % 5;
        return hash;
    }

    bool PartialManhattanMapEqual::operator() (const std::pair<MapPlane*, MapPlane*>& a, const std::pair<MapPlane*, MapPlane*>& b) const {
        MapPlane* pMP11, *pMP12, *pMP21, *pMP22;
        if (a.first->mnId > a.second->mnId) {
            pMP11 = a.second;
            pMP12 = a.first;
        } else {
            pMP11 = a.first;
            pMP12 = a.second;
        }

        if (b.first->mnId > b.second->mnId) {
            pMP21 = b.second;
            pMP22 = b.first;
        } else {
            pMP21 = b.first;
            pMP22 = b.second;
        }

        std::pair<MapPlane*, MapPlane*> p1 = std::make_pair(pMP11, pMP12);
        std::pair<MapPlane*, MapPlane*> p2 = std::make_pair(pMP21, pMP22);

        return p1 == p2;
    }

    size_t ManhattanMapHash::operator() (const std::tuple<MapPlane*, MapPlane*, MapPlane*>& key) const {
        vector<int> ids;
        ids.push_back(get<0>(key)->mnId);
        ids.push_back(get<1>(key)->mnId);
        ids.push_back(get<2>(key)->mnId);
        sort(ids.begin(), ids.end());

        size_t hash = 0;
        hash += (71*hash + ids[0]) % 5;
        hash += (71*hash + ids[1]) % 5;
        hash += (71*hash + ids[2]) % 5;
        return hash;
    }

    bool ManhattanMapEqual::operator() (const std::tuple<MapPlane*, MapPlane*, MapPlane*>& a,
                                        const std::tuple<MapPlane*, MapPlane*, MapPlane*>& b) const {
        MapPlane* pMP11, *pMP12, *pMP13, *pMP21, *pMP22, *pMP23;

        pMP11 = get<0>(a);
        pMP12 = get<1>(a);
        pMP13 = get<2>(a);

        if (pMP11 > pMP12)
        {
            std::swap(pMP11, pMP12);
        }
        if (pMP12 > pMP13)
        {
            std::swap(pMP12, pMP13);
        }
        if (pMP11 > pMP12)
        {
            std::swap(pMP11, pMP12);
        }

        pMP21 = get<0>(b);
        pMP22 = get<1>(b);
        pMP23 = get<2>(b);

        if (pMP21 > pMP22)
        {
            std::swap(pMP21, pMP22);
        }
        if (pMP22 > pMP23)
        {
            std::swap(pMP22, pMP23);
        }
        if (pMP21 > pMP22)
        {
            std::swap(pMP21, pMP22);
        }

        std::tuple<MapPlane*, MapPlane*, MapPlane*> t1 = std::make_tuple(pMP11, pMP12, pMP13);
        std::tuple<MapPlane*, MapPlane*, MapPlane*> t2 = std::make_tuple(pMP21, pMP22, pMP23);

        return t1 == t2;
    }

    Map::Map() : mnMaxKFid(0), mnBigChangeIdx(0), mbIsInertial(false), mbImuInitialized(false), mbIMU_BA1(false), mbIMU_BA2(false) {
    }

    void Map::AddKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);
        if (pKF->mnId > mnMaxKFid)
            mnMaxKFid = pKF->mnId;
    }

    void Map::AddMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    void Map::SetImuInitialized(bool state)
    {
        unique_lock<mutex> lock(mMutexMap);
        mbImuInitialized = state;
        
        if (state == false){
            mbIMU_BA1 = false;
            mbIMU_BA2 = false;
        }
    }

    bool Map::isImuInitialized()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mbImuInitialized;
    }

    void Map::EraseMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.erase(pMP);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::EraseKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapPoints = vpMPs;
    }

    void Map::InformNewBigChange() {
        unique_lock<mutex> lock(mMutexMap);
        mnBigChangeIdx++;
    }

    int Map::GetLastBigChangeIdx() {
        unique_lock<mutex> lock(mMutexMap);
        return mnBigChangeIdx;
    }

    vector<KeyFrame *> Map::GetAllKeyFrames() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    vector<MapPoint *> Map::GetAllMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
    }

    long unsigned int Map::MapPointsInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPoints.size();
    }

    long unsigned int Map::KeyFramesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspKeyFrames.size();
    }

    vector<MapPoint *> Map::GetReferenceMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapPoints;
    }

    long unsigned int Map::GetMaxKFid() {
        unique_lock<mutex> lock(mMutexMap);
        return mnMaxKFid;
    }

    void Map::clear() {
        for (auto mspMapPoint : mspMapPoints)
            delete mspMapPoint;
        for (auto mspMapLine : mspMapLines)
            delete mspMapLine;
        for (auto mspMapPlane : mspMapPlanes)
            delete mspMapPlane;

        for (auto mspKeyFrame : mspKeyFrames)
            delete mspKeyFrame;

        mspMapPlanes.clear();
        mspMapPoints.clear();
        mspKeyFrames.clear();
        mspMapLines.clear();
        mnMaxKFid = 0;
        mvpReferenceMapPoints.clear();
        mvpReferenceMapLines.clear();
        mvpKeyFrameOrigins.clear();
        mbImuInitialized = false;
        mbIMU_BA1 = false;
        mbIMU_BA2 = false;
    }

    void Map::RotateMap(const cv::Mat &R)
    {
        unique_lock<mutex> lock(mMutexMap);

        cv::Mat Txw = cv::Mat::eye(4,4,CV_32F);
        R.copyTo(Txw.rowRange(0,3).colRange(0,3));

        KeyFrame* pKFini = mvpKeyFrameOrigins[0];
        cv::Mat Twc_0 = pKFini->GetPoseInverse();
        cv::Mat Txc_0 = Txw*Twc_0;
        cv::Mat Txb_0 = Txc_0*pKFini->mImuCalib.Tcb;
        cv::Mat Tyx = cv::Mat::eye(4,4,CV_32F);
        Tyx.rowRange(0,3).col(3) = -Txb_0.rowRange(0,3).col(3);
        cv::Mat Tyw = Tyx*Txw;
        cv::Mat Ryw = Tyw.rowRange(0,3).colRange(0,3);
        cv::Mat tyw = Tyw.rowRange(0,3).col(3);

        for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(); sit!=mspKeyFrames.end(); sit++)
        {
            KeyFrame* pKF = *sit;
            cv::Mat Twc = pKF->GetPoseInverse();
            cv::Mat Tyc = Tyw*Twc;
            cv::Mat Tcy = cv::Mat::eye(4,4,CV_32F);
            Tcy.rowRange(0,3).colRange(0,3) = Tyc.rowRange(0,3).colRange(0,3).t();
            Tcy.rowRange(0,3).col(3) = -Tcy.rowRange(0,3).colRange(0,3)*Tyc.rowRange(0,3).col(3);
            pKF->SetPose(Tcy);
            cv::Mat Vw = pKF->GetVelocity();
            pKF->SetVelocity(Ryw*Vw);
        }
        for(set<MapPoint*>::iterator sit=mspMapPoints.begin(); sit!=mspMapPoints.end(); sit++)
        {
            MapPoint* pMP = *sit;
            pMP->SetWorldPos(Ryw*pMP->GetWorldPos()+tyw);
            pMP->UpdateNormalAndDepth();
        }
    }

    void Map::ApplyScaledRotation(const cv::Mat &R, const float s, Frame& F, const bool bScaledVel)
    {
        unique_lock<mutex> lock(mMutexMap);

        // Body position (IMU) of first keyframe is fixed to (0,0,0)
        cv::Mat Txw = cv::Mat::eye(4,4,CV_32F);
        R.copyTo(Txw.rowRange(0,3).colRange(0,3));

        cv::Mat Tyx = cv::Mat::eye(4,4,CV_32F);

        cv::Mat Tyw = Tyx*Txw;
        // Tyw.rowRange(0,3).col(3) = Tyw.rowRange(0,3).col(3)+t;
        cv::Mat Ryw = Tyw.rowRange(0,3).colRange(0,3);
        // cv::Mat tyw = Tyw.rowRange(0,3).col(3);

        cv::Mat Twc = F.mTcw.clone().t();
        Twc.rowRange(0,3).col(3)*=s;
        cv::Mat Tyc = Tyw*Twc;
        cv::Mat Tcy = cv::Mat::eye(4,4,CV_32F);
        Tcy.rowRange(0,3).colRange(0,3) = Tyc.rowRange(0,3).colRange(0,3).t();
        Tcy.rowRange(0,3).col(3) = -Tcy.rowRange(0,3).colRange(0,3)*Tyc.rowRange(0,3).col(3);
        F.SetPose(Tcy);
        // cv::Mat Vw = F.mVw;
        // cout << Vw.size() << endl;
        // cout << Vw.t() << endl;
        // if(!bScaledVel)
        //     F.SetVelocity(Ryw*Vw);
        // else
        //     F.SetVelocity(Ryw*Vw*s);

        for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(); sit!=mspKeyFrames.end(); sit++)
        {
            KeyFrame* pKF = *sit;
            cv::Mat Twc = pKF->GetPoseInverse();
            Twc.rowRange(0,3).col(3)*=s;
            cv::Mat Tyc = Tyw*Twc;
            cv::Mat Tcy = cv::Mat::eye(4,4,CV_32F);
            Tcy.rowRange(0,3).colRange(0,3) = Tyc.rowRange(0,3).colRange(0,3).t();
            Tcy.rowRange(0,3).col(3) = -Tcy.rowRange(0,3).colRange(0,3)*Tyc.rowRange(0,3).col(3);
            pKF->SetPose(Tcy);
            cv::Mat Vw = pKF->GetVelocity();
            if(!bScaledVel)
                pKF->SetVelocity(Ryw*Vw);
            else
                pKF->SetVelocity(Ryw*Vw*s);

        }
        for(set<MapPoint*>::iterator sit=mspMapPoints.begin(); sit!=mspMapPoints.end(); sit++)
        {
            MapPoint* pMP = *sit;
            pMP->SetWorldPos(s*Ryw*pMP->GetWorldPos());
            pMP->UpdateNormalAndDepth();
        }
        for(set<MapLine*>::iterator sit=mspMapLines.begin(); sit!=mspMapLines.end(); sit++)
        {
            MapLine* pML = *sit;

            Eigen::Matrix<double,6,1> pos = pML->GetWorldPos();
            Eigen::Vector3d sp = pos.head<3>();
            Eigen::Vector3d ep = pos.tail<3>();
            cv::Mat spm,epm;
            eigen2cv(sp,spm);
            eigen2cv(ep,epm);
            spm.convertTo(spm, CV_32F);
            epm.convertTo(epm, CV_32F);
            spm=s*Ryw*spm;
            epm=s*Ryw*epm;
            cv2eigen(spm,sp);
            cv2eigen(epm,ep);
            pos.head<3>()=sp;
            pos.tail<3>()=ep;
            pML->SetWorldPos(pos);
            pML->UpdateAverageDir();
        }
        for(set<MapPlane*>::iterator sit=mspMapPlanes.begin(); sit!=mspMapPlanes.end(); sit++)
        {
            MapPlane* pMP = *sit;
            cv::Mat pos4=pMP->GetWorldPos();
            cv::Mat pos3=s*Ryw*pos4.rowRange(0,3);
            pos3.copyTo(pos4.rowRange(0,3));
            pMP->SetWorldPos(pos4);
            pMP->UpdateCoefficientsAndPoints();
        }
        mnMapChange++;
    }

    void Map::SetInertialSensor()
    {
        unique_lock<mutex> lock(mMutexMap);
        mbIsInertial = true;
    }

    bool Map::IsInertial()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mbIsInertial;
    }

    void Map::SetIniertialBA1()
    {
        unique_lock<mutex> lock(mMutexMap);
        mbIMU_BA1 = true;
    }

    void Map::SetIniertialBA2()
    {
        unique_lock<mutex> lock(mMutexMap);
        mbIMU_BA2 = true;
    }

    bool Map::GetIniertialBA1()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mbIMU_BA1;
    }

    bool Map::GetIniertialBA2()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mbIMU_BA2;
    }

    int Map::GetMapChangeIndex()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mnMapChange;
    }

    void Map::IncreaseChangeIndex()
    {
        unique_lock<mutex> lock(mMutexMap);
        mnMapChange++;
    }

    int Map::GetLastMapChange()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mnMapChangeNotified;
    }

    void Map::SetLastMapChange(int currentChangeId)
    {
        unique_lock<mutex> lock(mMutexMap);
        mnMapChangeNotified = currentChangeId;
    }

    void Map::AddMapLine(MapLine *pML) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.insert(pML);
    }

    void Map::EraseMapLine(MapLine *pML) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.erase(pML);
    }

    /**
     * @brief 设置参考MapLines，将用于DrawMapLines函数画图
     * @param vpMLs Local MapLines
     */
    void Map::SetReferenceMapLines(const std::vector<MapLine *> &vpMLs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapLines = vpMLs;
    }

    vector<MapLine *> Map::GetAllMapLines() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapLine *>(mspMapLines.begin(), mspMapLines.end());
    }

    vector<MapLine *> Map::GetReferenceMapLines() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapLines;
    }

    long unsigned int Map::MapLinesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapLines.size();
    }

    void Map::AddMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.insert(pMP);
    }

    void Map::EraseMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.erase(pMP);
    }

    vector<MapPlane *> Map::GetAllMapPlanes() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPlane *>(mspMapPlanes.begin(), mspMapPlanes.end());
    }

    long unsigned int Map::MapPlanesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPlanes.size();
    }

    void Map::FlagMatchedPlanePoints(ORB_SLAM2::Frame &pF, const float &dTh) {

        unique_lock<mutex> lock(mMutexMap);
        int nMatches = 0;

        for (int i = 0; i < pF.mnPlaneNum; ++i) {

            cv::Mat pM = pF.ComputePlaneWorldCoeff(i);

            if (pF.mvpMapPlanes[i]) {
                for (auto mapPoint : mspMapPoints) {
                    cv::Mat pW = mapPoint->GetWorldPos();

                    double dis = abs(pM.at<float>(0, 0) * pW.at<float>(0, 0) +
                                     pM.at<float>(1, 0) * pW.at<float>(1, 0) +
                                     pM.at<float>(2, 0) * pW.at<float>(2, 0) +
                                     pM.at<float>(3, 0));

                    if (dis < 0.5) {
                        mapPoint->SetAssociatedWithPlaneFlag(true);
                        nMatches++;
                    }
                }
            }
        }

//        cout << "Point matches: " << nMatches << endl;
    }

    void Map::AddManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3, KeyFrame* pKF) {
        unique_lock<mutex> lock(mMutexMap);

        Manhattan manhattan = std::make_tuple(pMP1, pMP2, pMP3);
        if (mmpManhattanObservations.count(manhattan) != 0)
            return;
//        cout << "Insert Manhattan3 pMP1: " << pMP1->mnId << endl;
//        cout << "Insert Manhattan3 pMP2: " << pMP2->mnId << endl;
//        cout << "Insert Manhattan3 pMP3: " << pMP3->mnId << endl;
        pKF->SetNotErase();
        mmpManhattanObservations[manhattan] = pKF;
    }

    KeyFrame* Map::GetManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3) {
        unique_lock<mutex> lock(mMutexMap);
        Manhattan manhattan = std::make_tuple(pMP1, pMP2, pMP3);
        if (mmpManhattanObservations.count(manhattan)) {
            return mmpManhattanObservations[manhattan];
        } else {
            return static_cast<KeyFrame*>(nullptr);
        }
    }

    Map::Manhattans Map::GetAllManhattanObservations() {
        return mmpManhattanObservations;
    }

    void Map::AddPartialManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, KeyFrame* pKF) {
        unique_lock<mutex> lock(mMutexMap);

        PartialManhattan manhattan = std::make_pair(pMP1, pMP2);
//        cout << "Insert Manhattan2 pMP1: " << pMP1->mnId << endl;
//        cout << "Insert Manhattan2 pMP2: " << pMP2->mnId << endl;
        if (mmpPartialManhattanObservations.count(manhattan) != 0)
            return;
        pKF->SetNotErase();
        mmpPartialManhattanObservations[manhattan] = pKF;
    }

    KeyFrame* Map::GetPartialManhattanObservation(MapPlane *pMP1, MapPlane *pMP2) {
        unique_lock<mutex> lock(mMutexMap);
        PartialManhattan manhattan = std::make_pair(pMP1, pMP2);
        if (mmpPartialManhattanObservations.count(manhattan)) {
            return mmpPartialManhattanObservations[manhattan];
        } else {
            return static_cast<KeyFrame*>(nullptr);
        }
    }

    Map::PartialManhattans Map::GetAllPartialManhattanObservations() {
        return mmpPartialManhattanObservations;
    }

} //namespace ORB_SLAM
