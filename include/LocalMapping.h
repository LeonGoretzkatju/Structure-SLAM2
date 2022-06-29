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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

    class Tracking;
    class LoopClosing;
    class Map;

    class LocalMapping
    {
    public:
        typedef std::pair<MapPlane*, MapPlane*> Manhattan;
        typedef std::unordered_map<Manhattan, KeyFrame*, ManhattanMapHash, ManhattanMapEqual> Manhattans;

        LocalMapping(Map* pMap, const float bMonocular, bool bInertial);

        void SetLoopCloser(LoopClosing* pLoopCloser);

        void SetTracker(Tracking* pTracker);

        // Main function
        void Run();

        void InsertKeyFrame(KeyFrame* pKF);

        // Thread Synch
        void RequestStop();
        void RequestReset();
        bool Stop();
        void Release();
        bool isStopped();
        bool stopRequested();
        bool AcceptKeyFrames();
        void SetAcceptKeyFrames(bool flag);
        bool SetNotStop(bool flag);

        void InterruptBA();

        void RequestFinish();
        bool isFinished();

        int KeyframesInQueue(){
            unique_lock<std::mutex> lock(mMutexNewKFs);
            return mlNewKeyFrames.size();
        }

        std::mutex mMutexImuInit;

        bool IsInitializing();

        Eigen::MatrixXd mcovInertial;
        Eigen::Matrix3d mRwg;
        Eigen::Vector3d mbg;
        Eigen::Vector3d mba;
        double mScale;
        double mInitTime;
        double mCostTime;
        bool mbNewInit;
        unsigned int mInitSect;
        unsigned int mIdxInit;
        unsigned int mnKFs;
        double mFirstTs;
        int mnMatchesInliers;

        bool mbBadImu;

    protected:

        bool CheckNewKeyFrames();
        void ProcessNewKeyFrame();
        void CreateNewMapPoints();
        void CreateNewMapLines1();
        void CreateNewMapLines2();

        void MapPointCulling();
        void SearchInNeighbors();

        void MapLineCulling();
        void MapPlaneCulling();

        void KeyFrameCulling();

        cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

        cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

        bool mbMonocular;
        bool mbInertial;

        void ResetIfRequested();
        bool mbResetRequested;
        std::mutex mMutexReset;

        bool CheckFinish();
        void SetFinish();
        bool mbFinishRequested;
        bool mbFinished;
        std::mutex mMutexFinish;

        Map* mpMap;

        LoopClosing* mpLoopCloser;
        Tracking* mpTracker;

        std::list<KeyFrame*> mlNewKeyFrames;

        KeyFrame* mpCurrentKeyFrame;

        std::list<MapPoint*> mlpRecentAddedMapPoints;

        std::list<MapLine*> mlpRecentAddedMapLines;

        std::list<MapPlane*> mlpRecentAddedMapPlanes;

        std::mutex mMutexNewKFs;

        bool mbAbortBA;

        bool mbStopped;
        bool mbStopRequested;
        bool mbNotStop;
        std::mutex mMutexStop;

        bool mbAcceptKeyFrames;
        std::mutex mMutexAccept;

        void InitializeIMU(float priorG = 1e2, float priorA = 1e6, bool bFirst = false);
        // void ScaleRefinement();

        bool bInitializing;

        Eigen::MatrixXd infoInertial;

        float mTinit;
    };

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
