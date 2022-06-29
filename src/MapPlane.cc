//
// Created by fishmarch on 19-5-24.
//


#include "MapPlane.h"

#include<mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2{
    long unsigned int MapPlane::nNextId = 0;
    mutex MapPlane::mGlobalMutex;

    MapPlane::MapPlane(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
            mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), mpRefKF(pRefKF), mnVisible(1), mnFound(1),
            mnBALocalForKF(0), mnBAGlobalForKF(0), mvPlanePoints(new PointCloud()), mpMap(pMap), nObs(0),
            mbBad(false), mnFuseCandidateForKF(0), mpReplaced(static_cast<MapPlane*>(NULL)), mnLoopPlaneForKF(0),
            mnLoopVerticalPlaneForKF(0), mnLoopParallelPlaneForKF(0), mnCorrectedByKF(0),
            mnCorrectedReference(0) {
        mnId = nNextId++;

        Pos.copyTo(mWorldPos);

        rand();
        mRed = rand() % 256;
        mBlue = rand() % 256;
        mGreen = rand() % 256;
    }

    void MapPlane::AddObservation(KeyFrame *pKF, int idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return;
        mObservations[pKF] = idx;
        nObs++;
    }

    void MapPlane::AddVerObservation(KeyFrame *pKF, int idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mVerObservations.count(pKF))
            return;
        mVerObservations[pKF] = idx;
    }

    void MapPlane::AddParObservation(KeyFrame *pKF, int idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mParObservations.count(pKF))
            return;
        mParObservations[pKF] = idx;
    }

    void MapPlane::EraseObservation(KeyFrame *pKF) {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF)) {
                mObservations.erase(pKF);
                nObs--;

                if (mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;

                if (nObs <= 2)
                    bBad = true;
            }
        }

        if (bBad) {
            SetBadFlag();
        }
    }

    void MapPlane::EraseVerObservation(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mVerObservations.count(pKF)){
            mVerObservations.erase(pKF);
        }
    }

    void MapPlane::EraseParObservation(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mParObservations.count(pKF)){
            mParObservations.erase(pKF);
        }
    }

    map<KeyFrame*, size_t> MapPlane::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    map<KeyFrame*, size_t> MapPlane::GetVerObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mVerObservations;
    }

    map<KeyFrame*, size_t> MapPlane::GetParObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mParObservations;
    }

    int MapPlane::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    KeyFrame* MapPlane::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    int MapPlane::GetIndexInKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    int MapPlane::GetIndexInVerticalKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mVerObservations.count(pKF))
            return mVerObservations[pKF];
        else
            return -1;
    }

    int MapPlane::GetIndexInParallelKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mParObservations.count(pKF))
            return mParObservations[pKF];
        else
            return -1;
    }

    bool MapPlane::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }

    bool MapPlane::IsVerticalInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mVerObservations.count(pKF));
    }

    bool MapPlane::IsParallelInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mParObservations.count(pKF));
    }

    void MapPlane::SetWorldPos(const cv::Mat &Pos)
    {
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        Pos.copyTo(mWorldPos);
    }

    cv::Mat MapPlane::GetWorldPos(){
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    bool MapPlane::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }

    void MapPlane::SetBadFlag()
    {
        map<KeyFrame*,size_t> obs, verObs, parObs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad=true;
            obs = mObservations;
            mObservations.clear();
            verObs = mVerObservations;
            mVerObservations.clear();
            parObs = mParObservations;
            mParObservations.clear();
        }
        for(auto & ob : obs)
        {
            KeyFrame* pKF = ob.first;
            pKF->EraseMapPlaneMatch(ob.second);
        }
        for(auto & verOb : verObs)
        {
            KeyFrame* pKF = verOb.first;
            pKF->EraseMapVerticalPlaneMatch(verOb.second);
        }
        for(auto & parOb : parObs)
        {
            KeyFrame* pKF = parOb.first;
            pKF->EraseMapParallelPlaneMatch(parOb.second);
        }

        mpMap->EraseMapPlane(this);
    }

    void MapPlane::Replace(MapPlane* pMP) {
        if(pMP->mnId==this->mnId)
            return;

        int nvisible, nfound;
        map<KeyFrame*,size_t> obs, verObs, parObs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad=true;
            obs = mObservations;
            mObservations.clear();
            verObs = mVerObservations;
            mVerObservations.clear();
            parObs = mParObservations;
            mParObservations.clear();
            nvisible = mnVisible;
            nfound = mnFound;
            mpReplaced = pMP;
        }

        for(auto & ob : obs)
        {
            // Replace measurement in keyframe
            KeyFrame* pKF = ob.first;

            *pMP->mvPlanePoints += pKF->mvPlanePoints[ob.second];

            if(!pMP->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapPlaneMatch(ob.second, pMP);
                pMP->AddObservation(pKF,ob.second);
            }
            else
            {
                pKF->EraseMapPlaneMatch(ob.second);
            }
        }
        for(auto & ob : verObs)
        {
            // Replace measurement in keyframe
            KeyFrame* pKF = ob.first;

            if(!pMP->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapVerticalPlaneMatch(ob.second, pMP);
                pMP->AddVerObservation(pKF,ob.second);
            }
            else
            {
                pKF->EraseMapVerticalPlaneMatch(ob.second);
            }
        }
        for(auto & ob : parObs)
        {
            // Replace measurement in keyframe
            KeyFrame* pKF = ob.first;

            if(!pMP->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapParallelPlaneMatch(ob.second, pMP);
                pMP->AddParObservation(pKF,ob.second);
            }
            else
            {
                pKF->EraseMapParallelPlaneMatch(ob.second);
            }
        }

        pMP->IncreaseFound(nfound);
        pMP->IncreaseVisible(nvisible);
        pMP->UpdateCoefficientsAndPoints();

        mpMap->EraseMapPlane(this);
    }

    void MapPlane::ReplaceVerticalObservations(MapPlane* pMP) {
        if(pMP->mnId==this->mnId)
            return;

        map<KeyFrame*,size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            obs = mVerObservations;
            mVerObservations.clear();
        }

        for(auto & ob : obs)
        {
            // Replace measurement in keyframe
            KeyFrame* pKF = ob.first;

            if(!pMP->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapVerticalPlaneMatch(ob.second, pMP);
                pMP->AddVerObservation(pKF,ob.second);
            }
            else
            {
                pKF->mvPlanePoints[pMP->GetIndexInVerticalKeyFrame(pKF)] += *(pKF->GetMapVerticalPlane(ob.second)->mvPlanePoints);
                pKF->EraseMapVerticalPlaneMatch(ob.second);
            }
        }
    }

    void MapPlane::ReplaceParallelObservations(MapPlane* pMP) {
        if(pMP->mnId==this->mnId)
            return;

        map<KeyFrame*,size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            obs = mParObservations;
            mParObservations.clear();
        }

        for(auto & ob : obs)
        {
            // Replace measurement in keyframe
            KeyFrame* pKF = ob.first;

            if(!pMP->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapParallelPlaneMatch(ob.second, pMP);
                pMP->AddParObservation(pKF,ob.second);
            }
            else
            {
                pKF->mvPlanePoints[pMP->GetIndexInParallelKeyFrame(pKF)] += *(pKF->GetMapParallelPlane(ob.second)->mvPlanePoints);
                pKF->EraseMapParallelPlaneMatch(ob.second);
            }
        }
    }

    MapPlane* MapPlane::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapPlane::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible+=n;
    }

    void MapPlane::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound+=n;
    }

    float MapPlane::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound)/mnVisible;
    }

    void MapPlane::UpdateCoefficientsAndPoints() {
        PointCloud::Ptr combinedPoints (new PointCloud());
        map<KeyFrame*, size_t> observations = GetObservations();
        for(auto & observation : observations){
            KeyFrame* frame = observation.first;
            int id = observation.second;

            PointCloud::Ptr points (new PointCloud());
            // transform with Twc
            pcl::transformPointCloud(frame->mvPlanePoints[id], *points, Converter::toMatrix4d(frame->GetPoseInverse()));

            *combinedPoints += *points;
        }

        pcl::VoxelGrid<PointT>  voxel;
        voxel.setLeafSize(0.2, 0.2, 0.2);

        PointCloud::Ptr coarseCloud(new PointCloud());
        voxel.setInputCloud(combinedPoints);
        voxel.filter(*coarseCloud);

//        MaxPointDistanceFromPlane(mWorldPos, coarseCloud);

        mvPlanePoints = coarseCloud;

//        if (mvPlanePoints->points.size() > 4) {
//            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//            pcl::SACSegmentation<PointT> seg;
//            seg.setOptimizeCoefficients(true);
//            seg.setModelType(pcl::SACMODEL_PLANE);
//            seg.setMethodType(pcl::SAC_RANSAC);
//            seg.setDistanceThreshold(0.1);
//
//            seg.setInputCloud(mvPlanePoints);
//            seg.segment(*inliers, *coefficients);
//
//            mWorldPos.at<float>(0, 0) = coefficients->values[0];
//            mWorldPos.at<float>(1, 0) = coefficients->values[1];
//            mWorldPos.at<float>(2, 0) = coefficients->values[2];
//            mWorldPos.at<float>(3, 0) = coefficients->values[3];
//
//            PointCloud::Ptr inlierCloud (new PointCloud());
//            std::vector<int> indices = inliers->indices;
//            std::vector<PointT, Eigen::aligned_allocator<PointT>> points = coarseCloud->points;
//            for (auto &i : indices) {
//                inlierCloud->push_back(points[i]);
//            }
//
//            mvPlanePoints = inlierCloud;
//        }
    }

    void MapPlane::UpdateCoefficientsAndPoints(ORB_SLAM2::Frame &pF, int id) {

        PointCloud::Ptr combinedPoints (new PointCloud());

        Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( pF.mTcw );
        pcl::transformPointCloud(pF.mvPlanePoints[id], *combinedPoints, T.inverse().matrix());

        *combinedPoints += *mvPlanePoints;

        pcl::VoxelGrid<PointT>  voxel;
        voxel.setLeafSize(0.2, 0.2, 0.2);

        PointCloud::Ptr coarseCloud(new PointCloud());
        voxel.setInputCloud(combinedPoints);
        voxel.filter(*coarseCloud);

//        MaxPointDistanceFromPlane(mWorldPos, coarseCloud);

        mvPlanePoints = coarseCloud;

//        if (mvPlanePoints->points.size() > 4) {
//            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//            pcl::SACSegmentation<PointT> seg;
//            seg.setOptimizeCoefficients(true);
//            seg.setModelType(pcl::SACMODEL_PLANE);
//            seg.setMethodType(pcl::SAC_RANSAC);
//            seg.setDistanceThreshold(0.1);
//
//            seg.setInputCloud(mvPlanePoints);
//            seg.segment(*inliers, *coefficients);
//
//            mWorldPos.at<float>(0, 0) = coefficients->values[0];
//            mWorldPos.at<float>(1, 0) = coefficients->values[1];
//            mWorldPos.at<float>(2, 0) = coefficients->values[2];
//            mWorldPos.at<float>(3, 0) = coefficients->values[3];
//
//            PointCloud::Ptr inlierCloud (new PointCloud());
//            std::vector<int> indices = inliers->indices;
//            std::vector<PointT, Eigen::aligned_allocator<PointT>> points = coarseCloud->points;
//            for (auto &i : indices) {
//                inlierCloud->push_back(points[i]);
//            }
//
//            mvPlanePoints = inlierCloud;
//        }
    }

    void MapPlane::UpdateCoefficientsAndPoints(KeyFrame *pKF, int id) {

        PointCloud::Ptr combinedPoints (new PointCloud());

        Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( pKF->GetPose() );
        pcl::transformPointCloud(pKF->mvPlanePoints[id], *combinedPoints, T.inverse().matrix());

        *combinedPoints += *mvPlanePoints;

        pcl::VoxelGrid<PointT>  voxel;
        voxel.setLeafSize( 0.2, 0.2, 0.2);

        PointCloud::Ptr coarseCloud(new PointCloud());
        voxel.setInputCloud(combinedPoints);
        voxel.filter(*coarseCloud);

        mvPlanePoints = coarseCloud;

//        if (mvPlanePoints->points.size() > 4) {
//            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//            pcl::SACSegmentation<PointT> seg;
//            seg.setOptimizeCoefficients(true);
//            seg.setModelType(pcl::SACMODEL_PLANE);
//            seg.setMethodType(pcl::SAC_RANSAC);
//            seg.setDistanceThreshold(0.01);
//
//            seg.setInputCloud(mvPlanePoints);
//            seg.segment(*inliers, *coefficients);
//
//            mWorldPos.at<float>(0, 0) = coefficients->values[0];
//            mWorldPos.at<float>(1, 0) = coefficients->values[1];
//            mWorldPos.at<float>(2, 0) = coefficients->values[2];
//            mWorldPos.at<float>(3, 0) = coefficients->values[3];
//        }
    }

    bool MapPlane::MaxPointDistanceFromPlane(cv::Mat &plane, PointCloud::Ptr pointCloud) {
        auto disTh = Config::Get<double>("Plane.DistanceThreshold");
        bool erased = false;
        double max = -1;
        double threshold = 0.005;
        int i = 0;

        PointCloud::Ptr tempCloud (new PointCloud(*pointCloud));

        auto &points = tempCloud->points;

        std::cout << "points before: " << points.size() << std::endl;
        for (auto &p : points) {
            double dis = abs(plane.at<float>(0) * p.x +
                             plane.at<float>(1) * p.y +
                             plane.at<float>(2) * p.z +
                             plane.at<float>(3));
//            if (dis > disTh)
//                return false;

            if (dis > threshold) {
//                p.r = 0;
//                p.g = 0;
//                p.b = 0;
                points.erase(points.begin() + i);
                erased = true;
                continue;
            } else {
                p.r = mRed;
                p.g = mGreen;
                p.b = mBlue;
            }

            i++;
        }

        std::cout << "points after: " << points.size() << std::endl;

//        if (erased) {
        if (points.size() < 3) {
            return false;
        }
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<PointT> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(threshold);

        seg.setInputCloud(tempCloud);
        seg.segment(*inliers, *coefficients);

        float oldVal = plane.at<float>(3);
        float newVal = coefficients->values[3];

        cv::Mat oldPlane = plane.clone();

        std::cout << "old plane: " << plane.at<float>(0) << " "
                  << plane.at<float>(1) << " "
                  << plane.at<float>(2) << " "
                  << plane.at<float>(3) << std::endl;

        std::cout << "new plane: " << coefficients->values[0] << " "
                  << coefficients->values[1] << " "
                  << coefficients->values[2] << " "
                  << coefficients->values[3] << std::endl;

        plane.at<float>(0) = coefficients->values[0];
        plane.at<float>(1) = coefficients->values[1];
        plane.at<float>(2) = coefficients->values[2];
        plane.at<float>(3) = coefficients->values[3];

        if ((newVal < 0 && oldVal > 0) || (newVal > 0 && oldVal < 0)) {
            plane = -plane;
            double dotProduct = plane.dot(oldPlane) / sqrt(plane.dot(plane) * oldPlane.dot(oldPlane));
            std::cout << "Flipped plane: " << plane.t() << std::endl;
            std::cout << "Flip plane: " << dotProduct << std::endl;
        }
//        }

        return true;
    }
}