#include "MapVanishingDirection.h"

namespace ORB_SLAM2{

    long unsigned int MapVanishingDirection::nNextId = 0;
    mutex MapVanishingDirection::mGlobalMutex;


    MapVanishingDirection::MapVanishingDirection(const map<int, cv::Point3d> &VD, KeyFrame *pRefKF, ORB_SLAM2::Map *pMap) :
            mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), mpRefKF(pRefKF), mnVisible(1), mnFound(1),
            mnBALocalForKF(0), mnBAGlobalForKF(0), mpMap(pMap), nObs(0),
            mbBad(false), mnFuseCandidateForKF(0), mpReplaced(static_cast<MapVanishingDirection*>(NULL)),  mnCorrectedByKF(0),
            mnCorrectedReference(0) {
        mnId = nNextId++;
        mVD = VD;
        map<int, cv::Point3d>::iterator it= mVD.begin();  //
        direction = it->second; // world coordinate
        idOfLine = it->first;

        //cout<<"VD: "<<it->first<<", points:"<<it->second<<endl;

    }

    void MapVanishingDirection::AddObservation(KeyFrame * pKF, int idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        cout<<""<<mObservations.size()<<endl;
        if(mObservations.count(pKF))
            return;
        cout<<"yes"<<endl;
        mObservations[pKF] = idx;
        nObs++;
    }

    MapVanishingDirection* MapVanishingDirection::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapVanishingDirection::Replace(MapVanishingDirection *pMVD)
    {
        if(pMVD->mnId == this->mnId)
            return;

        int nvisible, nfound;
        map<KeyFrame*,size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            obs=mObservations;
            mObservations.clear();
            mbBad=true;
            nvisible = mnVisible;
            nfound = mnFound;
            mpReplaced = pMVD;
        }

        for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
        {
            // Replace measurement in keyframe
            KeyFrame* pKF = mit->first;

            if(!pMVD->IsInKeyFrame(pKF))
            {
                //TODO: waiting for checking
                pKF->ReplaceMapVDMatch(mit->second, pMVD);
                pMVD->AddObservation(pKF,mit->second);
            }
            else
            {
                pKF->EraseMapPointMatch(mit->second);
            }
        }
        pMVD->IncreaseFound(nfound);
        pMVD->IncreaseVisible(nvisible);

        mpMap->EraseMapVD(this);

    }


    void MapVanishingDirection::UpdateVanishingDirection(ORB_SLAM2::KeyFrame *pKF, int id)
    {
        // TODO update the vanishing direction


    }

    void MapVanishingDirection::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound+=n;
    }

    void MapVanishingDirection::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible+=n;
    }

    bool MapVanishingDirection::IsInKeyFrame(KeyFrame * pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }

    void MapVanishingDirection::EraseObservation(KeyFrame* pKF)
    {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if(mObservations.count(pKF))
            {
                mObservations.erase(pKF);
                nObs--;

                if(mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;

                if(nObs<=2)
                    bBad = true;
            }
        }

        if(bBad)
        {
            SetBadFlag();
        }

    }

    KeyFrame* MapVanishingDirection::GetAnchorKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;//mpAnchorKF;
    }

    std::map<int, cv::Point3d> MapVanishingDirection::GetMapVD()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mVD;

    }



    std::map<KeyFrame*, size_t> MapVanishingDirection::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    void MapVanishingDirection::SetBadFlag()
    {
        map<KeyFrame*,size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad=true;
            obs = mObservations;
            mObservations.clear();
        }
        for(auto & ob : obs)
        {
            KeyFrame* pKF = ob.first;
            // todo
            pKF->EraseMapVDMatches(ob.second);
        }

        // todo erase Map vanishing direction
        mpMap->EraseMapVD(this);

    }

    KeyFrame* MapVanishingDirection::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    int MapVanishingDirection::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }


    }
