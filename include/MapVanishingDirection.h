/** Yanyan
 *
 * */

#ifndef MAPVANISHINGDIRECTION_H
#define MAPVANISHINGDIRECTION_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "mutex"
namespace ORB_SLAM2 {

    typedef struct VanishingDirection {
        cv::Point3d direction;
        int mnID;

        //std::map<keylines, size_t>;
        VanishingDirection() {}
    } VanishDirection;

    class KeyFrame;

    class Frame;

    class Map;

    class MapVanishingDirection {

    public:
        // initialize
        MapVanishingDirection(const map<int, cv::Point3d> &VD, KeyFrame *pRefKF, Map *pMap);

        // add observation
        void AddObservation(KeyFrame *pKF, int idx);

        void EraseObservation(KeyFrame *pKF);

        std::map<KeyFrame *, size_t> GetObservations();

        KeyFrame* GetAnchorKeyFrame();

        void SetBadFlag();

        bool isBad();

        MapVanishingDirection* GetReplaced();

        bool IsInKeyFrame(KeyFrame *pKF);

        void Replace(MapVanishingDirection *pMVD);

        void IncreaseFound(int n=1);

        void IncreaseVisible(int n);




        KeyFrame *GetReferenceKeyFrame();


        void UpdateVanishingDirection(KeyFrame *pKF, int id);

        int Observations();

        std::map<int, cv::Point3d> GetMapVD();

        cv::Point3d GetMapVD_direction();



    public:
        long unsigned int mnId; ///< Global ID for Map vanishing direction;
        static long unsigned int nNextId;
        long int mnFirstKFid; // anchor keyframe frame
        long int mnFirstFrame; // anchor frame
        int nObs;

        static std::mutex mGlobalMutex;

        long unsigned int mnBALocalForKF; //used in local BA

        long unsigned int mnFuseCandidateForKF;

        // Variables used by loop closing
        long unsigned int mnLoopPlaneForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        cv::Mat mPosGBA;
        long unsigned int mnBAGlobalForKF;

        //used for visualization
        int mRed;
        int mGreen;
        int mBlue;

        //Tracking counters
        int mnVisible;
        int mnFound;

        cv::Point3d direction;
        int idOfLine;

    protected:
        // todo
        std::map<int, cv::Point3d> mVD;

        cv::Mat mWorldPos; ///< Position in absolute coordinates

        std::map<KeyFrame *, size_t> mObservations;


        std::mutex mMutexPos;
        std::mutex mMutexFeatures;

        KeyFrame *mpRefKF;
        KeyFrame *mpAnchorKF; // extensibility map


        bool mbBad;

        // todo not finished
        MapVanishingDirection *mpReplaced; // replace with null

        Map *mpMap;

    };
}
#endif  //MAPVANISHINGDIRECTION_H