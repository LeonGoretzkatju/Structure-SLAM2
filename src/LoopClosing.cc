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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2 {

    LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale) :
            mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
            mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0),
            mbRunningGBA(false), mbFinishedGBA(true),
            mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0) {
        mnCovisibilityConsistencyTh = 3;
    }

    void LoopClosing::SetTracker(Tracking *pTracker) {
        mpTracker = pTracker;
    }

    void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper) {
        mpLocalMapper = pLocalMapper;
    }


    void LoopClosing::Run() {
        mbFinished = false;

        while (1) {
            // Check if there are keyframes in the queue
//            if (CheckNewKeyFrames()) {
//                // Detect loop candidates and check covisibility consistency
//                if (DetectLoop()) {
//                    // Compute similarity transformation [sR|t]
//                    // In the stereo/RGBD case s=1
//                    if (ComputeSim3()) {
//                        // Perform loop fusion and pose graph optimization
//                        CorrectLoop();
//                    }
//                }
//            }

            ResetIfRequested();

            if (CheckFinish())
                break;

            usleep(5000);
        }

        SetFinish();
    }

    void LoopClosing::InsertKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexLoopQueue);
        if (pKF->mnId != 0)
            mlpLoopKeyFrameQueue.push_back(pKF);
    }

    bool LoopClosing::CheckNewKeyFrames() {
        unique_lock<mutex> lock(mMutexLoopQueue);
        return (!mlpLoopKeyFrameQueue.empty());
    }

    bool LoopClosing::DetectLoop() {
        {
            unique_lock<mutex> lock(mMutexLoopQueue);
            mpCurrentKF = mlpLoopKeyFrameQueue.front();
            mlpLoopKeyFrameQueue.pop_front();
            // Avoid that a keyframe can be erased while it is being process by this thread
            mpCurrentKF->SetNotErase();
        }

        //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
        if (mpCurrentKF->mnId < mLastLoopKFid + 10) {
            mpKeyFrameDB->add(mpCurrentKF);
            mpCurrentKF->SetErase();
            return false;
        }

        // Compute reference BoW similarity score
        // This is the lowest score to a connected keyframe in the covisibility graph
        // We will impose loop candidates to have a higher similarity than this
        const vector<KeyFrame *> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
        const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
        float minScore = 1;
        for (size_t i = 0; i < vpConnectedKeyFrames.size(); i++) {
            KeyFrame *pKF = vpConnectedKeyFrames[i];
            if (pKF->isBad())
                continue;
            const DBoW2::BowVector &BowVec = pKF->mBowVec;

            float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

            if (score < minScore)
                minScore = score;
        }

        // Query the database imposing the minimum score
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

        // If there are no loop candidates, just add new keyframe and return false
        if (vpCandidateKFs.empty()) {
            mpKeyFrameDB->add(mpCurrentKF);
            mvConsistentGroups.clear();
            mpCurrentKF->SetErase();
            return false;
        }


        // For each loop candidate check consistency with previous loop candidates
        // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
        // A group is consistent with a previous group if they share at least a keyframe
        // We must detect a consistent loop in several consecutive keyframes to accept it
        mvpEnoughConsistentCandidates.clear();

        vector<ConsistentGroup> vCurrentConsistentGroups;
        vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);
        for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++) {
            KeyFrame *pCandidateKF = vpCandidateKFs[i];

            set<KeyFrame *> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
            spCandidateGroup.insert(pCandidateKF);

            bool bEnoughConsistent = false;
            bool bConsistentForSomeGroup = false;
            for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++) {
                set<KeyFrame *> sPreviousGroup = mvConsistentGroups[iG].first;

                bool bConsistent = false;
                for (set<KeyFrame *>::iterator sit = spCandidateGroup.begin(), send = spCandidateGroup.end();
                     sit != send; sit++) {
                    if (sPreviousGroup.count(*sit)) {
                        bConsistent = true;
                        bConsistentForSomeGroup = true;
                        break;
                    }
                }

                if (bConsistent) {
                    int nPreviousConsistency = mvConsistentGroups[iG].second;
                    int nCurrentConsistency = nPreviousConsistency + 1;
                    if (!vbConsistentGroup[iG]) {
                        ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
                        vCurrentConsistentGroups.push_back(cg);
                        vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
                    }
                    if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent) {
                        mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                        bEnoughConsistent = true; //this avoid to insert the same candidate more than once
                    }
                }
            }

            // If the group is not consistent with any previous group insert with consistency counter set to zero
            if (!bConsistentForSomeGroup) {
                ConsistentGroup cg = make_pair(spCandidateGroup, 0);
                vCurrentConsistentGroups.push_back(cg);
            }
        }

        // Update Covisibility Consistent Groups
        mvConsistentGroups = vCurrentConsistentGroups;


        // Add Current Keyframe to database
        mpKeyFrameDB->add(mpCurrentKF);

        if (mvpEnoughConsistentCandidates.empty()) {
            mpCurrentKF->SetErase();
            return false;
        } else {
            return true;
        }

        mpCurrentKF->SetErase();
        return false;
    }

    bool LoopClosing::ComputeSim3() {
        // For each consistent loop candidate we try to compute a Sim3
        cout << "loop computeSim3" << endl;
        const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

        // We compute first ORB matches for each candidate
        // If enough matches are found, we setup a Sim3Solver
        ORBmatcher matcher(0.75, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(Config::Get<double>("Plane.AssociationDisRef"), 0.985,
                              Config::Get<double>("Plane.VerticalThreshold"),
                              Config::Get<double>("Plane.ParallelThreshold"));

        vector<Sim3Solver *> vpSim3Solvers;
        vpSim3Solvers.resize(nInitialCandidates);

        vector<vector<MapPoint *>> vvpMapPointMatches;
        vvpMapPointMatches.resize(nInitialCandidates);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nInitialCandidates);

        int nCandidates = 0; //candidates with enough matches

        for (int i = 0; i < nInitialCandidates; i++) {
            KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

            // avoid that local mapping erase it while it is being processed in this thread
            pKF->SetNotErase();

            if (pKF->isBad()) {
                vbDiscarded[i] = true;
                continue;
            }

            int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, vvpMapPointMatches[i]);

            if (nmatches < 20) {
                vbDiscarded[i] = true;
                continue;
            } else {
                Sim3Solver *pSolver = new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], mbFixScale);
                pSolver->SetRansacParameters(0.99, 20, 300);
                vpSim3Solvers[i] = pSolver;
            }

            nCandidates++;
        }

        bool bMatch = false;

        // Perform alternatively RANSAC iterations for each candidate
        // until one is succesful or all fail
        while (nCandidates > 0 && !bMatch) {
            for (int i = 0; i < nInitialCandidates; i++) {
                if (vbDiscarded[i])
                    continue;

                KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                Sim3Solver *pSolver = vpSim3Solvers[i];
                cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
                if (!Scm.empty()) {
                    vector<MapPoint *>
                            vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint *>(NULL));
                    for (size_t j = 0, jend = vbInliers.size(); j < jend; j++) {
                        if (vbInliers[j])
                            vpMapPointMatches[j] = vvpMapPointMatches[i][j];
                    }

                    cv::Mat R = pSolver->GetEstimatedRotation();
                    cv::Mat t = pSolver->GetEstimatedTranslation();
                    const float s = pSolver->GetEstimatedScale();
                    matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, s, R, t, 7.5);

                    vector<MapLine *> vpMapLineMatches;

                    int nlmatches = lmatcher.SearchByDescriptor(mpCurrentKF, pKF, vpMapLineMatches);
                    std::cout << "Line matches: " << nlmatches << std::endl;
                    int linesFound = lmatcher.SearchBySim3(mpCurrentKF, pKF, vpMapLineMatches, s, R, t, 7.5);
                    std::cout << "Line SearchBySim3 new: " << linesFound << ", total: "
                              << vpMapLineMatches.size() << std::endl;

                    vector<MapPlane *> vpMapPlaneMatches, vpMapVerticalPlaneMatches, vpMapParallelPlaneMatches;

                    int planesFound = pmatcher.SearchBySim3(mpCurrentKF, pKF, vpMapPlaneMatches,
                                                            vpMapVerticalPlaneMatches, vpMapParallelPlaneMatches,
                                                            s, R, t);
                    std::cout << "Plane SearchBySim3: Matches: " << planesFound
                              << ", direct: " << vpMapPlaneMatches.size()
                              << ", vertical: " << vpMapVerticalPlaneMatches.size()
                              << ", parallel: " << vpMapParallelPlaneMatches.size() << std::endl;

                    g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);

                    std::cout << "gScm: " << gScm << endl;
                    const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches,
                                                                 vpMapLineMatches,
                                                                 vpMapPlaneMatches, vpMapVerticalPlaneMatches,
                                                                 vpMapParallelPlaneMatches,
                                                                 gScm, 10, mbFixScale);

                    std::cout << "gScm: " << gScm << endl;

                    // If optimization is succesful stop ransacs and continue
                    if (nInliers >= 20) {
                        bMatch = true;
                        mpMatchedKF = pKF;
                        g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),
                                       Converter::toVector3d(pKF->GetTranslation()), 1.0);
                        mg2oScw = gScm * gSmw;
                        mScw = Converter::toCvMat(mg2oScw);

                        mvpCurrentMatchedPoints = vpMapPointMatches;
                        mvpCurrentMatchedLines = vpMapLineMatches;
                        mvpCurrentMatchedPlanes = vpMapPlaneMatches;
                        mvpCurrentMatchedVerticalPlanes = vpMapVerticalPlaneMatches;
                        mvpCurrentMatchedParallelPlanes = vpMapParallelPlaneMatches;
                        break;
                    }
                }
            }
        }

        if (!bMatch) {
            for (int i = 0; i < nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            mpCurrentKF->SetErase();
            return false;
        }

        // Retrieve MapPoints seen in Loop Keyframe and neighbors
        vector<KeyFrame *> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
        vpLoopConnectedKFs.push_back(mpMatchedKF);
        mvpLoopMapPoints.clear();
        mvpLoopMapLines.clear();
        mvpLoopMapPlanes.clear();
        mvpLoopMapVerticalPlanes.clear();
        mvpLoopMapParallelPlanes.clear();
        for (auto pKF : vpLoopConnectedKFs) {
            vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();
            for (auto pMP : vpMapPoints) {
                if (pMP) {
                    if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId) {
                        mvpLoopMapPoints.push_back(pMP);
                        pMP->mnLoopPointForKF = mpCurrentKF->mnId;
                    }
                }
            }

            vector<MapLine *> vpMapLines = pKF->GetMapLineMatches();
            for (auto pML : vpMapLines) {
                if (pML) {
                    if (!pML->isBad() && pML->mnLoopLineForKF != mpCurrentKF->mnId) {
                        mvpLoopMapLines.push_back(pML);
                        pML->mnLoopLineForKF = mpCurrentKF->mnId;
                    }
                }
            }

            vector<MapPlane *> vpMapPlanes = pKF->GetMapPlaneMatches();
            for (auto pMP : vpMapPlanes) {
                if (pMP) {
                    if (!pMP->isBad() && pMP->mnLoopPlaneForKF != mpCurrentKF->mnId) {
                        mvpLoopMapPlanes.push_back(pMP);
                        pMP->mnLoopPlaneForKF = mpCurrentKF->mnId;
                    }
                }
            }

            vector<MapPlane *> vpMapVerticalPlanes = pKF->GetMapVerticalPlaneMatches();
            for (auto pMP : vpMapVerticalPlanes) {
                if (pMP) {
                    if (!pMP->isBad() && pMP->mnLoopVerticalPlaneForKF != mpCurrentKF->mnId) {
                        mvpLoopMapVerticalPlanes.push_back(pMP);
                        pMP->mnLoopVerticalPlaneForKF = mpCurrentKF->mnId;
                    }
                }
            }

            vector<MapPlane *> vpMapParallelPlanes = pKF->GetMapParallelPlaneMatches();
            for (auto pMP : vpMapParallelPlanes) {
                if (pMP) {
                    if (!pMP->isBad() && pMP->mnLoopParallelPlaneForKF != mpCurrentKF->mnId) {
                        mvpLoopMapParallelPlanes.push_back(pMP);
                        pMP->mnLoopParallelPlaneForKF = mpCurrentKF->mnId;
                    }
                }
            }
        }

        // Find more matches projecting with the computed Sim3
        matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints, 10);
        int lmatches = lmatcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapLines, mvpCurrentMatchedLines, 10);
        std::cout << "Line matches (Neighbours): " << lmatches << ", total: " << mvpCurrentMatchedLines.size()
                  << std::endl;
        int pmatches = pmatcher.SearchByCoefficients(mpCurrentKF, mScw, mvpLoopMapPlanes, mvpLoopMapVerticalPlanes,
                                                     mvpLoopMapParallelPlanes, mvpCurrentMatchedPlanes,
                                                     mvpCurrentMatchedVerticalPlanes, mvpCurrentMatchedParallelPlanes);
        std::cout << "Plane matches (Neighbours): " << pmatches << ", total: " << mvpCurrentMatchedPlanes.size()
                  << std::endl;

        // If enough matches accept Loop
        int nTotalMatches = 0;
        for (auto &mvpCurrentMatchedPoint : mvpCurrentMatchedPoints) {
            if (mvpCurrentMatchedPoint)
                nTotalMatches++;
        }
        for (auto &mvpCurrentMatchedLine : mvpCurrentMatchedLines) {
            if (mvpCurrentMatchedLine)
                nTotalMatches++;
        }
        for (auto &mvpCurrentMatchedPlane : mvpCurrentMatchedPlanes) {
            if (mvpCurrentMatchedPlane)
                nTotalMatches++;
        }

        if (nTotalMatches >= 40) {
            for (int i = 0; i < nInitialCandidates; i++)
                if (mvpEnoughConsistentCandidates[i] != mpMatchedKF)
                    mvpEnoughConsistentCandidates[i]->SetErase();
            return true;
        } else {
            for (int i = 0; i < nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            mpCurrentKF->SetErase();
            return false;
        }

    }

    void LoopClosing::CorrectLoop() {
        cout << "Loop detected!" << endl;

        // Send a stop signal to Local Mapping
        // Avoid new keyframes are inserted while correcting the loop
        mpLocalMapper->RequestStop();

        // If a Global Bundle Adjustment is running, abort it
        if (isRunningGBA()) {
            unique_lock<mutex> lock(mMutexGBA);
            mbStopGBA = true;

            mnFullBAIdx++;

            if (mpThreadGBA) {
                mpThreadGBA->detach();
                delete mpThreadGBA;
            }
        }

        // Wait until Local Mapping has effectively stopped
        while (!mpLocalMapper->isStopped()) {
            usleep(1000);
        }

        // Ensure current keyframe is updated
        mpCurrentKF->UpdateConnections();

        // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
        mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
        mvpCurrentConnectedKFs.push_back(mpCurrentKF);

        KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
        CorrectedSim3[mpCurrentKF] = mg2oScw;
        cv::Mat Twc = mpCurrentKF->GetPoseInverse();


        {
            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            for (vector<KeyFrame *>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end();
                 vit != vend; vit++) {
                KeyFrame *pKFi = *vit;

                cv::Mat Tiw = pKFi->GetPose();

                if (pKFi != mpCurrentKF) {
                    cv::Mat Tic = Tiw * Twc;
                    cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
                    cv::Mat tic = Tic.rowRange(0, 3).col(3);
                    g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
                    g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;
                    //Pose corrected with the Sim3 of the loop closure
                    CorrectedSim3[pKFi] = g2oCorrectedSiw;
                }

                cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
                cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
                g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
                //Pose without correction
                NonCorrectedSim3[pKFi] = g2oSiw;
            }

            // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
            for (auto &mit : CorrectedSim3) {
                KeyFrame *pKFi = mit.first;
                g2o::Sim3 g2oCorrectedSiw = mit.second;
                g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

                g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

                vector<MapPoint *> vpMPsi = pKFi->GetMapPointMatches();
                for (auto pMPi : vpMPsi) {
                    if (!pMPi)
                        continue;
                    if (pMPi->isBad())
                        continue;
                    if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                        continue;

                    // Project with non-corrected pose and project back with corrected pose
                    cv::Mat P3Dw = pMPi->GetWorldPos();
                    Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
                    Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                    pMPi->SetWorldPos(cvCorrectedP3Dw);
                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMPi->mnCorrectedReference = pKFi->mnId;
                    pMPi->UpdateNormalAndDepth();
                }

                vector<MapLine *> vpMLsi = pKFi->GetMapLineMatches();
                for (auto pMLi : vpMLsi) {
                    if (!pMLi)
                        continue;
                    if (pMLi->isBad())
                        continue;
                    if (pMLi->mnCorrectedByKF == mpCurrentKF->mnId)
                        continue;

                    // Project with non-corrected pose and project back with corrected pose
                    Eigen::Vector3d eigSP3Dw = pMLi->mWorldPos.head(3);
                    Eigen::Vector3d eigEP3Dw = pMLi->mWorldPos.tail(3);

                    Eigen::Matrix<double, 3, 1> eigCorrectedSP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigSP3Dw));
                    Eigen::Matrix<double, 3, 1> eigCorrectedEP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigEP3Dw));

                    Vector6d linePos;
                    linePos << eigCorrectedSP3Dw(0), eigCorrectedSP3Dw(1), eigCorrectedSP3Dw(2), eigCorrectedEP3Dw(
                            0), eigCorrectedEP3Dw(1), eigCorrectedEP3Dw(2);
                    pMLi->SetWorldPos(linePos);
                    pMLi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMLi->mnCorrectedReference = pKFi->mnId;
                    pMLi->ComputeDistinctiveDescriptors();
                    pMLi->UpdateAverageDir();
                }

                cv::Mat cvSiw = Converter::toCvMat(g2oSiw);
                cv::Mat cvCorrectedSwi = Converter::toCvMat(g2oCorrectedSwi);

                cv::Mat sRSiw = cvSiw.rowRange(0, 3).colRange(0, 3);
                cv::Mat tSiw = cvSiw.rowRange(0, 3).col(3);

                cv::Mat sRCorrectedSiw = cvCorrectedSwi.rowRange(0, 3).colRange(0, 3);
                cv::Mat tCorrectedSiw = cvCorrectedSwi.rowRange(0, 3).col(3);

                vector<MapPlane *> vpMPLsi = pKFi->GetMapPlaneMatches();
                for (auto pMPi : vpMPLsi) {
                    if (!pMPi)
                        continue;
                    if (pMPi->isBad())
                        continue;
                    if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                        continue;

                    // Project with non-corrected pose and project back with corrected pose
                    cv::Mat P3Dw = pMPi->GetWorldPos();

                    cv::Mat correctedP3Dw = cv::Mat::eye(4, 1, CV_32F);

                    correctedP3Dw.rowRange(0, 3).col(0) = sRSiw * P3Dw.rowRange(0, 3).col(0);
                    correctedP3Dw.at<float>(3, 0) =
                            P3Dw.at<float>(3, 0) - tSiw.dot(correctedP3Dw.rowRange(0, 3).col(0));
                    if (correctedP3Dw.at<float>(3, 0) < 0.0)
                        correctedP3Dw = -correctedP3Dw;

                    correctedP3Dw.rowRange(0, 3).col(0) = sRCorrectedSiw * correctedP3Dw.rowRange(0, 3).col(0);
                    correctedP3Dw.at<float>(3, 0) =
                            correctedP3Dw.at<float>(3, 0) - tCorrectedSiw.dot(correctedP3Dw.rowRange(0, 3).col(0));
                    if (correctedP3Dw.at<float>(3, 0) < 0.0)
                        correctedP3Dw = -correctedP3Dw;

                    pMPi->SetWorldPos(correctedP3Dw);
                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMPi->mnCorrectedReference = pKFi->mnId;
                    pMPi->UpdateCoefficientsAndPoints();
                }

                vector<MapPlane *> vpMVPLsi = pKFi->GetMapVerticalPlaneMatches();
                for (auto pMPi : vpMVPLsi) {
                    if (!pMPi)
                        continue;
                    if (pMPi->isBad())
                        continue;
                    if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                        continue;

                    // Project with non-corrected pose and project back with corrected pose
                    cv::Mat P3Dw = pMPi->GetWorldPos();

                    cv::Mat correctedP3Dw = cv::Mat::eye(4, 1, CV_32F);

                    correctedP3Dw.rowRange(0, 3).col(0) = sRSiw * P3Dw.rowRange(0, 3).col(0);
                    correctedP3Dw.at<float>(3, 0) =
                            P3Dw.at<float>(3, 0) - tSiw.dot(correctedP3Dw.rowRange(0, 3).col(0));
                    if (correctedP3Dw.at<float>(3, 0) < 0.0)
                        correctedP3Dw = -correctedP3Dw;

                    correctedP3Dw.rowRange(0, 3).col(0) = sRCorrectedSiw * correctedP3Dw.rowRange(0, 3).col(0);
                    correctedP3Dw.at<float>(3, 0) =
                            correctedP3Dw.at<float>(3, 0) - tCorrectedSiw.dot(correctedP3Dw.rowRange(0, 3).col(0));
                    if (correctedP3Dw.at<float>(3, 0) < 0.0)
                        correctedP3Dw = -correctedP3Dw;

                    pMPi->SetWorldPos(correctedP3Dw);
                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMPi->mnCorrectedReference = pKFi->mnId;
                    pMPi->UpdateCoefficientsAndPoints();
                }

                vector<MapPlane *> vpMPPLsi = pKFi->GetMapParallelPlaneMatches();
                for (auto pMPi : vpMPPLsi) {
                    if (!pMPi)
                        continue;
                    if (pMPi->isBad())
                        continue;
                    if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                        continue;

                    // Project with non-corrected pose and project back with corrected pose
                    cv::Mat P3Dw = pMPi->GetWorldPos();

                    cv::Mat correctedP3Dw = cv::Mat::eye(4, 1, CV_32F);

                    correctedP3Dw.rowRange(0, 3).col(0) = sRSiw * P3Dw.rowRange(0, 3).col(0);
                    correctedP3Dw.at<float>(3, 0) =
                            P3Dw.at<float>(3, 0) - tSiw.dot(correctedP3Dw.rowRange(0, 3).col(0));
                    if (correctedP3Dw.at<float>(3, 0) < 0.0)
                        correctedP3Dw = -correctedP3Dw;

                    correctedP3Dw.rowRange(0, 3).col(0) = sRCorrectedSiw * correctedP3Dw.rowRange(0, 3).col(0);
                    correctedP3Dw.at<float>(3, 0) =
                            correctedP3Dw.at<float>(3, 0) - tCorrectedSiw.dot(correctedP3Dw.rowRange(0, 3).col(0));
                    if (correctedP3Dw.at<float>(3, 0) < 0.0)
                        correctedP3Dw = -correctedP3Dw;

                    pMPi->SetWorldPos(correctedP3Dw);
                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMPi->mnCorrectedReference = pKFi->mnId;
                    pMPi->UpdateCoefficientsAndPoints();
                }

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
                Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
                double s = g2oCorrectedSiw.scale();

                eigt *= (1. / s); //[R t/s;0 1]

                cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

                pKFi->SetPose(correctedTiw);

                // Make sure connections are updated
                pKFi->UpdateConnections();
            }

            // Start Loop Fusion
            // Update matched map points and replace if duplicated
            for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++) {
                if (mvpCurrentMatchedPoints[i]) {
                    MapPoint *pLoopMP = mvpCurrentMatchedPoints[i];
                    MapPoint *pCurMP = mpCurrentKF->GetMapPoint(i);
                    if (pCurMP)
                        pCurMP->Replace(pLoopMP);
                    else {
                        mpCurrentKF->AddMapPoint(pLoopMP, i);
                        pLoopMP->AddObservation(mpCurrentKF, i);
                        pLoopMP->ComputeDistinctiveDescriptors();
                    }
                }
            }

            for (size_t i = 0; i < mvpCurrentMatchedLines.size(); i++) {
                if (mvpCurrentMatchedLines[i]) {
                    MapLine *pLoopML = mvpCurrentMatchedLines[i];
                    MapLine *pCurML = mpCurrentKF->GetMapLine(i);
                    if (pCurML)
                        pCurML->Replace(pLoopML);
                    else {
                        mpCurrentKF->AddMapLine(pLoopML, i);
                        pLoopML->AddObservation(mpCurrentKF, i);
                        pLoopML->ComputeDistinctiveDescriptors();
                    }
                }
            }

            for (size_t i = 0; i < mvpCurrentMatchedPlanes.size(); i++) {
                if (mvpCurrentMatchedPlanes[i]) {
                    MapPlane *pLoopMP = mvpCurrentMatchedPlanes[i];
                    MapPlane *pCurMP = mpCurrentKF->GetMapPlane(i);
                    if (pCurMP) {
                        cout << "Plane fuse loop Replace: i: " << i << " - " << pCurMP->mRed << ", " << pCurMP->mGreen
                             << ", " << pCurMP->mBlue
                             << " with: " << pLoopMP->mRed << ", " << pLoopMP->mGreen
                             << ", " << pLoopMP->mBlue << endl;
                        pCurMP->Replace(pLoopMP);
                    } else {
                        cout << "Plane fuse loop Replace Add: " << i << " - " << pLoopMP->mRed << ", "
                             << pLoopMP->mGreen << ", " << pLoopMP->mBlue << endl;
                        mpCurrentKF->AddMapPlane(pLoopMP, i);
                        pLoopMP->AddObservation(mpCurrentKF, i);
                        pLoopMP->UpdateCoefficientsAndPoints();
                    }
                }
            }

//            for (size_t i = 0; i < mvpCurrentMatchedVerticalPlanes.size(); i++) {
//                if (mvpCurrentMatchedVerticalPlanes[i]) {
//                    MapPlane *pLoopMP = mvpCurrentMatchedVerticalPlanes[i];
//                    MapPlane *pCurMP = mpCurrentKF->GetMapVerticalPlane(i);
//                    if (pCurMP)
//                        pCurMP->ReplaceVerticalObservations(pLoopMP);
//                    else {
//                        mpCurrentKF->AddMapVerticalPlane(pLoopMP, i);
//                        pLoopMP->AddVerObservation(mpCurrentKF, i);
//                    }
//                }
//            }

//            for (size_t i = 0; i < mvpCurrentMatchedParallelPlanes.size(); i++) {
//                if (mvpCurrentMatchedParallelPlanes[i]) {
//                    MapPlane *pLoopMP = mvpCurrentMatchedParallelPlanes[i];
//                    MapPlane *pCurMP = mpCurrentKF->GetMapParallelPlane(i);
//                    if (pCurMP)
//                        pCurMP->ReplaceParallelObservations(pLoopMP);
//                    else {
//                        mpCurrentKF->AddMapParallelPlane(pLoopMP, i);
//                        pLoopMP->AddParObservation(mpCurrentKF, i);
//                    }
//                }
//            }
        }

        // Project MapPoints observed in the neighborhood of the loop keyframe
        // into the current keyframe and neighbors using corrected poses.
        // Fuse duplications.
        SearchAndFuse(CorrectedSim3);


        // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
        map<KeyFrame *, set<KeyFrame *> > LoopConnections;

        for (vector<KeyFrame *>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end();
             vit != vend; vit++) {
            KeyFrame *pKFi = *vit;
            vector<KeyFrame *> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

            // Update connections. Detect new links.
            pKFi->UpdateConnections();
            LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
            for (auto &vpPreviousNeighbor : vpPreviousNeighbors) {
                LoopConnections[pKFi].erase(vpPreviousNeighbor);
            }
            for (auto &mvpCurrentConnectedKF : mvpCurrentConnectedKFs) {
                LoopConnections[pKFi].erase(mvpCurrentConnectedKF);
            }
        }

        // Optimize graph
        Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3,
                                          LoopConnections, mbFixScale);

        mpMap->InformNewBigChange();

        // Add loop edge
        mpMatchedKF->AddLoopEdge(mpCurrentKF);
        mpCurrentKF->AddLoopEdge(mpMatchedKF);

        // Launch a new thread to perform Global Bundle Adjustment
//        mbRunningGBA = true;
//        mbFinishedGBA = false;
//        mbStopGBA = false;
//        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);

        // Loop closed. Release Local Mapping.
        mpLocalMapper->Release();

        mLastLoopKFid = mpCurrentKF->mnId;
    }


    void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap) {
        ORBmatcher matcher(0.8);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(Config::Get<double>("Plane.AssociationDisRef"),
                              Config::Get<double>("Plane.AssociationAngRef"),
                              Config::Get<double>("Plane.VerticalThreshold"),
                              Config::Get<double>("Plane.ParallelThreshold"));

        for (const auto &mit : CorrectedPosesMap) {
            KeyFrame *pKF = mit.first;

            g2o::Sim3 g2oScw = mit.second;
            cv::Mat cvScw = Converter::toCvMat(g2oScw);

            vector<MapPoint *> vpReplacePoints(mvpLoopMapPoints.size(), static_cast<MapPoint *>(nullptr));
            matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
            const int nLP = mvpLoopMapPoints.size();
            for (int i = 0; i < nLP; i++) {
                MapPoint *pRep = vpReplacePoints[i];
                if (pRep) {
                    pRep->Replace(mvpLoopMapPoints[i]);
                }
            }

            vector<MapLine *> vpReplaceLines(mvpLoopMapLines.size(), static_cast<MapLine *>(nullptr));
            lmatcher.Fuse(pKF, cvScw, mvpLoopMapLines, 4, vpReplaceLines);

            const int nLL = mvpLoopMapLines.size();
            for (int i = 0; i < nLL; i++) {
                MapLine *pRep = vpReplaceLines[i];
                if (pRep) {
                    pRep->Replace(mvpLoopMapLines[i]);
                }
            }

            vector<MapPlane *> vpReplacePlanes(mvpLoopMapPlanes.size(), static_cast<MapPlane *>(nullptr));
            vector<MapPlane *> vpReplaceVerticalPlanes(mvpLoopMapVerticalPlanes.size(),
                                                       static_cast<MapPlane *>(nullptr));
            vector<MapPlane *> vpReplaceParallelPlanes(mvpLoopMapParallelPlanes.size(),
                                                       static_cast<MapPlane *>(nullptr));

            cout << "Plane fuse loop map planes: " << mvpLoopMapPlanes.size() << endl;

//            pmatcher.Fuse(pKF, cvScw, mvpLoopMapPlanes, mvpLoopMapVerticalPlanes, mvpLoopMapParallelPlanes,
//                          vpReplacePlanes, vpReplaceVerticalPlanes, vpReplaceParallelPlanes);
//
//            const int nLPL = mvpLoopMapPlanes.size();
//            for (int i = 0; i < nLPL; i++) {
//                MapPlane *pRep = vpReplacePlanes[i];
//                if (pRep) {
//                    cout << "Plane fuse Replace: " << pRep->mRed << ", " << pRep->mGreen << ", " << pRep->mBlue
//                         << " with: " << mvpLoopMapPlanes[i]->mRed << ", " << mvpLoopMapPlanes[i]->mGreen
//                         << ", " << mvpLoopMapPlanes[i]->mBlue << endl;
//                    pRep->Replace(mvpLoopMapPlanes[i]);
//                }
//            }

//            const int nLVPL = mvpLoopMapVerticalPlanes.size();
//            for (int i = 0; i < nLVPL; i++) {
//                MapPlane *pRep = vpReplaceVerticalPlanes[i];
//                if (pRep) {
//                    pRep->ReplaceVerticalObservations(mvpLoopMapVerticalPlanes[i]);
//                }
//            }
//
//            const int nLPPL = mvpLoopMapParallelPlanes.size();
//            for (int i = 0; i < nLPPL; i++) {
//                MapPlane *pRep = vpReplaceParallelPlanes[i];
//                if (pRep) {
//                    pRep->ReplaceParallelObservations(mvpLoopMapParallelPlanes[i]);
//                }
//            }
        }
    }


    void LoopClosing::RequestReset() {
        {
            unique_lock<mutex> lock(mMutexReset);
            mbResetRequested = true;
        }

        while (1) {
            {
                unique_lock<mutex> lock2(mMutexReset);
                if (!mbResetRequested)
                    break;
            }
            usleep(5000);
        }
    }

    void LoopClosing::ResetIfRequested() {
        unique_lock<mutex> lock(mMutexReset);
        if (mbResetRequested) {
            mlpLoopKeyFrameQueue.clear();
            mLastLoopKFid = 0;
            mbResetRequested = false;
        }

    }

    void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF) {
        cout << "Starting Global Bundle Adjustment" << endl;

        int idx = mnFullBAIdx;

        Optimizer::GlobalBundleAdjustemnt(mpMap, 10, &mbStopGBA, nLoopKF, false);

        // Update all MapPoints and KeyFrames
        // Local Mapping was active during BA, that means that there might be new keyframes
        // not included in the Global BA and they are not consistent with the updated map.
        // We need to propagate the correction through the spanning tree
        {
            unique_lock<mutex> lock(mMutexGBA);
            if (idx != mnFullBAIdx)
                return;

            if (!mbStopGBA) {
                cout << "Global Bundle Adjustment finished" << endl;
                cout << "Updating map ..." << endl;
                mpLocalMapper->RequestStop();
                // Wait until Local Mapping has effectively stopped

                while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished()) {
                    usleep(1000);
                }

                // Get Map Mutex
                unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

                // Correct keyframes starting at map first keyframe
                list<KeyFrame *> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(), mpMap->mvpKeyFrameOrigins.end());

                while (!lpKFtoCheck.empty()) {
                    KeyFrame *pKF = lpKFtoCheck.front();
                    const set<KeyFrame *> sChilds = pKF->GetChilds();
                    cv::Mat Twc = pKF->GetPoseInverse();
                    for (set<KeyFrame *>::const_iterator sit = sChilds.begin(); sit != sChilds.end(); sit++) {
                        KeyFrame *pChild = *sit;
                        if (pChild->mnBAGlobalForKF != nLoopKF) {
                            cv::Mat Tchildc = pChild->GetPose() * Twc;
                            pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;
                            pChild->mnBAGlobalForKF = nLoopKF;

                        }
                        lpKFtoCheck.push_back(pChild);
                    }

                    pKF->mTcwBefGBA = pKF->GetPose();
                    pKF->SetPose(pKF->mTcwGBA);
                    lpKFtoCheck.pop_front();
                }

                // Correct MapPoints
                const vector<MapPoint *> vpMPs = mpMap->GetAllMapPoints();

                for (auto pMP : vpMPs) {
                    if (pMP->isBad())
                        continue;

                    if (pMP->mnBAGlobalForKF == nLoopKF) {
                        // If optimized by Global BA, just update
                        pMP->SetWorldPos(pMP->mPosGBA);
                    } else {
                        // Update according to the correction of its reference keyframe
                        KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();

                        if (pRefKF->mnBAGlobalForKF != nLoopKF)
                            continue;

                        // Map to non-corrected camera
                        cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
                        cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
                        cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

                        // Backproject using corrected camera
                        cv::Mat Twc = pRefKF->GetPoseInverse();
                        cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
                        cv::Mat twc = Twc.rowRange(0, 3).col(3);

                        pMP->SetWorldPos(Rwc * Xc + twc);
                    }
                }

                // Correct MapLines
                const vector<MapLine *> vpMLs = mpMap->GetAllMapLines();

                for (auto pML : vpMLs) {
                    if (pML->isBad())
                        continue;

                    if (pML->mnBAGlobalForKF == nLoopKF) {
                        // If optimized by Global BA, just update
                        cv::Mat pos = pML->mPosGBA;
                        Vector6d linePos;
                        linePos << pos.at<float>(0, 0), pos.at<float>(1, 0), pos.at<float>(2, 0),
                                pos.at<float>(3, 0), pos.at<float>(4, 0), pos.at<float>(5, 0);
                        pML->SetWorldPos(linePos);
                    } else {
                        // Update according to the correction of its reference keyframe
                        KeyFrame *pRefKF = pML->GetReferenceKeyFrame();

                        if (pRefKF->mnBAGlobalForKF != nLoopKF)
                            continue;

                        Eigen::Vector3d eigenSp = pML->GetWorldPos().head(3);
                        Eigen::Vector3d eigenEp = pML->GetWorldPos().tail(3);
                        cv::Mat sp = Converter::toCvMat(eigenSp);
                        cv::Mat ep = Converter::toCvMat(eigenEp);

                        // Map to non-corrected camera
                        cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
                        cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
                        cv::Mat Xsc = Rcw * sp + tcw;
                        cv::Mat Xec = Rcw * ep + tcw;

                        // Backproject using corrected camera
                        cv::Mat Twc = pRefKF->GetPoseInverse();
                        cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
                        cv::Mat twc = Twc.rowRange(0, 3).col(3);

                        Vector6d linePos;
                        linePos << Converter::toVector3d(Rwc * Xsc + twc), Converter::toVector3d(Rwc * Xec + twc);
                        pML->SetWorldPos(linePos);
                    }
                }

                // Correct MapPlanes
                const vector<MapPlane *> vpMPLs = mpMap->GetAllMapPlanes();

                for (auto pMP : vpMPLs) {
                    if (pMP->isBad())
                        continue;

                    if (pMP->mnBAGlobalForKF == nLoopKF) {
                        // If optimized by Global BA, just update
                        pMP->SetWorldPos(pMP->mPosGBA);
                    } else {
                        // Update according to the correction of its reference keyframe
                        KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();

                        if (pRefKF->mnBAGlobalForKF != nLoopKF)
                            continue;

                        cv::Mat sRTcwBefGBA = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
                        cv::Mat tTcwBefGBA = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);

                        cv::Mat sRCorrectedWi = pRefKF->GetPoseInverse().rowRange(0, 3).colRange(0, 3);
                        cv::Mat tCorrectedWi = pRefKF->GetPoseInverse().rowRange(0, 3).col(3);

                        cv::Mat P3Dw = pMP->GetWorldPos();

                        cv::Mat correctedP3Dw = cv::Mat::eye(4, 1, CV_32F);

                        correctedP3Dw.rowRange(0, 3).col(0) = sRTcwBefGBA * P3Dw.rowRange(0, 3).col(0);
                        correctedP3Dw.at<float>(3, 0) =
                                P3Dw.at<float>(3, 0) - tTcwBefGBA.dot(correctedP3Dw.rowRange(0, 3).col(0));
                        if (correctedP3Dw.at<float>(3, 0) < 0.0)
                            correctedP3Dw = -correctedP3Dw;

                        correctedP3Dw.rowRange(0, 3).col(0) = sRCorrectedWi * correctedP3Dw.rowRange(0, 3).col(0);
                        correctedP3Dw.at<float>(3, 0) =
                                correctedP3Dw.at<float>(3, 0) - tCorrectedWi.dot(correctedP3Dw.rowRange(0, 3).col(0));
                        if (correctedP3Dw.at<float>(3, 0) < 0.0)
                            correctedP3Dw = -correctedP3Dw;

                        pMP->SetWorldPos(correctedP3Dw);
                    }

                    pMP->UpdateCoefficientsAndPoints();
                }

                mpMap->InformNewBigChange();

                mpLocalMapper->Release();

                cout << "Map updated!" << endl;
            }

            mbFinishedGBA = true;
            mbRunningGBA = false;
        }
    }

    void LoopClosing::RequestFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool LoopClosing::CheckFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void LoopClosing::SetFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    bool LoopClosing::isFinished() {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }


} //namespace ORB_SLAM
