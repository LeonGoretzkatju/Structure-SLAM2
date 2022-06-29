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

#include "Optimizer.h"
#include "G2oTypes.h"
#include "OptimizableTypes.h"
#include "Converter.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <Eigen/StdVector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "include/Plane3D.h"
#include "include/EdgePlane.h"
#include "include/EdgePlanePoint.h"
#include "include/VertexPlane.h"
#include "include/EdgeVerticalPlane.h"
#include "include/EdgeParallelPlane.h"

#include <mutex>
#include <ctime>
#include <complex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;
using namespace g2o;

namespace ORB_SLAM2 {

    bool sortByVal(const pair<MapPoint*, int> &a, const pair<MapPoint*, int> &b)
    {
        return (a.second < b.second);
    }

    void Optimizer::GlobalBundleAdjustemnt(Map *pMap, int nIterations, bool *pbStopFlag,
                                           const unsigned long nLoopKF, const bool bRobust) {
        vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
        vector<MapPoint *> vpMP = pMap->GetAllMapPoints();
        vector<MapLine *> vpML = pMap->GetAllMapLines();
        vector<MapPlane *> vpMPL = pMap->GetAllMapPlanes();
        BundleAdjustment(vpKFs, vpMP, vpML, vpMPL, nIterations, pbStopFlag, nLoopKF, bRobust);
    }

    void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                     const vector<MapLine *> &vpML, const vector<MapPlane *> &vpMPL,
                                     int nIterations, bool *pbStopFlag,
                                     const unsigned long nLoopKF, const bool bRobust) {
        vector<bool> vbNotIncludedMP, vbNotIncludedML, vbNotIncludedMPL;
        vbNotIncludedMP.resize(vpMP.size());
        vbNotIncludedML.resize(vpML.size());
        vbNotIncludedMPL.resize(vpMPL.size());

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if (pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        long unsigned int maxKFid = 0;

        // Set KeyFrame vertices
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];
            if (pKF->isBad())
                continue;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
            vSE3->setId(pKF->mnId);
            vSE3->setFixed(pKF->mnId == 0);
            optimizer.addVertex(vSE3);
            if (pKF->mnId > maxKFid)
                maxKFid = pKF->mnId;
        }

        const float thHuber2D = sqrt(5.99);
        const float thHuber3D = sqrt(7.815);

        vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
        vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;

        long unsigned int maxMapPointId = maxKFid;

        int totalEdges = 0;
        // Set MapPoint vertices
        for (size_t i = 0; i < vpMP.size(); i++) {
            MapPoint *pMP = vpMP[i];

            if (pMP->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            const int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            if (id > maxMapPointId)
                maxMapPointId = id;

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            int nEdges = 0;
            //SET EDGES
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {

                KeyFrame *pKF = mit->first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

                if (pKF->mvuRight[mit->second] < 0) {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    if (bRobust) {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                } else {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKF->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    if (bRobust) {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber3D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;
                    e->bf = pKF->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                }
            }

            totalEdges += nEdges;

            // fix the edge if there is no observation of this mp
            if (nEdges == 0) {
                optimizer.removeVertex(vPoint);
                vbNotIncludedMP[i] = true;
            } else {
                vbNotIncludedMP[i] = false;
            }
        }

        cout << "GBA: Total point edges: " << totalEdges << endl;
        cout << "GBA: Max Point id: " << maxMapPointId << endl;

        int maxMapLineId = maxMapPointId;

        // Set line vertices and edges
        for (size_t i = 0; i < vpML.size(); i++) {
            MapLine *pML = vpML[i];

            if (pML->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vStartPoint = new g2o::VertexSBAPointXYZ();
            vStartPoint->setEstimate(pML->GetWorldPos().head(3));
            const int id1 = (2 * pML->mnId) + 1 + maxMapPointId;
            vStartPoint->setId(id1);
            vStartPoint->setMarginalized(true);
            optimizer.addVertex(vStartPoint);

            g2o::VertexSBAPointXYZ *vEndPoint = new g2o::VertexSBAPointXYZ();
            vEndPoint->setEstimate(pML->GetWorldPos().tail(3));
            const int id2 = (2 * (pML->mnId + 1)) + maxMapPointId;
            vEndPoint->setId(id2);
            vEndPoint->setMarginalized(true);
            optimizer.addVertex(vEndPoint);

            if (id2 > maxMapLineId) {
                maxMapLineId = id2;
            }

            cout << "GBA: Line id1: " << id1 << ", id2: " << id2 << ", Max: " << maxMapLineId << endl;

            const map<KeyFrame *, size_t> observations = pML->GetObservations();

            int nEdges = 0;

            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {
                KeyFrame *pKF = mit->first;

                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                Eigen::Vector3d lineObs = pKF->mvKeyLineFunctions[mit->second];

                EdgeLineProjectXYZ *es = new EdgeLineProjectXYZ();
                es->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
                es->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                es->setMeasurement(lineObs);
                es->setInformation(Eigen::Matrix3d::Identity());

                if (bRobust) {
                    g2o::RobustKernelHuber *rks = new g2o::RobustKernelHuber;
                    es->setRobustKernel(rks);
                    rks->setDelta(thHuber3D);
                }

                es->fx = pKF->fx;
                es->fy = pKF->fy;
                es->cx = pKF->cx;
                es->cy = pKF->cy;

                optimizer.addEdge(es);

                EdgeLineProjectXYZ *ee = new EdgeLineProjectXYZ();
                ee->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
                ee->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                ee->setMeasurement(lineObs);
                ee->setInformation(Eigen::Matrix3d::Identity());

                if (bRobust) {
                    g2o::RobustKernelHuber *rke = new g2o::RobustKernelHuber;
                    ee->setRobustKernel(rke);
                    rke->setDelta(thHuber3D);
                }

                ee->fx = pKF->fx;
                ee->fy = pKF->fy;
                ee->cx = pKF->cx;
                ee->cy = pKF->cy;

                optimizer.addEdge(ee);
            }

            if (nEdges == 0) {
                optimizer.removeVertex(vStartPoint);
                optimizer.removeVertex(vEndPoint);
                vbNotIncludedML[i] = true;
            } else {
                vbNotIncludedML[i] = false;
            }
        }

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        // Set MapPlane vertices
        for (size_t i = 0; i < vpMPL.size(); i++) {
            MapPlane *pMP = vpMPL[i];
            if (pMP->isBad())
                continue;

            g2o::VertexPlane *vPlane = new g2o::VertexPlane();
            vPlane->setEstimate(Converter::toPlane3D(pMP->GetWorldPos()));
            const int id = pMP->mnId + maxMapLineId + 1;
            vPlane->setId(id);
            vPlane->setMarginalized(true);
            optimizer.addVertex(vPlane);

            cout << "GBA: Plane id: " << id << endl;

            int nEdges = 0;

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            for (const auto &observation : observations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                g2o::EdgePlane *e = new g2o::EdgePlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                //TODO
                Eigen::Matrix3d Info;
                Info << angleInfo, 0, 0,
                        0, angleInfo, 0,
                        0, 0, disInfo;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaPlane);

                optimizer.addEdge(e);
            }

            const map<KeyFrame *, size_t> verObservations = pMP->GetVerObservations();
            for (const auto &observation : verObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                g2o::EdgeVerticalPlane *e = new g2o::EdgeVerticalPlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                //TODO
                Eigen::Matrix2d Info;
                Info << angleInfo, 0,
                        0, angleInfo;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
            }

            const map<KeyFrame *, size_t> parObservations = pMP->GetParObservations();
            for (const auto &observation : parObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                g2o::EdgeParallelPlane *e = new g2o::EdgeParallelPlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                //TODO
                Eigen::Matrix2d Info;
                Info << angleInfo, 0,
                        0, angleInfo;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
            }

            if (nEdges == 0) {
                optimizer.removeVertex(vPlane);
                vbNotIncludedMPL[i] = true;
            } else {
                vbNotIncludedMPL[i] = false;
            }
        }

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);

        int bad = 0;
        int PNMono = 0;
        double PEMono = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];


            const float chi2 = e->chi2();
            //cout<<"optimize chi2"<<chi2<<endl;
            PNMono++;
            PEMono += chi2;

            if (chi2 > thHuber2D * thHuber2D) {
                bad++;
                cout << " GBA: Bad point: " << chi2 << endl;
            }
        }

        if (PNMono == 0)
            cout << "GBA: No mono points " << " ";
        else
            cout << "GBA: Mono points: " << PEMono / PNMono << " ";

        int PNStereo = 0;
        double PEStereo = 0;
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];

            const float chi2 = e->chi2();
            //cout<<"optimize chi2"<<chi2<<endl;
            PNStereo++;
            PEStereo += chi2;

            if (chi2 > thHuber3D * thHuber3D) {
                bad++;
                cout << "GBA: Bad stereo point: " << chi2 << endl;
            }
        }
        if (PNStereo == 0)
            cout << "GBA: No stereo points " << " ";
        else
            cout << "GBA: Stereo points: " << PEStereo / PNStereo << endl;

        cout << "GBA: Total bad point edges: " << bad << endl;

        // Recover optimized data

        //Keyframes
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];
            if (pKF->isBad())
                continue;
            g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            if (nLoopKF == 0) {
                pKF->SetPose(Converter::toCvMat(SE3quat));
            } else {
                pKF->mTcwGBA.create(4, 4, CV_32F);
                Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
                pKF->mnBAGlobalForKF = nLoopKF;
            }
        }

        //Points
        for (size_t i = 0; i < vpMP.size(); i++) {
            if (vbNotIncludedMP[i])
                continue;

            MapPoint *pMP = vpMP[i];

            if (pMP->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));

            double dis = cv::norm(Converter::toCvMat(vPoint->estimate()) - pMP->GetWorldPos());
            if (dis > 0.5) {
                std::cout << "Point id: " << pMP->mnId << ", bad: " << pMP->isBad()  << ", pose - before: " << pMP->GetWorldPos().t()
                          << ", after: " << Converter::toCvMat(vPoint->estimate()).t() << std::endl;
            }

            if (nLoopKF == 0) {
                pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pMP->UpdateNormalAndDepth();
            } else {
                pMP->mPosGBA.create(3, 1, CV_32F);
                Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }

        //Lines
        for (size_t i = 0; i < vpML.size(); i++) {

            if (vbNotIncludedML[i])
                continue;

            MapLine *pML = vpML[i];

            if (pML->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vStartPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                    (2 * pML->mnId) + 1 + maxMapPointId));
            g2o::VertexSBAPointXYZ *vEndPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                    2 * (pML->mnId + 1) + maxMapPointId));

            if (nLoopKF == 0) {
                Vector6d linePos;
                linePos << vStartPoint->estimate(), vEndPoint->estimate();
                pML->SetWorldPos(linePos);
                pML->UpdateAverageDir();
            } else {
                pML->mPosGBA.create(6, 1, CV_32F);
                Converter::toCvMat(vStartPoint->estimate()).copyTo(pML->mPosGBA.rowRange(0, 3));
                Converter::toCvMat(vEndPoint->estimate()).copyTo(pML->mPosGBA.rowRange(3, 6));
                pML->mnBAGlobalForKF = nLoopKF;
            }
        }

        //Planes
        for (size_t i = 0; i < vpMPL.size(); i++) {
            if (vbNotIncludedMPL[i])
                continue;

            MapPlane *pMP = vpMPL[i];

            if (pMP->isBad())
                continue;

            g2o::VertexPlane *vPlane = static_cast<g2o::VertexPlane *>(optimizer.vertex(
                    pMP->mnId + maxMapLineId + 1));

            if (nLoopKF == 0) {
                pMP->SetWorldPos(Converter::toCvMat(vPlane->estimate()));
                pMP->UpdateCoefficientsAndPoints();
            } else {
                pMP->mPosGBA.create(4, 1, CV_32F);
                Converter::toCvMat(vPlane->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }
    }

    void Optimizer::FullInertialBA(Map *pMap, int its, const bool bFixLocal, const long unsigned int nLoopId, bool *pbStopFlag, bool bInit, float priorG, float priorA, Eigen::VectorXd *vSingVal, bool *bHess)
    {
        long unsigned int maxKFid = pMap->GetMaxKFid();
        const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
        const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
        const vector<MapLine *> vpMLs = pMap->GetAllMapLines();
        const vector<MapPlane *> vpMPLs = pMap->GetAllMapPlanes();

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e-5);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        int nNonFixed = 0;

        // Set KeyFrame vertices
        KeyFrame* pIncKF;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId>maxKFid)
                continue;
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            pIncKF=pKFi;
            bool bFixed = false;
            if(bFixLocal)
            {
                bFixed = (pKFi->mnBALocalForKF>=(maxKFid-1)) || (pKFi->mnBAFixedForKF>=(maxKFid-1));
                if(!bFixed)
                    nNonFixed++;
                VP->setFixed(bFixed);
            }
            optimizer.addVertex(VP);

            if(pKFi->bImu)
            {
                VertexVelocity* VV = new VertexVelocity(pKFi);
                VV->setId(maxKFid+3*(pKFi->mnId)+1);
                VV->setFixed(bFixed);
                optimizer.addVertex(VV);
                if (!bInit)
                {
                    VertexGyroBias* VG = new VertexGyroBias(pKFi);
                    VG->setId(maxKFid+3*(pKFi->mnId)+2);
                    VG->setFixed(bFixed);
                    optimizer.addVertex(VG);
                    VertexAccBias* VA = new VertexAccBias(pKFi);
                    VA->setId(maxKFid+3*(pKFi->mnId)+3);
                    VA->setFixed(bFixed);
                    optimizer.addVertex(VA);
                }
            }
        }

        if (bInit)
        {
            // IMU parameters of the lastest frame (end of vpKFs)
            VertexGyroBias* VG = new VertexGyroBias(pIncKF);
            VG->setId(4*maxKFid+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pIncKF);
            VA->setId(4*maxKFid+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }

        if(bFixLocal)
        {
            if(nNonFixed<3)
                return;
        }

        // IMU links
        for(size_t i=0;i<vpKFs.size();i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            if(!pKFi->mPrevKF)
            {
                Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
                continue;
            }

            if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
            {
                if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                    continue;
                if(pKFi->bImu && pKFi->mPrevKF->bImu)
                {
                    pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                    g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                    g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);

                    g2o::HyperGraph::Vertex* VG1; // gyro
                    g2o::HyperGraph::Vertex* VA1; // acc
                    g2o::HyperGraph::Vertex* VG2;
                    g2o::HyperGraph::Vertex* VA2;
                    if (!bInit)
                    {
                        VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
                        VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
                        VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
                        VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);
                    }
                    else
                    {
                        VG1 = optimizer.vertex(4*maxKFid+2);
                        VA1 = optimizer.vertex(4*maxKFid+3);
                    }

                    g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                    g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);

                    if (!bInit)
                    {
                        if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                        {
                            cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;

                            continue;
                        }
                    }
                    else
                    {
                        if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
                        {
                            cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<endl;

                            continue;
                        }
                    }

                    EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
                    ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                    ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                    ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                    ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                    ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                    ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                    g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                    ei->setRobustKernel(rki);
                    rki->setDelta(sqrt(16.92));

                    optimizer.addEdge(ei);

                    if (!bInit)
                    {
                        EdgeGyroRW* egr= new EdgeGyroRW();
                        egr->setVertex(0,VG1);
                        egr->setVertex(1,VG2);
                        cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
                        Eigen::Matrix3d InfoG;
                        for(int r=0;r<3;r++)
                            for(int c=0;c<3;c++)
                                InfoG(r,c)=cvInfoG.at<float>(r,c);
                        egr->setInformation(InfoG);
                        egr->computeError();
                        g2o::RobustKernelHuber *regr = new g2o::RobustKernelHuber;
                        egr->setRobustKernel(regr);
                        regr->setDelta(5);
                        optimizer.addEdge(egr);

                        EdgeAccRW* ear = new EdgeAccRW();
                        ear->setVertex(0,VA1);
                        ear->setVertex(1,VA2);
                        cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
                        Eigen::Matrix3d InfoA;
                        for(int r=0;r<3;r++)
                            for(int c=0;c<3;c++)
                                InfoA(r,c)=cvInfoA.at<float>(r,c);
                        ear->setInformation(InfoA);
                        ear->computeError();
                        g2o::RobustKernelHuber *rear = new g2o::RobustKernelHuber;
                        ear->setRobustKernel(rear);
                        rear->setDelta(5);
                        optimizer.addEdge(ear);
                    }
                }
                else
                {
                    cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
                }
            }
        }

        if (bInit)
        {
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(4*maxKFid+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(4*maxKFid+3);

            // Add prior to comon biases
            EdgePriorAcc* epa = new EdgePriorAcc(cv::Mat::zeros(3,1,CV_32F));
            epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            double infoPriorA = priorA; //
            epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
            optimizer.addEdge(epa);

            EdgePriorGyro* epg = new EdgePriorGyro(cv::Mat::zeros(3,1,CV_32F));
            epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            double infoPriorG = priorG; //
            epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
            optimizer.addEdge(epg);
        }

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);
        const float thHuberLD = sqrt(3.84);

        const unsigned long iniMPid = maxKFid*5;
        long unsigned int maxMapPointId = iniMPid;

        vector<bool> vbNotIncludedMP(vpMPs.size(),false);

        for(size_t i=0; i<vpMPs.size(); i++)
        {
            MapPoint* pMP = vpMPs[i];
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            unsigned long id = pMP->mnId+iniMPid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            if (id > maxMapPointId)
                maxMapPointId = id;

            // const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
            std::map<KeyFrame*,size_t> observations = pMP->GetObservations();

            // Fix the map point (not included in optimization) if all the keyframe observations are fixed
            bool bAllFixed = true;

            //Set edges
            for(std::map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnId>maxKFid)
                    continue;

                if(!pKFi->isBad())
                {
                    // const int leftIndex = get<0>(mit->second);
                    const int leftIndex = mit->second;
                    cv::KeyPoint kpUn;

                    if(pKFi->mvuRight[leftIndex]<0) // Monocular observation
                    {
                        kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMono* e = new EdgeMono(0);

                        g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                        if(bAllFixed)
                            if(!VP->fixed())
                                bAllFixed=false;

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, VP);
                        e->setMeasurement(obs);
                        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                    }
                    else // stereo observation
                    {
                        kpUn = pKFi->mvKeysUn[leftIndex];
                        const float kp_ur = pKFi->mvuRight[leftIndex];
                        Eigen::Matrix<double,3,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        EdgeStereo* e = new EdgeStereo(0);

                        g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                        if(bAllFixed)
                            if(!VP->fixed())
                                bAllFixed=false;

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, VP);
                        e->setMeasurement(obs);
                        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                        e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        optimizer.addEdge(e);
                    }

                    // if(pKFi->mpCamera2){ // Monocular right observation
                    //     int rightIndex = get<1>(mit->second);

                    //     if(rightIndex != -1 && rightIndex < pKFi->mvKeysRight.size()){
                    //         rightIndex -= pKFi->NLeft;

                    //         Eigen::Matrix<double,2,1> obs;
                    //         kpUn = pKFi->mvKeysRight[rightIndex];
                    //         obs << kpUn.pt.x, kpUn.pt.y;

                    //         EdgeMono *e = new EdgeMono(1);

                    //         g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    //         if(bAllFixed)
                    //             if(!VP->fixed())
                    //                 bAllFixed=false;

                    //         e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    //         e->setVertex(1, VP);
                    //         e->setMeasurement(obs);
                    //         const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    //         e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    //         g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    //         e->setRobustKernel(rk);
                    //         rk->setDelta(thHuberMono);

                    //         optimizer.addEdge(e);
                    //     }
                    // }
                }
            }

            if(bAllFixed)
            {
                optimizer.removeVertex(vPoint);
                vbNotIncludedMP[i]=true;
            }
        }

        int maxMapLineId = maxMapPointId;
        // vector<bool> vbNotIncludedML(vpMLs.size(),false);

        // // Set line vertices and edges
        // for (size_t i = 0; i < vpMLs.size(); i++) {
        //     MapLine *pML = vpMLs[i];

        //     if (pML->isBad())
        //         continue;

        //     g2o::VertexSBAPointXYZ *vStartPoint = new g2o::VertexSBAPointXYZ();
        //     vStartPoint->setEstimate(pML->GetWorldPos().head(3));
        //     const int id1 = (2 * pML->mnId) + 1 + maxMapPointId;
        //     vStartPoint->setId(id1);
        //     vStartPoint->setMarginalized(true);
        //     optimizer.addVertex(vStartPoint);

        //     g2o::VertexSBAPointXYZ *vEndPoint = new g2o::VertexSBAPointXYZ();
        //     vEndPoint->setEstimate(pML->GetWorldPos().tail(3));
        //     const int id2 = (2 * (pML->mnId + 1)) + maxMapPointId;
        //     vEndPoint->setId(id2);
        //     vEndPoint->setMarginalized(true);
        //     optimizer.addVertex(vEndPoint);

        //     if (id2 > maxMapLineId) {
        //         maxMapLineId = id2;
        //     }

        //     const map<KeyFrame *, size_t> observations = pML->GetObservations();

        //     int nEdges = 0;
        //     bool bAllFixed = true;

        //     for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {
        //         KeyFrame *pKF = mit->first;

        //         if (pKF->mnId > maxKFid)
        //             continue;

        //         if(!pKF->isBad()){
        //             nEdges++;

        //             Eigen::Vector3d lineObs = pKF->mvKeyLineFunctions[mit->second];

        //             EdgeInertialLineProjectXYZ *es = new EdgeInertialLineProjectXYZ();
        //             es->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));

        //             g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId));
        //             es->setVertex(1, VP);
                    
        //             es->setMeasurement(lineObs);
        //             es->setInformation(Eigen::Matrix3d::Identity());

        //             if(bAllFixed)
        //                 if(!VP->fixed())
        //                     bAllFixed=false;

        //             g2o::RobustKernelHuber *rks = new g2o::RobustKernelHuber;
        //             es->setRobustKernel(rks);
        //             rks->setDelta(thHuberLD);

        //             es->fx = pKF->fx;
        //             es->fy = pKF->fy;
        //             es->cx = pKF->cx;
        //             es->cy = pKF->cy;

        //             optimizer.addEdge(es);

        //             EdgeInertialLineProjectXYZ *ee = new EdgeInertialLineProjectXYZ();
        //             ee->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
        //             ee->setVertex(1, VP);
        //             ee->setMeasurement(lineObs);
        //             ee->setInformation(Eigen::Matrix3d::Identity());

        //             g2o::RobustKernelHuber *rke = new g2o::RobustKernelHuber;
        //             ee->setRobustKernel(rke);
        //             rke->setDelta(thHuberLD);

        //             ee->fx = pKF->fx;
        //             ee->fy = pKF->fy;
        //             ee->cx = pKF->cx;
        //             ee->cy = pKF->cy;

        //             optimizer.addEdge(ee);
        //         }
        //     }

        //     if (nEdges == 0 || bAllFixed) {
        //         optimizer.removeVertex(vStartPoint);
        //         optimizer.removeVertex(vEndPoint);
        //         vbNotIncludedML[i] = true;
        //     } else {
        //         vbNotIncludedML[i] = false;
        //     }
        // }

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        vector<bool> vbNotIncludedMPL(vpMPLs.size(), false);

        // Set MapPlane vertices
        for (size_t i = 0; i < vpMPLs.size(); i++) {
            MapPlane *pMP = vpMPLs[i];
            if (pMP->isBad())
                continue;

            g2o::VertexPlane *vPlane = new g2o::VertexPlane();
            vPlane->setEstimate(Converter::toPlane3D(pMP->GetWorldPos()));
            const int id = pMP->mnId + maxMapLineId + 1;
            vPlane->setId(id);
            vPlane->setMarginalized(true);
            optimizer.addVertex(vPlane);

            int nEdges = 0;
            bool bAllFixed1 = true, bAllFixed2 = true, bAllFixed3 = true; 

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            for (const auto &observation : observations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                g2o::EdgeInertialPlane *e = new g2o::EdgeInertialPlane();

                g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId));
                e->setVertex(1, VP);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));

                if(bAllFixed1)
                    if(!VP->fixed())
                        bAllFixed1=false;

                //TODO
                Eigen::Matrix3d Info;
                Info << angleInfo, 0, 0,
                        0, angleInfo, 0,
                        0, 0, disInfo;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaPlane);

                // e->planePoints = pMP->mvPlanePoints;

                optimizer.addEdge(e);
            }

            const map<KeyFrame *, size_t> verObservations = pMP->GetVerObservations();
            for (const auto &observation : verObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                g2o::EdgeInertialVerticalPlane *e = new g2o::EdgeInertialVerticalPlane();

                g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId));
                e->setVertex(1, VP);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));

                if(bAllFixed2)
                    if(!VP->fixed())
                        bAllFixed2=false;

                //TODO
                Eigen::Matrix2d Info;
                Info << angleInfo, 0,
                        0, angleInfo;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
            }

            const map<KeyFrame *, size_t> parObservations = pMP->GetParObservations();
            for (const auto &observation : parObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                g2o::EdgeInertialParallelPlane *e = new g2o::EdgeInertialParallelPlane();

                g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId));
                e->setVertex(1, VP);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));

                if(bAllFixed3)
                    if(!VP->fixed())
                        bAllFixed3=false;

                //TODO
                Eigen::Matrix2d Info;
                Info << angleInfo, 0,
                        0, angleInfo;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
            }

            // Fix the plane only if all keyframe observations are fixed
            if (nEdges == 0 || (bAllFixed1 && bAllFixed2 && bAllFixed3)) {
                optimizer.removeVertex(vPlane);
                vbNotIncludedMPL[i] = true;
            } else {
                vbNotIncludedMPL[i] = false;
            }
        }

        if(pbStopFlag)
            if(*pbStopFlag)
                return;


        optimizer.initializeOptimization();
        optimizer.optimize(its);


        // Recover optimized data
        //Keyframes
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId>maxKFid)
                continue;
            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
            if(nLoopId==0)
            {
                cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
                pKFi->SetPose(Tcw);
            }
            else
            {
                pKFi->mTcwGBA = cv::Mat::eye(4,4,CV_32F);
                Converter::toCvMat(VP->estimate().Rcw[0]).copyTo(pKFi->mTcwGBA.rowRange(0,3).colRange(0,3));
                Converter::toCvMat(VP->estimate().tcw[0]).copyTo(pKFi->mTcwGBA.rowRange(0,3).col(3));
                pKFi->mnBAGlobalForKF = nLoopId;

            }
            if(pKFi->bImu)
            {
                VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
                if(nLoopId==0)
                {
                    pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
                }
                else
                {
                    pKFi->mVwbGBA = Converter::toCvMat(VV->estimate());
                }

                VertexGyroBias* VG;
                VertexAccBias* VA;
                if (!bInit)
                {
                    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
                    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
                }
                else
                {
                    VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*maxKFid+2));
                    VA = static_cast<VertexAccBias*>(optimizer.vertex(4*maxKFid+3));
                }

                Vector6d vb;
                vb << VG->estimate(), VA->estimate();
                IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
                if(nLoopId==0)
                {
                    pKFi->SetNewBias(b);
                }
                else
                {
                    pKFi->mBiasGBA = b;
                }
            }
        }

        //Points
        for(size_t i=0; i<vpMPs.size(); i++)
        {
            if(vbNotIncludedMP[i])
                continue;

            MapPoint* pMP = vpMPs[i];
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));

            if(nLoopId==0)
            {
                pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pMP->UpdateNormalAndDepth();
            }
            else
            {
                pMP->mPosGBA.create(3,1,CV_32F);
                Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopId;
            }

        }

        // //Lines
        // for (size_t i = 0; i < vpMLs.size(); i++) {

        //     if (vbNotIncludedML[i])
        //         continue;

        //     MapLine *pML = vpMLs[i];

        //     if (pML->isBad())
        //         continue;

        //     g2o::VertexSBAPointXYZ *vStartPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
        //             (2 * pML->mnId) + 1 + maxMapPointId));
        //     g2o::VertexSBAPointXYZ *vEndPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
        //             2 * (pML->mnId + 1) + maxMapPointId));

        //     if (nLoopId == 0) {
        //         Vector6d linePos;
        //         linePos << vStartPoint->estimate(), vEndPoint->estimate();
        //         pML->SetWorldPos(linePos);
        //         pML->UpdateAverageDir();
        //     } else {
        //         pML->mPosGBA.create(6, 1, CV_32F);
        //         Converter::toCvMat(vStartPoint->estimate()).copyTo(pML->mPosGBA.rowRange(0, 3));
        //         Converter::toCvMat(vEndPoint->estimate()).copyTo(pML->mPosGBA.rowRange(3, 6));
        //         pML->mnBAGlobalForKF = nLoopId;
        //     }
        // }

        //Planes
        for (size_t i = 0; i < vpMPLs.size(); i++) {
            if (vbNotIncludedMPL[i])
                continue;

            MapPlane *pMP = vpMPLs[i];

            if (pMP->isBad())
                continue;

            g2o::VertexPlane *vPlane = static_cast<g2o::VertexPlane *>(optimizer.vertex(
                    pMP->mnId + maxMapLineId + 1));

            if (nLoopId == 0) {
                pMP->SetWorldPos(Converter::toCvMat(vPlane->estimate()));
                pMP->UpdateCoefficientsAndPoints();
            } else {
                pMP->mPosGBA.create(4, 1, CV_32F);
                Converter::toCvMat(vPlane->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopId;
            }
        }

        pMap->IncreaseChangeIndex();
    }

    int Optimizer::PoseOptimization(Frame *pFrame) {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);

        vector<double> vMonoPointInfo(N, 1);
        vector<double> vSteroPointInfo(N, 1);

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; i++) {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (pMP) {
                    // Monocular observation
                    if (pFrame->mvuRight[i] < 0) {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    } else  // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        //SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }

            }
        }

        const int NL = pFrame->NL;

        vector<EdgeLineProjectXYZOnlyPose *> vpEdgesLineSp;
        vector<size_t> vnIndexLineEdgeSp;
        vpEdgesLineSp.reserve(NL);
        vnIndexLineEdgeSp.reserve(NL);

        vector<EdgeLineProjectXYZOnlyPose *> vpEdgesLineEp;
        vector<size_t> vnIndexLineEdgeEp;
        vpEdgesLineEp.reserve(NL);
        vnIndexLineEdgeEp.reserve(NL);

        vector<double> vMonoStartPointInfo(NL, 1);
        vector<double> vMonoEndPointInfo(NL, 1);
        vector<double> vSteroStartPointInfo(NL, 1);
        vector<double> vSteroEndPointInfo(NL, 1);

        // Set MapLine vertices
        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            for (int i = 0; i < NL; i++) {
                MapLine *pML = pFrame->mvpMapLines[i];
                if (pML) {
                    nInitialCorrespondences++;
                    pFrame->mvbLineOutlier[i] = false;

                    Eigen::Vector3d line_obs;
                    line_obs = pFrame->mvKeyLineFunctions[i];

                    EdgeLineProjectXYZOnlyPose *els = new EdgeLineProjectXYZOnlyPose();

                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity() * 1);

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(deltaStereo);

                    els->fx = pFrame->fx;
                    els->fy = pFrame->fy;
                    els->cx = pFrame->cx;
                    els->cy = pFrame->cy;

                    els->Xw = pML->mWorldPos.head(3);
                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    EdgeLineProjectXYZOnlyPose *ele = new EdgeLineProjectXYZOnlyPose();

                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity() * 1);

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(deltaStereo);

                    ele->fx = pFrame->fx;
                    ele->fy = pFrame->fy;
                    ele->cx = pFrame->cx;
                    ele->cy = pFrame->cy;

                    ele->Xw = pML->mWorldPos.tail(3);

                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                    vnIndexLineEdgeEp.push_back(i);
                }
            }
        }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;

        vector<g2o::EdgePlaneOnlyPose *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

        vector<vector<g2o::EdgePlanePoint *>> vpEdgesPlanePoint;
        vector<vector<size_t>> vnIndexEdgePlanePoint;
        vpEdgesPlanePoint = vector<vector<g2o::EdgePlanePoint *>>(M);
        vnIndexEdgePlanePoint = vector<vector<size_t>>(M);

        vector<g2o::EdgeParallelPlaneOnlyPose *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeVerticalPlaneOnlyPose *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        auto aTh = ORB_SLAM2::Config::Get<double>("Plane.AssociationAngRef");
        auto parTh = ORB_SLAM2::Config::Get<double>("Plane.ParallelThreshold");

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            int PNum = 0;
            double PEror = 0, PMax = 0;
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgePlaneOnlyPose *e = new g2o::EdgePlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    //TODO
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //TODO
                    rk->setDelta(deltaPlane);

                    Isometry3D trans = static_cast<const VertexSE3Expmap *>(optimizer.vertex(0))->estimate();
                    cv::Mat Pc3D = pFrame->mvPlaneCoefficients[i];
                    Plane3D Pw3D = Converter::toPlane3D(pMP->GetWorldPos());
                    Vector4D Pw = Pw3D._coeffs;
                    Vector4D Pc;
                    Matrix3D R = trans.rotation();
                    Pc.head<3>() = R * Pw.head<3>();
                    Pc(3) = Pw(3) - trans.translation().dot(Pc.head<3>());

                    double angle = Pc(0) * Pc3D.at<float>(0) +
                            Pc(1) * Pc3D.at<float>(1) +
                            Pc(2) * Pc3D.at<float>(2);
                    if (angle < -aTh) {
                        Pw = -Pw;
                        Pw3D.fromVector(Pw);
                    }

                    e->Xw = Pw3D;

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);

//                    int nPointMatches = pFrame->mvPlanePointMatches[i].size();
//
//                    vector<g2o::EdgePlanePoint*> edgesPlanePoint;
//                    vector<size_t> indexEdgePlanePoint;
//                    for (int j = 0; j < nPointMatches; j++) {
//                        MapPoint *mapPoint = pFrame->mvPlanePointMatches[i][j];
//                        if (mapPoint) {
//                            g2o::EdgePlanePoint *edge = new g2o::EdgePlanePoint();
//                            edge->setVertex(0, optimizer.vertex(0));
//                            edge->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edge->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            cv::Mat Pw = mapPoint->GetWorldPos();
//                            edge->Xw[0] = Pw.at<float>(0);
//                            edge->Xw[1] = Pw.at<float>(1);
//                            edge->Xw[2] = Pw.at<float>(2);
//
//                            g2o::RobustKernelHuber *rkEdge = new g2o::RobustKernelHuber;
//                            edge->setRobustKernel(rkEdge);
//                            rkEdge->setDelta(deltaMono);
//
//                            optimizer.addEdge(edge);
//
//                            edgesPlanePoint.push_back(edge);
//                            indexEdgePlanePoint.push_back(j);
//                        }
//                    }
//
//                    int pointEdges = edgesPlanePoint.size();
//                    int nLineMatches = pFrame->mvPlaneLineMatches[i].size();
//
//                    for (int j = 0, index = pointEdges; j < nLineMatches; j++) {
//                        MapLine *mapLine = pFrame->mvPlaneLineMatches[i][j];
//                        if (mapLine) {
//                            g2o::EdgePlanePoint *edgeStart = new g2o::EdgePlanePoint();
//                            edgeStart->setVertex(0, optimizer.vertex(0));
//                            edgeStart->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edgeStart->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            Vector3d startPoint = mapLine->mWorldPos.head(3);
//                            edgeStart->Xw[0] = startPoint(0);
//                            edgeStart->Xw[1] = startPoint(1);
//                            edgeStart->Xw[2] = startPoint(2);
//
//                            g2o::RobustKernelHuber *rkEdgeStart = new g2o::RobustKernelHuber;
//                            edgeStart->setRobustKernel(rkEdgeStart);
//                            rkEdgeStart->setDelta(deltaMono);
//
//                            optimizer.addEdge(edgeStart);
//
//                            edgesPlanePoint.push_back(edgeStart);
//                            indexEdgePlanePoint.push_back(index++);
//
//                            g2o::EdgePlanePoint *edgeEnd = new g2o::EdgePlanePoint();
//                            edgeEnd->setVertex(0, optimizer.vertex(0));
//                            edgeEnd->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edgeEnd->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            Vector3d endPoint = mapLine->mWorldPos.tail(3);
//                            edgeEnd->Xw[0] = endPoint(0);
//                            edgeEnd->Xw[1] = endPoint(1);
//                            edgeEnd->Xw[2] = endPoint(2);
//
//                            g2o::RobustKernelHuber *rkEdgeEnd = new g2o::RobustKernelHuber;
//                            edgeEnd->setRobustKernel(rkEdgeEnd);
//                            rkEdgeEnd->setDelta(deltaMono);
//
//                            optimizer.addEdge(edgeEnd);
//
//                            edgesPlanePoint.push_back(edgeEnd);
//                            indexEdgePlanePoint.push_back(index++);
//                        }
//                    }
//
//                    vpEdgesPlanePoint[i] = edgesPlanePoint;
//                    vnIndexEdgePlanePoint[i] = indexEdgePlanePoint;


                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            // cout << " Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;
            for (int i = 0; i < M; ++i) {
                // add parallel planes!
                MapPlane *pMP = pFrame->mvpParallelPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbParPlaneOutlier[i] = false;

                    g2o::EdgeParallelPlaneOnlyPose *e = new g2o::EdgeParallelPlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << parInfo, 0,
                            0, parInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    Isometry3D trans = static_cast<const VertexSE3Expmap *>(optimizer.vertex(0))->estimate();
                    cv::Mat Pc3D = pFrame->mvPlaneCoefficients[i];
                    Plane3D Pw3D = Converter::toPlane3D(pMP->GetWorldPos());
                    Vector4D Pw = Pw3D._coeffs;
                    Vector4D Pc;
                    Matrix3D R = trans.rotation();
                    Pc.head<3>() = R * Pw.head<3>();
                    Pc(3) = Pw(3) - trans.translation().dot(Pc.head<3>());

                    double angle = Pc(0) * Pc3D.at<float>(0) +
                                   Pc(1) * Pc3D.at<float>(1) +
                                   Pc(2) * Pc3D.at<float>(2);
                    if (angle < -parTh) {
                        Pw = -Pw;
                        Pw3D.fromVector(Pw);
                    }

                    e->Xw = Pw3D;

                    optimizer.addEdge(e);

                    vpEdgesParPlane.push_back(e);
                    vnIndexEdgeParPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            // cout << " Par Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;

            for (int i = 0; i < M; ++i) {
                // add vertical planes!
                MapPlane *pMP = pFrame->mvpVerticalPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbVerPlaneOutlier[i] = false;

                    g2o::EdgeVerticalPlaneOnlyPose *e = new g2o::EdgeVerticalPlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << verInfo, 0,
                            0, verInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesVerPlane.push_back(e);
                    vnIndexEdgeVerPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            // cout << " Ver Plane: " << PEror / PNum << endl;
        }

        if (nInitialCorrespondences < 3)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;

        for (size_t it = 0; it < 4; it++) {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;

//            int PNMono = 0;
//            double PEMono = 0, PMaxMono = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
                g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                //cout<<"optimize chi2"<<chi2<<endl;
//                PNMono++;
//                PEMono += chi2;
//                PMaxMono = PMaxMono > chi2 ? PMaxMono : chi2;

                if (chi2 > chi2Mono[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbOutlier[idx] = false;
                    vMonoPointInfo[i] = 1.0 / sqrt(chi2);
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

//            if (PNMono == 0)
//                cout << "No mono points " << " ";
//            else
//                cout << " Mono points: " << PEMono / PNMono << " ";

//            int PNStereo = 0;
//            double PEStereo = 0, PMaxStereo = 0;
            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                //cout<<"optimize chi2"<<chi2<<endl;
//                PNStereo++;
//                PEStereo += chi2;
//                PMaxStereo = PMaxStereo > chi2 ? PMaxStereo : chi2;

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                    vSteroPointInfo[i] = 1.0 / sqrt(chi2);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PNStereo == 0)
//                cout << "No stereo points " << " ";
//            else
//                cout << " Stereo points: " << PEStereo / PNStereo << endl;

//            int PNLine = 0;
//            double PELine = 0, PMaxLine = 0;
            for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
                EdgeLineProjectXYZOnlyPose *e1 = vpEdgesLineSp[i];  //线段起始点
                EdgeLineProjectXYZOnlyPose *e2 = vpEdgesLineEp[i];  //线段终止点

                const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

                if (pFrame->mvbLineOutlier[idx]) {
                    e1->computeError();
                    e2->computeError();
                }
                e1->computeError();
                e2->computeError();

                const float chi2_s = e1->chiline();//e1->chi2();
                const float chi2_e = e2->chiline();//e2->chi2();
//                cout<<"Optimization: chi2_s "<<chi2_s<<", chi2_e "<<chi2_e<<endl;

//                PNLine++;
//                PELine += chi2_s + chi2_e;
//                PMaxLine = PMaxLine > chi2_s + chi2_e ? PMaxLine : chi2_s + chi2_e;


                if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
                    pFrame->mvbLineOutlier[idx] = true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbLineOutlier[idx] = false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                    vSteroEndPointInfo[i] = 1.0 / sqrt(chi2_e);
                    vSteroStartPointInfo[i] = 1.0 / sqrt(chi2_s);
                }

                if (it == 2) {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }

//            if (PNLine == 0)
//                cout << "No lines " << " ";
//            else
//                cout << " Lines: " << PELine / PNLine << endl;

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgePlaneOnlyPose *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                    // cout << "bad: " << chi2 << ", id: " << idx << "  Pc : " << pFrame->mvPlaneCoefficients[idx].t()
                    //      << "  Pw :" << (pFrame->mTwc.t() * pFrame->mvpMapPlanes[idx]->GetWorldPos()).t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);

//                if (vpEdgesPlanePoint[i].size() > 0) {
//                    int PPN = 0;
//                    double PPE = 0, PPMax = 0;
//                    for (size_t j = 0, jend = vpEdgesPlanePoint[i].size(); j < jend; j++) {
//                        g2o::EdgePlanePoint *edge = vpEdgesPlanePoint[i][j];
//
//                        const size_t index = vnIndexEdgePlanePoint[i][j];
//
//                        const float chi2 = edge->chi2();
////                    cout<<"optimize chi2"<<chi2<<endl;
//                        PPN++;
//                        PPE += chi2;
//                        PPMax = PPMax > chi2 ? PPMax : chi2;
//
//                        if (chi2 > chi2Mono[it]) {
//                            edge->setLevel(1);
//                            nBad++;
//                        } else {
//                            edge->setLevel(0);
//                        }
//
//                        if (it == 2)
//                            edge->setRobustKernel(0);
//                    }
//
//                    if (PPN == 0)
//                        cout << "planetest No plane point matches " << " ";
//                    else
//                        cout << "planetest  Plane point matches: " << PPE / PPN << " "; //<< " Max: " << PMax << endl;
//                }
            }
            // if (PN == 0)
            //     cout << "No plane " << " ";
            // else
            //     cout << " Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;
            for (size_t i = 0, iend = vpEdgesParPlane.size(); i < iend; i++) {
                g2o::EdgeParallelPlaneOnlyPose *e = vpEdgesParPlane[i];

                const size_t idx = vnIndexEdgeParPlane[i];

                if (pFrame->mvbParPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbParPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                    // cout << "bad Par: " << chi2 << ", id: " << idx << "  Pc : "
                        //  << pFrame->mvPlaneCoefficients[idx].t() << "  Pw :"
                        //  << (pFrame->mTwc.t() * pFrame->mvpParallelPlanes[idx]->GetWorldPos()).t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbParPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
            // if (PN == 0)
            //     cout << "No par plane " << " ";
            // else
            //     cout << "par Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;

            for (size_t i = 0, iend = vpEdgesVerPlane.size(); i < iend; i++) {
                g2o::EdgeVerticalPlaneOnlyPose *e = vpEdgesVerPlane[i];

                const size_t idx = vnIndexEdgeVerPlane[i];

                if (pFrame->mvbVerPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbVerPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                    // cout << "bad Ver: " << chi2 << ", id: " << idx << "  Pc : "
                    //      << pFrame->mvPlaneCoefficients[idx].t() << "  Pw :"
                    //      << (pFrame->mTwc.t() * pFrame->mvpVerticalPlanes[idx]->GetWorldPos()).t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbVerPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
            // if (PN == 0)
            //     cout << "No Ver plane " << endl;
            // else
            //     cout << "Ver Plane: " << PE / PN << endl; //<< " Max: " << PMax << endl;

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        pFrame->SetPose(Converter::toCvMat(SE3quat_recov));

        return nInitialCorrespondences - nBad;
    }

    void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap) {
        // Local KeyFrames: First Breath Search from Current Keyframe
        list<KeyFrame*> lLocalKeyFrames;

        lLocalKeyFrames.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;

        const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
        {
            KeyFrame* pKFi = vNeighKFs[i];
            pKFi->mnBALocalForKF = pKF->mnId;
            if(!pKFi->isBad())
                lLocalKeyFrames.push_back(pKFi);
        }

        // Local MapPoints seen in Local KeyFrames
        list<MapPoint*> lLocalMapPoints;
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
            }
        }

        // Local MapLines seen in Local KeyFrames
        list<MapLine *> lLocalMapLines;
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
            vector<MapLine *> vpMLs = (*lit)->GetMapLineMatches();
            for (vector<MapLine *>::iterator vit = vpMLs.begin(), vend = vpMLs.end(); vit != vend; vit++) {

                MapLine *pML = *vit;
                if (pML) {
                    if (!pML->isBad()) {
                        if (pML->mnBALocalForKF != pKF->mnId) {
                            lLocalMapLines.push_back(pML);
                            pML->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        // Local MapPlanes seen in Local KeyFrames
        list<MapPlane *> lLocalMapPlanes;
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
            vector<MapPlane *> vpMPs = (*lit)->GetMapPlaneMatches();
            for (vector<MapPlane *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {

                MapPlane *pMP = *vit;
                if (pMP) {
                    if (!pMP->isBad()) {
                        if (pMP->mnBALocalForKF != pKF->mnId) {
                            lLocalMapPlanes.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
        list<KeyFrame*> lFixedCameras;
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }
        for(list<MapLine*>::iterator lit=lLocalMapLines.begin(), lend=lLocalMapLines.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }
        for(list<MapPlane*>::iterator lit=lLocalMapPlanes.begin(), lend=lLocalMapPlanes.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;

        // Set Local KeyFrame vertices
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId==0);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }

        // Set Fixed KeyFrame vertices
        for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }

        // Set MapPoint vertices
        const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

        vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        long unsigned int maxMapPointId = maxKFid;

        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            int id = pMP->mnId+maxKFid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            if (id > maxMapPointId) {
                maxMapPointId = id;
            }

            const map<KeyFrame*,size_t> observations = pMP->GetObservations();

            //Set edges
            for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(!pKFi->isBad())
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                    // Monocular observation
                    if(pKFi->mvuRight[mit->second]<0)
                    {
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                    else // Stereo observation
                    {
                        Eigen::Matrix<double,3,1> obs;
                        const float kp_ur = pKFi->mvuRight[mit->second];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;
                        e->bf = pKFi->mbf;

                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);
                    }
                }
            }
        }

        const int nLineExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapLines.size();

        vector<EdgeLineProjectXYZ*> vpLineEdgesStart;
        vpLineEdgesStart.reserve(nLineExpectedSize);

        vector<EdgeLineProjectXYZ*> vpLineEdgesEnd;
        vpLineEdgesEnd.reserve(nLineExpectedSize);

        vector<KeyFrame*> vpLineEdgeKF;
        vpLineEdgeKF.reserve(nLineExpectedSize);

        vector<MapLine*> vpMapLineEdge;
        vpMapLineEdge.reserve(nLineExpectedSize);

        long unsigned int maxMapLineId = maxMapPointId;

        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
            MapLine *pML = *lit;
            g2o::VertexSBAPointXYZ *vStartPoint = new g2o::VertexSBAPointXYZ();
            vStartPoint->setEstimate(pML->GetWorldPos().head(3));
            int id1 = (2 * pML->mnId) + 1 + maxMapPointId;
            vStartPoint->setId(id1);
            vStartPoint->setMarginalized(true);
            optimizer.addVertex(vStartPoint);

            g2o::VertexSBAPointXYZ *vEndPoint = new VertexSBAPointXYZ();
            vEndPoint->setEstimate(pML->GetWorldPos().tail(3));
            int id2 = (2 * (pML->mnId + 1)) + maxMapPointId;
            vEndPoint->setId(id2);
            vEndPoint->setMarginalized(true);
            optimizer.addVertex(vEndPoint);

            if (id2 > maxMapLineId) {
                maxMapLineId = id2;
            }

            const map<KeyFrame *, size_t> observations = pML->GetObservations();

            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end();
                 mit != mend; mit++) {
                KeyFrame *pKFi = mit->first;

                if (!pKFi->isBad()) {

                    Eigen::Vector3d lineObs = pKF->mvKeyLineFunctions[mit->second];

                    EdgeLineProjectXYZ *es = new EdgeLineProjectXYZ();
                    es->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
                    es->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    es->setMeasurement(lineObs);
                    es->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rks = new g2o::RobustKernelHuber;
                    es->setRobustKernel(rks);
                    rks->setDelta(thHuberStereo);

                    es->fx = pKF->fx;
                    es->fy = pKF->fy;
                    es->cx = pKF->cx;
                    es->cy = pKF->cy;

                    optimizer.addEdge(es);

                    EdgeLineProjectXYZ *ee = new EdgeLineProjectXYZ();
                    ee->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
                    ee->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    ee->setMeasurement(lineObs);
                    ee->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rke = new g2o::RobustKernelHuber;
                    ee->setRobustKernel(rke);
                    rke->setDelta(thHuberStereo);

                    ee->fx = pKF->fx;
                    ee->fy = pKF->fy;
                    ee->cx = pKF->cx;
                    ee->cy = pKF->cy;

                    optimizer.addEdge(ee);

                    vpLineEdgesStart.push_back(es);
                    vpLineEdgesEnd.push_back(ee);
                    vpLineEdgeKF.push_back(pKFi);
                    vpMapLineEdge.push_back(pML);
                }
            }
        }

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        const int nPlaneExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPlanes.size();

        vector<g2o::EdgePlane*> vpPlaneEdges;
        vpPlaneEdges.reserve(nPlaneExpectedSize);

        vector<g2o::EdgeVerticalPlane*> vpVerPlaneEdges;
        vpVerPlaneEdges.reserve(nPlaneExpectedSize);

        vector<g2o::EdgeParallelPlane*> vpParPlaneEdges;
        vpParPlaneEdges.reserve(nPlaneExpectedSize);

        vector<KeyFrame*> vpPlaneEdgeKF;
        vpLineEdgeKF.reserve(nPlaneExpectedSize);

        vector<KeyFrame*> vpVerPlaneEdgeKF;
        vpVerPlaneEdgeKF.reserve(nPlaneExpectedSize);

        vector<KeyFrame*> vpParPlaneEdgeKF;
        vpParPlaneEdgeKF.reserve(nPlaneExpectedSize);

        vector<MapPlane*> vpMapPlaneEdge;
        vpMapPlaneEdge.reserve(nPlaneExpectedSize);

        vector<MapPlane*> vpVerMapPlaneEdge;
        vpVerMapPlaneEdge.reserve(nPlaneExpectedSize);

        vector<MapPlane*> vpParMapPlaneEdge;
        vpParMapPlaneEdge.reserve(nPlaneExpectedSize);

        // Set MapPlane vertices
        for (list<MapPlane *>::iterator lit = lLocalMapPlanes.begin(), lend = lLocalMapPlanes.end(); lit != lend; lit++) {
            MapPlane *pMP = *lit;

            g2o::VertexPlane *vPlane = new g2o::VertexPlane();
            vPlane->setEstimate(Converter::toPlane3D(pMP->GetWorldPos()));
            const int id = pMP->mnId + maxMapLineId + 1;
            vPlane->setId(id);
            vPlane->setMarginalized(true);
            optimizer.addVertex(vPlane);

            Eigen::Matrix3d Info;
            Info << angleInfo, 0, 0,
                    0, angleInfo, 0,
                    0, 0, disInfo;

            Eigen::Matrix2d VPInfo;
            VPInfo << angleInfo, 0,
                    0, angleInfo;

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            for (const auto &observation : observations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                g2o::EdgePlane *e = new g2o::EdgePlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaPlane);

                optimizer.addEdge(e);
                vpPlaneEdges.push_back(e);
                vpPlaneEdgeKF.push_back(pKF);
                vpMapPlaneEdge.push_back(pMP);
            }

            const map<KeyFrame *, size_t> verObservations = pMP->GetVerObservations();
            for (const auto &observation : verObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                if (optimizer.vertex(pKF->mnId) == 0)
                {
                    cout << "WHY?" << endl;
                    // TODO: check!
                    // maybe something wrong in adding observation?
                    continue;
                }

                g2o::EdgeVerticalPlane *e = new g2o::EdgeVerticalPlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                e->setInformation(VPInfo);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
                vpVerPlaneEdges.push_back(e);
                vpVerPlaneEdgeKF.push_back(pKF);
                vpVerMapPlaneEdge.push_back(pMP);
            }

            const map<KeyFrame *, size_t> parObservations = pMP->GetParObservations();
            for (const auto &observation : parObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                if (optimizer.vertex(pKF->mnId) == 0)
                {
                    cout << "WHY?" << endl;
                    // TODO: check!
                    // maybe something wrong in adding observation?
                    continue;
                }

                g2o::EdgeParallelPlane *e = new g2o::EdgeParallelPlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                e->setInformation(VPInfo);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
                vpParPlaneEdges.push_back(e);
                vpParPlaneEdgeKF.push_back(pKF);
                vpParMapPlaneEdge.push_back(pMP);
            }
        }

        if(pbStopFlag)
            if(*pbStopFlag)
                return;

        optimizer.initializeOptimization();
        optimizer.optimize(5);

        bool bDoMore= true;

        if(pbStopFlag)
            if(*pbStopFlag)
                bDoMore = false;

        if(bDoMore)
        {

            // Check inlier observations
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
            {
                g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
                MapPoint* pMP = vpMapPointEdgeMono[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>5.991 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
            {
                g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
                MapPoint* pMP = vpMapPointEdgeStereo[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>7.815 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpLineEdgesStart.size(); i < iend; i++) {
                EdgeLineProjectXYZ *es = vpLineEdgesStart[i];
                EdgeLineProjectXYZ *ee = vpLineEdgesEnd[i];
                MapLine *pML = vpMapLineEdge[i];

                if (pML->isBad())
                    continue;

                if (es->chi2() > 7.815 || ee->chi2() > 7.815) {
                    es->setLevel(1);
                    ee->setLevel(1);
                }

                es->setRobustKernel(0);
                ee->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpPlaneEdges.size(); i<iend;i++)
            {
                g2o::EdgePlane* e = vpPlaneEdges[i];
                MapPlane* pMP = vpMapPlaneEdge[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>planeChi)
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpVerPlaneEdges.size(); i<iend;i++)
            {
                g2o::EdgeVerticalPlane* e = vpVerPlaneEdges[i];
                MapPlane* pMP = vpVerMapPlaneEdge[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>VPplaneChi)
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpParPlaneEdges.size(); i<iend;i++)
            {
                g2o::EdgeParallelPlane* e = vpParPlaneEdges[i];
                MapPlane* pMP = vpParMapPlaneEdge[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>VPplaneChi)
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            // Optimize again without the outliers

            optimizer.initializeOptimization(0);
            optimizer.optimize(10);

        }

        vector<pair<KeyFrame*,MapPoint*> > vToErase;
        vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

        // Check inlier observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFMono[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        vector<pair<KeyFrame *, MapLine *>> vLineToErase;
        vLineToErase.reserve(vpLineEdgesStart.size());

        for (size_t i = 0, iend = vpLineEdgesStart.size(); i < iend; i++) {
            EdgeLineProjectXYZ *es = vpLineEdgesStart[i];
            EdgeLineProjectXYZ *ee = vpLineEdgesEnd[i];
            MapLine *pML = vpMapLineEdge[i];

            if (pML->isBad())
                continue;

            if (es->chi2() > 7.815 || ee->chi2() > 7.815) {
                KeyFrame *pKFi = vpLineEdgeKF[i];
                vLineToErase.push_back(make_pair(pKFi, pML));
            }
        }

        vector<pair<KeyFrame*,MapPlane*> > vPlaneToErase;
        vPlaneToErase.reserve(vpPlaneEdges.size());

        for(size_t i=0, iend=vpPlaneEdges.size(); i<iend;i++)
        {
            g2o::EdgePlane* e = vpPlaneEdges[i];
            MapPlane* pMP = vpMapPlaneEdge[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>planeChi)
            {
                KeyFrame* pKFi = vpPlaneEdgeKF[i];
                vPlaneToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        vector<pair<KeyFrame*,MapPlane*> > vVerPlaneToErase;
        vVerPlaneToErase.reserve(vpVerPlaneEdges.size());

        for(size_t i=0, iend=vpVerPlaneEdges.size(); i<iend;i++)
        {
            g2o::EdgeVerticalPlane* e = vpVerPlaneEdges[i];
            MapPlane* pMP = vpVerMapPlaneEdge[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>VPplaneChi)
            {
                KeyFrame* pKFi = vpVerPlaneEdgeKF[i];
                vVerPlaneToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        vector<pair<KeyFrame*,MapPlane*> > vParPlaneToErase;
        vParPlaneToErase.reserve(vpParPlaneEdges.size());

        for(size_t i=0, iend=vpParPlaneEdges.size(); i<iend;i++)
        {
            g2o::EdgeParallelPlane* e = vpParPlaneEdges[i];
            MapPlane* pMP = vpParMapPlaneEdge[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>VPplaneChi)
            {
                KeyFrame* pKFi = vpParPlaneEdgeKF[i];
                vParPlaneToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        if(!vToErase.empty())
        {
            for(size_t i=0;i<vToErase.size();i++)
            {
                KeyFrame* pKFi = vToErase[i].first;
                MapPoint* pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        if(!vLineToErase.empty())
        {
            for(size_t i=0;i<vLineToErase.size();i++)
            {
                KeyFrame* pKFi = vLineToErase[i].first;
                MapLine* pMLi = vLineToErase[i].second;
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseObservation(pKFi);
            }
        }

        if(!vPlaneToErase.empty())
        {
            for(size_t i=0;i<vPlaneToErase.size();i++)
            {
                KeyFrame* pKFi = vPlaneToErase[i].first;
                MapPlane* pMPi = vPlaneToErase[i].second;
                pKFi->EraseMapPlaneMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        if(!vVerPlaneToErase.empty())
        {
            for(size_t i=0;i<vVerPlaneToErase.size();i++)
            {
                KeyFrame* pKFi = vVerPlaneToErase[i].first;
                MapPlane* pMPi = vVerPlaneToErase[i].second;
                pKFi->EraseMapVerticalPlaneMatch(pMPi);
                pMPi->EraseVerObservation(pKFi);
            }
        }

        if(!vParPlaneToErase.empty())
        {
            for(size_t i=0;i<vParPlaneToErase.size();i++)
            {
                KeyFrame* pKFi = vParPlaneToErase[i].first;
                MapPlane* pMPi = vParPlaneToErase[i].second;
                pKFi->EraseMapParallelPlaneMatch(pMPi);
                pMPi->EraseParObservation(pKFi);
            }
        }

        // Recover optimized data

        //Keyframes
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKF = *lit;
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }

        //Points
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }

        // Lines
        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
            MapLine *pML = *lit;

            g2o::VertexSBAPointXYZ *vStartPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                    (2 * pML->mnId) + 1 + maxMapPointId));
            g2o::VertexSBAPointXYZ *vEndPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                    (2 * (pML->mnId + 1)) + maxMapPointId));

            Vector6d LinePos;
            LinePos << Converter::toVector3d(Converter::toCvMat(vStartPoint->estimate())), Converter::toVector3d(
                    Converter::toCvMat(vEndPoint->estimate()));
            pML->SetWorldPos(LinePos);
            pML->UpdateAverageDir();
        }

        //Planes
        for (list<MapPlane *>::iterator lit = lLocalMapPlanes.begin(), lend = lLocalMapPlanes.end(); lit != lend; lit++) {
            MapPlane *pMP = *lit;
            g2o::VertexPlane *vPlane = static_cast<g2o::VertexPlane *>(optimizer.vertex(
                    pMP->mnId + maxMapLineId + 1));
            pMP->SetWorldPos(Converter::toCvMat(vPlane->estimate()));
            pMP->UpdateCoefficientsAndPoints();
        }

        pMap->IncreaseChangeIndex();
    }

    void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                           const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                           const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                           const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                           const bool &bFixScale) {
        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(false);
        g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
                new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
        g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        solver->setUserLambdaInit(1e-16);
        optimizer.setAlgorithm(solver);

        const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
        const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();
        const vector<MapLine *> vpMLs = pMap->GetAllMapLines();
        const vector<MapPlane *> vpMPLs = pMap->GetAllMapPlanes();

        const unsigned int nMaxKFid = pMap->GetMaxKFid();

        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid + 1);
        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid + 1);
        vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

        const int minFeat = 100;

        // Set KeyFrame vertices
        for (auto pKF : vpKFs) {
            if (pKF->isBad())
                continue;
            g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

            const int nIDi = pKF->mnId;

            auto it = CorrectedSim3.find(pKF);

            if (it != CorrectedSim3.end()) {
                vScw[nIDi] = it->second;
                VSim3->setEstimate(it->second);
            } else {
                Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
                Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
                g2o::Sim3 Siw(Rcw, tcw, 1.0);
                vScw[nIDi] = Siw;
                VSim3->setEstimate(Siw);
            }

            if (pKF == pLoopKF)
                VSim3->setFixed(true);

            VSim3->setId(nIDi);
            VSim3->setMarginalized(false);
            VSim3->_fix_scale = bFixScale;

            optimizer.addVertex(VSim3);

            vpVertices[nIDi] = VSim3;
        }


        set<pair<long unsigned int, long unsigned int> > sInsertedEdges;

        const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

        // Set Loop edges
        for (const auto &LoopConnection : LoopConnections) {
            KeyFrame *pKF = LoopConnection.first;
            const long unsigned int nIDi = pKF->mnId;
            const set<KeyFrame *> &spConnections = LoopConnection.second;
            const g2o::Sim3 Siw = vScw[nIDi];
            const g2o::Sim3 Swi = Siw.inverse();

            for (auto spConnection : spConnections) {
                const long unsigned int nIDj = spConnection->mnId;
                if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(spConnection) < minFeat)
                    continue;

                const g2o::Sim3 Sjw = vScw[nIDj];
                const g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;

                optimizer.addEdge(e);

                sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
            }
        }

        // Set normal edges
        for (auto pKF : vpKFs) {
            const int nIDi = pKF->mnId;

            g2o::Sim3 Swi;

            auto iti = NonCorrectedSim3.find(pKF);

            if (iti != NonCorrectedSim3.end())
                Swi = (iti->second).inverse();
            else
                Swi = vScw[nIDi].inverse();

            KeyFrame *pParentKF = pKF->GetParent();

            // Spanning tree edge
            if (pParentKF) {
                int nIDj = pParentKF->mnId;

                g2o::Sim3 Sjw;

                auto itj = NonCorrectedSim3.find(pParentKF);

                if (itj != NonCorrectedSim3.end())
                    Sjw = itj->second;
                else
                    Sjw = vScw[nIDj];

                g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;
                optimizer.addEdge(e);
            }

            // Loop edges
            const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
            for (auto pLKF : sLoopEdges) {
                if (pLKF->mnId < pKF->mnId) {
                    g2o::Sim3 Slw;

                    auto itl = NonCorrectedSim3.find(pLKF);

                    if (itl != NonCorrectedSim3.end())
                        Slw = itl->second;
                    else
                        Slw = vScw[pLKF->mnId];

                    g2o::Sim3 Sli = Slw * Swi;
                    g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                }
            }

            // Covisibility graph edges
            const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
            for (auto pKFn : vpConnectedKFs) {
                if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn)) {
                    if (!pKFn->isBad() && pKFn->mnId < pKF->mnId) {
                        if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId))))
                            continue;

                        g2o::Sim3 Snw;

                        auto itn = NonCorrectedSim3.find(pKFn);

                        if (itn != NonCorrectedSim3.end())
                            Snw = itn->second;
                        else
                            Snw = vScw[pKFn->mnId];

                        g2o::Sim3 Sni = Snw * Swi;

                        g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                    }
                }
            }

            // inertial edges
            if(pKF->bImu && pKF->mPrevKF)
            {
                g2o::Sim3 Spw;
                LoopClosing::KeyFrameAndPose::const_iterator itp = NonCorrectedSim3.find(pKF->mPrevKF);
                if(itp!=NonCorrectedSim3.end())
                    Spw = itp->second;
                else
                    Spw = vScw[pKF->mPrevKF->mnId];

                g2o::Sim3 Spi = Spw * Swi;
                g2o::EdgeSim3* ep = new g2o::EdgeSim3();
                ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mPrevKF->mnId)));
                ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                ep->setMeasurement(Spi);
                ep->information() = matLambda;
                optimizer.addEdge(ep);
            }
        }

        

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(20);

        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
        for (auto pKFi : vpKFs) {
            const int nIDi = pKFi->mnId;

            g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
            g2o::Sim3 CorrectedSiw = VSim3->estimate();
            vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
            Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = CorrectedSiw.translation();
            double s = CorrectedSiw.scale();

            eigt *= (1. / s); //[R t/s;0 1]

            cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

            pKFi->SetPose(Tiw);
        }

        // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
        for (auto pMP : vpMPs) {
            if (pMP->isBad())
                continue;

            int nIDr;
            if (pMP->mnCorrectedByKF == pCurKF->mnId) {
                nIDr = pMP->mnCorrectedReference;
            } else {
                KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;
            }


            g2o::Sim3 Srw = vScw[nIDr];
            g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

            cv::Mat P3Dw = pMP->GetWorldPos();
            Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
            Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

            cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
            pMP->SetWorldPos(cvCorrectedP3Dw);

            pMP->UpdateNormalAndDepth();
        }

        // Correct lines. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
        for (auto pML : vpMLs) {
            if (pML->isBad())
                continue;

            int nIDr;
            if (pML->mnCorrectedByKF == pCurKF->mnId) {
                nIDr = pML->mnCorrectedReference;
            } else {
                KeyFrame *pRefKF = pML->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;
            }


            g2o::Sim3 Srw = vScw[nIDr];
            g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

            Eigen::Vector3d eigSP3Dw = pML->mWorldPos.head(3);
            Eigen::Vector3d eigEP3Dw = pML->mWorldPos.tail(3);

            Eigen::Matrix<double, 3, 1> eigCorrectedSP3Dw = correctedSwr.map(Srw.map(eigSP3Dw));
            Eigen::Matrix<double, 3, 1> eigCorrectedEP3Dw = correctedSwr.map(Srw.map(eigEP3Dw));

            Vector6d linePos;
            linePos << eigCorrectedSP3Dw(0), eigCorrectedSP3Dw(1), eigCorrectedSP3Dw(2), eigCorrectedEP3Dw(
                    0), eigCorrectedEP3Dw(1), eigCorrectedEP3Dw(2);
            pML->SetWorldPos(linePos);
            pML->ComputeDistinctiveDescriptors();
            pML->UpdateAverageDir();
        }

        // Correct planes. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
        for (auto pMP : vpMPLs) {
            if (pMP->isBad())
                continue;

            int nIDr;
            if (pMP->mnCorrectedByKF == pCurKF->mnId) {
                nIDr = pMP->mnCorrectedReference;
            } else {
                KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;
            }

            cv::Mat Srw = Converter::toCvMat(vScw[nIDr]);
            cv::Mat correctedSwr = Converter::toCvMat(vCorrectedSwc[nIDr]);

            cv::Mat sRSrw = Srw.rowRange(0, 3).colRange(0, 3);
            cv::Mat tSrw = Srw.rowRange(0, 3).col(3);

            cv::Mat sRCorrectedSwr = correctedSwr.rowRange(0, 3).colRange(0, 3);
            cv::Mat tCorrectedSwr = correctedSwr.rowRange(0, 3).col(3);

            cv::Mat P3Dw = pMP->GetWorldPos();

            cv::Mat correctedP3Dw = cv::Mat::eye(4, 1, CV_32F);

            correctedP3Dw.rowRange(0, 3).col(0) = sRSrw * P3Dw.rowRange(0, 3).col(0);
            correctedP3Dw.at<float>(3, 0) =
                    P3Dw.at<float>(3, 0) - tSrw.dot(correctedP3Dw.rowRange(0, 3).col(0));
            if (correctedP3Dw.at<float>(3, 0) < 0.0)
                correctedP3Dw = -correctedP3Dw;

            correctedP3Dw.rowRange(0, 3).col(0) = sRCorrectedSwr * correctedP3Dw.rowRange(0, 3).col(0);
            correctedP3Dw.at<float>(3, 0) =
                    correctedP3Dw.at<float>(3, 0) - tCorrectedSwr.dot(correctedP3Dw.rowRange(0, 3).col(0));
            if (correctedP3Dw.at<float>(3, 0) < 0.0)
                correctedP3Dw = -correctedP3Dw;

            pMP->SetWorldPos(correctedP3Dw);
            pMP->UpdateCoefficientsAndPoints();
        }

        // Not sure if it's being used
        pMap->IncreaseChangeIndex();
    }

    int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpPointMatches1,
                                vector<MapLine *> &vpLineMatches1,
                                vector<MapPlane *> &vpPlaneMatches1, vector<MapPlane *> &vpVerticalPlaneMatches1,
                                vector<MapPlane *> &vpParallelPlaneMatches1,
                                g2o::Sim3 &g2oS12, const float th2, const bool bFixScale) {

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Calibration
        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;

        // Camera poses
        const cv::Mat R1w = pKF1->GetRotation();
        const cv::Mat t1w = pKF1->GetTranslation();
        const cv::Mat R2w = pKF2->GetRotation();
        const cv::Mat t2w = pKF2->GetTranslation();

        // Set Sim3 vertex
        g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
        vSim3->_fix_scale = bFixScale;
        vSim3->setEstimate(g2oS12);
        vSim3->setId(0);
        vSim3->setFixed(false);
        vSim3->_principle_point1[0] = K1.at<float>(0, 2);
        vSim3->_principle_point1[1] = K1.at<float>(1, 2);
        vSim3->_focal_length1[0] = K1.at<float>(0, 0);
        vSim3->_focal_length1[1] = K1.at<float>(1, 1);
        vSim3->_principle_point2[0] = K2.at<float>(0, 2);
        vSim3->_principle_point2[1] = K2.at<float>(1, 2);
        vSim3->_focal_length2[0] = K2.at<float>(0, 0);
        vSim3->_focal_length2[1] = K2.at<float>(1, 1);
        optimizer.addVertex(vSim3);

        // Set MapPoint vertices
        const int N = vpPointMatches1.size();
        const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
        vector<g2o::EdgeSim3ProjectXYZ *> vpEdges12;
        vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdges21;
        vector<size_t> vnIndexEdge;

        vnIndexEdge.reserve(2 * N);
        vpEdges12.reserve(2 * N);
        vpEdges21.reserve(2 * N);

        const float deltaHuber = sqrt(th2);

        int nCorrespondences = 0, maxPointId = 0;

        for (int i = 0; i < N; i++) {
            if (!vpPointMatches1[i])
                continue;

            MapPoint *pMP1 = vpMapPoints1[i];
            MapPoint *pMP2 = vpPointMatches1[i];

            const int id1 = 2 * i + 1;
            const int id2 = 2 * (i + 1);

            const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

            if (pMP1 && pMP2) {
                if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0) {
                    maxPointId = id2;

                    g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
                    cv::Mat P3D1w = pMP1->GetWorldPos();
                    cv::Mat P3D1c = R1w * P3D1w + t1w;
                    vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                    vPoint1->setId(id1);
                    vPoint1->setFixed(true);
                    optimizer.addVertex(vPoint1);

                    g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                    cv::Mat P3D2w = pMP2->GetWorldPos();
                    cv::Mat P3D2c = R2w * P3D2w + t2w;
                    vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                    vPoint2->setId(id2);
                    vPoint2->setFixed(true);
                    optimizer.addVertex(vPoint2);
                } else
                    continue;
            } else
                continue;

            nCorrespondences++;

            // Set edge x1 = S12*X2
            Eigen::Matrix<double, 2, 1> obs1;
            const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
            obs1 << kpUn1.pt.x, kpUn1.pt.y;

            g2o::EdgeSim3ProjectXYZ *e12 = new g2o::EdgeSim3ProjectXYZ();
            e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
            e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e12->setMeasurement(obs1);
            const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
            e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

            g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
            e12->setRobustKernel(rk1);
            rk1->setDelta(deltaHuber);
            optimizer.addEdge(e12);

            // Set edge x2 = S21*X1
            Eigen::Matrix<double, 2, 1> obs2;
            const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;

            g2o::EdgeInverseSim3ProjectXYZ *e21 = new g2o::EdgeInverseSim3ProjectXYZ();

            e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
            e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e21->setMeasurement(obs2);
            float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
            e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

            g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
            e21->setRobustKernel(rk2);
            rk2->setDelta(deltaHuber);
            optimizer.addEdge(e21);

            vpEdges12.push_back(e12);
            vpEdges21.push_back(e21);
            vnIndexEdge.push_back(i);
        }

        // Set Line vertices
        const int NL = vpLineMatches1.size();
        const vector<MapLine *> vpMapLines1 = pKF1->GetMapLineMatches();
        vector<EdgeLineSim3Project *> vpLineStartEdges12, vpLineEndEdges12;
        vector<EdgeLineInverseSim3Project *> vpLineStartEdges21, vpLineEndEdges21;
        vector<size_t> vnIndexLineEdge;

        vnIndexLineEdge.reserve(2 * NL);
        vpLineStartEdges12.reserve(2 * NL);
        vpLineEndEdges12.reserve(2 * NL);
        vpLineStartEdges21.reserve(2 * NL);
        vpLineEndEdges21.reserve(2 * NL);

        int maxLineId = maxPointId;

        for (int i = 0; i < NL; i++) {
            if (!vpLineMatches1[i])
                continue;

            MapLine *pML1 = vpMapLines1[i];
            MapLine *pML2 = vpLineMatches1[i];

            const int id1 = maxPointId + (4 * i + 1);
            const int id2 = maxPointId + (4 * i + 2);
            const int id3 = maxPointId + (4 * i + 3);
            const int id4 = maxPointId + (4 * i + 4);

            const int i2 = pML2->GetIndexInKeyFrame(pKF2);

            if (pML1 && pML2) {
                if (!pML1->isBad() && !pML2->isBad() && i2 >= 0) {
                    if (id4 > maxLineId) {
                        maxLineId = id4;
                    }

                    g2o::VertexSBAPointXYZ *vLineStartPoint1 = new g2o::VertexSBAPointXYZ();
                    Vector3d pMLSP1 = pML1->mWorldPos.head(3);
                    cv::Mat LSP3D1w = Converter::toCvMat(pMLSP1);
                    cv::Mat LSP3D1c = R1w * LSP3D1w + t1w;
                    vLineStartPoint1->setEstimate(Converter::toVector3d(LSP3D1c));
                    vLineStartPoint1->setId(id1);
                    vLineStartPoint1->setFixed(true);
                    optimizer.addVertex(vLineStartPoint1);

                    g2o::VertexSBAPointXYZ *vLineEndPoint1 = new g2o::VertexSBAPointXYZ();
                    Vector3d pMLEP1 = pML1->mWorldPos.tail(3);
                    cv::Mat LEP3D1w = Converter::toCvMat(pMLEP1);
                    cv::Mat LEP3D1c = R1w * LEP3D1w + t1w;
                    vLineEndPoint1->setEstimate(Converter::toVector3d(LEP3D1c));
                    vLineEndPoint1->setId(id2);
                    vLineEndPoint1->setFixed(true);
                    optimizer.addVertex(vLineEndPoint1);

                    g2o::VertexSBAPointXYZ *vLineStartPoint2 = new g2o::VertexSBAPointXYZ();
                    Vector3d pMLSP2 = pML2->mWorldPos.head(3);
                    cv::Mat LSP3D2w = Converter::toCvMat(pMLSP2);
                    cv::Mat LSP3D2c = R2w * LSP3D2w + t2w;
                    vLineStartPoint2->setEstimate(Converter::toVector3d(LSP3D2c));
                    vLineStartPoint2->setId(id3);
                    vLineStartPoint2->setFixed(true);
                    optimizer.addVertex(vLineStartPoint2);

                    g2o::VertexSBAPointXYZ *vLineEndPoint2 = new g2o::VertexSBAPointXYZ();
                    Vector3d pMLEP2 = pML2->mWorldPos.tail(3);
                    cv::Mat LEP3D2w = Converter::toCvMat(pMLEP2);
                    cv::Mat LEP3D2c = R2w * LEP3D2w + t2w;
                    vLineEndPoint2->setEstimate(Converter::toVector3d(LEP3D2c));
                    vLineEndPoint2->setId(id4);
                    vLineEndPoint2->setFixed(true);
                    optimizer.addVertex(vLineEndPoint2);
                } else
                    continue;
            } else
                continue;

            nCorrespondences++;

            // Set edge x1 = S12*X2
            Eigen::Vector3d lineObs1 = pKF1->mvKeyLineFunctions[i];
            const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[pKF1->mvKeyLines[i].octave];

            EdgeLineSim3Project *es12 = new EdgeLineSim3Project();
            es12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id3)));
            es12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            es12->setMeasurement(lineObs1);
            es12->setInformation(Eigen::Matrix3d::Identity() * invSigmaSquare1);

            g2o::RobustKernelHuber *rks1 = new g2o::RobustKernelHuber;
            es12->setRobustKernel(rks1);
            rks1->setDelta(deltaHuber);
            optimizer.addEdge(es12);

            EdgeLineSim3Project *ee12 = new EdgeLineSim3Project();
            ee12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id4)));
            ee12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            ee12->setMeasurement(lineObs1);
            ee12->setInformation(Eigen::Matrix3d::Identity() * invSigmaSquare1);

            g2o::RobustKernelHuber *rke1 = new g2o::RobustKernelHuber;
            ee12->setRobustKernel(rke1);
            rke1->setDelta(deltaHuber);
            optimizer.addEdge(ee12);

            // Set edge x2 = S21*X1
            Eigen::Vector3d lineObs2 = pKF2->mvKeyLineFunctions[i];
            const float &invSigmaSquare2 = pKF2->mvInvLevelSigma2[pKF2->mvKeyLines[i].octave];

            EdgeLineInverseSim3Project *es21 = new EdgeLineInverseSim3Project();
            es21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
            es21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            es21->setMeasurement(lineObs2);
            es21->setInformation(Eigen::Matrix3d::Identity() * invSigmaSquare2);

            g2o::RobustKernelHuber *rks2 = new g2o::RobustKernelHuber;
            es21->setRobustKernel(rks2);
            rks2->setDelta(deltaHuber);
            optimizer.addEdge(es21);

            EdgeLineInverseSim3Project *ee21 = new EdgeLineInverseSim3Project();
            ee21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
            ee21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            ee21->setMeasurement(lineObs2);
            ee21->setInformation(Eigen::Matrix3d::Identity() * invSigmaSquare2);

            g2o::RobustKernelHuber *rke2 = new g2o::RobustKernelHuber;
            ee21->setRobustKernel(rke2);
            rke2->setDelta(deltaHuber);
            optimizer.addEdge(ee21);

            vpLineStartEdges12.push_back(es12);
            vpLineEndEdges12.push_back(ee12);
            vpLineStartEdges21.push_back(es21);
            vpLineEndEdges21.push_back(ee21);
            vnIndexLineEdge.push_back(i);
        }

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        std::map<int, int> vnVertexId;

        // Set MapPlane vertices
        const int NP = vpPlaneMatches1.size();
        const vector<MapPlane *> vpMapPlanes1 = pKF1->GetMapPlaneMatches();
        vector<g2o::EdgePlaneSim3Project *> vpPlaneEdges12;
        vector<g2o::EdgePlaneInverseSim3Project *> vpPlaneEdges21;
        vector<size_t> vnIndexPlaneEdge;

        vnIndexPlaneEdge.reserve(2 * NP);
        vpPlaneEdges12.reserve(2 * NP);
        vpPlaneEdges21.reserve(2 * NP);

        int maxPlaneId = maxLineId;

        for (int i = 0; i < NP; i++) {
            if (!vpPlaneMatches1[i])
                continue;

            MapPlane *pMP1 = vpMapPlanes1[i];
            MapPlane *pMP2 = vpPlaneMatches1[i];

            int id1 = maxLineId + 2 * i + 1;
            int id2 = maxLineId + 2 * (i + 1);

            const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

            if (pMP1 && pMP2) {
                if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0) {
                    if (vnVertexId.count(pMP1->mnId) == 0) {
                        g2o::VertexPlane *vPlane1 = new g2o::VertexPlane();
                        cv::Mat P3D1w = pMP1->GetWorldPos();

                        cv::Mat P3D1c = cv::Mat::eye(4, 1, CV_32F);
                        P3D1c.rowRange(0, 3).col(0) = R1w * P3D1w.rowRange(0, 3).col(0);
                        P3D1c.at<float>(3, 0) = P3D1w.at<float>(3, 0) - t1w.dot(P3D1c.rowRange(0, 3).col(0));
                        if (P3D1c.at<float>(3, 0) < 0.0)
                            P3D1c = -P3D1c;

                        vPlane1->setEstimate(Converter::toPlane3D(P3D1c));
                        vPlane1->setId(id1);
                        vPlane1->setFixed(true);
                        optimizer.addVertex(vPlane1);
                        vnVertexId[pMP1->mnId] = id1;
                    } else {
                        id1 = vnVertexId[pMP1->mnId];
                    }

                    if (vnVertexId.count(pMP2->mnId) == 0) {
                        g2o::VertexPlane *vPlane2 = new g2o::VertexPlane();
                        cv::Mat P3D2w = pMP2->GetWorldPos();

                        cv::Mat P3D2c = cv::Mat::eye(4, 1, CV_32F);
                        P3D2c.rowRange(0, 3).col(0) = R2w * P3D2w.rowRange(0, 3).col(0);
                        P3D2c.at<float>(3, 0) = P3D2w.at<float>(3, 0) - t2w.dot(P3D2c.rowRange(0, 3).col(0));
                        if (P3D2c.at<float>(3, 0) < 0.0)
                            P3D2c = -P3D2c;

                        vPlane2->setEstimate(Converter::toPlane3D(P3D2c));
                        vPlane2->setId(id2);
                        vPlane2->setFixed(true);
                        optimizer.addVertex(vPlane2);
                        vnVertexId[pMP2->mnId] = id2;
                    } else {
                        id2 = vnVertexId[pMP2->mnId];
                    }

                    int id = std::max(id1, id2);
                    if (id > maxPlaneId) {
                        maxPlaneId = id;
                    }
                } else
                    continue;
            } else
                continue;

            std::cout << "Plane sim3 optim" << std::endl;

            nCorrespondences++;

            g2o::EdgePlaneSim3Project *e12 = new g2o::EdgePlaneSim3Project();
            e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
            e12->setMeasurement(Converter::toPlane3D(pKF1->mvPlaneCoefficients[i]));

            Eigen::Matrix3d Info;
            Info << angleInfo, 0, 0,
                    0, angleInfo, 0,
                    0, 0, disInfo;
            e12->setInformation(Info);

            g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
            e12->setRobustKernel(rk1);
            rk1->setDelta(deltaPlane);

            // PointCloud::Ptr localPoints2(new PointCloud());
//            pcl::transformPointCloud(*(pMP2->mvPlanePoints), *localPoints2, Converter::toMatrix4d(pKF2->GetPose()));

            // e12->planePoints = localPoints2;

            optimizer.addEdge(e12);

            g2o::EdgePlaneInverseSim3Project *e21 = new g2o::EdgePlaneInverseSim3Project();
            e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
            e21->setMeasurement(Converter::toPlane3D(pKF2->mvPlaneCoefficients[i2]));
            e21->setInformation(Info);

            g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
            e21->setRobustKernel(rk2);
            rk2->setDelta(deltaPlane);

            // PointCloud::Ptr localPoints1(new PointCloud());
//            pcl::transformPointCloud(*(pMP1->mvPlanePoints), *localPoints1, Converter::toMatrix4d(pKF1->GetPose()));

            // e21->planePoints = localPoints1;

            optimizer.addEdge(e21);

            vpPlaneEdges12.push_back(e12);
            vpPlaneEdges21.push_back(e21);
            vnIndexPlaneEdge.push_back(i);
        }

        const int NVP = vpVerticalPlaneMatches1.size();
        const vector<MapPlane *> vpMapVerticalPlanes1 = pKF1->GetMapVerticalPlaneMatches();
        vector<g2o::EdgeVerticalPlaneSim3Project *> vpVerticalPlaneEdges12;
        vector<g2o::EdgeVerticalPlaneInverseSim3Project *> vpVerticalPlaneEdges21;
        vector<size_t> vnIndexVerticalPlaneEdge;

        vnIndexVerticalPlaneEdge.reserve(2 * NVP);
        vpVerticalPlaneEdges12.reserve(2 * NVP);
        vpVerticalPlaneEdges21.reserve(2 * NVP);

        int maxVerticalPlaneId = maxPlaneId;

        for (int i = 0; i < NVP; i++) {
            if (!vpVerticalPlaneMatches1[i])
                continue;

            MapPlane *pMP1 = vpMapVerticalPlanes1[i];
            MapPlane *pMP2 = vpVerticalPlaneMatches1[i];

            int id1 = maxPlaneId + 2 * i + 1;
            int id2 = maxPlaneId + 2 * (i + 1);

            const int i2 = pMP2->GetIndexInVerticalKeyFrame(pKF2);

            if (pMP1 && pMP2) {
                if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0) {
                    if (vnVertexId.count(pMP1->mnId) == 0) {
                        g2o::VertexPlane *vPlane1 = new g2o::VertexPlane();
                        cv::Mat P3D1w = pMP1->GetWorldPos();
                        cv::Mat P3D1c = cv::Mat::eye(4, 1, CV_32F);
                        P3D1c.rowRange(0, 3).col(0) = R1w * P3D1w.rowRange(0, 3).col(0);
                        P3D1c.at<float>(3, 0) = P3D1w.at<float>(3, 0) - t1w.dot(P3D1c.rowRange(0, 3).col(0));
                        if (P3D1c.at<float>(3, 0) < 0.0)
                            P3D1c = -P3D1c;
                        vPlane1->setEstimate(Converter::toPlane3D(P3D1c));
                        vPlane1->setId(id1);
                        vPlane1->setFixed(true);
                        optimizer.addVertex(vPlane1);
                        vnVertexId[pMP1->mnId] = id1;
                    } else {
                        id1 = vnVertexId[pMP1->mnId];
                    }

                    if (vnVertexId.count(pMP2->mnId) == 0) {
                        g2o::VertexPlane *vPlane2 = new g2o::VertexPlane();
                        cv::Mat P3D2w = pMP2->GetWorldPos();
                        cv::Mat P3D2c = cv::Mat::eye(4, 1, CV_32F);
                        P3D2c.rowRange(0, 3).col(0) = R2w * P3D2w.rowRange(0, 3).col(0);
                        P3D2c.at<float>(3, 0) = P3D2w.at<float>(3, 0) - t2w.dot(P3D2c.rowRange(0, 3).col(0));
                        if (P3D2c.at<float>(3, 0) < 0.0)
                            P3D2c = -P3D2c;
                        vPlane2->setEstimate(Converter::toPlane3D(P3D2c));
                        vPlane2->setId(id2);
                        vPlane2->setFixed(true);
                        optimizer.addVertex(vPlane2);
                        vnVertexId[pMP2->mnId] = id2;
                    } else {
                        id2 = vnVertexId[pMP2->mnId];
                    }

                    int id = std::max(id1, id2);
                    if (id > maxVerticalPlaneId) {
                        maxVerticalPlaneId = id;
                    }
                } else
                    continue;
            } else
                continue;

            nCorrespondences++;

            std::cout << "Plane sim3 optim vertical" << std::endl;

            g2o::EdgeVerticalPlaneSim3Project *e12 = new g2o::EdgeVerticalPlaneSim3Project();
            e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
            e12->setMeasurement(Converter::toPlane3D(pKF1->mvPlaneCoefficients[i]));

            Eigen::Matrix2d Info;
            Info << angleInfo, 0,
                    0, angleInfo;
            e12->setInformation(Info);

            g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
            e12->setRobustKernel(rk1);
            //TODO
            rk1->setDelta(VPdeltaPlane);

            optimizer.addEdge(e12);

            g2o::EdgeVerticalPlaneInverseSim3Project *e21 = new g2o::EdgeVerticalPlaneInverseSim3Project();
            e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
            e21->setMeasurement(Converter::toPlane3D(pKF2->mvPlaneCoefficients[i2]));
            e21->setInformation(Info);

            g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
            e21->setRobustKernel(rk2);
            //TODO
            rk2->setDelta(VPdeltaPlane);

            optimizer.addEdge(e21);

            vpVerticalPlaneEdges12.push_back(e12);
            vpVerticalPlaneEdges21.push_back(e21);
            vnIndexVerticalPlaneEdge.push_back(i);
        }

        const int NPP = vpParallelPlaneMatches1.size();
        const vector<MapPlane *> vpMapParallelPlanes1 = pKF1->GetMapParallelPlaneMatches();
        vector<g2o::EdgeParallelPlaneSim3Project *> vpParallelPlaneEdges12;
        vector<g2o::EdgeParallelPlaneInverseSim3Project *> vpParallelPlaneEdges21;
        vector<size_t> vnIndexParallelPlaneEdge;

        vnIndexPlaneEdge.reserve(2 * NPP);
        vpPlaneEdges12.reserve(2 * NPP);
        vpPlaneEdges21.reserve(2 * NPP);

        for (int i = 0; i < NPP; i++) {
            if (!vpParallelPlaneMatches1[i])
                continue;

            MapPlane *pMP1 = vpMapParallelPlanes1[i];
            MapPlane *pMP2 = vpParallelPlaneMatches1[i];

            int id1 = maxVerticalPlaneId + 2 * i + 1;
            int id2 = maxVerticalPlaneId + 2 * (i + 1);

            const int i2 = pMP2->GetIndexInParallelKeyFrame(pKF2);

            if (pMP1 && pMP2) {
                if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0) {
                    if (vnVertexId.count(pMP1->mnId) == 0) {
                        g2o::VertexPlane *vPlane1 = new g2o::VertexPlane();
                        cv::Mat P3D1w = pMP1->GetWorldPos();
                        cv::Mat P3D1c = cv::Mat::eye(4, 1, CV_32F);
                        P3D1c.rowRange(0, 3).col(0) = R1w * P3D1w.rowRange(0, 3).col(0);
                        P3D1c.at<float>(3, 0) = P3D1w.at<float>(3, 0) - t1w.dot(P3D1c.rowRange(0, 3).col(0));
                        if (P3D1c.at<float>(3, 0) < 0.0)
                            P3D1c = -P3D1c;
                        vPlane1->setEstimate(Converter::toPlane3D(P3D1c));
                        vPlane1->setId(id1);
                        vPlane1->setFixed(true);
                        optimizer.addVertex(vPlane1);
                        vnVertexId[pMP1->mnId] = id1;
                    } else {
                        id1 = vnVertexId[pMP1->mnId];
                    }

                    if (vnVertexId.count(pMP2->mnId) == 0) {
                        g2o::VertexPlane *vPlane2 = new g2o::VertexPlane();
                        cv::Mat P3D2w = pMP2->GetWorldPos();
                        cv::Mat P3D2c = cv::Mat::eye(4, 1, CV_32F);
                        P3D2c.rowRange(0, 3).col(0) = R2w * P3D2w.rowRange(0, 3).col(0);
                        P3D2c.at<float>(3, 0) = P3D2w.at<float>(3, 0) - t2w.dot(P3D2c.rowRange(0, 3).col(0));
                        if (P3D2c.at<float>(3, 0) < 0.0)
                            P3D2c = -P3D2c;
                        vPlane2->setEstimate(Converter::toPlane3D(P3D2c));
                        vPlane2->setId(id2);
                        vPlane2->setFixed(true);
                        optimizer.addVertex(vPlane2);
                        vnVertexId[pMP2->mnId] = id2;
                    } else {
                        id2 = vnVertexId[pMP2->mnId];
                    }
                } else
                    continue;
            } else
                continue;

            nCorrespondences++;

            std::cout << "Plane sim3 optim parallel" << std::endl;

            g2o::EdgeParallelPlaneSim3Project *e12 = new g2o::EdgeParallelPlaneSim3Project();
            e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
            e12->setMeasurement(Converter::toPlane3D(pKF1->mvPlaneCoefficients[i]));

            Eigen::Matrix2d Info;
            Info << angleInfo, 0,
                    0, angleInfo;
            e12->setInformation(Info);

            g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
            e12->setRobustKernel(rk1);
            //TODO
            rk1->setDelta(VPdeltaPlane);

            optimizer.addEdge(e12);

            g2o::EdgeParallelPlaneInverseSim3Project *e21 = new g2o::EdgeParallelPlaneInverseSim3Project();
            e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
            e21->setMeasurement(Converter::toPlane3D(pKF2->mvPlaneCoefficients[i2]));
            e21->setInformation(Info);

            g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
            e21->setRobustKernel(rk2);
            //TODO
            rk2->setDelta(VPdeltaPlane);

            optimizer.addEdge(e21);

            vpParallelPlaneEdges12.push_back(e12);
            vpParallelPlaneEdges21.push_back(e21);
            vnIndexParallelPlaneEdge.push_back(i);
        }

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(5);

        // Check inliers
        int nBad = 0;
        for (size_t i = 0; i < vpEdges12.size(); i++) {
            g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
            g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
            if (!e12 || !e21)
                continue;

            if (e12->chi2() > th2 || e21->chi2() > th2) {
                cout << "Bad point: " << e12->chi2() << ", " << e21->chi2() << endl;
                size_t idx = vnIndexEdge[i];
                vpPointMatches1[idx] = static_cast<MapPoint *>(NULL);
                optimizer.removeEdge(e12);
                optimizer.removeEdge(e21);
                vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(NULL);
                vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(NULL);
                nBad++;
            }
        }

        for (size_t i = 0; i < vpLineStartEdges12.size(); i++) {
            EdgeLineSim3Project *es12 = vpLineStartEdges12[i];
            EdgeLineSim3Project *ee12 = vpLineEndEdges12[i];
            EdgeLineInverseSim3Project *es21 = vpLineStartEdges21[i];
            EdgeLineInverseSim3Project *ee21 = vpLineEndEdges21[i];

            if (!es12 || !ee12 || !es21 || !ee21)
                continue;

            if (es12->chi2() > th2 || ee12->chi2() > th2 || es21->chi2() > th2 || ee21->chi2() > th2) {
                cout << "Bad line: " << es12->chi2() + ee12->chi2() << ", " << es21->chi2() + ee21->chi2() << endl;
                size_t idx = vnIndexLineEdge[i];
                vpLineMatches1[idx] = static_cast<MapLine *>(NULL);
                optimizer.removeEdge(es12);
                optimizer.removeEdge(ee12);
                optimizer.removeEdge(es21);
                optimizer.removeEdge(ee21);
                vpLineStartEdges12[i] = static_cast<EdgeLineSim3Project *>(NULL);
                vpLineEndEdges12[i] = static_cast<EdgeLineSim3Project *>(NULL);
                vpLineStartEdges21[i] = static_cast<EdgeLineInverseSim3Project *>(NULL);
                vpLineEndEdges21[i] = static_cast<EdgeLineInverseSim3Project *>(NULL);
                nBad++;
            }
        }

        for (size_t i = 0; i < vpPlaneEdges12.size(); i++) {
            g2o::EdgePlaneSim3Project *e12 = vpPlaneEdges12[i];
            g2o::EdgePlaneInverseSim3Project *e21 = vpPlaneEdges21[i];
            if (!e12 || !e21)
                continue;

            if (e12->chi2() > planeChi || e21->chi2() > planeChi) {
                // cout << "Bad plane: " << e12->chi2() << ", " << e21->chi2() << endl;
                size_t idx = vnIndexPlaneEdge[i];
                vpPlaneMatches1[idx] = static_cast<MapPlane *>(NULL);
                optimizer.removeEdge(e12);
                optimizer.removeEdge(e21);
                vpPlaneEdges12[i] = static_cast<g2o::EdgePlaneSim3Project *>(NULL);
                vpPlaneEdges21[i] = static_cast<g2o::EdgePlaneInverseSim3Project *>(NULL);
                nBad++;
            }
        }

        for (size_t i = 0; i < vpVerticalPlaneEdges12.size(); i++) {
            g2o::EdgeVerticalPlaneSim3Project *e12 = vpVerticalPlaneEdges12[i];
            g2o::EdgeVerticalPlaneInverseSim3Project *e21 = vpVerticalPlaneEdges21[i];
            if (!e12 || !e21)
                continue;

            if (e12->chi2() > VPplaneChi || e21->chi2() > VPplaneChi) {
                // cout << "Bad vertical plane: " << e12->chi2() << ", " << e21->chi2() << endl;
                size_t idx = vnIndexVerticalPlaneEdge[i];
                vpVerticalPlaneMatches1[idx] = static_cast<MapPlane *>(NULL);
                optimizer.removeEdge(e12);
                optimizer.removeEdge(e21);
                vpVerticalPlaneEdges12[i] = static_cast<g2o::EdgeVerticalPlaneSim3Project *>(NULL);
                vpVerticalPlaneEdges21[i] = static_cast<g2o::EdgeVerticalPlaneInverseSim3Project *>(NULL);
                nBad++;
            }
        }

        for (size_t i = 0; i < vpParallelPlaneEdges12.size(); i++) {
            g2o::EdgeParallelPlaneSim3Project *e12 = vpParallelPlaneEdges12[i];
            g2o::EdgeParallelPlaneInverseSim3Project *e21 = vpParallelPlaneEdges21[i];
            if (!e12 || !e21)
                continue;

            if (e12->chi2() > th2 || e21->chi2() > th2) {
                cout << "Bad parallel plane: " << e12->chi2() << ", " << e21->chi2() << endl;
                size_t idx = vnIndexParallelPlaneEdge[i];
                vpParallelPlaneMatches1[idx] = static_cast<MapPlane *>(NULL);
                optimizer.removeEdge(e12);
                optimizer.removeEdge(e21);
                vpParallelPlaneEdges12[i] = static_cast<g2o::EdgeParallelPlaneSim3Project *>(NULL);
                vpParallelPlaneEdges21[i] = static_cast<g2o::EdgeParallelPlaneInverseSim3Project *>(NULL);
                nBad++;
            }
        }

        int nMoreIterations;
        if (nBad > 0)
            nMoreIterations = 10;
        else
            nMoreIterations = 5;

        if (nCorrespondences - nBad < 10)
            return 0;

        // Optimize again only with inliers

        optimizer.initializeOptimization();
        optimizer.optimize(nMoreIterations);

        int nIn = 0;
        int PNPoint = 0;
        double PEPoint = 0;
        for (size_t i = 0; i < vpEdges12.size(); i++) {
            g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
            g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
            if (!e12 || !e21)
                continue;

            const float chi21 = e12->chi2();
            const float chi22 = e21->chi2();

            PNPoint++;
            PEPoint += chi21 + chi22;

            if (e12->chi2() > th2 || e21->chi2() > th2) {
                cout << "Bad point " << endl;
                size_t idx = vnIndexEdge[i];
                vpPointMatches1[idx] = static_cast<MapPoint *>(NULL);
            } else
                nIn++;
        }
        if (PNPoint == 0)
            cout << "No points " << endl;
        else
            cout << "Points: " << PNPoint / PEPoint << endl;

        int PNLine = 0;
        double PELine = 0;
        for (size_t i = 0; i < vpLineStartEdges12.size(); i++) {
            EdgeLineSim3Project *es12 = vpLineStartEdges12[i];
            EdgeLineSim3Project *ee12 = vpLineEndEdges12[i];
            EdgeLineInverseSim3Project *es21 = vpLineStartEdges21[i];
            EdgeLineInverseSim3Project *ee21 = vpLineEndEdges21[i];

            if (!es12 || !ee12 || !es21 || !ee21)
                continue;

            const float chi2_s1 = es12->chi2();
            const float chi2_e1 = ee12->chi2();
            const float chi2_s2 = es21->chi2();
            const float chi2_e2 = ee21->chi2();

            PNLine++;
            PELine += chi2_s1 + chi2_e1 + chi2_s2 + chi2_e2;

            if (es12->chi2() > th2 || ee12->chi2() > th2 || es21->chi2() > th2 || ee21->chi2() > th2) {
                cout << "Bad line " << endl;
                size_t idx = vnIndexLineEdge[i];
                vpLineMatches1[idx] = static_cast<MapLine *>(NULL);
            } else
                nIn++;
        }
        if (PNLine == 0)
            cout << "No lines " << endl;
        else
            cout << " Lines: " << PELine / PNLine << endl;

        int PNPlane = 0;
        double PEPlane = 0;
        for (size_t i = 0; i < vpPlaneEdges12.size(); i++) {
            g2o::EdgePlaneSim3Project *e12 = vpPlaneEdges12[i];
            g2o::EdgePlaneInverseSim3Project *e21 = vpPlaneEdges21[i];
            if (!e12 || !e21)
                continue;

            const float chi21 = e12->chi2();
            const float chi22 = e21->chi2();

            PNPlane++;
            PEPlane += chi21 + chi22;

            if (e12->chi2() > planeChi || e21->chi2() > planeChi) {
                // cout << "Bad plane: " << e12->chi2() << ", " << e21->chi2() << endl;
                size_t idx = vnIndexPlaneEdge[i];
                vpPlaneMatches1[idx] = static_cast<MapPlane *>(NULL);
            } else
                nIn++;
        }
        if (PNPlane == 0)
            cout << "No plane " << endl;
        else
            cout << "Planes: " << PEPlane / PNPlane << endl;

        int PNVPlane = 0;
        double PEVPlane = 0;
        for (size_t i = 0; i < vpVerticalPlaneEdges12.size(); i++) {
            g2o::EdgeVerticalPlaneSim3Project *e12 = vpVerticalPlaneEdges12[i];
            g2o::EdgeVerticalPlaneInverseSim3Project *e21 = vpVerticalPlaneEdges21[i];
            if (!e12 || !e21)
                continue;

            const float chi21 = e12->chi2();
            const float chi22 = e21->chi2();

            PNVPlane++;
            PEVPlane += chi21 + chi22;

            if (e12->chi2() > VPplaneChi || e21->chi2() > VPplaneChi) {
                cout << "Bad vertical plane: " << e12->chi2() << ", " << e21->chi2() << endl;
                size_t idx = vnIndexVerticalPlaneEdge[i];
                vpVerticalPlaneMatches1[idx] = static_cast<MapPlane *>(NULL);
            } else
                nIn++;
        }
        if (PNVPlane == 0)
            cout << "No vertical plane " << endl;
        else
            cout << "Vertical Planes: " << PEVPlane / PNVPlane << endl;

        int PNPPlane = 0;
        double PEPPlane = 0;
        for (size_t i = 0; i < vpParallelPlaneEdges12.size(); i++) {
            g2o::EdgeParallelPlaneSim3Project *e12 = vpParallelPlaneEdges12[i];
            g2o::EdgeParallelPlaneInverseSim3Project *e21 = vpParallelPlaneEdges21[i];
            if (!e12 || !e21)
                continue;

            const float chi21 = e12->chi2();
            const float chi22 = e21->chi2();

            PNPPlane++;
            PEPPlane += chi21 + chi22;

            if (e12->chi2() > VPplaneChi || e21->chi2() > VPplaneChi) {
                cout << "Bad parallel plane: " << e12->chi2() << ", " << e21->chi2() << endl;
                size_t idx = vnIndexParallelPlaneEdge[i];
                vpParallelPlaneMatches1[idx] = static_cast<MapPlane *>(NULL);
            } else
                nIn++;
        }
        if (PNPPlane == 0)
            cout << "No parallel plane " << endl;
        else
            cout << "Parallel Planes: " << PEPPlane / PNPPlane << endl;

//         Recover optimized Sim3
        g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
        g2oS12 = vSim3_recov->estimate();

        return nIn;
    }

    int Optimizer::TranslationOptimization(ORB_SLAM2::Frame *pFrame) {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        //rotation
        cv::Mat R_cw = pFrame->mTcw.rowRange(0, 3).colRange(0, 3).clone();

        vector<g2o::EdgeSE3ProjectXYZOnlyTranslation *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyTranslation *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);


        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; i++) {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (pMP) {
                    // Monocular observation
                    if (pFrame->mvuRight[i] < 0) {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZOnlyTranslation *e = new g2o::EdgeSE3ProjectXYZOnlyTranslation();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        cv::Mat Xw = pMP->GetWorldPos();
                        cv::Mat Xc = R_cw * Xw;

                        e->Xc[0] = Xc.at<float>(0);
                        e->Xc[1] = Xc.at<float>(1);
                        e->Xc[2] = Xc.at<float>(2);


                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    } else  // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        //SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyTranslation *e = new g2o::EdgeStereoSE3ProjectXYZOnlyTranslation();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        cv::Mat Xc = R_cw * Xw;

                        e->Xc[0] = Xc.at<float>(0);
                        e->Xc[1] = Xc.at<float>(1);
                        e->Xc[2] = Xc.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }

            }
        }

        const int NL = pFrame->NL;

        vector<EdgeLineProjectXYZOnlyTranslation *> vpEdgesLineSp;
        vector<size_t> vnIndexLineEdgeSp;
        vpEdgesLineSp.reserve(NL);
        vnIndexLineEdgeSp.reserve(NL);

        vector<EdgeLineProjectXYZOnlyTranslation *> vpEdgesLineEp;
        vpEdgesLineEp.reserve(NL);

        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            for (int i = 0; i < NL; i++) {
                MapLine *pML = pFrame->mvpMapLines[i];
                if (pML) {
                    pFrame->mvbLineOutlier[i] = false;

                    Eigen::Vector3d line_obs;
                    line_obs = pFrame->mvKeyLineFunctions[i];

                    EdgeLineProjectXYZOnlyTranslation *els = new EdgeLineProjectXYZOnlyTranslation();

                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity() * 1);//*vSteroStartPointInfo[i]);

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(deltaStereo);

                    els->fx = pFrame->fx;
                    els->fy = pFrame->fy;
                    els->cx = pFrame->cx;
                    els->cy = pFrame->cy;

                    cv::Mat Xw = Converter::toCvVec(pML->mWorldPos.head(3));
                    cv::Mat Xc = R_cw * Xw;
                    els->Xc[0] = Xc.at<float>(0);
                    els->Xc[1] = Xc.at<float>(1);
                    els->Xc[2] = Xc.at<float>(2);

                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    EdgeLineProjectXYZOnlyTranslation *ele = new EdgeLineProjectXYZOnlyTranslation();

                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity() * 1);//vSteroEndPointInfo[i]);

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(deltaStereo);

                    ele->fx = pFrame->fx;
                    ele->fy = pFrame->fy;
                    ele->cx = pFrame->cx;
                    ele->cy = pFrame->cy;

                    Xw = Converter::toCvVec(pML->mWorldPos.tail(3));
                    Xc = R_cw * Xw;
                    ele->Xc[0] = Xc.at<float>(0);
                    ele->Xc[1] = Xc.at<float>(1);
                    ele->Xc[2] = Xc.at<float>(2);


                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                }
            }
        }


        if (nInitialCorrespondences < 3) {
            return 0;
        }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;
        vector<g2o::EdgePlaneOnlyTranslation *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

//        vector<vector<g2o::EdgePlanePointTranslationOnly *>> vpEdgesPlanePoint;
//        vector<vector<size_t>> vnIndexEdgePlanePoint;
//        vpEdgesPlanePoint = vector<vector<g2o::EdgePlanePointTranslationOnly *>>(M);
//        vnIndexEdgePlanePoint = vector<vector<size_t>>(M);

        vector<g2o::EdgeParallelPlaneOnlyTranslation *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeVerticalPlaneOnlyTranslation *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        auto aTh = ORB_SLAM2::Config::Get<double>("Plane.AssociationAngRef");

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            int PNum = 0;
            double PEror = 0, PMax = 0;
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgePlaneOnlyTranslation *e = new g2o::EdgePlaneOnlyTranslation();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    //TODO
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    Isometry3D trans = static_cast<const VertexSE3Expmap *>(optimizer.vertex(0))->estimate();
                    cv::Mat Pc3D = pFrame->mvPlaneCoefficients[i];
                    Plane3D Pw3D = Converter::toPlane3D(pMP->GetWorldPos());
                    Vector4D Pw = Pw3D._coeffs;
                    Vector4D Pc;
                    Matrix3D R = trans.rotation();
                    Pc.head<3>() = R * Pw.head<3>();
                    Pc(3) = Pw(3) - trans.translation().dot(Pc.head<3>());

                    double angle = Pc(0) * Pc3D.at<float>(0) +
                                   Pc(1) * Pc3D.at<float>(1) +
                                   Pc(2) * Pc3D.at<float>(2);
                    if (angle < -aTh) {
                        Pw = -Pw;
                        Pw3D.fromVector(Pw);
                    }

                    Pw3D.rotateNormal(Converter::toMatrix3d(R_cw));
                    e->Xc = Pw3D;

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //TODO
                    rk->setDelta(deltaPlane);

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);

//                    int nMatches = pFrame->mvPlanePointMatches[i].size();
//
//                    vector<g2o::EdgePlanePointTranslationOnly *> edgesPlanePoint;
//                    vector<size_t> indexEdgePlanePoint;
//                    for (int j = 0; j < nMatches; j++) {
//                        MapPoint *mapPoint = pFrame->mvPlanePointMatches[i][j];
//                        if (mapPoint) {
//                            g2o::EdgePlanePointTranslationOnly *edge = new g2o::EdgePlanePointTranslationOnly();
//                            edge->setVertex(0, optimizer.vertex(0));
//                            edge->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edge->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            cv::Mat Pw = mapPoint->GetWorldPos();
//                            cv::Mat Pc = R_cw * Pw;
//                            edge->Xc[0] = Pc.at<float>(0);
//                            edge->Xc[1] = Pc.at<float>(1);
//                            edge->Xc[2] = Pc.at<float>(2);
//
//                            g2o::RobustKernelHuber *rkEdge = new g2o::RobustKernelHuber;
//                            edge->setRobustKernel(rkEdge);
//                            rkEdge->setDelta(deltaMono);
//
//                            optimizer.addEdge(edge);
//
//                            edgesPlanePoint.push_back(edge);
//                            indexEdgePlanePoint.push_back(j);
//                        }
//                    }
//
//                    int pointEdges = edgesPlanePoint.size();
//                    int nLineMatches = pFrame->mvPlaneLineMatches[i].size();
//
//                    for (int j = 0, index = pointEdges; j < nLineMatches; j++) {
//                        MapLine *mapLine = pFrame->mvPlaneLineMatches[i][j];
//                        if (mapLine) {
//                            g2o::EdgePlanePointTranslationOnly *edgeStart = new g2o::EdgePlanePointTranslationOnly();
//                            edgeStart->setVertex(0, optimizer.vertex(0));
//                            edgeStart->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edgeStart->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            cv::Mat Xw = Converter::toCvVec(mapLine->mWorldPos.head(3));
//                            cv::Mat Xc = R_cw * Xw;
//                            edgeStart->Xc[0] = Xc.at<float>(0);
//                            edgeStart->Xc[1] = Xc.at<float>(1);
//                            edgeStart->Xc[2] = Xc.at<float>(2);
//
//                            g2o::RobustKernelHuber *rkEdgeStart = new g2o::RobustKernelHuber;
//                            edgeStart->setRobustKernel(rkEdgeStart);
//                            rkEdgeStart->setDelta(deltaMono);
//
//                            optimizer.addEdge(edgeStart);
//
//                            edgesPlanePoint.push_back(edgeStart);
//                            indexEdgePlanePoint.push_back(index++);
//
//                            g2o::EdgePlanePointTranslationOnly *edgeEnd = new g2o::EdgePlanePointTranslationOnly();
//                            edgeEnd->setVertex(0, optimizer.vertex(0));
//                            edgeEnd->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edgeEnd->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            Xw = Converter::toCvVec(mapLine->mWorldPos.tail(3));
//                            Xc = R_cw * Xw;
//                            edgeEnd->Xc[0] = Xc.at<float>(0);
//                            edgeEnd->Xc[1] = Xc.at<float>(1);
//                            edgeEnd->Xc[2] = Xc.at<float>(2);
//
//                            g2o::RobustKernelHuber *rkEdgeEnd = new g2o::RobustKernelHuber;
//                            edgeEnd->setRobustKernel(rkEdgeEnd);
//                            rkEdgeEnd->setDelta(deltaMono);
//
//                            optimizer.addEdge(edgeEnd);
//
//                            edgesPlanePoint.push_back(edgeEnd);
//                            indexEdgePlanePoint.push_back(index++);
//                        }
//                    }
//
//                    vpEdgesPlanePoint[i] = edgesPlanePoint;
//                    vnIndexEdgePlanePoint[i] = indexEdgePlanePoint;


                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
//                cout << "  done!" << endl;
                }
            }
            // cout << " Plane: " << PEror / PNum << " ";//" Max: " << PMax << " ";

//            PNum = 0;
//            PEror = 0;
//            PMax = 0;
//            for (int i = 0; i < M; ++i) {
//                // add parallel planes!
//                MapPlane *pMP = pFrame->mvpParallelPlanes[i];
//                if (pMP) {
//                    pFrame->mvbParPlaneOutlier[i] = false;
//
//                    g2o::EdgeParallelPlaneOnlyTranslation *e = new g2o::EdgeParallelPlaneOnlyTranslation();
//                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
//                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                    //TODO
//                    Eigen::Matrix2d Info;
//                    Info << parInfo, 0,
//                            0, parInfo;
////                    Info << 0, 0,
////                            0, 0;
//
//                    e->setInformation(Info);
//
//                    Plane3D Xw = Converter::toPlane3D(pMP->GetWorldPos());
//                    Xw.rotateNormal(Converter::toMatrix3d(R_cw));
//                    e->Xc = Xw;
//
//                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                    e->setRobustKernel(rk);
//                    //TODO
//                    rk->setDelta(VPdeltaPlane);
//                    optimizer.addEdge(e);
//
//                    vpEdgesParPlane.push_back(e);
//                    vnIndexEdgeParPlane.push_back(i);
//
//                    e->computeError();
//                    double chi = e->chi2();
//                    PEror += chi;
//                    PMax = PMax > chi ? PMax : chi;
//                    PNum++;
//                }
//            }
//            cout << " Par Plane: " << PEror / PNum << " ";//" Max: " << PMax << " ";
//            PNum = 0;
//            PEror = 0;
//            PMax = 0;
//
//            for (int i = 0; i < M; ++i) {
//                // add vertical planes!
//                MapPlane *pMP = pFrame->mvpVerticalPlanes[i];
//                if (pMP) {
//                    pFrame->mvbVerPlaneOutlier[i] = false;
//
//                    g2o::EdgeVerticalPlaneOnlyTranslation *e = new g2o::EdgeVerticalPlaneOnlyTranslation();
//                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
//                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                    //TODO
//                    Eigen::Matrix2d Info;
//                    Info << verInfo, 0,
//                            0, verInfo;
////                    Info << 0, 0,
////                            0, 0;
//
//                    e->setInformation(Info);
//
//                    Plane3D Xw = Converter::toPlane3D(pMP->GetWorldPos());
//                    Xw.rotateNormal(Converter::toMatrix3d(R_cw));
//                    e->Xc = Xw;
//
//                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                    e->setRobustKernel(rk);
//                    //TODO
//                    rk->setDelta(VPdeltaPlane);
//                    optimizer.addEdge(e);
//
//                    vpEdgesVerPlane.push_back(e);
//                    vnIndexEdgeVerPlane.push_back(i);
//
//                    e->computeError();
//                    double chi = e->chi2();
//                    PEror += chi;
//                    PMax = PMax > chi ? PMax : chi;
//                    PNum++;
//                }
//            }
//            cout << " Ver Plane: " << PEror / PNum << endl;//" Max: " << PMax << endl;
        }

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        int nLineBad = 0;
        for (size_t it = 0; it < 4; it++) {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;

//            int PNMono = 0;
//            double PEMono = 0, PMaxMono = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
                g2o::EdgeSE3ProjectXYZOnlyTranslation *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
//                cout<<"optimize mono point chi2, "<<chi2<<endl;
//                PNMono++;
//                PEMono += chi2;
//                PMaxMono = PMaxMono > chi2 ? PMaxMono : chi2;

                if (chi2 > chi2Mono[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PNMono == 0)
//                cout << "No mono points " << " ";
//            else
//                cout << " Mono points: " << PEMono / PNMono << " "; //<< " Max: " << PMax << endl;

//            int PNStereo = 0;
//            double PEStereo = 0, PMaxStereo = 0;

//            cout << "Opti:vpEdgesMono:" << vpEdgesMono.size() << "," << vpEdgesStereo.size() << endl;
            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyTranslation *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
//                cout<<"optimize stereo point chi2, "<<chi2<<endl;
//                PNStereo++;
//                PEStereo += chi2;
//                PMaxStereo = PMaxStereo > chi2 ? PMaxStereo : chi2;

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

//            if (PNStereo == 0)
//                cout << "No stereo points " << " ";
//            else
//                cout << " Stereo points: " << PEStereo / PNStereo << endl;

            nLineBad = 0;
//            int PNLine= 0;
//            double PELine = 0, PMaxLine = 0;
//            cout << "Opti:vpEdgesLine:" << vpEdgesLineSp.size() << endl;
            for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
                EdgeLineProjectXYZOnlyTranslation *e1 = vpEdgesLineSp[i];  //线段起始点
                EdgeLineProjectXYZOnlyTranslation *e2 = vpEdgesLineEp[i];  //线段终止点

                const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

                if (pFrame->mvbLineOutlier[idx]) {
                    e1->computeError();
                    e2->computeError();
                }

                const float chi2_s = e1->chiline();//e1->chi2();
                const float chi2_e = e2->chiline();//e2->chi2();
//                cout<<"Optimization: chi2_s "<<chi2_s<<", chi2_e "<<chi2_e<<endl;

//                PNLine++;
//                PELine += chi2_s + chi2_e;
//                PMaxLine = PMaxLine > chi2_s + chi2_e ? PMaxLine : chi2_s + chi2_e;


                if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
                    pFrame->mvbLineOutlier[idx] = true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nLineBad++;
                } else {
                    pFrame->mvbLineOutlier[idx] = false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                }

                if (it == 2) {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }

//            if (PNLine == 0)
//                cout << "No lines " << " ";
//            else
//                cout << " Lines: " << PELine / PNLine << endl;

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgePlaneOnlyTranslation *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                    // cout << "planetest bad: " << chi2 << ", id: " << idx << "  Pc : "
                    //      << pFrame->mvPlaneCoefficients[idx].t() << "  Pw :"
                    //      << (pFrame->mTwc.t() * pFrame->mvpMapPlanes[idx]->GetWorldPos()).t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);

//                if (vpEdgesPlanePoint[i].size() > 0) {
//                    int PPN = 0;
//                    double PPE = 0, PPMax = 0;
//                    for (size_t j = 0, jend = vpEdgesPlanePoint[i].size(); j < jend; j++) {
//                        g2o::EdgePlanePointTranslationOnly *edge = vpEdgesPlanePoint[i][j];
//
//                        const size_t index = vnIndexEdgePlanePoint[i][j];
//
//                        const float chi2 = edge->chi2();
////                    cout<<"optimize chi2"<<chi2<<endl;
//                        PPN++;
//                        PPE += chi2;
//                        PPMax = PPMax > chi2 ? PPMax : chi2;
//
//                        if (chi2 > chi2Mono[it]) {
//                            edge->setLevel(1);
//                            nBad++;
//                        } else {
//                            edge->setLevel(0);
//                        }
//
//                        if (it == 2)
//                            edge->setRobustKernel(0);
//                    }
//
//                    if (PPN == 0)
//                        cout << "planetest No plane point matches " << " ";
//                    else
//                        cout << "planetest  Plane point matches: " << PPE / PPN << " "; //<< " Max: " << PMax << endl;
//                }
            }
            // if (PN == 0)
            //     cout << "planetest No plane " << " ";
            // else
            //     cout << "planetest  Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

//            PN = 0;
//            PE = 0;
//            PMax = 0;
//            for (size_t i = 0, iend = vpEdgesParPlane.size(); i < iend; i++) {
//                g2o::EdgeParallelPlaneOnlyTranslation *e = vpEdgesParPlane[i];
//
//                const size_t idx = vnIndexEdgeParPlane[i];
//
//                if (pFrame->mvbParPlaneOutlier[idx]) {
//                    e->computeError();
//                }
//
//                const float chi2 = e->chi2();
//                PN++;
//                PE += chi2;
//                PMax = PMax > chi2 ? PMax : chi2;
//
//                if (chi2 > VPplaneChi) {
//                    pFrame->mvbParPlaneOutlier[idx] = true;
//                    e->setLevel(1);
//                    nBad++;
//                    cout << "planetest bad Par: " << chi2 << ", id: " << idx << "  Pc : "
//                         << pFrame->ComputePlaneWorldCoeff(idx).t() << "  Pw :"
//                         << pFrame->mvpParallelPlanes[idx]->GetWorldPos().t() << endl;
//                } else {
//                    e->setLevel(0);
//                    pFrame->mvbParPlaneOutlier[idx] = false;
//                }
//
//                if (it == 2)
//                    e->setRobustKernel(0);
//            }
//            if (PN == 0)
//                cout << "planetest No par plane " << " ";
//            else
//                cout << "planetest par Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;
//
//            PN = 0;
//            PE = 0;
//            PMax = 0;
//
//            for (size_t i = 0, iend = vpEdgesVerPlane.size(); i < iend; i++) {
//                g2o::EdgeVerticalPlaneOnlyTranslation *e = vpEdgesVerPlane[i];
//
//                const size_t idx = vnIndexEdgeVerPlane[i];
//
//                if (pFrame->mvbVerPlaneOutlier[idx]) {
//                    e->computeError();
//                }
//
//                const float chi2 = e->chi2();
//                PN++;
//                PE += chi2;
//                PMax = PMax > chi2 ? PMax : chi2;
//
//                if (chi2 > VPplaneChi) {
//                    pFrame->mvbVerPlaneOutlier[idx] = true;
//                    e->setLevel(1);
//                    nBad++;
//                    cout << "planetest bad Ver: " << chi2 << ", id: " << idx << "  Pc : "
//                         << pFrame->ComputePlaneWorldCoeff(idx).t() << "  Pw :"
//                         << pFrame->mvpVerticalPlanes[idx]->GetWorldPos().t() << endl;
//                } else {
//                    e->setLevel(0);
//                    pFrame->mvbVerPlaneOutlier[idx] = false;
//                }
//
//                if (it == 2)
//                    e->setRobustKernel(0);
//            }
//            if (PN == 0)
//                cout << "planetest No Ver plane " << endl;
//            else
//                cout << "planetest Ver Plane: " << PE / PN << endl; //<< " Max: " << PMax << endl;

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nInitialCorrespondences - nBad;
    }

    void Optimizer::OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections)
    {
        typedef g2o::BlockSolver< g2o::BlockSolverTraits<4, 4> > BlockSolver_4_4;

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(false);
        g2o::BlockSolverX::LinearSolverType * linearSolver =
                new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        optimizer.setAlgorithm(solver);

        const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
        const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

        const unsigned int nMaxKFid = pMap->GetMaxKFid();

        vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
        vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);

        vector<VertexPose4DoF*> vpVertices(nMaxKFid+1);

        const int minFeat = 100;
        // Set KeyFrame vertices
        for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
        {
            KeyFrame* pKF = vpKFs[i];
            if(pKF->isBad())
                continue;

            VertexPose4DoF* V4DoF;

            const int nIDi = pKF->mnId;

            LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

            if(it!=CorrectedSim3.end())
            {
                vScw[nIDi] = it->second;
                const g2o::Sim3 Swc = it->second.inverse();
                Eigen::Matrix3d Rwc = Swc.rotation().toRotationMatrix();
                Eigen::Vector3d twc = Swc.translation();
                V4DoF = new VertexPose4DoF(Rwc, twc, pKF);
            }
            else
            {
                Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
                Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
                g2o::Sim3 Siw(Rcw,tcw,1.0);
                vScw[nIDi] = Siw;
                V4DoF = new VertexPose4DoF(pKF);
            }

            if(pKF==pLoopKF)
                V4DoF->setFixed(true);

            V4DoF->setId(nIDi);
            V4DoF->setMarginalized(false);

            optimizer.addVertex(V4DoF);
            vpVertices[nIDi]=V4DoF;
        }
        cout << "PoseGraph4DoF: KFs loaded" << endl;

        set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

        // Edge used in posegraph has still 6Dof, even if updates of camera poses are just in 4DoF
        Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
        matLambda(0,0) = 1e3;
        matLambda(1,1) = 1e3;
        matLambda(0,0) = 1e3;

        // Set Loop edges
        Edge4DoF* e_loop;
        for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;
            const long unsigned int nIDi = pKF->mnId;
            const set<KeyFrame*> &spConnections = mit->second;
            const g2o::Sim3 Siw = vScw[nIDi];
            const g2o::Sim3 Swi = Siw.inverse();

            for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
            {
                const long unsigned int nIDj = (*sit)->mnId;
                if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                    continue;

                const g2o::Sim3 Sjw = vScw[nIDj];
                const g2o::Sim3 Sij = Siw * Sjw.inverse();
                Eigen::Matrix4d Tij;
                Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
                Tij.block<3,1>(0,3) = Sij.translation();
                Tij(3,3) = 1.;

                Edge4DoF* e = new Edge4DoF(Tij);
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));

                e->information() = matLambda;
                e_loop = e;
                optimizer.addEdge(e);

                sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
            }
        }
        cout << "PoseGraph4DoF: Loop edges loaded" << endl;

        // 1. Set normal edges
        for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
        {
            KeyFrame* pKF = vpKFs[i];

            const int nIDi = pKF->mnId;

            g2o::Sim3 Siw;

            // Use noncorrected poses for posegraph edges
            LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

            if(iti!=NonCorrectedSim3.end())
                Siw = iti->second;
            else
                Siw = vScw[nIDi];


            // 1.1.0 Spanning tree edge
            KeyFrame* pParentKF = static_cast<KeyFrame*>(NULL);
            if(pParentKF)
            {
                int nIDj = pParentKF->mnId;

                g2o::Sim3 Swj;

                LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

                if(itj!=NonCorrectedSim3.end())
                    Swj = (itj->second).inverse();
                else
                    Swj =  vScw[nIDj].inverse();

                g2o::Sim3 Sij = Siw * Swj;
                Eigen::Matrix4d Tij;
                Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
                Tij.block<3,1>(0,3) = Sij.translation();
                Tij(3,3)=1.;

                Edge4DoF* e = new Edge4DoF(Tij);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                e->information() = matLambda;
                optimizer.addEdge(e);
            }

            // 1.1.1 Inertial edges
            KeyFrame* prevKF = pKF->mPrevKF;
            if(prevKF)
            {
                int nIDj = prevKF->mnId;

                g2o::Sim3 Swj;

                LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(prevKF);

                if(itj!=NonCorrectedSim3.end())
                    Swj = (itj->second).inverse();
                else
                    Swj =  vScw[nIDj].inverse();

                g2o::Sim3 Sij = Siw * Swj;
                Eigen::Matrix4d Tij;
                Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
                Tij.block<3,1>(0,3) = Sij.translation();
                Tij(3,3)=1.;

                Edge4DoF* e = new Edge4DoF(Tij);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                e->information() = matLambda;
                optimizer.addEdge(e);
            }

            // 1.2 Loop edges
            const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
            for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
            {
                KeyFrame* pLKF = *sit;
                if(pLKF->mnId<pKF->mnId)
                {
                    g2o::Sim3 Swl;

                    LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                    if(itl!=NonCorrectedSim3.end())
                        Swl = itl->second.inverse();
                    else
                        Swl = vScw[pLKF->mnId].inverse();

                    g2o::Sim3 Sil = Siw * Swl;
                    Eigen::Matrix4d Til;
                    Til.block<3,3>(0,0) = Sil.rotation().toRotationMatrix();
                    Til.block<3,1>(0,3) = Sil.translation();
                    Til(3,3) = 1.;

                    Edge4DoF* e = new Edge4DoF(Til);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                    e->information() = matLambda;
                    optimizer.addEdge(e);
                }
            }

            // 1.3 Covisibility graph edges
            const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
            for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
            {
                KeyFrame* pKFn = *vit;
                if(pKFn && pKFn!=pParentKF && pKFn!=prevKF && pKFn!=pKF->mNextKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
                {
                    if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                    {
                        if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                            continue;

                        g2o::Sim3 Swn;

                        LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                        if(itn!=NonCorrectedSim3.end())
                            Swn = itn->second.inverse();
                        else
                            Swn = vScw[pKFn->mnId].inverse();

                        g2o::Sim3 Sin = Siw * Swn;
                        Eigen::Matrix4d Tin;
                        Tin.block<3,3>(0,0) = Sin.rotation().toRotationMatrix();
                        Tin.block<3,1>(0,3) = Sin.translation();
                        Tin(3,3) = 1.;
                        Edge4DoF* e = new Edge4DoF(Tin);
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                        e->information() = matLambda;
                        optimizer.addEdge(e);
                    }
                }
            }
        }
        cout << "PoseGraph4DoF: Covisibility edges loaded" << endl;

        optimizer.initializeOptimization();
        optimizer.computeActiveErrors();
        optimizer.optimize(20);

        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
        for(size_t i=0;i<vpKFs.size();i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            const int nIDi = pKFi->mnId;

            VertexPose4DoF* Vi = static_cast<VertexPose4DoF*>(optimizer.vertex(nIDi));
            Eigen::Matrix3d Ri = Vi->estimate().Rcw[0];
            Eigen::Vector3d ti = Vi->estimate().tcw[0];

            g2o::Sim3 CorrectedSiw = g2o::Sim3(Ri,ti,1.);
            vCorrectedSwc[nIDi]=CorrectedSiw.inverse();

            cv::Mat Tiw = Converter::toCvSE3(Ri,ti);
            pKFi->SetPose(Tiw);
        }

        // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
        for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMPs[i];

            if(pMP->isBad())
                continue;

            int nIDr;

            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;

            g2o::Sim3 Srw = vScw[nIDr];
            g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

            cv::Mat P3Dw = pMP->GetWorldPos();
            Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
            Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

            cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
            pMP->SetWorldPos(cvCorrectedP3Dw);

            pMP->UpdateNormalAndDepth();
        }
        pMap->IncreaseChangeIndex();
    }

    int Optimizer::TranslationInertialOptimizationLastKeyFrame(Frame *pFrame, bool bRecInit)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
        optimizer.setVerbose(false);
        optimizer.setAlgorithm(solver);

        int nInitialMonoCorrespondences=0;
        int nInitialStereoCorrespondences=0;
        int nInitialCorrespondences=0;

        // Set Frame vertex
        VertexPose* VP = new VertexPose(pFrame);
        VP->setId(0);
        VP->setFixed(false);
        optimizer.addVertex(VP);
        VertexVelocity* VV = new VertexVelocity(pFrame);
        VV->setId(1);
        VV->setFixed(false);
        optimizer.addVertex(VV);
        VertexGyroBias* VG = new VertexGyroBias(pFrame);
        VG->setId(2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pFrame);
        VA->setId(3);
        VA->setFixed(false);
        optimizer.addVertex(VA);

        // Set MapPoint vertices
        const int N = pFrame->N;
        // const int Nleft = pFrame->Nleft;
        // const bool bRight = (Nleft!=-1);

        vector<EdgeMonoOnlyTranslation*> vpEdgesMono;
        vector<EdgeStereoOnlyTranslation*> vpEdgesStereo;
        vector<size_t> vnIndexEdgeMono;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesMono.reserve(N);
        vpEdgesStereo.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);


        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<N; i++)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(pMP)
                {
                    cv::KeyPoint kpUn;

                    // Left monocular observation
                    // if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                    {   
                        // TODO: IMU Check mvKeys or mvKeys
                        // if(i < Nleft) // pair left-right
                            kpUn = pFrame->mvKeys[i];
                        // else
                        // kpUn = pFrame->mvKeysUn[i];

                        nInitialMonoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyTranslation* e = new EdgeMonoOnlyTranslation(pMP->GetWorldPos(),0);

                        e->setVertex(0,VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    // Stereo observation
                    // else if(!bRight)
                    // {
                    //     nInitialStereoCorrespondences++;
                    //     pFrame->mvbOutlier[i] = false;

                    //     kpUn = pFrame->mvKeysUn[i];
                    //     const float kp_ur = pFrame->mvuRight[i];
                    //     Eigen::Matrix<double,3,1> obs;
                    //     obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    //     EdgeStereoOnlyTranslation* e = new EdgeStereoOnlyTranslation(pMP->GetWorldPos());

                    //     e->setVertex(0, VP);
                    //     e->setMeasurement(obs);

                    //     // Add here uncerteinty
                    //     const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    //     const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    //     e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    //     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    //     e->setRobustKernel(rk);
                    //     rk->setDelta(thHuberStereo);

                    //     optimizer.addEdge(e);

                    //     vpEdgesStereo.push_back(e);
                    //     vnIndexEdgeStereo.push_back(i);
                    // }

                    // Right monocular observation
                    // if(bRight && i >= Nleft)
                    // {
                    //     nInitialMonoCorrespondences++;
                    //     pFrame->mvbOutlier[i] = false;

                    //     kpUn = pFrame->mvKeysRight[i - Nleft];
                    //     Eigen::Matrix<double,2,1> obs;
                    //     obs << kpUn.pt.x, kpUn.pt.y;

                    //     EdgeMonoOnlyTranslation* e = new EdgeMonoOnlyTranslation(pMP->GetWorldPos(),1);

                    //     e->setVertex(0,VP);
                    //     e->setMeasurement(obs);

                    //     // Add here uncerteinty
                    //     const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    //     const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    //     e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    //     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    //     e->setRobustKernel(rk);
                    //     rk->setDelta(thHuberMono);

                    //     optimizer.addEdge(e);

                    //     vpEdgesMono.push_back(e);
                    //     vnIndexEdgeMono.push_back(i);
                    // }
                }
            }
        }
        
        const int NL = pFrame->NL;

        // vector<EdgeInertialLineProjectXYZOnlyTranslation *> vpEdgesLineSp;
        // vector<size_t> vnIndexLineEdgeSp;
        // vpEdgesLineSp.reserve(NL);
        // vnIndexLineEdgeSp.reserve(NL);

        // vector<EdgeInertialLineProjectXYZOnlyTranslation *> vpEdgesLineEp;
        // vector<size_t> vnIndexLineEdgeEp;
        // vpEdgesLineEp.reserve(NL);
        // vnIndexLineEdgeEp.reserve(NL);

        // vector<double> vMonoStartPointInfo(NL, 1);
        // vector<double> vMonoEndPointInfo(NL, 1);
        // vector<double> vSteroStartPointInfo(NL, 1);
        // vector<double> vSteroEndPointInfo(NL, 1);

        // // Set MapLine vertices
        // {
        //     unique_lock<mutex> lock(MapLine::mGlobalMutex);

        //     for (int i = 0; i < NL; i++) {
        //         MapLine *pML = pFrame->mvpMapLines[i];
        //         if (pML) {
        //             nInitialCorrespondences++;
        //             pFrame->mvbLineOutlier[i] = false;

        //             Eigen::Vector3d line_obs;
        //             line_obs = pFrame->mvKeyLineFunctions[i];

        //             EdgeInertialLineProjectXYZOnlyTranslation *els = new EdgeInertialLineProjectXYZOnlyTranslation(pML->mWorldPos.head(3), 0);

        //             els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        //             els->setMeasurement(line_obs);
        //             els->setInformation(Eigen::Matrix3d::Identity());

        //             g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
        //             els->setRobustKernel(rk_line_s);
        //             rk_line_s->setDelta(thHuberStereo);

        //             els->fx = pFrame->fx;
        //             els->fy = pFrame->fy;
        //             els->cx = pFrame->cx;
        //             els->cy = pFrame->cy;

        //             // els->Xw = pML->mWorldPos.head(3);
        //             optimizer.addEdge(els);

        //             vpEdgesLineSp.push_back(els);
        //             vnIndexLineEdgeSp.push_back(i);

        //             EdgeInertialLineProjectXYZOnlyTranslation *ele = new EdgeInertialLineProjectXYZOnlyTranslation(pML->mWorldPos.tail(3), 0);

        //             ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        //             ele->setMeasurement(line_obs);
        //             ele->setInformation(Eigen::Matrix3d::Identity());

        //             g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
        //             ele->setRobustKernel(rk_line_e);
        //             rk_line_e->setDelta(thHuberStereo);

        //             ele->fx = pFrame->fx;
        //             ele->fy = pFrame->fy;
        //             ele->cx = pFrame->cx;
        //             ele->cy = pFrame->cy;

        //             // ele->Xw = pML->mWorldPos.tail(3);
        //             optimizer.addEdge(ele);

        //             vpEdgesLineEp.push_back(ele);
        //             vnIndexLineEdgeEp.push_back(i);
        //         }
        //     }
        // }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;

        vector<g2o::EdgeInertialPlaneOnlyTranslation *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

        vector<vector<g2o::EdgePlanePoint *>> vpEdgesPlanePoint;
        vector<vector<size_t>> vnIndexEdgePlanePoint;
        vpEdgesPlanePoint = vector<vector<g2o::EdgePlanePoint *>>(M);
        vnIndexEdgePlanePoint = vector<vector<size_t>>(M);

        vector<g2o::EdgeInertialParallelPlaneOnlyTranslation *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeInertialVerticalPlaneOnlyTranslation *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            int PNum = 0;
            double PEror = 0, PMax = 0;
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgeInertialPlaneOnlyTranslation *e = new g2o::EdgeInertialPlaneOnlyTranslation(pMP->GetWorldPos(),0);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    //TODO
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //TODO
                    rk->setDelta(deltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());
                    // e->planePoints = pMP->mvPlanePoints;

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;
            for (int i = 0; i < M; ++i) {
                // add parallel planes!
                MapPlane *pMP = pFrame->mvpParallelPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbParPlaneOutlier[i] = false;

                    g2o::EdgeInertialParallelPlaneOnlyTranslation *e = new g2o::EdgeInertialParallelPlaneOnlyTranslation(pMP->GetWorldPos(),0);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << parInfo, 0,
                            0, parInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesParPlane.push_back(e);
                    vnIndexEdgeParPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Par Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;

            for (int i = 0; i < M; ++i) {
                // add vertical planes!
                MapPlane *pMP = pFrame->mvpVerticalPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbVerPlaneOutlier[i] = false;

                    g2o::EdgeInertialVerticalPlaneOnlyTranslation *e = new g2o::EdgeInertialVerticalPlaneOnlyTranslation(pMP->GetWorldPos(), 0);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << verInfo, 0,
                            0, verInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesVerPlane.push_back(e);
                    vnIndexEdgeVerPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Ver Plane: " << PEror / PNum << endl;
        }

        nInitialCorrespondences += nInitialMonoCorrespondences + nInitialStereoCorrespondences;

        KeyFrame* pKF = pFrame->mpLastKeyFrame;
        VertexPose* VPk = new VertexPose(pKF);
        VPk->setId(4);
        VPk->setFixed(true);
        optimizer.addVertex(VPk);
        VertexVelocity* VVk = new VertexVelocity(pKF);
        VVk->setId(5);
        VVk->setFixed(true);
        optimizer.addVertex(VVk);
        VertexGyroBias* VGk = new VertexGyroBias(pKF);
        VGk->setId(6);
        VGk->setFixed(true);
        optimizer.addVertex(VGk);
        VertexAccBias* VAk = new VertexAccBias(pKF);
        VAk->setId(7);
        VAk->setFixed(true);
        optimizer.addVertex(VAk);

        EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegrated);

        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        g2o::RobustKernelHuber* rei = new g2o::RobustKernelHuber;
        ei->setRobustKernel(rei);
        rei->setDelta(thHuberMono);
        optimizer.addEdge(ei);

        EdgeGyroRW* egr = new EdgeGyroRW();
        egr->setVertex(0,VGk);
        egr->setVertex(1,VG);
        cv::Mat cvInfoG = pFrame->mpImuPreintegrated->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoG;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoG(r,c)=cvInfoG.at<float>(r,c);
        egr->setInformation(InfoG);
        g2o::RobustKernelHuber* regr = new g2o::RobustKernelHuber;
        egr->setRobustKernel(regr);
        regr->setDelta(thHuberMono);
        optimizer.addEdge(egr);

        EdgeAccRW* ear = new EdgeAccRW();
        ear->setVertex(0,VAk);
        ear->setVertex(1,VA);
        cv::Mat cvInfoA = pFrame->mpImuPreintegrated->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoA;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoA(r,c)=cvInfoA.at<float>(r,c);
        ear->setInformation(InfoA);
        g2o::RobustKernelHuber* rear = new g2o::RobustKernelHuber;
        ear->setRobustKernel(rear);
        rear->setDelta(thHuberMono);
        optimizer.addEdge(ear);

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        float chi2Mono[4]={12,7.5,5.991,5.991};
        float chi2Stereo[4]={15.6,9.8,7.815,7.815};

        int its[4]={10,10,10,10};

        int nBad=0;
        int nBadMono = 0;
        int nBadStereo = 0;
        int nInliersMono = 0;
        int nInliersStereo = 0;
        int nInliers=0;
        bool bOut = false;
        for(size_t it=0; it<4; it++)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad=0;
            nBadMono = 0;
            nBadStereo = 0;
            nInliers=0;
            nInliersMono=0;
            nInliersStereo=0;
            float chi2close = 1.5*chi2Mono[it];

            // For monocular observations
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                EdgeMonoOnlyTranslation* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

                if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
                {
                    pFrame->mvbOutlier[idx]=true;
                    e->setLevel(1);
                    nBadMono++;
                }
                else
                {
                    pFrame->mvbOutlier[idx]=false;
                    e->setLevel(0);
                    nInliersMono++;
                }

                if (it==2)
                    e->setRobustKernel(0);
            }

            // For stereo observations
            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
            {
                EdgeStereoOnlyTranslation* e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Stereo[it])
                {
                    pFrame->mvbOutlier[idx]=true;
                    e->setLevel(1); // not included in next optimization
                    nBadStereo++;
                }
                else
                {
                    pFrame->mvbOutlier[idx]=false;
                    e->setLevel(0);
                    nInliersStereo++;
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

//             int PNLine = 0;
//             double PELine = 0, PMaxLine = 0;
//             for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
//                 EdgeInertialLineProjectXYZOnlyTranslation *e1 = vpEdgesLineSp[i];  //线段起始点
//                 EdgeInertialLineProjectXYZOnlyTranslation *e2 = vpEdgesLineEp[i];  //线段终止点

//                 const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

//                 if (pFrame->mvbLineOutlier[idx]) {
//                     e1->computeError();
//                     e2->computeError();
//                 }
//                 e1->computeError();
//                 e2->computeError();

//                 const float chi2_s = e1->chiline();//e1->chi2();
//                 const float chi2_e = e2->chiline();//e2->chi2();
// //                cout<<"Optimization: chi2_s "<<chi2_s<<", chi2_e "<<chi2_e<<endl;

//                 PNLine++;
//                 PELine += chi2_s + chi2_e;
//                 PMaxLine = PMaxLine > chi2_s + chi2_e ? PMaxLine : chi2_s + chi2_e;


//                 if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
//                     pFrame->mvbLineOutlier[idx] = true;
//                     e1->setLevel(1);
//                     e2->setLevel(1);
//                     nBad++;
//                 } else {
//                     pFrame->mvbLineOutlier[idx] = false;
//                     e1->setLevel(0);
//                     e2->setLevel(0);
//                     vSteroEndPointInfo[i] = 1.0 / sqrt(chi2_e);
//                     vSteroStartPointInfo[i] = 1.0 / sqrt(chi2_s);
//                 }

//                 if (it == 2) {
//                     e1->setRobustKernel(0);
//                     e2->setRobustKernel(0);
//                 }
//             }

//            if (PNLine == 0)
//                cout << "No lines " << " ";
//            else
//                cout << " Lines: " << PELine / PNLine << endl;

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgeInertialPlaneOnlyTranslation *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No plane " << " ";
//            else
//                cout << " Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;
            for (size_t i = 0, iend = vpEdgesParPlane.size(); i < iend; i++) {
                g2o::EdgeInertialParallelPlaneOnlyTranslation *e = vpEdgesParPlane[i];

                const size_t idx = vnIndexEdgeParPlane[i];

                if (pFrame->mvbParPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbParPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbParPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No par plane " << " ";
//            else
//                cout << "par Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;

            for (size_t i = 0, iend = vpEdgesVerPlane.size(); i < iend; i++) {
                g2o::EdgeInertialVerticalPlaneOnlyTranslation *e = vpEdgesVerPlane[i];

                const size_t idx = vnIndexEdgeVerPlane[i];

                if (pFrame->mvbVerPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbVerPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbVerPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            nInliers = nInliersMono + nInliersStereo;
            nBad += nBadMono + nBadStereo;

            if(optimizer.edges().size()<10)
            {
                cout << "PIOLKF: NOT ENOUGH EDGES" << endl;
                return -1;
            }

        }

        // If not too much tracks, recover not too bad points
        if ((nInliers<30) && !bRecInit)
        {
            nBad=0;
            const float chi2MonoOut = 18.f;
            const float chi2StereoOut = 24.f;
            EdgeMonoOnlyTranslation* e1;
            EdgeStereoOnlyTranslation* e2;
            for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeMono[i];
                e1 = vpEdgesMono[i];
                e1->computeError();
                if (e1->chi2()<chi2MonoOut)
                    pFrame->mvbOutlier[idx]=false;
                else
                    nBad++;
            }
            for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeStereo[i];
                e2 = vpEdgesStereo[i];
                e2->computeError();
                if (e2->chi2()<chi2StereoOut)
                    pFrame->mvbOutlier[idx]=false;
                else
                    nBad++;
            }
        }

        // Recover optimized pose, velocity and biases
        pFrame->SetImuPoseVelocity(Converter::toCvMat(VP->estimate().Rwb),Converter::toCvMat(VP->estimate().twb),Converter::toCvMat(VV->estimate()));
        Vector6d b;
        b << VG->estimate(), VA->estimate();
        pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

        // Recover Hessian, marginalize keyFframe states and generate new prior for frame
        Eigen::Matrix<double,15,15> H;
        H.setZero();

        H.block<9,9>(0,0)+= ei->GetHessian2();
        H.block<3,3>(9,9) += egr->GetHessian2();
        H.block<3,3>(12,12) += ear->GetHessian2();

        int tot_in = 0, tot_out = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyTranslation* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6,6>(0,0) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyTranslation* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6,6>(0,0) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H);

        return nInitialCorrespondences-nBad;
    }

    int Optimizer::PoseInertialOptimizationLastKeyFrame(Frame *pFrame, bool bRecInit)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
        optimizer.setVerbose(false);
        optimizer.setAlgorithm(solver);

        int nInitialMonoCorrespondences=0;
        int nInitialStereoCorrespondences=0;
        int nInitialCorrespondences=0;

        // Set Frame vertex
        VertexPose* VP = new VertexPose(pFrame);
        VP->setId(0);
        VP->setFixed(false);
        optimizer.addVertex(VP);
        VertexVelocity* VV = new VertexVelocity(pFrame);
        VV->setId(1);
        VV->setFixed(false);
        optimizer.addVertex(VV);
        VertexGyroBias* VG = new VertexGyroBias(pFrame);
        VG->setId(2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pFrame);
        VA->setId(3);
        VA->setFixed(false);
        optimizer.addVertex(VA);

        // Set MapPoint vertices
        const int N = pFrame->N;
        // const int Nleft = pFrame->Nleft;
        // const bool bRight = (Nleft!=-1);

        vector<EdgeMonoOnlyPose*> vpEdgesMono;
        vector<EdgeStereoOnlyPose*> vpEdgesStereo;
        vector<size_t> vnIndexEdgeMono;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesMono.reserve(N);
        vpEdgesStereo.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);


        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<N; i++)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(pMP)
                {
                    cv::KeyPoint kpUn;

                    // Left monocular observation
                    if(pFrame->mvuRight[i]<0)
                    {   
                        // TODO: IMU Check mvKeys or mvKeys
                        // if(i < Nleft) // pair left-right
                            kpUn = pFrame->mvKeys[i];
                        // else
                        // kpUn = pFrame->mvKeysUn[i];

                        nInitialMonoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                        e->setVertex(0,VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    // Stereo observation
                    else
                    {
                        nInitialStereoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        kpUn = pFrame->mvKeysUn[i];
                        const float kp_ur = pFrame->mvuRight[i];
                        Eigen::Matrix<double,3,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                        e->setVertex(0, VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                        const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }

                    // Right monocular observation
                    // if(bRight && i >= Nleft)
                    // {
                    //     nInitialMonoCorrespondences++;
                    //     pFrame->mvbOutlier[i] = false;

                    //     kpUn = pFrame->mvKeysRight[i - Nleft];
                    //     Eigen::Matrix<double,2,1> obs;
                    //     obs << kpUn.pt.x, kpUn.pt.y;

                    //     EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                    //     e->setVertex(0,VP);
                    //     e->setMeasurement(obs);

                    //     // Add here uncerteinty
                    //     const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    //     const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    //     e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    //     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    //     e->setRobustKernel(rk);
                    //     rk->setDelta(thHuberMono);

                    //     optimizer.addEdge(e);

                    //     vpEdgesMono.push_back(e);
                    //     vnIndexEdgeMono.push_back(i);
                    // }
                }
            }
        }
        
        const int NL = pFrame->NL;

        vector<EdgeInertialLineProjectXYZOnlyPose *> vpEdgesLineSp;
        vector<size_t> vnIndexLineEdgeSp;
        vpEdgesLineSp.reserve(NL);
        vnIndexLineEdgeSp.reserve(NL);

        vector<EdgeInertialLineProjectXYZOnlyPose *> vpEdgesLineEp;
        vector<size_t> vnIndexLineEdgeEp;
        vpEdgesLineEp.reserve(NL);
        vnIndexLineEdgeEp.reserve(NL);

        vector<double> vMonoStartPointInfo(NL, 1);
        vector<double> vMonoEndPointInfo(NL, 1);
        vector<double> vSteroStartPointInfo(NL, 1);
        vector<double> vSteroEndPointInfo(NL, 1);

        // Set MapLine vertices
        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            for (int i = 0; i < NL; i++) {
                MapLine *pML = pFrame->mvpMapLines[i];
                if (pML) {
                    nInitialCorrespondences++;
                    pFrame->mvbLineOutlier[i] = false;

                    Eigen::Vector3d line_obs;
                    line_obs = pFrame->mvKeyLineFunctions[i];

                    EdgeInertialLineProjectXYZOnlyPose *els = new EdgeInertialLineProjectXYZOnlyPose(pML->mWorldPos.head(3), 0);

                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(thHuberStereo);

                    els->fx = pFrame->fx;
                    els->fy = pFrame->fy;
                    els->cx = pFrame->cx;
                    els->cy = pFrame->cy;

                    // els->Xw = pML->mWorldPos.head(3);
                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    EdgeInertialLineProjectXYZOnlyPose *ele = new EdgeInertialLineProjectXYZOnlyPose(pML->mWorldPos.tail(3), 0);

                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(thHuberStereo);

                    ele->fx = pFrame->fx;
                    ele->fy = pFrame->fy;
                    ele->cx = pFrame->cx;
                    ele->cy = pFrame->cy;

                    // ele->Xw = pML->mWorldPos.tail(3);
                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                    vnIndexLineEdgeEp.push_back(i);
                }
            }
        }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;

        vector<g2o::EdgeInertialPlaneOnlyPose *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

        vector<vector<g2o::EdgePlanePoint *>> vpEdgesPlanePoint;
        vector<vector<size_t>> vnIndexEdgePlanePoint;
        vpEdgesPlanePoint = vector<vector<g2o::EdgePlanePoint *>>(M);
        vnIndexEdgePlanePoint = vector<vector<size_t>>(M);

        vector<g2o::EdgeInertialParallelPlaneOnlyPose *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeInertialVerticalPlaneOnlyPose *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            int PNum = 0;
            double PEror = 0, PMax = 0;
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgeInertialPlaneOnlyPose *e = new g2o::EdgeInertialPlaneOnlyPose(pMP->GetWorldPos(),0);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    //TODO
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //TODO
                    rk->setDelta(deltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());
                    // e->planePoints = pMP->mvPlanePoints;

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;
            for (int i = 0; i < M; ++i) {
                // add parallel planes!
                MapPlane *pMP = pFrame->mvpParallelPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbParPlaneOutlier[i] = false;

                    g2o::EdgeInertialParallelPlaneOnlyPose *e = new g2o::EdgeInertialParallelPlaneOnlyPose(pMP->GetWorldPos(),0);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << parInfo, 0,
                            0, parInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesParPlane.push_back(e);
                    vnIndexEdgeParPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Par Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;

            for (int i = 0; i < M; ++i) {
                // add vertical planes!
                MapPlane *pMP = pFrame->mvpVerticalPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbVerPlaneOutlier[i] = false;

                    g2o::EdgeInertialVerticalPlaneOnlyPose *e = new g2o::EdgeInertialVerticalPlaneOnlyPose(pMP->GetWorldPos(), 0);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << verInfo, 0,
                            0, verInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesVerPlane.push_back(e);
                    vnIndexEdgeVerPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Ver Plane: " << PEror / PNum << endl;
        }

        nInitialCorrespondences += nInitialMonoCorrespondences + nInitialStereoCorrespondences;

        KeyFrame* pKF = pFrame->mpLastKeyFrame;
        VertexPose* VPk = new VertexPose(pKF);
        VPk->setId(4);
        VPk->setFixed(true);
        optimizer.addVertex(VPk);
        VertexVelocity* VVk = new VertexVelocity(pKF);
        VVk->setId(5);
        VVk->setFixed(true);
        optimizer.addVertex(VVk);
        VertexGyroBias* VGk = new VertexGyroBias(pKF);
        VGk->setId(6);
        VGk->setFixed(true);
        optimizer.addVertex(VGk);
        VertexAccBias* VAk = new VertexAccBias(pKF);
        VAk->setId(7);
        VAk->setFixed(true);
        optimizer.addVertex(VAk);

        EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegrated);

        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        g2o::RobustKernelHuber* rei = new g2o::RobustKernelHuber;
        ei->setRobustKernel(rei);
        rei->setDelta(thHuberMono);
        optimizer.addEdge(ei);

        EdgeGyroRW* egr = new EdgeGyroRW();
        egr->setVertex(0,VGk);
        egr->setVertex(1,VG);
        cv::Mat cvInfoG = pFrame->mpImuPreintegrated->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoG;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoG(r,c)=cvInfoG.at<float>(r,c);
        egr->setInformation(InfoG);
        g2o::RobustKernelHuber* regr = new g2o::RobustKernelHuber;
        egr->setRobustKernel(regr);
        regr->setDelta(thHuberMono);
        optimizer.addEdge(egr);

        EdgeAccRW* ear = new EdgeAccRW();
        ear->setVertex(0,VAk);
        ear->setVertex(1,VA);
        cv::Mat cvInfoA = pFrame->mpImuPreintegrated->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoA;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoA(r,c)=cvInfoA.at<float>(r,c);
        ear->setInformation(InfoA);
        g2o::RobustKernelHuber* rear = new g2o::RobustKernelHuber;
        ear->setRobustKernel(rear);
        rear->setDelta(thHuberMono);
        optimizer.addEdge(ear);

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        float chi2Mono[4]={12,7.5,5.991,5.991};
        float chi2Stereo[4]={15.6,9.8,7.815,7.815};

        int its[4]={10,10,10,10};

        int nBad=0;
        int nBadMono = 0;
        int nBadStereo = 0;
        int nInliersMono = 0;
        int nInliersStereo = 0;
        int nInliers=0;
        bool bOut = false;
        for(size_t it=0; it<4; it++)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad=0;
            nBadMono = 0;
            nBadStereo = 0;
            nInliers=0;
            nInliersMono=0;
            nInliersStereo=0;
            float chi2close = 1.5*chi2Mono[it];

            // For monocular observations
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                EdgeMonoOnlyPose* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

                if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
                {
                    pFrame->mvbOutlier[idx]=true;
                    e->setLevel(1);
                    nBadMono++;
                }
                else
                {
                    pFrame->mvbOutlier[idx]=false;
                    e->setLevel(0);
                    nInliersMono++;
                }

                if (it==2)
                    e->setRobustKernel(0);
            }

            // For stereo observations
            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
            {
                EdgeStereoOnlyPose* e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Stereo[it])
                {
                    pFrame->mvbOutlier[idx]=true;
                    e->setLevel(1); // not included in next optimization
                    nBadStereo++;
                }
                else
                {
                    pFrame->mvbOutlier[idx]=false;
                    e->setLevel(0);
                    nInliersStereo++;
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

            int PNLine = 0;
            double PELine = 0, PMaxLine = 0;
            for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
                EdgeInertialLineProjectXYZOnlyPose *e1 = vpEdgesLineSp[i];  //线段起始点
                EdgeInertialLineProjectXYZOnlyPose *e2 = vpEdgesLineEp[i];  //线段终止点

                const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

                if (pFrame->mvbLineOutlier[idx]) {
                    e1->computeError();
                    e2->computeError();
                }
                e1->computeError();
                e2->computeError();

                const float chi2_s = e1->chiline();//e1->chi2();
                const float chi2_e = e2->chiline();//e2->chi2();
//                cout<<"Optimization: chi2_s "<<chi2_s<<", chi2_e "<<chi2_e<<endl;

                PNLine++;
                PELine += chi2_s + chi2_e;
                PMaxLine = PMaxLine > chi2_s + chi2_e ? PMaxLine : chi2_s + chi2_e;


                if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
                    pFrame->mvbLineOutlier[idx] = true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbLineOutlier[idx] = false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                    vSteroEndPointInfo[i] = 1.0 / sqrt(chi2_e);
                    vSteroStartPointInfo[i] = 1.0 / sqrt(chi2_s);
                }

                if (it == 2) {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }

//            if (PNLine == 0)
//                cout << "No lines " << " ";
//            else
//                cout << " Lines: " << PELine / PNLine << endl;

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgeInertialPlaneOnlyPose *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No plane " << " ";
//            else
//                cout << " Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;
            for (size_t i = 0, iend = vpEdgesParPlane.size(); i < iend; i++) {
                g2o::EdgeInertialParallelPlaneOnlyPose *e = vpEdgesParPlane[i];

                const size_t idx = vnIndexEdgeParPlane[i];

                if (pFrame->mvbParPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbParPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbParPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No par plane " << " ";
//            else
//                cout << "par Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;

            for (size_t i = 0, iend = vpEdgesVerPlane.size(); i < iend; i++) {
                g2o::EdgeInertialVerticalPlaneOnlyPose *e = vpEdgesVerPlane[i];

                const size_t idx = vnIndexEdgeVerPlane[i];

                if (pFrame->mvbVerPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbVerPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbVerPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            nInliers = nInliersMono + nInliersStereo;
            nBad += nBadMono + nBadStereo;

            if(optimizer.edges().size()<10)
            {
                cout << "PIOLKF: NOT ENOUGH EDGES" << endl;
                return -1;
            }

        }

        // If not too much tracks, recover not too bad points
        if ((nInliers<30) && !bRecInit)
        {
            nBad=0;
            const float chi2MonoOut = 18.f;
            const float chi2StereoOut = 24.f;
            EdgeMonoOnlyPose* e1;
            EdgeStereoOnlyPose* e2;
            for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeMono[i];
                e1 = vpEdgesMono[i];
                e1->computeError();
                if (e1->chi2()<chi2MonoOut)
                    pFrame->mvbOutlier[idx]=false;
                else
                    nBad++;
            }
            for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeStereo[i];
                e2 = vpEdgesStereo[i];
                e2->computeError();
                if (e2->chi2()<chi2StereoOut)
                    pFrame->mvbOutlier[idx]=false;
                else
                    nBad++;
            }
        }

        // Recover optimized pose, velocity and biases
        pFrame->SetImuPoseVelocity(Converter::toCvMat(VP->estimate().Rwb),Converter::toCvMat(VP->estimate().twb),Converter::toCvMat(VV->estimate()));
        Vector6d b;
        b << VG->estimate(), VA->estimate();
        pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

        // Recover Hessian, marginalize keyFframe states and generate new prior for frame
        Eigen::Matrix<double,15,15> H;
        H.setZero();

        H.block<9,9>(0,0)+= ei->GetHessian2();
        H.block<3,3>(9,9) += egr->GetHessian2();
        H.block<3,3>(12,12) += ear->GetHessian2();

        int tot_in = 0, tot_out = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6,6>(0,0) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6,6>(0,0) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H);

        return nInitialCorrespondences-nBad;
    }

    int Optimizer::PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        int nInitialMonoCorrespondences=0;
        int nInitialStereoCorrespondences=0;
        int nInitialCorrespondences=0;

        // Set Current Frame vertex
        VertexPose* VP = new VertexPose(pFrame);
        VP->setId(0);
        VP->setFixed(false);
        optimizer.addVertex(VP);
        VertexVelocity* VV = new VertexVelocity(pFrame);
        VV->setId(1);
        VV->setFixed(false);
        optimizer.addVertex(VV);
        VertexGyroBias* VG = new VertexGyroBias(pFrame);
        VG->setId(2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pFrame);
        VA->setId(3);
        VA->setFixed(false);
        optimizer.addVertex(VA);

        // Set MapPoint vertices
        const int N = pFrame->N;
        // const int Nleft = pFrame->Nleft;
        // const bool bRight = (Nleft!=-1);

        vector<EdgeMonoOnlyPose*> vpEdgesMono;
        vector<EdgeStereoOnlyPose*> vpEdgesStereo;
        vector<size_t> vnIndexEdgeMono;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesMono.reserve(N);
        vpEdgesStereo.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<N; i++)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(pMP)
                {
                    cv::KeyPoint kpUn;
                    // Left monocular observation
                    if(pFrame->mvuRight[i]<0)
                    {
                        // if(i < Nleft) // pair left-right
                            // kpUn = pFrame->mvKeys[i];
                        // else
                            kpUn = pFrame->mvKeysUn[i];

                        nInitialMonoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                        e->setVertex(0,VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    // Stereo observation
                    else
                    {
                        nInitialStereoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        kpUn = pFrame->mvKeysUn[i];
                        const float kp_ur = pFrame->mvuRight[i];
                        Eigen::Matrix<double,3,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                        e->setVertex(0, VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                        const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }

                    // Right monocular observation
                    // if(bRight && i >= Nleft)
                    // {
                    //     nInitialMonoCorrespondences++;
                    //     pFrame->mvbOutlier[i] = false;

                    //     kpUn = pFrame->mvKeysRight[i - Nleft];
                    //     Eigen::Matrix<double,2,1> obs;
                    //     obs << kpUn.pt.x, kpUn.pt.y;

                    //     EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                    //     e->setVertex(0,VP);
                    //     e->setMeasurement(obs);

                    //     // Add here uncerteinty
                    //     const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    //     const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    //     e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    //     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    //     e->setRobustKernel(rk);
                    //     rk->setDelta(thHuberMono);

                    //     optimizer.addEdge(e);

                    //     vpEdgesMono.push_back(e);
                    //     vnIndexEdgeMono.push_back(i);
                    // }
                }
            }
        }

        const int NL = pFrame->NL;

        vector<EdgeInertialLineProjectXYZOnlyPose *> vpEdgesLineSp;
        vector<size_t> vnIndexLineEdgeSp;
        vpEdgesLineSp.reserve(NL);
        vnIndexLineEdgeSp.reserve(NL);

        vector<EdgeInertialLineProjectXYZOnlyPose *> vpEdgesLineEp;
        vector<size_t> vnIndexLineEdgeEp;
        vpEdgesLineEp.reserve(NL);
        vnIndexLineEdgeEp.reserve(NL);

        vector<double> vMonoStartPointInfo(NL, 1);
        vector<double> vMonoEndPointInfo(NL, 1);
        vector<double> vSteroStartPointInfo(NL, 1);
        vector<double> vSteroEndPointInfo(NL, 1);

        // Set MapLine vertices
        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            for (int i = 0; i < NL; i++) {
                MapLine *pML = pFrame->mvpMapLines[i];
                if (pML) {
                    nInitialCorrespondences++;
                    pFrame->mvbLineOutlier[i] = false;

                    Eigen::Vector3d line_obs;
                    line_obs = pFrame->mvKeyLineFunctions[i];

                    EdgeInertialLineProjectXYZOnlyPose *els = new EdgeInertialLineProjectXYZOnlyPose(pML->mWorldPos.head(3), 0);

                    els->setVertex(0, VP);
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(thHuberStereo);

                    els->fx = pFrame->fx;
                    els->fy = pFrame->fy;
                    els->cx = pFrame->cx;
                    els->cy = pFrame->cy;

                    // els->Xw = pML->mWorldPos.head(3);
                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    EdgeInertialLineProjectXYZOnlyPose *ele = new EdgeInertialLineProjectXYZOnlyPose(pML->mWorldPos.tail(3), 0);

                    ele->setVertex(0, VP);
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(thHuberStereo);

                    ele->fx = pFrame->fx;
                    ele->fy = pFrame->fy;
                    ele->cx = pFrame->cx;
                    ele->cy = pFrame->cy;

                    // ele->Xw = pML->mWorldPos.tail(3);
                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                    vnIndexLineEdgeEp.push_back(i);
                }
            }
        }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;

        vector<g2o::EdgeInertialPlaneOnlyPose *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

        vector<vector<g2o::EdgePlanePoint *>> vpEdgesPlanePoint;
        vector<vector<size_t>> vnIndexEdgePlanePoint;
        vpEdgesPlanePoint = vector<vector<g2o::EdgePlanePoint *>>(M);
        vnIndexEdgePlanePoint = vector<vector<size_t>>(M);

        vector<g2o::EdgeInertialParallelPlaneOnlyPose *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeInertialVerticalPlaneOnlyPose *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            int PNum = 0;
            double PEror = 0, PMax = 0;
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgeInertialPlaneOnlyPose *e = new g2o::EdgeInertialPlaneOnlyPose(pMP->GetWorldPos(),0);
                    e->setVertex(0, VP);
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    //TODO
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //TODO
                    rk->setDelta(deltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());
                    // e->planePoints = pMP->mvPlanePoints;

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);




                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;
            for (int i = 0; i < M; ++i) {
                // add parallel planes!
                MapPlane *pMP = pFrame->mvpParallelPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbParPlaneOutlier[i] = false;

                    g2o::EdgeInertialParallelPlaneOnlyPose *e = new g2o::EdgeInertialParallelPlaneOnlyPose(pMP->GetWorldPos(), 0);
                    e->setVertex(0, VP);
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << parInfo, 0,
                            0, parInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesParPlane.push_back(e);
                    vnIndexEdgeParPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Par Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;

            for (int i = 0; i < M; ++i) {
                // add vertical planes!
                MapPlane *pMP = pFrame->mvpVerticalPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbVerPlaneOutlier[i] = false;

                    g2o::EdgeInertialVerticalPlaneOnlyPose *e = new g2o::EdgeInertialVerticalPlaneOnlyPose(pMP->GetWorldPos(), 0);
                    e->setVertex(0, VP);
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << verInfo, 0,
                            0, verInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesVerPlane.push_back(e);
                    vnIndexEdgeVerPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Ver Plane: " << PEror / PNum << endl;
        }

        nInitialCorrespondences += nInitialMonoCorrespondences + nInitialStereoCorrespondences;

        // Set Previous Frame Vertex
        Frame* pFp = pFrame->mpPrevFrame;

        VertexPose* VPk = new VertexPose(pFp);
        VPk->setId(4);
        VPk->setFixed(false);
        optimizer.addVertex(VPk);
        VertexVelocity* VVk = new VertexVelocity(pFp);
        VVk->setId(5);
        VVk->setFixed(false);
        optimizer.addVertex(VVk);
        VertexGyroBias* VGk = new VertexGyroBias(pFp);
        VGk->setId(6);
        VGk->setFixed(false);
        optimizer.addVertex(VGk);
        VertexAccBias* VAk = new VertexAccBias(pFp);
        VAk->setId(7);
        VAk->setFixed(false);
        optimizer.addVertex(VAk);

        EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);

        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        g2o::RobustKernelHuber* rei = new g2o::RobustKernelHuber;
        ei->setRobustKernel(rei);
        rei->setDelta(thHuberMono);
        optimizer.addEdge(ei);

        EdgeGyroRW* egr = new EdgeGyroRW();
        egr->setVertex(0,VGk);
        egr->setVertex(1,VG);
        cv::Mat cvInfoG = pFrame->mpImuPreintegratedFrame->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoG;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoG(r,c)=cvInfoG.at<float>(r,c);
        egr->setInformation(InfoG);
        g2o::RobustKernelHuber* regr = new g2o::RobustKernelHuber;
        egr->setRobustKernel(regr);
        regr->setDelta(thHuberMono);
        optimizer.addEdge(egr);

        EdgeAccRW* ear = new EdgeAccRW();
        ear->setVertex(0,VAk);
        ear->setVertex(1,VA);
        cv::Mat cvInfoA = pFrame->mpImuPreintegratedFrame->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoA;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoA(r,c)=cvInfoA.at<float>(r,c);
        ear->setInformation(InfoA);
        g2o::RobustKernelHuber* rear = new g2o::RobustKernelHuber;
        ear->setRobustKernel(rear);
        rear->setDelta(thHuberMono);
        optimizer.addEdge(ear);

        if (!pFp->mpcpi){
            Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId), Verbose::VERBOSITY_NORMAL);
            return -1;
        }

        EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);

        ep->setVertex(0,VPk);
        ep->setVertex(1,VVk);
        ep->setVertex(2,VGk);
        ep->setVertex(3,VAk);
        g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
        ep->setRobustKernel(rkp);
        rkp->setDelta(thHuberMono);
        optimizer.addEdge(ep);

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.

        const float chi2Mono[4]={5.991,5.991,5.991,5.991};
        const float chi2Stereo[4]={15.6f,9.8f,7.815f,7.815f};
        const int its[4]={10,10,10,10};

        int nBad=0;
        int nBadMono = 0;
        int nBadStereo = 0;
        int nInliersMono = 0;
        int nInliersStereo = 0;
        int nInliers=0;
        for(size_t it=0; it<4; it++)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad=0;
            nBadMono = 0;
            nBadStereo = 0;
            nInliers=0;
            nInliersMono=0;
            nInliersStereo=0;
            float chi2close = 1.5*chi2Mono[it];

            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                EdgeMonoOnlyPose* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];
                bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
                {
                    pFrame->mvbOutlier[idx]=true;
                    e->setLevel(1);
                    nBadMono++;
                }
                else
                {
                    pFrame->mvbOutlier[idx]=false;
                    e->setLevel(0);
                    nInliersMono++;
                }

                if (it==2)
                    e->setRobustKernel(0);

            }

            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
            {
                EdgeStereoOnlyPose* e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Stereo[it])
                {
                    pFrame->mvbOutlier[idx]=true;
                    e->setLevel(1);
                    nBadStereo++;
                }
                else
                {
                    pFrame->mvbOutlier[idx]=false;
                    e->setLevel(0);
                    nInliersStereo++;
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

            int PNLine = 0;
            double PELine = 0, PMaxLine = 0;
            for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
                EdgeInertialLineProjectXYZOnlyPose *e1 = vpEdgesLineSp[i];  //线段起始点
                EdgeInertialLineProjectXYZOnlyPose *e2 = vpEdgesLineEp[i];  //线段终止点

                const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

                if (pFrame->mvbLineOutlier[idx]) {
                    e1->computeError();
                    e2->computeError();
                }
                e1->computeError();
                e2->computeError();

                const float chi2_s = e1->chiline();//e1->chi2();
                const float chi2_e = e2->chiline();//e2->chi2();
//                cout<<"Optimization: chi2_s "<<chi2_s<<", chi2_e "<<chi2_e<<endl;

                PNLine++;
                PELine += chi2_s + chi2_e;
                PMaxLine = PMaxLine > chi2_s + chi2_e ? PMaxLine : chi2_s + chi2_e;


                if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
                    pFrame->mvbLineOutlier[idx] = true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbLineOutlier[idx] = false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                    vSteroEndPointInfo[i] = 1.0 / sqrt(chi2_e);
                    vSteroStartPointInfo[i] = 1.0 / sqrt(chi2_s);
                }

                if (it == 2) {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgeInertialPlaneOnlyPose *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No plane " << " ";
//            else
//                cout << " Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;
            for (size_t i = 0, iend = vpEdgesParPlane.size(); i < iend; i++) {
                g2o::EdgeInertialParallelPlaneOnlyPose *e = vpEdgesParPlane[i];

                const size_t idx = vnIndexEdgeParPlane[i];

                if (pFrame->mvbParPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbParPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbParPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No par plane " << " ";
//            else
//                cout << "par Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;

            for (size_t i = 0, iend = vpEdgesVerPlane.size(); i < iend; i++) {
                g2o::EdgeInertialVerticalPlaneOnlyPose *e = vpEdgesVerPlane[i];

                const size_t idx = vnIndexEdgeVerPlane[i];

                if (pFrame->mvbVerPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbVerPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbVerPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            nInliers = nInliersMono + nInliersStereo;
            nBad += nBadMono + nBadStereo;

            if(optimizer.edges().size()<10)
            {
                cout << "PIOLF: NOT ENOUGH EDGES" << endl;
                break;
            }
        }


        if ((nInliers<30) && !bRecInit)
        {
            nBad=0;
            const float chi2MonoOut = 18.f;
            const float chi2StereoOut = 24.f;
            EdgeMonoOnlyPose* e1;
            EdgeStereoOnlyPose* e2;
            for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeMono[i];
                e1 = vpEdgesMono[i];
                e1->computeError();
                if (e1->chi2()<chi2MonoOut)
                    pFrame->mvbOutlier[idx]=false;
                else
                    nBad++;

            }
            for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeStereo[i];
                e2 = vpEdgesStereo[i];
                e2->computeError();
                if (e2->chi2()<chi2StereoOut)
                    pFrame->mvbOutlier[idx]=false;
                else
                    nBad++;
            }
        }

        nInliers = nInliersMono + nInliersStereo;


        // Recover optimized pose, velocity and biases
        pFrame->SetImuPoseVelocity(Converter::toCvMat(VP->estimate().Rwb),Converter::toCvMat(VP->estimate().twb),Converter::toCvMat(VV->estimate()));
        Vector6d b;
        b << VG->estimate(), VA->estimate();
        pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

        // Recover Hessian, marginalize previous frame states and generate new prior for frame
        Eigen::Matrix<double,30,30> H;
        H.setZero();

        H.block<24,24>(0,0)+= ei->GetHessian();

        Eigen::Matrix<double,6,6> Hgr = egr->GetHessian();
        H.block<3,3>(9,9) += Hgr.block<3,3>(0,0);
        H.block<3,3>(9,24) += Hgr.block<3,3>(0,3);
        H.block<3,3>(24,9) += Hgr.block<3,3>(3,0);
        H.block<3,3>(24,24) += Hgr.block<3,3>(3,3);

        Eigen::Matrix<double,6,6> Har = ear->GetHessian();
        H.block<3,3>(12,12) += Har.block<3,3>(0,0);
        H.block<3,3>(12,27) += Har.block<3,3>(0,3);
        H.block<3,3>(27,12) += Har.block<3,3>(3,0);
        H.block<3,3>(27,27) += Har.block<3,3>(3,3);

        H.block<15,15>(0,0) += ep->GetHessian();

        int tot_in = 0, tot_out = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6,6>(15,15) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6,6>(15,15) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        H = Marginalize(H,0,14);

        pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H.block<15,15>(15,15));
        delete pFp->mpcpi;
        pFp->mpcpi = NULL;

        return nInitialCorrespondences-nBad;
    }

    int Optimizer::TranslationInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        int nInitialMonoCorrespondences=0;
        int nInitialStereoCorrespondences=0;
        int nInitialCorrespondences=0;

        // Set Current Frame vertex
        VertexPose* VP = new VertexPose(pFrame);
        VP->setId(0);
        VP->setFixed(false);
        optimizer.addVertex(VP);
        VertexVelocity* VV = new VertexVelocity(pFrame);
        VV->setId(1);
        VV->setFixed(false);
        optimizer.addVertex(VV);
        VertexGyroBias* VG = new VertexGyroBias(pFrame);
        VG->setId(2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pFrame);
        VA->setId(3);
        VA->setFixed(false);
        optimizer.addVertex(VA);

        // Set MapPoint vertices
        const int N = pFrame->N;
        // const int Nleft = pFrame->Nleft;
        // const bool bRight = (Nleft!=-1);

        vector<EdgeMonoOnlyTranslation*> vpEdgesMono;
        vector<EdgeStereoOnlyTranslation*> vpEdgesStereo;
        vector<size_t> vnIndexEdgeMono;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesMono.reserve(N);
        vpEdgesStereo.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<N; i++)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(pMP)
                {
                    cv::KeyPoint kpUn;
                    // Left monocular observation
                    // if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                    {
                        // TODO: IMU Check mvKeys or mvKeys
                        // if(i < Nleft) // pair left-right
                            // kpUn = pFrame->mvKeys[i];
                        // else
                            kpUn = pFrame->mvKeysUn[i];

                        nInitialMonoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyTranslation* e = new EdgeMonoOnlyTranslation(pMP->GetWorldPos(),0);

                        e->setVertex(0,VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    // Stereo observation
                    // else if(!bRight)
                    // {
                    //     nInitialStereoCorrespondences++;
                    //     pFrame->mvbOutlier[i] = false;

                    //     kpUn = pFrame->mvKeysUn[i];
                    //     const float kp_ur = pFrame->mvuRight[i];
                    //     Eigen::Matrix<double,3,1> obs;
                    //     obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    //     EdgeStereoOnlyTranslation* e = new EdgeStereoOnlyTranslation(pMP->GetWorldPos());

                    //     e->setVertex(0, VP);
                    //     e->setMeasurement(obs);

                    //     // Add here uncerteinty
                    //     const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    //     const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    //     e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    //     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    //     e->setRobustKernel(rk);
                    //     rk->setDelta(thHuberStereo);

                    //     optimizer.addEdge(e);

                    //     vpEdgesStereo.push_back(e);
                    //     vnIndexEdgeStereo.push_back(i);
                    // }

                    // Right monocular observation
                    // if(bRight && i >= Nleft)
                    // {
                    //     nInitialMonoCorrespondences++;
                    //     pFrame->mvbOutlier[i] = false;

                    //     kpUn = pFrame->mvKeysRight[i - Nleft];
                    //     Eigen::Matrix<double,2,1> obs;
                    //     obs << kpUn.pt.x, kpUn.pt.y;

                    //     EdgeMonoOnlyTranslation* e = new EdgeMonoOnlyTranslation(pMP->GetWorldPos(),1);

                    //     e->setVertex(0,VP);
                    //     e->setMeasurement(obs);

                    //     // Add here uncerteinty
                    //     const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    //     const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    //     e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    //     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    //     e->setRobustKernel(rk);
                    //     rk->setDelta(thHuberMono);

                    //     optimizer.addEdge(e);

                    //     vpEdgesMono.push_back(e);
                    //     vnIndexEdgeMono.push_back(i);
                    // }
                }
            }
        }

        const int NL = pFrame->NL;

        // vector<EdgeInertialLineProjectXYZOnlyTranslation *> vpEdgesLineSp;
        // vector<size_t> vnIndexLineEdgeSp;
        // vpEdgesLineSp.reserve(NL);
        // vnIndexLineEdgeSp.reserve(NL);

        // vector<EdgeInertialLineProjectXYZOnlyTranslation *> vpEdgesLineEp;
        // vector<size_t> vnIndexLineEdgeEp;
        // vpEdgesLineEp.reserve(NL);
        // vnIndexLineEdgeEp.reserve(NL);

        // vector<double> vMonoStartPointInfo(NL, 1);
        // vector<double> vMonoEndPointInfo(NL, 1);
        // vector<double> vSteroStartPointInfo(NL, 1);
        // vector<double> vSteroEndPointInfo(NL, 1);

        // // Set MapLine vertices
        // {
        //     unique_lock<mutex> lock(MapLine::mGlobalMutex);

        //     for (int i = 0; i < NL; i++) {
        //         MapLine *pML = pFrame->mvpMapLines[i];
        //         if (pML) {
        //             nInitialCorrespondences++;
        //             pFrame->mvbLineOutlier[i] = false;

        //             Eigen::Vector3d line_obs;
        //             line_obs = pFrame->mvKeyLineFunctions[i];

        //             EdgeInertialLineProjectXYZOnlyTranslation *els = new EdgeInertialLineProjectXYZOnlyTranslation(pML->mWorldPos.head(3), 0);

        //             els->setVertex(0, VP);
        //             els->setMeasurement(line_obs);
        //             els->setInformation(Eigen::Matrix3d::Identity());

        //             g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
        //             els->setRobustKernel(rk_line_s);
        //             rk_line_s->setDelta(thHuberStereo);

        //             els->fx = pFrame->fx;
        //             els->fy = pFrame->fy;
        //             els->cx = pFrame->cx;
        //             els->cy = pFrame->cy;

        //             // els->Xw = pML->mWorldPos.head(3);
        //             optimizer.addEdge(els);

        //             vpEdgesLineSp.push_back(els);
        //             vnIndexLineEdgeSp.push_back(i);

        //             EdgeInertialLineProjectXYZOnlyTranslation *ele = new EdgeInertialLineProjectXYZOnlyTranslation(pML->mWorldPos.tail(3), 0);

        //             ele->setVertex(0, VP);
        //             ele->setMeasurement(line_obs);
        //             ele->setInformation(Eigen::Matrix3d::Identity());

        //             g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
        //             ele->setRobustKernel(rk_line_e);
        //             rk_line_e->setDelta(thHuberStereo);

        //             ele->fx = pFrame->fx;
        //             ele->fy = pFrame->fy;
        //             ele->cx = pFrame->cx;
        //             ele->cy = pFrame->cy;

        //             // ele->Xw = pML->mWorldPos.tail(3);
        //             optimizer.addEdge(ele);

        //             vpEdgesLineEp.push_back(ele);
        //             vnIndexLineEdgeEp.push_back(i);
        //         }
        //     }
        // }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;

        vector<g2o::EdgeInertialPlaneOnlyTranslation *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

        vector<vector<g2o::EdgePlanePoint *>> vpEdgesPlanePoint;
        vector<vector<size_t>> vnIndexEdgePlanePoint;
        vpEdgesPlanePoint = vector<vector<g2o::EdgePlanePoint *>>(M);
        vnIndexEdgePlanePoint = vector<vector<size_t>>(M);

        vector<g2o::EdgeInertialParallelPlaneOnlyTranslation *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeInertialVerticalPlaneOnlyTranslation *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            int PNum = 0;
            double PEror = 0, PMax = 0;
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgeInertialPlaneOnlyTranslation *e = new g2o::EdgeInertialPlaneOnlyTranslation(pMP->GetWorldPos(),0);
                    e->setVertex(0, VP);
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    //TODO
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //TODO
                    rk->setDelta(deltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());
                    // e->planePoints = pMP->mvPlanePoints;

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);




                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;
            for (int i = 0; i < M; ++i) {
                // add parallel planes!
                MapPlane *pMP = pFrame->mvpParallelPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbParPlaneOutlier[i] = false;

                    g2o::EdgeInertialParallelPlaneOnlyTranslation *e = new g2o::EdgeInertialParallelPlaneOnlyTranslation(pMP->GetWorldPos(), 0);
                    e->setVertex(0, VP);
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << parInfo, 0,
                            0, parInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesParPlane.push_back(e);
                    vnIndexEdgeParPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Par Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;

            for (int i = 0; i < M; ++i) {
                // add vertical planes!
                MapPlane *pMP = pFrame->mvpVerticalPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbVerPlaneOutlier[i] = false;

                    g2o::EdgeInertialVerticalPlaneOnlyTranslation *e = new g2o::EdgeInertialVerticalPlaneOnlyTranslation(pMP->GetWorldPos(), 0);
                    e->setVertex(0, VP);
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << verInfo, 0,
                            0, verInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    // e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesVerPlane.push_back(e);
                    vnIndexEdgeVerPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Ver Plane: " << PEror / PNum << endl;
        }

        nInitialCorrespondences += nInitialMonoCorrespondences + nInitialStereoCorrespondences;

        // Set Previous Frame Vertex
        Frame* pFp = pFrame->mpPrevFrame;

        VertexPose* VPk = new VertexPose(pFp);
        VPk->setId(4);
        VPk->setFixed(false);
        optimizer.addVertex(VPk);
        VertexVelocity* VVk = new VertexVelocity(pFp);
        VVk->setId(5);
        VVk->setFixed(false);
        optimizer.addVertex(VVk);
        VertexGyroBias* VGk = new VertexGyroBias(pFp);
        VGk->setId(6);
        VGk->setFixed(false);
        optimizer.addVertex(VGk);
        VertexAccBias* VAk = new VertexAccBias(pFp);
        VAk->setId(7);
        VAk->setFixed(false);
        optimizer.addVertex(VAk);

        EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);

        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        g2o::RobustKernelHuber* rei = new g2o::RobustKernelHuber;
        ei->setRobustKernel(rei);
        rei->setDelta(thHuberMono);
        optimizer.addEdge(ei);

        EdgeGyroRW* egr = new EdgeGyroRW();
        egr->setVertex(0,VGk);
        egr->setVertex(1,VG);
        cv::Mat cvInfoG = pFrame->mpImuPreintegratedFrame->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoG;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoG(r,c)=cvInfoG.at<float>(r,c);
        egr->setInformation(InfoG);
        g2o::RobustKernelHuber* regr = new g2o::RobustKernelHuber;
        egr->setRobustKernel(regr);
        regr->setDelta(thHuberMono);
        optimizer.addEdge(egr);

        EdgeAccRW* ear = new EdgeAccRW();
        ear->setVertex(0,VAk);
        ear->setVertex(1,VA);
        cv::Mat cvInfoA = pFrame->mpImuPreintegratedFrame->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoA;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoA(r,c)=cvInfoA.at<float>(r,c);
        ear->setInformation(InfoA);
        g2o::RobustKernelHuber* rear = new g2o::RobustKernelHuber;
        ear->setRobustKernel(rear);
        rear->setDelta(thHuberMono);
        optimizer.addEdge(ear);

        if (!pFp->mpcpi){
            Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId), Verbose::VERBOSITY_NORMAL);
            return -1;
        }
            

        EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);

        ep->setVertex(0,VPk);
        ep->setVertex(1,VVk);
        ep->setVertex(2,VGk);
        ep->setVertex(3,VAk);
        g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
        ep->setRobustKernel(rkp);
        rkp->setDelta(thHuberMono);
        optimizer.addEdge(ep);

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.

        const float chi2Mono[4]={5.991,5.991,5.991,5.991};
        const float chi2Stereo[4]={15.6f,9.8f,7.815f,7.815f};
        const int its[4]={10,10,10,10};

        int nBad=0;
        int nBadMono = 0;
        int nBadStereo = 0;
        int nInliersMono = 0;
        int nInliersStereo = 0;
        int nInliers=0;
        for(size_t it=0; it<4; it++)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad=0;
            nBadMono = 0;
            nBadStereo = 0;
            nInliers=0;
            nInliersMono=0;
            nInliersStereo=0;
            float chi2close = 1.5*chi2Mono[it];

            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                EdgeMonoOnlyTranslation* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];
                bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
                {
                    pFrame->mvbOutlier[idx]=true;
                    e->setLevel(1);
                    nBadMono++;
                }
                else
                {
                    pFrame->mvbOutlier[idx]=false;
                    e->setLevel(0);
                    nInliersMono++;
                }

                if (it==2)
                    e->setRobustKernel(0);

            }

            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
            {
                EdgeStereoOnlyTranslation* e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Stereo[it])
                {
                    pFrame->mvbOutlier[idx]=true;
                    e->setLevel(1);
                    nBadStereo++;
                }
                else
                {
                    pFrame->mvbOutlier[idx]=false;
                    e->setLevel(0);
                    nInliersStereo++;
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

//             int PNLine = 0;
//             double PELine = 0, PMaxLine = 0;
//             for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
//                 EdgeInertialLineProjectXYZOnlyTranslation *e1 = vpEdgesLineSp[i];  //线段起始点
//                 EdgeInertialLineProjectXYZOnlyTranslation *e2 = vpEdgesLineEp[i];  //线段终止点

//                 const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

//                 if (pFrame->mvbLineOutlier[idx]) {
//                     e1->computeError();
//                     e2->computeError();
//                 }
//                 e1->computeError();
//                 e2->computeError();

//                 const float chi2_s = e1->chiline();//e1->chi2();
//                 const float chi2_e = e2->chiline();//e2->chi2();
// //                cout<<"Optimization: chi2_s "<<chi2_s<<", chi2_e "<<chi2_e<<endl;

//                 PNLine++;
//                 PELine += chi2_s + chi2_e;
//                 PMaxLine = PMaxLine > chi2_s + chi2_e ? PMaxLine : chi2_s + chi2_e;


//                 if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
//                     pFrame->mvbLineOutlier[idx] = true;
//                     e1->setLevel(1);
//                     e2->setLevel(1);
//                     nBad++;
//                 } else {
//                     pFrame->mvbLineOutlier[idx] = false;
//                     e1->setLevel(0);
//                     e2->setLevel(0);
//                     vSteroEndPointInfo[i] = 1.0 / sqrt(chi2_e);
//                     vSteroStartPointInfo[i] = 1.0 / sqrt(chi2_s);
//                 }

//                 if (it == 2) {
//                     e1->setRobustKernel(0);
//                     e2->setRobustKernel(0);
//                 }
//             }

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgeInertialPlaneOnlyTranslation *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No plane " << " ";
//            else
//                cout << " Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;
            for (size_t i = 0, iend = vpEdgesParPlane.size(); i < iend; i++) {
                g2o::EdgeInertialParallelPlaneOnlyTranslation *e = vpEdgesParPlane[i];

                const size_t idx = vnIndexEdgeParPlane[i];

                if (pFrame->mvbParPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbParPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbParPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No par plane " << " ";
//            else
//                cout << "par Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;

            for (size_t i = 0, iend = vpEdgesVerPlane.size(); i < iend; i++) {
                g2o::EdgeInertialVerticalPlaneOnlyTranslation *e = vpEdgesVerPlane[i];

                const size_t idx = vnIndexEdgeVerPlane[i];

                if (pFrame->mvbVerPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbVerPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbVerPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            nInliers = nInliersMono + nInliersStereo;
            nBad += nBadMono + nBadStereo;

            if(optimizer.edges().size()<10)
            {
                cout << "PIOLF: NOT ENOUGH EDGES" << endl;
                break;
            }
        }


        if ((nInliers<30) && !bRecInit)
        {
            nBad=0;
            const float chi2MonoOut = 18.f;
            const float chi2StereoOut = 24.f;
            EdgeMonoOnlyTranslation* e1;
            EdgeStereoOnlyTranslation* e2;
            for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeMono[i];
                e1 = vpEdgesMono[i];
                e1->computeError();
                if (e1->chi2()<chi2MonoOut)
                    pFrame->mvbOutlier[idx]=false;
                else
                    nBad++;

            }
            for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeStereo[i];
                e2 = vpEdgesStereo[i];
                e2->computeError();
                if (e2->chi2()<chi2StereoOut)
                    pFrame->mvbOutlier[idx]=false;
                else
                    nBad++;
            }
        }

        nInliers = nInliersMono + nInliersStereo;


        // Recover optimized pose, velocity and biases
        pFrame->SetImuPoseVelocity(Converter::toCvMat(VP->estimate().Rwb),Converter::toCvMat(VP->estimate().twb),Converter::toCvMat(VV->estimate()));
        Vector6d b;
        b << VG->estimate(), VA->estimate();
        pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

        // Recover Hessian, marginalize previous frame states and generate new prior for frame
        Eigen::Matrix<double,30,30> H;
        H.setZero();

        H.block<24,24>(0,0)+= ei->GetHessian();

        Eigen::Matrix<double,6,6> Hgr = egr->GetHessian();
        H.block<3,3>(9,9) += Hgr.block<3,3>(0,0);
        H.block<3,3>(9,24) += Hgr.block<3,3>(0,3);
        H.block<3,3>(24,9) += Hgr.block<3,3>(3,0);
        H.block<3,3>(24,24) += Hgr.block<3,3>(3,3);

        Eigen::Matrix<double,6,6> Har = ear->GetHessian();
        H.block<3,3>(12,12) += Har.block<3,3>(0,0);
        H.block<3,3>(12,27) += Har.block<3,3>(0,3);
        H.block<3,3>(27,12) += Har.block<3,3>(3,0);
        H.block<3,3>(27,27) += Har.block<3,3>(3,3);

        H.block<15,15>(0,0) += ep->GetHessian();

        int tot_in = 0, tot_out = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyTranslation* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6,6>(15,15) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyTranslation* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6,6>(15,15) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        H = Marginalize(H,0,14);

        pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H.block<15,15>(15,15));
        delete pFp->mpcpi;
        pFp->mpcpi = NULL;

        return nInitialCorrespondences-nBad;
    }

    Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
    {
        // Goal
        // a  | ab | ac       a*  | 0 | ac*
        // ba | b  | bc  -->  0   | 0 | 0
        // ca | cb | c        ca* | 0 | c*

        // Size of block before block to marginalize
        const int a = start;
        // Size of block to marginalize
        const int b = end-start+1;
        // Size of block after block to marginalize
        const int c = H.cols() - (end+1);

        // Reorder as follows:
        // a  | ab | ac       a  | ac | ab
        // ba | b  | bc  -->  ca | c  | cb
        // ca | cb | c        ba | bc | b

        Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(),H.cols());
        if(a>0)
        {
            Hn.block(0,0,a,a) = H.block(0,0,a,a);
            Hn.block(0,a+c,a,b) = H.block(0,a,a,b);
            Hn.block(a+c,0,b,a) = H.block(a,0,b,a);
        }
        if(a>0 && c>0)
        {
            Hn.block(0,a,a,c) = H.block(0,a+b,a,c);
            Hn.block(a,0,c,a) = H.block(a+b,0,c,a);
        }
        if(c>0)
        {
            Hn.block(a,a,c,c) = H.block(a+b,a+b,c,c);
            Hn.block(a,a+c,c,b) = H.block(a+b,a,c,b);
            Hn.block(a+c,a,b,c) = H.block(a,a+b,b,c);
        }
        Hn.block(a+c,a+c,b,b) = H.block(a,a,b,b);

        // Perform marginalization (Schur complement)
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a+c,a+c,b,b),Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv=svd.singularValues();
        for (int i=0; i<b; ++i)
        {
            if (singularValues_inv(i)>1e-6)
                singularValues_inv(i)=1.0/singularValues_inv(i);
            else singularValues_inv(i)=0;
        }
        Eigen::MatrixXd invHb = svd.matrixV()*singularValues_inv.asDiagonal()*svd.matrixU().transpose();
        Hn.block(0,0,a+c,a+c) = Hn.block(0,0,a+c,a+c) - Hn.block(0,a+c,a+c,b)*invHb*Hn.block(a+c,0,b,a+c);
        Hn.block(a+c,a+c,b,b) = Eigen::MatrixXd::Zero(b,b);
        Hn.block(0,a+c,a+c,b) = Eigen::MatrixXd::Zero(a+c,b);
        Hn.block(a+c,0,b,a+c) = Eigen::MatrixXd::Zero(b,a+c);

        // Inverse reorder
        // a*  | ac* | 0       a*  | 0 | ac*
        // ca* | c*  | 0  -->  0   | 0 | 0
        // 0   | 0   | 0       ca* | 0 | c*
        Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(),H.cols());
        if(a>0)
        {
            res.block(0,0,a,a) = Hn.block(0,0,a,a);
            res.block(0,a,a,b) = Hn.block(0,a+c,a,b);
            res.block(a,0,b,a) = Hn.block(a+c,0,b,a);
        }
        if(a>0 && c>0)
        {
            res.block(0,a+b,a,c) = Hn.block(0,a,a,c);
            res.block(a+b,0,c,a) = Hn.block(a,0,c,a);
        }
        if(c>0)
        {
            res.block(a+b,a+b,c,c) = Hn.block(a,a,c,c);
            res.block(a+b,a,c,b) = Hn.block(a,a+c,c,b);
            res.block(a,a+b,b,c) = Hn.block(a+c,a,b,c);
        }

        res.block(a,a,b,b) = Hn.block(a+c,a+c,b,b);

        return res;
    }

    Eigen::MatrixXd Optimizer::Condition(const Eigen::MatrixXd &H, const int &start, const int &end)
    {
        // Size of block before block to condition
        const int a = start;
        // Size of block to condition
        const int b = end+1-start;

        // Set to zero elements related to block b(start:end,start:end)
        // a  | ab | ac       a  | 0 | ac
        // ba | b  | bc  -->  0  | 0 | 0
        // ca | cb | c        ca | 0 | c

        Eigen::MatrixXd Hn = H;

        Hn.block(a,0,b,H.cols()) = Eigen::MatrixXd::Zero(b,H.cols());
        Hn.block(0,a,H.rows(),b) = Eigen::MatrixXd::Zero(H.rows(),b);

        return Hn;
    }

    Eigen::MatrixXd Optimizer::Sparsify(const Eigen::MatrixXd &H, const int &start1, const int &end1, const int &start2, const int &end2)
    {
        // Goal: remove link between a and b
        // p(a,b,c) ~ p(a,b,c)*p(a|c)/p(a|b,c) => H' = H + H1 - H2
        // H1: marginalize b and condition c
        // H2: condition b and c
        Eigen::MatrixXd Hac = Marginalize(H,start2,end2);
        Eigen::MatrixXd Hbc = Marginalize(H,start1,end1);
        Eigen::MatrixXd Hc = Marginalize(Hac,start1,end1);

        return Hac+Hbc-Hc;
    }

    void Optimizer::LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, bool bLarge, bool bRecInit)
    {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        Map* pCurrentMap = pKF->GetMap();

        int maxOpt=10;
        int opt_it=10;
        if(bLarge)
        {
            maxOpt=25;
            opt_it=4;
        }
        const int Nd = std::min((int)pCurrentMap->KeyFramesInMap()-2,maxOpt);
        const unsigned long maxKFid = pKF->mnId;

        vector<KeyFrame*> vpOptimizableKFs;
        const vector<KeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
        list<KeyFrame*> lpOptVisKFs;

        vpOptimizableKFs.reserve(Nd);
        vpOptimizableKFs.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;
        for(int i=1; i<Nd; i++)
        {
            if(vpOptimizableKFs.back()->mPrevKF)
            {
                vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
                vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
            }
            else
                break;
        }

        int N = vpOptimizableKFs.size();

        // Optimizable points seen by temporal optimizable keyframes
        list<MapPoint*> lLocalMapPoints;
        for(int i=0; i<N; i++)
        {
            vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId; // associate the point to the current frame in this local BA
                        }
            }
        }

        // // Local MapLines seen in Local KeyFrames
        // list<MapLine *> lLocalMapLines;
        // for(int i=0; i<N; i++) {
        //     vector<MapLine *> vpMLs = vpOptimizableKFs[i]->GetMapLineMatches();
        //     for (vector<MapLine *>::iterator vit = vpMLs.begin(), vend = vpMLs.end(); vit != vend; vit++) {

        //         MapLine *pML = *vit;
        //         if (pML) {
        //             if (!pML->isBad()) {
        //                 if (pML->mnBALocalForKF != pKF->mnId) {
        //                     lLocalMapLines.push_back(pML);
        //                     pML->mnBALocalForKF = pKF->mnId;
        //                 }
        //             }
        //         }
        //     }
        // }

        // Local MapPlanes seen in Local KeyFrames
        list<MapPlane *> lLocalMapPlanes;
        for(int i=0; i<N; i++) {
            vector<MapPlane *> vpMPs = vpOptimizableKFs[i]->GetMapPlaneMatches();
            for (vector<MapPlane *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {

                MapPlane *pMP = *vit;
                if (pMP) {
                    if (!pMP->isBad()) {
                        if (pMP->mnBALocalForKF != pKF->mnId) {
                            lLocalMapPlanes.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        // Fixed Keyframe: First frame previous KF to optimization window)
        list<KeyFrame*> lFixedKeyFrames;
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pKF->mnId;
        }
        else
        {
            vpOptimizableKFs.back()->mnBALocalForKF=0;
            vpOptimizableKFs.back()->mnBAFixedForKF=pKF->mnId;
            lFixedKeyFrames.push_back(vpOptimizableKFs.back());
            vpOptimizableKFs.pop_back();
        }

        // Optimizable visual KFs
        const int maxCovKF = 0;
        for(int i=0, iend=vpNeighsKFs.size(); i<iend; i++)
        {
            if(lpOptVisKFs.size() >= maxCovKF)
                break;

            KeyFrame* pKFi = vpNeighsKFs[i];
            if(pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId || pKFi->mnId>maxKFid)
                continue;
            pKFi->mnBALocalForKF = pKF->mnId;
            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                lpOptVisKFs.push_back(pKFi);

                vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
                for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
                {
                    MapPoint* pMP = *vit;
                    if(pMP)
                        if(!pMP->isBad())
                            if(pMP->mnBALocalForKF!=pKF->mnId)
                            {
                                lLocalMapPoints.push_back(pMP);
                                pMP->mnBALocalForKF=pKF->mnId;
                            }
                }
            }
        }

        // Fixed KFs which are not covisible optimizable
        const int maxFixKF = 200;

        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
            for(map<KeyFrame *, size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId && pKFi->mnId<=maxKFid)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad())
                    {
                        lFixedKeyFrames.push_back(pKFi);
                        break;
                    }
                }
            }
            if(lFixedKeyFrames.size()>=maxFixKF)
                break;
        }

        // for(list<MapLine*>::iterator lit=lLocalMapLines.begin(), lend=lLocalMapLines.end(); lit!=lend; lit++)
        // {
        //     map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        //     for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        //     {
        //         KeyFrame* pKFi = mit->first;

        //         if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId && pKFi->mnId<=maxKFid)
        //         {
        //             pKFi->mnBAFixedForKF=pKF->mnId;
        //             if(!pKFi->isBad())
        //                 lFixedKeyFrames.push_back(pKFi);
        //         }
        //     }
        // }

        for(list<MapPlane*>::iterator lit=lLocalMapPlanes.begin(), lend=lLocalMapPlanes.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId && pKFi->mnId<=maxKFid)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad())
                        lFixedKeyFrames.push_back(pKFi);
                }
            }
        }

        bool bNonFixed = (lFixedKeyFrames.size() == 0);

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        if(bLarge)
        {
            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
            solver->setUserLambdaInit(1e-2); // to avoid iterating for finding optimal lambda
            optimizer.setAlgorithm(solver);
        }
        else
        {
            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
            solver->setUserLambdaInit(1e0);
            optimizer.setAlgorithm(solver);
        }


        // Set Local temporal KeyFrame vertices
        N=vpOptimizableKFs.size();
        for(int i=0; i<N; i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];

            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(false);
            optimizer.addVertex(VP);

            if(pKFi->bImu)
            {
                VertexVelocity* VV = new VertexVelocity(pKFi);
                VV->setId(maxKFid+3*(pKFi->mnId)+1);
                VV->setFixed(false);
                optimizer.addVertex(VV);
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid+3*(pKFi->mnId)+2);
                VG->setFixed(false);
                optimizer.addVertex(VG);
                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid+3*(pKFi->mnId)+3);
                VA->setFixed(false);
                optimizer.addVertex(VA);
            }
        }

        // Set Local visual KeyFrame vertices
        for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
        {
            KeyFrame* pKFi = *it;
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(false);
            optimizer.addVertex(VP);
        }

        // Set Fixed KeyFrame vertices
        for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(true);
            optimizer.addVertex(VP);

            if(pKFi->bImu) // This should be done only for keyframe just before temporal window
            {
                VertexVelocity* VV = new VertexVelocity(pKFi);
                VV->setId(maxKFid+3*(pKFi->mnId)+1);
                VV->setFixed(true);
                optimizer.addVertex(VV);
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid+3*(pKFi->mnId)+2);
                VG->setFixed(true);
                optimizer.addVertex(VG);
                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid+3*(pKFi->mnId)+3);
                VA->setFixed(true);
                optimizer.addVertex(VA);
            }
        }

        // Create intertial constraints
        vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
        vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
        vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);

        for(int i=0;i<N;i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];

            if(!pKFi->mPrevKF)
            {
                cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << endl;
                continue;
            }
            if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
            {
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
                g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
                g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
                g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
                g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

                if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                {
                    cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                    continue;
                }

                vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

                vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                if(i==N-1 || bRecInit)
                {
                    // All inertial residuals are included without robust cost function, but not that one linking the
                    // last optimizable keyframe inside of the local window and the first fixed keyframe out. The
                    // information matrix for this measurement is also downweighted. This is done to avoid accumulating
                    // error due to fixing variables.
                    g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                    vei[i]->setRobustKernel(rki);
                    if(i==N-1)
                        vei[i]->setInformation(vei[i]->information()*1e-2);
                    rki->setDelta(sqrt(16.92));
                }
                optimizer.addEdge(vei[i]);

                vegr[i] = new EdgeGyroRW();
                vegr[i]->setVertex(0,VG1);
                vegr[i]->setVertex(1,VG2);
                cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
                Eigen::Matrix3d InfoG;

                for(int r=0;r<3;r++)
                    for(int c=0;c<3;c++)
                        InfoG(r,c)=cvInfoG.at<float>(r,c);
                vegr[i]->setInformation(InfoG);
                g2o::RobustKernelHuber* rvegr = new g2o::RobustKernelHuber;
                vegr[i]->setRobustKernel(rvegr);
                rvegr->setDelta(sqrt(5.991));
                optimizer.addEdge(vegr[i]);

                // cout << "b";
                vear[i] = new EdgeAccRW();
                vear[i]->setVertex(0,VA1);
                vear[i]->setVertex(1,VA2);
                cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
                Eigen::Matrix3d InfoA;
                for(int r=0;r<3;r++)
                    for(int c=0;c<3;c++)
                        InfoA(r,c)=cvInfoA.at<float>(r,c);

                g2o::RobustKernelHuber* rvear = new g2o::RobustKernelHuber;
                vear[i]->setRobustKernel(rvear);
                rvear->setDelta(sqrt(5.991));
                vear[i]->setInformation(InfoA);           

                optimizer.addEdge(vear[i]);
            }
            else
                cout << "ERROR building inertial edge" << endl;
        }

        // Set MapPoint vertices
        const int nExpectedSize = (N+lFixedKeyFrames.size())*lLocalMapPoints.size();

        // Mono
        vector<EdgeMono*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        // Stereo
        vector<EdgeStereo*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);



        const float thHuberMono = sqrt(5.991);
        const float chi2Mono2 = 5.991;
        const float thHuberStereo = sqrt(7.815);
        const float chi2Stereo2 = 7.815;
        const float thHuberLD = sqrt(3.84);

        const unsigned long iniMPid = maxKFid*5;
        long unsigned int maxMapPointId = iniMPid;

        map<int,int> mVisEdges;
        for(int i=0;i<N;i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];
            mVisEdges[pKFi->mnId] = 0;
        }
        for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
        {
            mVisEdges[(*lit)->mnId] = 0;
        }

        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

            unsigned long id = pMP->mnId+iniMPid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            if (id > maxMapPointId) 
                maxMapPointId = id;

            // Create visual constraints
            for(map<KeyFrame *, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                    continue;

                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                {
                    const int leftIndex = mit->second;

                    cv::KeyPoint kpUn;

                    // Monocular left observation
                    if(pKFi->mvuRight[leftIndex]<0)
                    {
                        mVisEdges[pKFi->mnId]++;

                        kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMono* e = new EdgeMono(0);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                    else// Stereo observation
                    {
                        kpUn = pKFi->mvKeysUn[leftIndex];
                        mVisEdges[pKFi->mnId]++;

                        const float kp_ur = pKFi->mvuRight[leftIndex];
                        Eigen::Matrix<double,3,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        EdgeStereo* e = new EdgeStereo(0);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2));

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);
                    }

                    // Monocular right observation
                    // if(pKFi->mpCamera2){
                    //     int rightIndex = get<1>(mit->second);

                    //     if(rightIndex != -1 ){
                    //         rightIndex -= pKFi->NLeft;
                    //         mVisEdges[pKFi->mnId]++;

                    //         Eigen::Matrix<double,2,1> obs;
                    //         cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                    //         obs << kp.pt.x, kp.pt.y;

                    //         EdgeMono* e = new EdgeMono(1);

                    //         e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    //         e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    //         e->setMeasurement(obs);

                    //         // Add here uncerteinty
                    //         const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                    //         const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                    //         e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    //         g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    //         e->setRobustKernel(rk);
                    //         rk->setDelta(thHuberMono);

                    //         optimizer.addEdge(e);
                    //         vpEdgesMono.push_back(e);
                    //         vpEdgeKFMono.push_back(pKFi);
                    //         vpMapPointEdgeMono.push_back(pMP);
                    //     }
                    // }
                }
            }
        }

        // const int nLineExpectedSize = (N+lFixedKeyFrames.size())*lLocalMapLines.size();

        // vector<EdgeInertialLineProjectXYZ*> vpLineEdgesStart;
        // vpLineEdgesStart.reserve(nLineExpectedSize);

        // vector<EdgeInertialLineProjectXYZ*> vpLineEdgesEnd;
        // vpLineEdgesEnd.reserve(nLineExpectedSize);

        // vector<KeyFrame*> vpLineEdgeKF;
        // vpLineEdgeKF.reserve(nLineExpectedSize);

        // vector<MapLine*> vpMapLineEdge;
        // vpMapLineEdge.reserve(nLineExpectedSize);

        long unsigned int maxMapLineId = maxMapPointId;

        // for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
        //     MapLine *pML = *lit;
        //     g2o::VertexSBAPointXYZ *vStartPoint = new g2o::VertexSBAPointXYZ();
        //     vStartPoint->setEstimate(pML->GetWorldPos().head(3));
        //     int id1 = (2 * pML->mnId) + 1 + maxMapPointId;
        //     vStartPoint->setId(id1);
        //     vStartPoint->setMarginalized(true);
        //     optimizer.addVertex(vStartPoint);

        //     g2o::VertexSBAPointXYZ *vEndPoint = new VertexSBAPointXYZ();
        //     vEndPoint->setEstimate(pML->GetWorldPos().tail(3));
        //     int id2 = (2 * pML->mnId) + 2 + maxMapPointId;
        //     vEndPoint->setId(id2);
        //     vEndPoint->setMarginalized(true);
        //     optimizer.addVertex(vEndPoint);

        //     if (id2 > maxMapLineId)
        //         maxMapLineId = id2;

        //     const map<KeyFrame *, size_t> observations = pML->GetObservations();

        //     for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end();
        //          mit != mend; mit++) {
        //         KeyFrame *pKFi = mit->first;

        //         if (!pKFi->isBad()) {
        //                 mVisEdges[pKF->mnId]++;

        //             Eigen::Vector3d lineObs = pKF->mvKeyLineFunctions[mit->second];

        //             EdgeInertialLineProjectXYZ *es = new EdgeInertialLineProjectXYZ(0);
        //             es->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
        //             es->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
        //             es->setMeasurement(lineObs);
        //             es->setInformation(Eigen::Matrix3d::Identity());

        //             g2o::RobustKernelHuber *rks = new g2o::RobustKernelHuber;
        //             es->setRobustKernel(rks);
        //             rks->setDelta(thHuberLD);

        //             es->fx = pKF->fx;
        //             es->fy = pKF->fy;
        //             es->cx = pKF->cx;
        //             es->cy = pKF->cy;

        //             optimizer.addEdge(es);

        //             EdgeInertialLineProjectXYZ *ee = new EdgeInertialLineProjectXYZ(0);
        //             ee->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
        //             ee->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
        //             ee->setMeasurement(lineObs);
        //             ee->setInformation(Eigen::Matrix3d::Identity());

        //             g2o::RobustKernelHuber *rke = new g2o::RobustKernelHuber;
        //             ee->setRobustKernel(rke);
        //             rke->setDelta(thHuberLD);

        //             ee->fx = pKF->fx;
        //             ee->fy = pKF->fy;
        //             ee->cx = pKF->cx;
        //             ee->cy = pKF->cy;

        //             optimizer.addEdge(ee);

        //             vpLineEdgesStart.push_back(es);
        //             vpLineEdgesEnd.push_back(ee);
        //             vpLineEdgeKF.push_back(pKFi);
        //             vpMapLineEdge.push_back(pML);
        //         }
        //     }
        // }

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        const int nPlaneExpectedSize = (N+lFixedKeyFrames.size())*lLocalMapPlanes.size();

        vector<g2o::EdgeInertialPlane*> vpPlaneEdges;
        vpPlaneEdges.reserve(nPlaneExpectedSize);

        vector<g2o::EdgeInertialVerticalPlane*> vpVerPlaneEdges;
        vpVerPlaneEdges.reserve(nPlaneExpectedSize);

        vector<g2o::EdgeInertialParallelPlane*> vpParPlaneEdges;
        vpParPlaneEdges.reserve(nPlaneExpectedSize);

        vector<KeyFrame*> vpPlaneEdgeKF;
        // vpLineEdgeKF.reserve(nPlaneExpectedSize);

        vector<KeyFrame*> vpVerPlaneEdgeKF;
        vpVerPlaneEdgeKF.reserve(nPlaneExpectedSize);

        vector<KeyFrame*> vpParPlaneEdgeKF;
        vpParPlaneEdgeKF.reserve(nPlaneExpectedSize);

        vector<MapPlane*> vpMapPlaneEdge;
        vpMapPlaneEdge.reserve(nPlaneExpectedSize);

        vector<MapPlane*> vpVerMapPlaneEdge;
        vpVerMapPlaneEdge.reserve(nPlaneExpectedSize);

        vector<MapPlane*> vpParMapPlaneEdge;
        vpParMapPlaneEdge.reserve(nPlaneExpectedSize);

        // Set MapPlane vertices
        for (list<MapPlane *>::iterator lit = lLocalMapPlanes.begin(), lend = lLocalMapPlanes.end(); lit != lend; lit++) {
            MapPlane *pMP = *lit;

            g2o::VertexPlane *vPlane = new g2o::VertexPlane();
            vPlane->setEstimate(Converter::toPlane3D(pMP->GetWorldPos()));
            const int id = pMP->mnId + maxMapLineId + 1;
            vPlane->setId(id);
            vPlane->setMarginalized(true);
            optimizer.addVertex(vPlane);

            Eigen::Matrix3d Info;
            Info << angleInfo, 0, 0,
                    0, angleInfo, 0,
                    0, 0, disInfo;

            Eigen::Matrix2d VPInfo;
            VPInfo << angleInfo, 0,
                    0, angleInfo;

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            for (const auto &observation : observations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                mVisEdges[pKF->mnId]++;

                g2o::EdgeInertialPlane *e = new g2o::EdgeInertialPlane(0);
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaPlane);

                // e->planePoints = pMP->mvPlanePoints;

                optimizer.addEdge(e);
                vpPlaneEdges.push_back(e);
                vpPlaneEdgeKF.push_back(pKF);
                vpMapPlaneEdge.push_back(pMP);
            }

            const map<KeyFrame *, size_t> verObservations = pMP->GetVerObservations();
            for (const auto &observation : verObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid || mVisEdges.find(pKF->mnId)==mVisEdges.end())
                    continue;

                mVisEdges[pKF->mnId]++;

                g2o::EdgeInertialVerticalPlane *e = new g2o::EdgeInertialVerticalPlane(0);
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                e->setInformation(VPInfo);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
                vpVerPlaneEdges.push_back(e);
                vpVerPlaneEdgeKF.push_back(pKF);
                vpVerMapPlaneEdge.push_back(pMP);
            }

            const map<KeyFrame *, size_t> parObservations = pMP->GetParObservations();
            for (const auto &observation : parObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid || mVisEdges.find(pKF->mnId)==mVisEdges.end())
                    continue;

                mVisEdges[pKF->mnId]++;

                g2o::EdgeInertialParallelPlane *e = new g2o::EdgeInertialParallelPlane(0);
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                e->setInformation(VPInfo);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
                vpParPlaneEdges.push_back(e);
                vpParPlaneEdgeKF.push_back(pKF);
                vpParMapPlaneEdge.push_back(pMP);
            }
        }

        // Why haveing this?
        // cout << "Total map points: " << lLocalMapPoints.size() << endl;
        // for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
        // {
        //     assert(mit->second>=3);
        // }

        if (optimizer.edges().size()<=10){
            cout << "LOCALBA: NOT EOUGH EDGES" << endl;
            return;
        }

        optimizer.initializeOptimization();
        optimizer.computeActiveErrors();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        float err = optimizer.activeRobustChi2();
        optimizer.optimize(opt_it); // Originally to 2
        float err_end = optimizer.activeRobustChi2();
        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);


        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        vector<pair<KeyFrame*,MapPoint*> > vToErase;
        vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

        // Check inlier observations
        // Mono
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            EdgeMono* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];
            bool bClose = pMP->mTrackDepth<10.f;

            if(pMP->isBad())
                continue;

            if((e->chi2()>chi2Mono2 && !bClose) || (e->chi2()>1.5f*chi2Mono2 && bClose) || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFMono[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }


        // Stereo
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            EdgeStereo* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>chi2Stereo2)
            {
                KeyFrame* pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        // vector<pair<KeyFrame *, MapLine *>> vLineToErase;
        // vLineToErase.reserve(vpLineEdgesStart.size());

        // for (size_t i = 0, iend = vpLineEdgesStart.size(); i < iend; i++) {
        //     EdgeInertialLineProjectXYZ *es = vpLineEdgesStart[i];
        //     EdgeInertialLineProjectXYZ *ee = vpLineEdgesEnd[i];
        //     MapLine *pML = vpMapLineEdge[i];

        //     if (pML->isBad())
        //         continue;

        //     if (es->chi2() > 7.815 || ee->chi2() > 7.815) {
        //         KeyFrame *pKFi = vpLineEdgeKF[i];
        //         vLineToErase.push_back(make_pair(pKFi, pML));
        //     }
        // }

        vector<pair<KeyFrame*,MapPlane*> > vPlaneToErase;
        vPlaneToErase.reserve(vpPlaneEdges.size());

        for(size_t i=0, iend=vpPlaneEdges.size(); i<iend;i++)
        {
            g2o::EdgeInertialPlane* e = vpPlaneEdges[i];
            MapPlane* pMP = vpMapPlaneEdge[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>planeChi)
            {
                KeyFrame* pKFi = vpPlaneEdgeKF[i];
                vPlaneToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        vector<pair<KeyFrame*,MapPlane*> > vVerPlaneToErase;
        vVerPlaneToErase.reserve(vpVerPlaneEdges.size());

        for(size_t i=0, iend=vpVerPlaneEdges.size(); i<iend;i++)
        {
            g2o::EdgeInertialVerticalPlane* e = vpVerPlaneEdges[i];
            MapPlane* pMP = vpVerMapPlaneEdge[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>VPplaneChi)
            {
                KeyFrame* pKFi = vpVerPlaneEdgeKF[i];
                vVerPlaneToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        vector<pair<KeyFrame*,MapPlane*> > vParPlaneToErase;
        vParPlaneToErase.reserve(vpParPlaneEdges.size());

        for(size_t i=0, iend=vpParPlaneEdges.size(); i<iend;i++)
        {
            g2o::EdgeInertialParallelPlane* e = vpParPlaneEdges[i];
            MapPlane* pMP = vpParMapPlaneEdge[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>VPplaneChi)
            {
                KeyFrame* pKFi = vpParPlaneEdgeKF[i];
                vParPlaneToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        // Get Map Mutex and erase outliers
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);


        // TODO: Some convergence problems have been detected here
        //cout << "err0 = " << err << endl;
        //cout << "err_end = " << err_end << endl;
        if((2*err < err_end || isnan(err) || isnan(err_end)) && !bLarge) //bGN)
        {
            cout << "FAIL LOCAL-INERTIAL BA!!!!" << endl;
            return;
        }



        if(!vToErase.empty())
        {
            for(size_t i=0;i<vToErase.size();i++)
            {
                KeyFrame* pKFi = vToErase[i].first;
                MapPoint* pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        // if(!vLineToErase.empty())
        // {
        //     for(size_t i=0;i<vLineToErase.size();i++)
        //     {
        //         KeyFrame* pKFi = vLineToErase[i].first;
        //         MapLine* pMLi = vLineToErase[i].second;
        //         pKFi->EraseMapLineMatch(pMLi);
        //         pMLi->EraseObservation(pKFi);
        //     }
        // }

        if(!vPlaneToErase.empty())
        {
            for(size_t i=0;i<vPlaneToErase.size();i++)
            {
                KeyFrame* pKFi = vPlaneToErase[i].first;
                MapPlane* pMPi = vPlaneToErase[i].second;
                pKFi->EraseMapPlaneMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        if(!vVerPlaneToErase.empty())
        {
            for(size_t i=0;i<vVerPlaneToErase.size();i++)
            {
                KeyFrame* pKFi = vVerPlaneToErase[i].first;
                MapPlane* pMPi = vVerPlaneToErase[i].second;
                pKFi->EraseMapVerticalPlaneMatch(pMPi);
                pMPi->EraseVerObservation(pKFi);
            }
        }

        if(!vParPlaneToErase.empty())
        {
            for(size_t i=0;i<vParPlaneToErase.size();i++)
            {
                KeyFrame* pKFi = vParPlaneToErase[i].first;
                MapPlane* pMPi = vParPlaneToErase[i].second;
                pKFi->EraseMapParallelPlaneMatch(pMPi);
                pMPi->EraseParObservation(pKFi);
            }
        }

        // Display main statistcis of optimization
        Verbose::PrintMess("LIBA KFs: " + to_string(N), Verbose::VERBOSITY_DEBUG);
        Verbose::PrintMess("LIBA bNonFixed?: " + to_string(bNonFixed), Verbose::VERBOSITY_DEBUG);
        Verbose::PrintMess("LIBA KFs visual outliers: " + to_string(vToErase.size()), Verbose::VERBOSITY_DEBUG);

        for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
            (*lit)->mnBAFixedForKF = 0;

        // Recover optimized data
        // Local temporal Keyframes
        N=vpOptimizableKFs.size();
        for(int i=0; i<N; i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];

            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
            cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
            pKFi->SetPose(Tcw);
            pKFi->mnBALocalForKF=0;

            if(pKFi->bImu)
            {
                VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
                pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
                VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
                VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
                Vector6d b;
                b << VG->estimate(), VA->estimate();
                pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));

            }
        }

        // Local visual KeyFrame
        for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
        {
            KeyFrame* pKFi = *it;
            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
            cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
            pKFi->SetPose(Tcw);
            pKFi->mnBALocalForKF=0;
        }

        //Points
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }

        // // Lines
        // for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
        //     MapLine *pML = *lit;

        //     g2o::VertexSBAPointXYZ *vStartPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
        //             (2 * pML->mnId) + 1 + maxMapPointId));
        //     g2o::VertexSBAPointXYZ *vEndPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
        //             (2 * (pML->mnId + 1)) + maxMapPointId));

        //     Vector6d LinePos;
        //     LinePos << vStartPoint->estimate(), vEndPoint->estimate();
        //     pML->SetWorldPos(LinePos);
        //     pML->UpdateAverageDir();
        // }

        //Planes
        for (list<MapPlane *>::iterator lit = lLocalMapPlanes.begin(), lend = lLocalMapPlanes.end(); lit != lend; lit++) {
            MapPlane *pMP = *lit;
            g2o::VertexPlane *vPlane = static_cast<g2o::VertexPlane *>(optimizer.vertex(
                    pMP->mnId + maxMapLineId + 1));
            pMP->SetWorldPos(Converter::toCvMat(vPlane->estimate()));
            pMP->UpdateCoefficientsAndPoints();
        }

        pMap->IncreaseChangeIndex();

        // std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

        /*double t_const = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
        double t_opt = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        double t_rec = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
        /*std::cout << " Construction time: " << t_const << std::endl;
        std::cout << " Optimization time: " << t_opt << std::endl;
        std::cout << " Recovery time: " << t_rec << std::endl;
        std::cout << " Total time: " << t_const+t_opt+t_rec << std::endl;
        std::cout << " Optimization iterations: " << opt_it << std::endl;*/

    }

    // void Optimizer::MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF, bool *pbStopFlag, Map *pMap, LoopClosing::KeyFrameAndPose &corrPoses)
    // {
    //     const int Nd = 6;
    //     const unsigned long maxKFid = pCurrKF->mnId;

    //     vector<KeyFrame*> vpOptimizableKFs;
    //     vpOptimizableKFs.reserve(2*Nd);

    //     // For cov KFS, inertial parameters are not optimized
    //     const int maxCovKF=30;
    //     vector<KeyFrame*> vpOptimizableCovKFs;
    //     vpOptimizableCovKFs.reserve(maxCovKF);

    //     // Add sliding window for current KF
    //     vpOptimizableKFs.push_back(pCurrKF);
    //     pCurrKF->mnBALocalForKF = pCurrKF->mnId;
    //     for(int i=1; i<Nd; i++)
    //     {
    //         if(vpOptimizableKFs.back()->mPrevKF)
    //         {
    //             vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
    //             vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
    //         }
    //         else
    //             break;
    //     }

    //     list<KeyFrame*> lFixedKeyFrames;
    //     if(vpOptimizableKFs.back()->mPrevKF)
    //     {
    //         vpOptimizableCovKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
    //         vpOptimizableKFs.back()->mPrevKF->mnBALocalForKF=pCurrKF->mnId;
    //     }
    //     else
    //     {
    //         vpOptimizableCovKFs.push_back(vpOptimizableKFs.back());
    //         vpOptimizableKFs.pop_back();
    //     }

    //     KeyFrame* pKF0 = vpOptimizableCovKFs.back();
    //     cv::Mat Twc0 = pKF0->GetPoseInverse();

    //     // Add temporal neighbours to merge KF (previous and next KFs)
    //     vpOptimizableKFs.push_back(pMergeKF);
    //     pMergeKF->mnBALocalForKF = pCurrKF->mnId;

    //     // Previous KFs
    //     for(int i=1; i<(Nd/2); i++)
    //     {
    //         if(vpOptimizableKFs.back()->mPrevKF)
    //         {
    //             vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
    //             vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
    //         }
    //         else
    //             break;
    //     }

    //     // We fix just once the old map
    //     if(vpOptimizableKFs.back()->mPrevKF)
    //     {
    //         lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
    //         vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pCurrKF->mnId;
    //     }
    //     else
    //     {
    //         vpOptimizableKFs.back()->mnBALocalForKF=0;
    //         vpOptimizableKFs.back()->mnBAFixedForKF=pCurrKF->mnId;
    //         lFixedKeyFrames.push_back(vpOptimizableKFs.back());
    //         vpOptimizableKFs.pop_back();
    //     }

    //     // Next KFs
    //     if(pMergeKF->mNextKF)
    //     {
    //         vpOptimizableKFs.push_back(pMergeKF->mNextKF);
    //         vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
    //     }

    //     while(vpOptimizableKFs.size()<(2*Nd))
    //     {
    //         if(vpOptimizableKFs.back()->mNextKF)
    //         {
    //             vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mNextKF);
    //             vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
    //         }
    //         else
    //             break;
    //     }

    //     int N = vpOptimizableKFs.size();

    //     // Optimizable points seen by optimizable keyframes
    //     list<MapPoint*> lLocalMapPoints;
    //     map<MapPoint*,int> mLocalObs;
    //     for(int i=0; i<N; i++)
    //     {
    //         vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
    //         for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
    //         {
    //             // Using mnBALocalForKF we avoid redundance here, one MP can not be added several times to lLocalMapPoints
    //             MapPoint* pMP = *vit;
    //             if(pMP)
    //                 if(!pMP->isBad())
    //                     if(pMP->mnBALocalForKF!=pCurrKF->mnId)
    //                     {
    //                         mLocalObs[pMP]=1;
    //                         lLocalMapPoints.push_back(pMP);
    //                         pMP->mnBALocalForKF=pCurrKF->mnId;
    //                     }
    //                     else
    //                         mLocalObs[pMP]++;
    //         }
    //     }

    //     std::vector<std::pair<MapPoint*, int>> pairs;
    //     pairs.reserve(mLocalObs.size());
    //     for (auto itr = mLocalObs.begin(); itr != mLocalObs.end(); ++itr)
    //         pairs.push_back(*itr);
    //     sort(pairs.begin(), pairs.end(),sortByVal);

    //     // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    //     int i=0;
    //     for(vector<pair<MapPoint*,int>>::iterator lit=pairs.begin(), lend=pairs.end(); lit!=lend; lit++, i++)
    //     {
    //         map<KeyFrame *, size_t> observations = lit->first->GetObservations();
    //         if(i>=maxCovKF)
    //             break;
    //         for(map<KeyFrame *, size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    //         {
    //             KeyFrame* pKFi = mit->first;

    //             if(pKFi->mnBALocalForKF!=pCurrKF->mnId && pKFi->mnBAFixedForKF!=pCurrKF->mnId) // If optimizable or already included...
    //             {
    //                 pKFi->mnBALocalForKF=pCurrKF->mnId;
    //                 if(!pKFi->isBad())
    //                 {
    //                     vpOptimizableCovKFs.push_back(pKFi);
    //                     break;
    //                 }
    //             }
    //         }
    //     }

    //     g2o::SparseOptimizer optimizer;
    //     g2o::BlockSolverX::LinearSolverType * linearSolver;
    //     linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    //     g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    //     g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    //     solver->setUserLambdaInit(1e3); // TODO uncomment

    //     optimizer.setAlgorithm(solver);
    //     optimizer.setVerbose(false);

    //     // Set Local KeyFrame vertices
    //     N=vpOptimizableKFs.size();
    //     for(int i=0; i<N; i++)
    //     {
    //         KeyFrame* pKFi = vpOptimizableKFs[i];

    //         VertexPose * VP = new VertexPose(pKFi);
    //         VP->setId(pKFi->mnId);
    //         VP->setFixed(false);
    //         optimizer.addVertex(VP);

    //         if(pKFi->bImu)
    //         {
    //             VertexVelocity* VV = new VertexVelocity(pKFi);
    //             VV->setId(maxKFid+3*(pKFi->mnId)+1);
    //             VV->setFixed(false);
    //             optimizer.addVertex(VV);
    //             VertexGyroBias* VG = new VertexGyroBias(pKFi);
    //             VG->setId(maxKFid+3*(pKFi->mnId)+2);
    //             VG->setFixed(false);
    //             optimizer.addVertex(VG);
    //             VertexAccBias* VA = new VertexAccBias(pKFi);
    //             VA->setId(maxKFid+3*(pKFi->mnId)+3);
    //             VA->setFixed(false);
    //             optimizer.addVertex(VA);
    //         }
    //     }

    //     // Set Local cov keyframes vertices
    //     int Ncov=vpOptimizableCovKFs.size();
    //     for(int i=0; i<Ncov; i++)
    //     {
    //         KeyFrame* pKFi = vpOptimizableCovKFs[i];

    //         VertexPose * VP = new VertexPose(pKFi);
    //         VP->setId(pKFi->mnId);
    //         VP->setFixed(false);
    //         optimizer.addVertex(VP);

    //         if(pKFi->bImu)
    //         {
    //             VertexVelocity* VV = new VertexVelocity(pKFi);
    //             VV->setId(maxKFid+3*(pKFi->mnId)+1);
    //             VV->setFixed(false);
    //             optimizer.addVertex(VV);
    //             VertexGyroBias* VG = new VertexGyroBias(pKFi);
    //             VG->setId(maxKFid+3*(pKFi->mnId)+2);
    //             VG->setFixed(false);
    //             optimizer.addVertex(VG);
    //             VertexAccBias* VA = new VertexAccBias(pKFi);
    //             VA->setId(maxKFid+3*(pKFi->mnId)+3);
    //             VA->setFixed(false);
    //             optimizer.addVertex(VA);
    //         }
    //     }

    //     // Set Fixed KeyFrame vertices
    //     for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    //     {
    //         KeyFrame* pKFi = *lit;
    //         VertexPose * VP = new VertexPose(pKFi);
    //         VP->setId(pKFi->mnId);
    //         VP->setFixed(true);
    //         optimizer.addVertex(VP);

    //         if(pKFi->bImu)
    //         {
    //             VertexVelocity* VV = new VertexVelocity(pKFi);
    //             VV->setId(maxKFid+3*(pKFi->mnId)+1);
    //             VV->setFixed(true);
    //             optimizer.addVertex(VV);
    //             VertexGyroBias* VG = new VertexGyroBias(pKFi);
    //             VG->setId(maxKFid+3*(pKFi->mnId)+2);
    //             VG->setFixed(true);
    //             optimizer.addVertex(VG);
    //             VertexAccBias* VA = new VertexAccBias(pKFi);
    //             VA->setId(maxKFid+3*(pKFi->mnId)+3);
    //             VA->setFixed(true);
    //             optimizer.addVertex(VA);
    //         }
    //     }

    //     // Create intertial constraints
    //     vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
    //     vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
    //     vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);
    //     for(int i=0;i<N;i++)
    //     {
    //         //cout << "inserting inertial edge " << i << endl;
    //         KeyFrame* pKFi = vpOptimizableKFs[i];

    //         if(!pKFi->mPrevKF)
    //         {
    //             Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!!!!", Verbose::VERBOSITY_NORMAL);
    //             continue;
    //         }
    //         if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
    //         {
    //             pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
    //             g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
    //             g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
    //             g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
    //             g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
    //             g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
    //             g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
    //             g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
    //             g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

    //             if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
    //             {
    //                 cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
    //                 continue;
    //             }

    //             vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

    //             vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
    //             vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
    //             vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
    //             vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
    //             vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
    //             vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

    //             // TODO Uncomment
    //             g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
    //             vei[i]->setRobustKernel(rki);
    //             rki->setDelta(sqrt(16.92));
    //             optimizer.addEdge(vei[i]);

    //             vegr[i] = new EdgeGyroRW();
    //             vegr[i]->setVertex(0,VG1);
    //             vegr[i]->setVertex(1,VG2);
    //             cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
    //             Eigen::Matrix3d InfoG;

    //             for(int r=0;r<3;r++)
    //                 for(int c=0;c<3;c++)
    //                     InfoG(r,c)=cvInfoG.at<float>(r,c);
    //             vegr[i]->setInformation(InfoG);
    //             optimizer.addEdge(vegr[i]);

    //             vear[i] = new EdgeAccRW();
    //             vear[i]->setVertex(0,VA1);
    //             vear[i]->setVertex(1,VA2);
    //             cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
    //             Eigen::Matrix3d InfoA;
    //             for(int r=0;r<3;r++)
    //                 for(int c=0;c<3;c++)
    //                     InfoA(r,c)=cvInfoA.at<float>(r,c);
    //             vear[i]->setInformation(InfoA);
    //             optimizer.addEdge(vear[i]);
    //         }
    //         else
    //             Verbose::PrintMess("ERROR building inertial edge", Verbose::VERBOSITY_NORMAL);
    //     }

    //     Verbose::PrintMess("end inserting inertial edges", Verbose::VERBOSITY_NORMAL);


    //     // Set MapPoint vertices
    //     const int nExpectedSize = (N+Ncov+lFixedKeyFrames.size())*lLocalMapPoints.size();

    //     // Mono
    //     vector<EdgeMono*> vpEdgesMono;
    //     vpEdgesMono.reserve(nExpectedSize);

    //     vector<KeyFrame*> vpEdgeKFMono;
    //     vpEdgeKFMono.reserve(nExpectedSize);

    //     vector<MapPoint*> vpMapPointEdgeMono;
    //     vpMapPointEdgeMono.reserve(nExpectedSize);

    //     // Stereo
    //     vector<EdgeStereo*> vpEdgesStereo;
    //     vpEdgesStereo.reserve(nExpectedSize);

    //     vector<KeyFrame*> vpEdgeKFStereo;
    //     vpEdgeKFStereo.reserve(nExpectedSize);

    //     vector<MapPoint*> vpMapPointEdgeStereo;
    //     vpMapPointEdgeStereo.reserve(nExpectedSize);

    //     const float thHuberMono = sqrt(5.991);
    //     const float chi2Mono2 = 5.991;
    //     const float thHuberStereo = sqrt(7.815);
    //     const float chi2Stereo2 = 7.815;




    //     const unsigned long iniMPid = maxKFid*5; // TODO: should be  maxKFid*4;


    //     Verbose::PrintMess("start inserting MPs", Verbose::VERBOSITY_NORMAL);
    //     for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    //     {
    //         MapPoint* pMP = *lit;
    //         if (!pMP)
    //             continue;

    //         g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
    //         vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

    //         unsigned long id = pMP->mnId+iniMPid+1;
    //         vPoint->setId(id);
    //         vPoint->setMarginalized(true);
    //         optimizer.addVertex(vPoint);

    //         const map<KeyFrame *, size_t> observations = pMP->GetObservations();

    //         // Create visual constraints
    //         for(map<KeyFrame *, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    //         {
    //             KeyFrame* pKFi = mit->first;

    //             if (!pKFi)
    //                 continue;

    //             if ((pKFi->mnBALocalForKF!=pCurrKF->mnId) && (pKFi->mnBAFixedForKF!=pCurrKF->mnId))
    //                 continue;

    //             if (pKFi->mnId>maxKFid){
    //                 Verbose::PrintMess("ID greater than current KF is", Verbose::VERBOSITY_NORMAL);
    //                 continue;
    //             }


    //             if(optimizer.vertex(id)==NULL || optimizer.vertex(pKFi->mnId)==NULL)
    //                 continue;

    //             if(!pKFi->isBad())
    //             {
    //                 const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

    //                 if(pKFi->mvuRight[mit->second]<0) // Monocular observation
    //                 {
    //                     Eigen::Matrix<double,2,1> obs;
    //                     obs << kpUn.pt.x, kpUn.pt.y;

    //                     EdgeMono* e = new EdgeMono();
    //                     e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
    //                     e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
    //                     e->setMeasurement(obs);
    //                     const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
    //                     e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

    //                     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    //                     e->setRobustKernel(rk);
    //                     rk->setDelta(thHuberMono);
    //                     optimizer.addEdge(e);
    //                     vpEdgesMono.push_back(e);
    //                     vpEdgeKFMono.push_back(pKFi);
    //                     vpMapPointEdgeMono.push_back(pMP);
    //                 }
    //                 else // stereo observation
    //                 {
    //                     const float kp_ur = pKFi->mvuRight[mit->second];
    //                     Eigen::Matrix<double,3,1> obs;
    //                     obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

    //                     EdgeStereo* e = new EdgeStereo();

    //                     e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
    //                     e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
    //                     e->setMeasurement(obs);
    //                     const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
    //                     e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

    //                     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    //                     e->setRobustKernel(rk);
    //                     rk->setDelta(thHuberStereo);

    //                     optimizer.addEdge(e);
    //                     vpEdgesStereo.push_back(e);
    //                     vpEdgeKFStereo.push_back(pKFi);
    //                     vpMapPointEdgeStereo.push_back(pMP);
    //                 }
    //             }
    //         }
    //     }

    //     if(pbStopFlag)
    //         if(*pbStopFlag)
    //             return;
    //     optimizer.initializeOptimization();
    //     optimizer.optimize(3);
    //     if(pbStopFlag)
    //         if(!*pbStopFlag)
    //             optimizer.optimize(5);

    //     optimizer.setForceStopFlag(pbStopFlag);

    //     vector<pair<KeyFrame*,MapPoint*> > vToErase;
    //     vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    //     // Check inlier observations
    //     // Mono
    //     for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    //     {
    //         EdgeMono* e = vpEdgesMono[i];
    //         MapPoint* pMP = vpMapPointEdgeMono[i];

    //         if(pMP->isBad())
    //             continue;

    //         if(e->chi2()>chi2Mono2)
    //         {
    //             KeyFrame* pKFi = vpEdgeKFMono[i];
    //             vToErase.push_back(make_pair(pKFi,pMP));
    //         }
    //     }


    //     // Stereo
    //     for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    //     {
    //         EdgeStereo* e = vpEdgesStereo[i];
    //         MapPoint* pMP = vpMapPointEdgeStereo[i];

    //         if(pMP->isBad())
    //             continue;

    //         if(e->chi2()>chi2Stereo2)
    //         {
    //             KeyFrame* pKFi = vpEdgeKFStereo[i];
    //             vToErase.push_back(make_pair(pKFi,pMP));
    //         }
    //     }

    //     // Get Map Mutex and erase outliers
    //     unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    //     if(!vToErase.empty())
    //     {
    //         for(size_t i=0;i<vToErase.size();i++)
    //         {
    //             KeyFrame* pKFi = vToErase[i].first;
    //             MapPoint* pMPi = vToErase[i].second;
    //             pKFi->EraseMapPointMatch(pMPi);
    //             pMPi->EraseObservation(pKFi);
    //         }
    //     }


    //     // Recover optimized data
    //     //Keyframes
    //     for(int i=0; i<N; i++)
    //     {
    //         KeyFrame* pKFi = vpOptimizableKFs[i];

    //         VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
    //         cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
    //         pKFi->SetPose(Tcw);

    //         cv::Mat Tiw=pKFi->GetPose();
    //         cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
    //         cv::Mat tiw = Tiw.rowRange(0,3).col(3);
    //         g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
    //         corrPoses[pKFi] = g2oSiw;


    //         if(pKFi->bImu)
    //         {
    //             VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
    //             pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
    //             VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
    //             VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
    //             Vector6d b;
    //             b << VG->estimate(), VA->estimate();
    //             pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
    //         }
    //     }

    //     for(int i=0; i<Ncov; i++)
    //     {
    //         KeyFrame* pKFi = vpOptimizableCovKFs[i];

    //         VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
    //         cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
    //         pKFi->SetPose(Tcw);

    //         cv::Mat Tiw=pKFi->GetPose();
    //         cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
    //         cv::Mat tiw = Tiw.rowRange(0,3).col(3);
    //         g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
    //         corrPoses[pKFi] = g2oSiw;

    //         if(pKFi->bImu)
    //         {
    //             VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
    //             pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
    //             VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
    //             VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
    //             Vector6d b;
    //             b << VG->estimate(), VA->estimate();
    //             pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
    //         }
    //     }

    //     //Points
    //     for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    //     {
    //         MapPoint* pMP = *lit;
    //         g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
    //         pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
    //         pMP->UpdateNormalAndDepth();
    //     }

    //     pMap->IncreaseChangeIndex();
    // }

    // void Optimizer::LocalBundleAdjustment(KeyFrame* pMainKF,vector<KeyFrame*> vpAdjustKF, vector<KeyFrame*> vpFixedKF, bool *pbStopFlag)
    // {
    //     bool bShowImages = false;

    //     vector<MapPoint*> vpMPs;

    //     g2o::SparseOptimizer optimizer;
    //     g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    //     linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    //     g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    //     g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    //     optimizer.setAlgorithm(solver);

    //     optimizer.setVerbose(false);

    //     if(pbStopFlag)
    //         optimizer.setForceStopFlag(pbStopFlag);

    //     long unsigned int maxKFid = 0;
    //     set<KeyFrame*> spKeyFrameBA;

    //     Map* pCurrentMap = pMainKF->GetMap();

    //     //set<MapPoint*> sNumObsMP;

    //     // Set fixed KeyFrame vertices
    //     for(KeyFrame* pKFi : vpFixedKF)
    //     {
    //         if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
    //         {
    //             Verbose::PrintMess("ERROR LBA: KF is bad or is not in the current map", Verbose::VERBOSITY_NORMAL);
    //             continue;
    //         }

    //         pKFi->mnBALocalForMerge = pMainKF->mnId;

    //         g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    //         vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
    //         vSE3->setId(pKFi->mnId);
    //         vSE3->setFixed(true);
    //         optimizer.addVertex(vSE3);
    //         if(pKFi->mnId>maxKFid)
    //             maxKFid=pKFi->mnId;

    //         set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
    //         for(MapPoint* pMPi : spViewMPs)
    //         {
    //             if(pMPi)
    //                 if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)

    //                     if(pMPi->mnBALocalForMerge!=pMainKF->mnId)
    //                     {
    //                         vpMPs.push_back(pMPi);
    //                         pMPi->mnBALocalForMerge=pMainKF->mnId;
    //                     }
    //                     /*if(sNumObsMP.find(pMPi) == sNumObsMP.end())
    //                     {
    //                         sNumObsMP.insert(pMPi);
    //                     }
    //                     else
    //                     {
    //                         if(pMPi->mnBALocalForMerge!=pMainKF->mnId)
    //                         {
    //                             vpMPs.push_back(pMPi);
    //                             pMPi->mnBALocalForMerge=pMainKF->mnId;
    //                         }
    //                     }*/
    //         }

    //         spKeyFrameBA.insert(pKFi);
    //     }

    //     //cout << "End to load Fixed KFs" << endl;

    //     // Set non fixed Keyframe vertices
    //     set<KeyFrame*> spAdjustKF(vpAdjustKF.begin(), vpAdjustKF.end());
    //     for(KeyFrame* pKFi : vpAdjustKF)
    //     {
    //         if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
    //             continue;

    //         pKFi->mnBALocalForKF = pMainKF->mnId;

    //         g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    //         vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
    //         vSE3->setId(pKFi->mnId);
    //         optimizer.addVertex(vSE3);
    //         if(pKFi->mnId>maxKFid)
    //             maxKFid=pKFi->mnId;

    //         set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
    //         for(MapPoint* pMPi : spViewMPs)
    //         {
    //             if(pMPi)
    //             {
    //                 if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)
    //                 {
    //                     /*if(sNumObsMP.find(pMPi) == sNumObsMP.end())
    //                     {
    //                         sNumObsMP.insert(pMPi);
    //                     }*/
    //                     if(pMPi->mnBALocalForMerge != pMainKF->mnId)
    //                     {
    //                         vpMPs.push_back(pMPi);
    //                         pMPi->mnBALocalForMerge = pMainKF->mnId;
    //                     }
    //                 }
    //             }
    //         }

    //         spKeyFrameBA.insert(pKFi);
    //     }

    //     //Verbose::PrintMess("LBA: There are " + to_string(vpMPs.size()) + " MPs to optimize", Verbose::VERBOSITY_NORMAL);

    //     //cout << "End to load KFs for position adjust" << endl;

    //     const int nExpectedSize = (vpAdjustKF.size()+vpFixedKF.size())*vpMPs.size();

    //     vector<ORB_SLAM2::EdgeSE3ProjectXYZ*> vpEdgesMono;
    //     vpEdgesMono.reserve(nExpectedSize);

    //     vector<KeyFrame*> vpEdgeKFMono;
    //     vpEdgeKFMono.reserve(nExpectedSize);

    //     vector<MapPoint*> vpMapPointEdgeMono;
    //     vpMapPointEdgeMono.reserve(nExpectedSize);

    //     vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    //     vpEdgesStereo.reserve(nExpectedSize);

    //     vector<KeyFrame*> vpEdgeKFStereo;
    //     vpEdgeKFStereo.reserve(nExpectedSize);

    //     vector<MapPoint*> vpMapPointEdgeStereo;
    //     vpMapPointEdgeStereo.reserve(nExpectedSize);

    //     const float thHuber2D = sqrt(5.99);
    //     const float thHuber3D = sqrt(7.815);

    //     // Set MapPoint vertices
    //     map<KeyFrame*, int> mpObsKFs;
    //     map<KeyFrame*, int> mpObsFinalKFs;
    //     map<MapPoint*, int> mpObsMPs;
    //     for(unsigned int i=0; i < vpMPs.size(); ++i)
    //     {
    //         MapPoint* pMPi = vpMPs[i];
    //         if(pMPi->isBad())
    //             continue;

    //         g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
    //         vPoint->setEstimate(Converter::toVector3d(pMPi->GetWorldPos()));
    //         const int id = pMPi->mnId+maxKFid+1;
    //         vPoint->setId(id);
    //         vPoint->setMarginalized(true);
    //         optimizer.addVertex(vPoint);


    //         const map<KeyFrame *, size_t> observations = pMPi->GetObservations();
    //         int nEdges = 0;
    //         //SET EDGES
    //         for(map<KeyFrame *, size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
    //         {
    //             //cout << "--KF view init" << endl;

    //             KeyFrame* pKF = mit->first;
    //             if(pKF->isBad() || pKF->mnId>maxKFid || pKF->mnBALocalForMerge != pMainKF->mnId || !pKF->GetMapPoint(mit->second))
    //                 continue;

    //             //cout << "-- KF view exists" << endl;
    //             nEdges++;

    //             const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];
    //             //cout << "-- KeyPoint loads" << endl;

    //             if(pKF->mvuRight[mit->second]<0) //Monocular
    //             {
    //                 mpObsMPs[pMPi]++;
    //                 Eigen::Matrix<double,2,1> obs;
    //                 obs << kpUn.pt.x, kpUn.pt.y;

    //                 ORB_SLAM2::EdgeSE3ProjectXYZ* e = new ORB_SLAM2::EdgeSE3ProjectXYZ();


    //                 e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
    //                 e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
    //                 e->setMeasurement(obs);
    //                 const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
    //                 e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
    //                 //cout << "-- Sigma loads" << endl;

    //                 g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    //                 e->setRobustKernel(rk);
    //                 rk->setDelta(thHuber2D);

    //                 e->pCamera = pKF->mpCamera;
    //                 //cout << "-- Calibration loads" << endl;

    //                 optimizer.addEdge(e);
    //                 //cout << "-- Edge added" << endl;

    //                 vpEdgesMono.push_back(e);
    //                 vpEdgeKFMono.push_back(pKF);
    //                 vpMapPointEdgeMono.push_back(pMPi);
    //                 //cout << "-- Added to vector" << endl;

    //                 mpObsKFs[pKF]++;
    //             }
    //             else // RGBD or Stereo
    //             {
    //                 mpObsMPs[pMPi]+=2;
    //                 Eigen::Matrix<double,3,1> obs;
    //                 const float kp_ur = pKF->mvuRight[mit->second];
    //                 obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

    //                 g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

    //                 e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
    //                 e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
    //                 e->setMeasurement(obs);
    //                 const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
    //                 Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
    //                 e->setInformation(Info);

    //                 g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    //                 e->setRobustKernel(rk);
    //                 rk->setDelta(thHuber3D);

    //                 e->fx = pKF->fx;
    //                 e->fy = pKF->fy;
    //                 e->cx = pKF->cx;
    //                 e->cy = pKF->cy;
    //                 e->bf = pKF->mbf;

    //                 optimizer.addEdge(e);

    //                 vpEdgesStereo.push_back(e);
    //                 vpEdgeKFStereo.push_back(pKF);
    //                 vpMapPointEdgeStereo.push_back(pMPi);

    //                 mpObsKFs[pKF]++;
    //             }
    //             //cout << "-- End to load point" << endl;
    //         }
    //     }
    //     //Verbose::PrintMess("LBA: number total of edged -> " + to_string(vpEdgeKFMono.size() + vpEdgeKFStereo.size()), Verbose::VERBOSITY_NORMAL);

    //     map<int, int> mStatsObs;
    //     for(map<MapPoint*, int>::iterator it = mpObsMPs.begin(); it != mpObsMPs.end(); ++it)
    //     {
    //         MapPoint* pMPi = it->first;
    //         int numObs = it->second;

    //         mStatsObs[numObs]++;
    //         /*if(numObs < 5)
    //         {
    //             cout << "LBA: MP " << pMPi->mnId << " has " << numObs << " observations" << endl;
    //         }*/
    //     }

    //     /*for(map<int, int>::iterator it = mStatsObs.begin(); it != mStatsObs.end(); ++it)
    //     {
    //         cout << "LBA: There are " << it->second << " MPs with " << it->first << " observations" << endl;
    //     }*/

    //     //cout << "End to load MPs" << endl;

    //     if(pbStopFlag)
    //         if(*pbStopFlag)
    //             return;

    //     optimizer.initializeOptimization();
    //     optimizer.optimize(5);

    //     //cout << "End the first optimization" << endl;

    //     bool bDoMore= true;

    //     if(pbStopFlag)
    //         if(*pbStopFlag)
    //             bDoMore = false;

    //     map<unsigned long int, int> mWrongObsKF;
    //     if(bDoMore)
    //     {

    //         // Check inlier observations
    //         int badMonoMP = 0, badStereoMP = 0;
    //         for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    //         {
    //             ORB_SLAM2::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
    //             MapPoint* pMP = vpMapPointEdgeMono[i];

    //             if(pMP->isBad())
    //                 continue;

    //             if(e->chi2()>5.991 || !e->isDepthPositive())
    //             {
    //                 e->setLevel(1);
    //                 badMonoMP++;
    //             }

    //             e->setRobustKernel(0);
    //         }

    //         for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    //         {
    //             g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
    //             MapPoint* pMP = vpMapPointEdgeStereo[i];

    //             if(pMP->isBad())
    //                 continue;

    //             if(e->chi2()>7.815 || !e->isDepthPositive())
    //             {
    //                 e->setLevel(1);
    //                 badStereoMP++;
    //             }

    //             e->setRobustKernel(0);
    //         }
    //         Verbose::PrintMess("LBA: First optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " sterero bad edges", Verbose::VERBOSITY_DEBUG);

    //     // Optimize again without the outliers

    //     optimizer.initializeOptimization(0);
    //     optimizer.optimize(10);

    //     //cout << "End the second optimization (without outliers)" << endl;
    //     }

    //     vector<pair<KeyFrame*,MapPoint*> > vToErase;
    //     vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    //     set<MapPoint*> spErasedMPs;
    //     set<KeyFrame*> spErasedKFs;

    //     // Check inlier observations
    //     int badMonoMP = 0, badStereoMP = 0;
    //     for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    //     {
    //         ORB_SLAM2::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
    //         MapPoint* pMP = vpMapPointEdgeMono[i];

    //         if(pMP->isBad())
    //             continue;

    //         if(e->chi2()>5.991 || !e->isDepthPositive())
    //         {
    //             KeyFrame* pKFi = vpEdgeKFMono[i];
    //             vToErase.push_back(make_pair(pKFi,pMP));
    //             mWrongObsKF[pKFi->mnId]++;
    //             badMonoMP++;

    //             spErasedMPs.insert(pMP);
    //             spErasedKFs.insert(pKFi);
    //         }
    //     }

    //     for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    //     {
    //         g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
    //         MapPoint* pMP = vpMapPointEdgeStereo[i];

    //         if(pMP->isBad())
    //             continue;

    //         if(e->chi2()>7.815 || !e->isDepthPositive())
    //         {
    //             KeyFrame* pKFi = vpEdgeKFStereo[i];
    //             vToErase.push_back(make_pair(pKFi,pMP));
    //             mWrongObsKF[pKFi->mnId]++;
    //             badStereoMP++;

    //             spErasedMPs.insert(pMP);
    //             spErasedKFs.insert(pKFi);
    //         }
    //     }
    //     Verbose::PrintMess("LBA: Second optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " sterero bad edges", Verbose::VERBOSITY_DEBUG);

    //     // Get Map Mutex
    //     unique_lock<mutex> lock(pMainKF->GetMap()->mMutexMapUpdate);

    //     if(!vToErase.empty())
    //     {
    //         map<KeyFrame*, int> mpMPs_in_KF;
    //         for(KeyFrame* pKFi : spErasedKFs)
    //         {
    //             int num_MPs = pKFi->GetMapPoints().size();
    //             mpMPs_in_KF[pKFi] = num_MPs;
    //         }

    //         Verbose::PrintMess("LBA: There are " + to_string(vToErase.size()) + " observations whose will be deleted from the map", Verbose::VERBOSITY_DEBUG);
    //         for(size_t i=0;i<vToErase.size();i++)
    //         {
    //             KeyFrame* pKFi = vToErase[i].first;
    //             MapPoint* pMPi = vToErase[i].second;
    //             pKFi->EraseMapPointMatch(pMPi);
    //             pMPi->EraseObservation(pKFi);
    //         }

    //         Verbose::PrintMess("LBA: " + to_string(spErasedMPs.size()) + " MPs had deleted observations", Verbose::VERBOSITY_DEBUG);
    //         // Verbose::PrintMess("LBA: Current map is " + to_string(pMainKF->GetMap()->GetId()), Verbose::VERBOSITY_DEBUG);
    //         int numErasedMP = 0;
    //         for(MapPoint* pMPi : spErasedMPs)
    //         {
    //             if(pMPi->isBad())
    //             {
    //                 // Verbose::PrintMess("LBA: MP " + to_string(pMPi->mnId) + " has lost almost all the observations, its origin map is " + to_string(pMPi->mnOriginMapId), Verbose::VERBOSITY_DEBUG);
    //                 numErasedMP++;
    //             }
    //         }
    //         Verbose::PrintMess("LBA: " + to_string(numErasedMP) + " MPs had deleted from the map", Verbose::VERBOSITY_DEBUG);

    //         for(KeyFrame* pKFi : spErasedKFs)
    //         {
    //             int num_MPs = pKFi->GetMapPoints().size();
    //             int num_init_MPs = mpMPs_in_KF[pKFi];
    //             Verbose::PrintMess("LBA: Initially KF " + to_string(pKFi->mnId) + " had " + to_string(num_init_MPs) + ", at the end has " + to_string(num_MPs), Verbose::VERBOSITY_DEBUG);
    //         }
    //     }
    //     for(unsigned int i=0; i < vpMPs.size(); ++i)
    //     {
    //         MapPoint* pMPi = vpMPs[i];
    //         if(pMPi->isBad())
    //             continue;

    //         const map<KeyFrame *, size_t> observations = pMPi->GetObservations();
    //         for(map<KeyFrame *, size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
    //         {
    //             //cout << "--KF view init" << endl;

    //             KeyFrame* pKF = mit->first;
    //             if(pKF->isBad() || pKF->mnId>maxKFid || pKF->mnBALocalForKF != pMainKF->mnId || !pKF->GetMapPoint(mit->second))
    //                 continue;

    //             const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];
    //             //cout << "-- KeyPoint loads" << endl;

    //             if(pKF->mvuRight[mit->second]<0) //Monocular
    //             {
    //                 mpObsFinalKFs[pKF]++;
    //             }
    //             else // RGBD or Stereo
    //             {

    //                 mpObsFinalKFs[pKF]++;
    //             }
    //             //cout << "-- End to load point" << endl;
    //         }
    //     }

    //     //cout << "End to erase observations" << endl;

    //     // Recover optimized data

    //     //Keyframes
    //     for(KeyFrame* pKFi : vpAdjustKF)
    //     {
    //         if(pKFi->isBad())
    //             continue;

    //         g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
    //         g2o::SE3Quat SE3quat = vSE3->estimate();
    //         cv::Mat Tiw = Converter::toCvMat(SE3quat);
    //         cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
    //         cv::Vec3d trasl = Tco_cn.rowRange(0,3).col(3);
    //         double dist = cv::norm(trasl);

    //         int numMonoBadPoints = 0, numMonoOptPoints = 0;
    //         int numStereoBadPoints = 0, numStereoOptPoints = 0;
    //         vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;
    //         vector<MapPoint*> vpMonoMPsBad, vpStereoMPsBad;

    //         for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    //         {
    //             ORB_SLAM2::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
    //             MapPoint* pMP = vpMapPointEdgeMono[i];
    //             KeyFrame* pKFedge = vpEdgeKFMono[i];

    //             if(pKFi != pKFedge)
    //             {
    //                 continue;
    //             }

    //             if(pMP->isBad())
    //                 continue;

    //             if(e->chi2()>5.991 || !e->isDepthPositive())
    //             {
    //                 numMonoBadPoints++;
    //                 vpMonoMPsBad.push_back(pMP);

    //             }
    //             else
    //             {
    //                 numMonoOptPoints++;
    //                 vpMonoMPsOpt.push_back(pMP);
    //             }

    //         }

    //         for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    //         {
    //             g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
    //             MapPoint* pMP = vpMapPointEdgeStereo[i];
    //             KeyFrame* pKFedge = vpEdgeKFMono[i];

    //             if(pKFi != pKFedge)
    //             {
    //                 continue;
    //             }

    //             if(pMP->isBad())
    //                 continue;

    //             if(e->chi2()>7.815 || !e->isDepthPositive())
    //             {
    //                 numStereoBadPoints++;
    //                 vpStereoMPsBad.push_back(pMP);
    //             }
    //             else
    //             {
    //                 numStereoOptPoints++;
    //                 vpStereoMPsOpt.push_back(pMP);
    //             }
    //         }


    //         if(numMonoOptPoints + numStereoOptPoints < 50)
    //         {
    //             Verbose::PrintMess("LBA ERROR: KF " + to_string(pKFi->mnId) + " has only " + to_string(numMonoOptPoints) + " monocular and " + to_string(numStereoOptPoints) + " stereo points", Verbose::VERBOSITY_DEBUG);
    //         }
    //         // if(dist > 1.0)
    //         // {
    //         //     if(bShowImages)
    //         //     {
    //         //         string strNameFile = pKFi->mNameFile;
    //         //         cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);

    //         //         cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);

    //         //         int numPointsMono = 0, numPointsStereo = 0;
    //         //         int numPointsMonoBad = 0, numPointsStereoBad = 0;
    //         //         for(int i=0; i<vpMonoMPsOpt.size(); ++i)
    //         //         {
    //         //             if(!vpMonoMPsOpt[i] || vpMonoMPsOpt[i]->isBad())
    //         //             {
    //         //                 continue;
    //         //             }
    //         //             int index = get<0>(vpMonoMPsOpt[i]->GetIndexInKeyFrame(pKFi));
    //         //             if(index < 0)
    //         //             {
    //         //                 //cout << "LBA ERROR: KF has a monocular observation which is not recognized by the MP" << endl;
    //         //                 //cout << "LBA: KF " << pKFi->mnId << " and MP " << vpMonoMPsOpt[i]->mnId << " with index " << endl;
    //         //                 continue;
    //         //             }

    //         //             //string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
    //         //             cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(255, 0, 0));
    //         //             //cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
    //         //             numPointsMono++;
    //         //         }

    //         //         for(int i=0; i<vpStereoMPsOpt.size(); ++i)
    //         //         {
    //         //             if(!vpStereoMPsOpt[i] || vpStereoMPsOpt[i]->isBad())
    //         //             {
    //         //                 continue;
    //         //             }
    //         //             int index = get<0>(vpStereoMPsOpt[i]->GetIndexInKeyFrame(pKFi));
    //         //             if(index < 0)
    //         //             {
    //         //                 //cout << "LBA: KF has a stereo observation which is not recognized by the MP" << endl;
    //         //                 //cout << "LBA: KF " << pKFi->mnId << " and MP " << vpStereoMPsOpt[i]->mnId << endl;
    //         //                 continue;
    //         //             }

    //         //             //string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
    //         //             cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(0, 255, 0));
    //         //             //cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
    //         //             numPointsStereo++;
    //         //         }

    //         //         for(int i=0; i<vpMonoMPsBad.size(); ++i)
    //         //         {
    //         //             if(!vpMonoMPsBad[i] || vpMonoMPsBad[i]->isBad())
    //         //             {
    //         //                 continue;
    //         //             }
    //         //             int index = vpMonoMPsBad[i]->GetIndexInKeyFrame(pKFi);
    //         //             if(index < 0)
    //         //             {
    //         //                 //cout << "LBA ERROR: KF has a monocular observation which is not recognized by the MP" << endl;
    //         //                 //cout << "LBA: KF " << pKFi->mnId << " and MP " << vpMonoMPsOpt[i]->mnId << " with index " << endl;
    //         //                 continue;
    //         //             }

    //         //             //string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
    //         //             cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(0, 0, 255));
    //         //             //cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
    //         //             numPointsMonoBad++;
    //         //         }
    //         //         for(int i=0; i<vpStereoMPsBad.size(); ++i)
    //         //         {
    //         //             if(!vpStereoMPsBad[i] || vpStereoMPsBad[i]->isBad())
    //         //             {
    //         //                 continue;
    //         //             }
    //         //             int index = vpStereoMPsBad[i]->GetIndexInKeyFrame(pKFi);
    //         //             if(index < 0)
    //         //             {
    //         //                 //cout << "LBA: KF has a stereo observation which is not recognized by the MP" << endl;
    //         //                 //cout << "LBA: KF " << pKFi->mnId << " and MP " << vpStereoMPsOpt[i]->mnId << endl;
    //         //                 continue;
    //         //             }

    //         //             //string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
    //         //             cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(0, 0, 255));
    //         //             //cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
    //         //             numPointsStereoBad++;
    //         //         }

    //         //         string namefile = "./test_LBA/LBA_KF" + to_string(pKFi->mnId) + "_" + to_string(numPointsMono + numPointsStereo) +"_D" + to_string(dist) +".png";
    //         //         cv::imwrite(namefile, imLeft);

    //         //         Verbose::PrintMess("--LBA in KF " + to_string(pKFi->mnId), Verbose::VERBOSITY_DEBUG);
    //         //         Verbose::PrintMess("--Distance: " + to_string(dist) + " meters", Verbose::VERBOSITY_DEBUG);
    //         //         Verbose::PrintMess("--Number of observations: " + to_string(numMonoOptPoints) + " in mono and " + to_string(numStereoOptPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
    //         //         Verbose::PrintMess("--Number of discarded observations: " + to_string(numMonoBadPoints) + " in mono and " + to_string(numStereoBadPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
    //         //         Verbose::PrintMess("--To much distance correction in LBA: It has " + to_string(mpObsKFs[pKFi]) + " observated MPs", Verbose::VERBOSITY_DEBUG);
    //         //         Verbose::PrintMess("--To much distance correction in LBA: It has " + to_string(mpObsFinalKFs[pKFi]) + " deleted observations", Verbose::VERBOSITY_DEBUG);
    //         //         Verbose::PrintMess("--------", Verbose::VERBOSITY_DEBUG);
    //         //     }
    //         // }
    //         pKFi->SetPose(Tiw);

    //     }
    //     //cout << "End to update the KeyFrames" << endl;

    //     //Points
    //     for(MapPoint* pMPi : vpMPs)
    //     {
    //         if(pMPi->isBad())
    //             continue;

    //         g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMPi->mnId+maxKFid+1));
    //         pMPi->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
    //         pMPi->UpdateNormalAndDepth();

    //     }
    //     //cout << "End to update MapPoint" << endl;
    // }

    void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd  &covInertial, bool bFixedVel, bool bGauss, float priorG, float priorA)
    {
        Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
        int its = 200; // Check number of iterations
        long unsigned int maxKFid = pMap->GetMaxKFid();
        const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        if (priorG!=0.f)
            solver->setUserLambdaInit(1e3);

        optimizer.setAlgorithm(solver);


        // Set KeyFrame vertices (fixed poses and optimizable velocities)
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId>maxKFid)
                continue;
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(true);
            optimizer.addVertex(VP);

            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+(pKFi->mnId)+1);
            if (bFixedVel)
                VV->setFixed(true);
            else
                VV->setFixed(false);

            optimizer.addVertex(VV);
        }

        // Biases
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(maxKFid*2+2);
        if (bFixedVel)
            VG->setFixed(true);
        else
            VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(maxKFid*2+3);
        if (bFixedVel)
            VA->setFixed(true);
        else
            VA->setFixed(false);

        optimizer.addVertex(VA);
        // prior acc bias
        EdgePriorAcc* epa = new EdgePriorAcc(cv::Mat::zeros(3,1,CV_32F));
        epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA;
        epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);
        EdgePriorGyro* epg = new EdgePriorGyro(cv::Mat::zeros(3,1,CV_32F));
        epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG;
        epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);

        // Gravity and scale
        VertexGDir* VGDir = new VertexGDir(Rwg);
        VGDir->setId(maxKFid*2+4);
        VGDir->setFixed(false);
        optimizer.addVertex(VGDir);
        VertexScale* VS = new VertexScale(scale);
        VS->setId(maxKFid*2+5);
        VS->setFixed(!bMono); // Fixed for stereo case
        optimizer.addVertex(VS);

        // Graph edges
        // IMU links with gravity and scale
        vector<EdgeInertialGS*> vpei;
        vpei.reserve(vpKFs.size());
        vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
        vppUsedKF.reserve(vpKFs.size());
        std::cout << "build optimization graph" << std::endl;

        for(size_t i=0;i<vpKFs.size();i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
            {
                if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                    continue;
                if(!pKFi->mpImuPreintegrated)
                    std::cout << "Not preintegrated measurement" << std::endl;

                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1);
                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);
                g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
                g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
                g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
                g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
                if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
                {
                    cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                    continue;
                }
                EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
                ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
                ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

                vpei.push_back(ei);

                vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi));
                optimizer.addEdge(ei);

            }
        }

        // Compute error for different scales
        std::set<g2o::HyperGraph::Edge*> setEdges = optimizer.edges();

        std::cout << "start optimization" << std::endl;
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(its);

        std::cout << "end optimization" << std::endl;

        scale = VS->estimate();

        // Recover optimized data
        // Biases
        VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
        VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
        Vector6d vb;
        vb << VG->estimate(), VA->estimate();
        bg << VG->estimate();
        ba << VA->estimate();
        scale = VS->estimate();


//        IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
        IMU::Bias b (0.0,0.0,0.0,vb[0],vb[1],vb[2]);
        Rwg = VGDir->estimate().Rwg;

        cv::Mat cvbg = Converter::toCvMat(bg);

        //Keyframes velocities and biases
        std::cout << "update Keyframes velocities and biases" << std::endl;

        const int N = vpKFs.size();
        for(size_t i=0; i<N; i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi && !pKFi->isBad() && pKFi->mnId>maxKFid)
                continue;

            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));
            Eigen::Vector3d Vw = VV->estimate(); // Velocity is scaled after
            pKFi->SetVelocity(Converter::toCvMat(Vw));

            if (cv::norm(pKFi->GetGyroBias()-cvbg)>0.01)
            {
                pKFi->SetNewBias(b);
                if (pKFi->mpImuPreintegrated)
                    pKFi->mpImuPreintegrated->Reintegrate();
            }
            else
                pKFi->SetNewBias(b);


        }
    }


    void Optimizer::InertialOptimization(Map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG, float priorA)
    {
        int its = 200; // Check number of iterations
        long unsigned int maxKFid = pMap->GetMaxKFid();
        const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e3);

        optimizer.setAlgorithm(solver);


        // Set KeyFrame vertices (fixed poses and optimizable velocities)
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId>maxKFid)
                continue;
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(true);
            optimizer.addVertex(VP);

            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+(pKFi->mnId)+1);
            VV->setFixed(false);

            optimizer.addVertex(VV);
        }

        // Biases
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(maxKFid*2+2);
        VG->setFixed(false);
        optimizer.addVertex(VG);

        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(maxKFid*2+3);
        VA->setFixed(false);

        optimizer.addVertex(VA);
        // prior acc bias
        EdgePriorAcc* epa = new EdgePriorAcc(cv::Mat::zeros(3,1,CV_32F));
        epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA;
        epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);
        EdgePriorGyro* epg = new EdgePriorGyro(cv::Mat::zeros(3,1,CV_32F));
        epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG;
        epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);

        // Gravity and scale
        VertexGDir* VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
        VGDir->setId(maxKFid*2+4);
        VGDir->setFixed(true);
        optimizer.addVertex(VGDir);
        VertexScale* VS = new VertexScale(1.0);
        VS->setId(maxKFid*2+5);
        VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
        optimizer.addVertex(VS);

        // Graph edges
        // IMU links with gravity and scale
        vector<EdgeInertialGS*> vpei;
        vpei.reserve(vpKFs.size());
        vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
        vppUsedKF.reserve(vpKFs.size());

        for(size_t i=0;i<vpKFs.size();i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
            {
                if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                    continue;

                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1);
                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);
                g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
                g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
                g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
                g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
                if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
                {
                    cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                    continue;
                }
                EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
                ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
                ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

                vpei.push_back(ei);

                vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi));
                optimizer.addEdge(ei);

            }
        }

        // Compute error for different scales
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(its);


        // Recover optimized data
        // Biases
        VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
        VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
        Vector6d vb;
        vb << VG->estimate(), VA->estimate();
        bg << VG->estimate();
        ba << VA->estimate();

        IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);

        cv::Mat cvbg = Converter::toCvMat(bg);

        //Keyframes velocities and biases
        const int N = vpKFs.size();
        for(size_t i=0; i<N; i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId>maxKFid)
                continue;

            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));
            Eigen::Vector3d Vw = VV->estimate();
            pKFi->SetVelocity(Converter::toCvMat(Vw));

            if (cv::norm(pKFi->GetGyroBias()-cvbg)>0.01)
            {
                pKFi->SetNewBias(b);
                if (pKFi->mpImuPreintegrated)
                    pKFi->mpImuPreintegrated->Reintegrate();
            }
            else
                pKFi->SetNewBias(b);
        }
    }

    void Optimizer::InertialOptimization(vector<KeyFrame*> vpKFs, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG, float priorA)
    {
        int its = 200; // Check number of iterations
        long unsigned int maxKFid = vpKFs[0]->GetMap()->GetMaxKFid();

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e3);

        optimizer.setAlgorithm(solver);


        // Set KeyFrame vertices (fixed poses and optimizable velocities)
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            //if(pKFi->mnId>maxKFid)
            //    continue;
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(true);
            optimizer.addVertex(VP);

            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+(pKFi->mnId)+1);
            VV->setFixed(false);

            optimizer.addVertex(VV);
        }

        // Biases
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(maxKFid*2+2);
        VG->setFixed(false);
        optimizer.addVertex(VG);

        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(maxKFid*2+3);
        VA->setFixed(false);

        optimizer.addVertex(VA);
        // prior acc bias
        EdgePriorAcc* epa = new EdgePriorAcc(cv::Mat::zeros(3,1,CV_32F));
        epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA;
        epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);
        EdgePriorGyro* epg = new EdgePriorGyro(cv::Mat::zeros(3,1,CV_32F));
        epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG;
        epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);

        // Gravity and scale
        VertexGDir* VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
        VGDir->setId(maxKFid*2+4);
        VGDir->setFixed(true);
        optimizer.addVertex(VGDir);
        VertexScale* VS = new VertexScale(1.0);
        VS->setId(maxKFid*2+5);
        VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
        optimizer.addVertex(VS);

        // Graph edges
        // IMU links with gravity and scale
        vector<EdgeInertialGS*> vpei;
        vpei.reserve(vpKFs.size());
        vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
        vppUsedKF.reserve(vpKFs.size());

        for(size_t i=0;i<vpKFs.size();i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
            {
                if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                    continue;

                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1);
                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);
                g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
                g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
                g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
                g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
                if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
                {
                    cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                    continue;
                }
                EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
                ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
                ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

                vpei.push_back(ei);

                vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi));
                optimizer.addEdge(ei);

            }
        }

        // Compute error for different scales
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(its);


        // Recover optimized data
        // Biases
        VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
        VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
        Vector6d vb;
        vb << VG->estimate(), VA->estimate();
        bg << VG->estimate();
        ba << VA->estimate();

        IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);

        cv::Mat cvbg = Converter::toCvMat(bg);

        //Keyframes velocities and biases
        const int N = vpKFs.size();
        for(size_t i=0; i<N; i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId>maxKFid)
                continue;

            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));
            Eigen::Vector3d Vw = VV->estimate();
            pKFi->SetVelocity(Converter::toCvMat(Vw));

            if (cv::norm(pKFi->GetGyroBias()-cvbg)>0.01)
            {
                pKFi->SetNewBias(b);
                if (pKFi->mpImuPreintegrated)
                    pKFi->mpImuPreintegrated->Reintegrate();
            }
            else
                pKFi->SetNewBias(b);
        }
    }


    void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale)
    {
        int its = 10;
        long unsigned int maxKFid = pMap->GetMaxKFid();
        const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Set KeyFrame vertices (all variables are fixed)
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId>maxKFid)
                continue;
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(true);
            optimizer.addVertex(VP);

            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+1+(pKFi->mnId));
            VV->setFixed(true);
            optimizer.addVertex(VV);

            // Vertex of fixed biases
            VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
            VG->setId(2*(maxKFid+1)+(pKFi->mnId));
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(vpKFs.front());
            VA->setId(3*(maxKFid+1)+(pKFi->mnId));
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }

        // Gravity and scale
        VertexGDir* VGDir = new VertexGDir(Rwg);
        VGDir->setId(4*(maxKFid+1));
        VGDir->setFixed(false);
        optimizer.addVertex(VGDir);
        VertexScale* VS = new VertexScale(scale);
        VS->setId(4*(maxKFid+1)+1);
        VS->setFixed(false);
        optimizer.addVertex(VS);

        // Graph edges
        for(size_t i=0;i<vpKFs.size();i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
            {
                if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                    continue;

                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex((maxKFid+1)+pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex((maxKFid+1)+pKFi->mnId);
                g2o::HyperGraph::Vertex* VG = optimizer.vertex(2*(maxKFid+1)+pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VA = optimizer.vertex(3*(maxKFid+1)+pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(4*(maxKFid+1));
                g2o::HyperGraph::Vertex* VS = optimizer.vertex(4*(maxKFid+1)+1);
                if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
                {
                    Verbose::PrintMess("Error" + to_string(VP1->id()) + ", " + to_string(VV1->id()) + ", " + to_string(VG->id()) + ", " + to_string(VA->id()) + ", " + to_string(VP2->id()) + ", " + to_string(VV2->id()) +  ", " + to_string(VGDir->id()) + ", " + to_string(VS->id()), Verbose::VERBOSITY_NORMAL);

                    continue;
                }
                EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
                ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
                ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

                optimizer.addEdge(ei);
            }
        }

        // Compute error for different scales
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(its);

        // Recover optimized data
        scale = VS->estimate();
        Rwg = VGDir->estimate().Rwg;
    }

} //namespace ORB_SLAM
