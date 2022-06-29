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


#include "Tracking.h"
#include "ORBmatcher.h"
#include <include/CameraModels/Pinhole.h>
// #include <include/CameraModels/KannalaBrandt8.h>

#include "Optimizer.h"
#include "PnPsolver.h"
#include "G2oTypes.h"


using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {

    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
                       KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor) :
            mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
            mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys),
            mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0),
            mbMapUpdated(false), time_recently_lost(5.0), t0IMU(numeric_limits<double>::infinity()) {
        // Load camera parameters from settings file
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        // Only consider Pinhole for now
        if(fSettings["Camera.type"] == "KannalaBrandt8")
            throw logic_error("Not implemented!");

        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        vector<float> vCamCalib{fx,fy,cx,cy};
        mpCamera = new Pinhole(vCamCalib);

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        int img_width = fSettings["Camera.width"];
        int img_height = fSettings["Camera.height"];

        cout << "img_width = " << img_width << endl;
        cout << "img_height = " << img_height << endl;

        // Is it used somewhere else?
        initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3, 3), mK, Size(img_width, img_height), CV_32F,
                                mUndistX, mUndistY);

        cout << "mUndistX size = " << mUndistX.size << endl;
        cout << "mUndistY size = " << mUndistY.size << endl;

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

// Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

// Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::MONOCULAR)
            mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        if (sensor == System::STEREO || sensor == System::RGBD || sensor == System::IMU_RGBD) {
            mThDepth = mbf * (float) fSettings["ThDepth"] / fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }

        if (sensor == System::RGBD || sensor == System::IMU_RGBD) {
            mDepthMapFactor = fSettings["DepthMapFactor"];
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }

        if(sensor==System::IMU_RGBD)
        {
            cv::Mat Tbc;
            fSettings["Tbc"] >> Tbc;

            cout << "Left camera to Imu Transform (Tbc): " << endl << Tbc << endl;

            float Nafreq, Ngfreq, Ng, Na, Ngw, Naw;
            fSettings["IMU.AccFrequency"] >> Nafreq;
            fSettings["IMU.GyroFrequency"] >> Ngfreq;
            fSettings["IMU.NoiseGyro"] >> Ng;
            fSettings["IMU.NoiseAcc"] >> Na;
            fSettings["IMU.GyroWalk"] >> Ngw;
            fSettings["IMU.AccWalk"] >> Naw;

            const float asf = sqrt(Nafreq);
            const float gsf = sqrt(Ngfreq);
            cout << endl;
            cout << "IMU accelerometer frequency: " << Nafreq << " Hz" << endl;
            cout << "IMU gyro frequency: " << Ngfreq << " Hz" << endl;
            cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
            cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
            cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
            cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

            mpImuCalib = new IMU::Calib(Tbc,Ng*gsf,Na*asf,Ngw/gsf,Naw/asf);

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);

            mnFramesToResetIMU = mMaxFrames;
        }

        // Plane and Line related settings
        mfDThRef = fSettings["Plane.AssociationDisRef"];
        mfDThMon = fSettings["Plane.AssociationDisMon"];
        mfAThRef = fSettings["Plane.AssociationAngRef"];
        mfAThMon = fSettings["Plane.AssociationAngMon"];

        mfVerTh = fSettings["Plane.VerticalThreshold"];
        mfParTh = fSettings["Plane.ParallelThreshold"];

        manhattanCount = 0;
        fullManhattanCount = 0;

        fullManhattanFound = false;

//        planeViewer = make_shared<PlaneViewer>();
    }


    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetSurfelMapper(SurfelMapping *pSurfelMapper) {
        mpSurfelMapper = pSurfelMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer *pViewer) {
        mpViewer = pViewer;
    }


    cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp) {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;

        if (mImGray.channels() == 3) {
            if (mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
            }
        } else if (mImGray.channels() == 4) {
            if (mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
            }
        }

        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary,
                              mK, mDistCoef, mbf, mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }


    cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp) {
        mImRGB = imRGB;
        mImGray = imRGB;
        mImDepth = imD;

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        if (mSensor == System::RGBD)
            mCurrentFrame = Frame(mImRGB, mImGray, mImDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK,
                                mDistCoef, mbf, mThDepth, mDepthMapFactor, mpCamera);
        else if (mSensor == System::IMU_RGBD)
            mCurrentFrame = Frame(mImRGB, mImGray, mImDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK,
                                mDistCoef, mbf, mThDepth, mDepthMapFactor, mpCamera, &mLastFrame, *mpImuCalib);
        else
            throw logic_error("Not implemented!");

        if (mDepthMapFactor != 1 || mImDepth.type() != CV_32F) {
            mImDepth.convertTo(mImDepth, CV_32F, mDepthMapFactor);
        }

        Track();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t12= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        std::ofstream fileWrite12("Track.dat", std::ios::binary | std::ios::app);
        fileWrite12.write((char*) &t12, sizeof(double));
        fileWrite12.close();

        cout << "Track time: " << t12 << endl;

        return mCurrentFrame.mTcw.clone();
    }


    cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp) {
        mImGray = im;

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        else
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                                  mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
    {
        unique_lock<mutex> lock(mMutexImuQueue);
        mlQueueImuData.push_back(imuMeasurement);
    }

    void Tracking::PreintegrateIMU()
    {
        //cout << "start preintegration" << endl;

        if(!mCurrentFrame.mpPrevFrame)
        {
            Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
            mCurrentFrame.setIntegrated();
            return;
        }

        // cout << "start loop. Total meas:" << mlQueueImuData.size() << endl;

        mvImuFromLastFrame.clear();
        mvImuFromLastFrame.reserve(mlQueueImuData.size());
        if(mlQueueImuData.size() == 0)
        {
            Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
            mCurrentFrame.setIntegrated();
            return;
        }

        // loop IMU records from front to last
        while(true)
        {
            bool bSleep = false;
            {
                unique_lock<mutex> lock(mMutexImuQueue);
                if(!mlQueueImuData.empty())
                {
                    IMU::Point* m = &mlQueueImuData.front();
                    cout.precision(17);
                    // if the timestamp of the current IMU record is less than the previous-0.001l
                    if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-0.001l)
                    {
                        // discard
                        mlQueueImuData.pop_front();
                    }
                    // if the timestamp of the current IMU record is less than the current-0.001l
                    else if(m->t<mCurrentFrame.mTimeStamp-0.001l)
                    {
                        mvImuFromLastFrame.push_back(*m);
                        mlQueueImuData.pop_front();
                    }
                    else
                    {
                        // IMU record from timestamp current--0.001l will be reused at next frame
                        mvImuFromLastFrame.push_back(*m);
                        break;
                    }
                }
                else
                {
                    break;
                    bSleep = true;
                }
            }
            if(bSleep)
                usleep(500);
        }


        const int n = mvImuFromLastFrame.size()-1;
        IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);

        for(int i=0; i<n; i++)
        {
            float tstep;
            cv::Point3f acc, angVel;
            if((i==0) && (i<(n-1)))
            {
                float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
                float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
                acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                        (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
                angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                        (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
                tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            }
            else if(i<(n-1))
            {
                acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
                angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
                tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            }
            else if((i>0) && (i==(n-1)))
            {
                float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
                float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
                acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                        (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
                angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                        (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
                tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
            }
            else if((i==0) && (i==(n-1)))
            {
                acc = mvImuFromLastFrame[i].a;
                angVel = mvImuFromLastFrame[i].w;
                tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
            }

            if (!mpImuPreintegratedFromLastKF)
                cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
            mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
            pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
        }

        mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
        mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

        mCurrentFrame.setIntegrated();

        Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
    }


    bool Tracking::PredictStateIMU()
    {
        if(!mCurrentFrame.mpPrevFrame)
        {
            Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
            return false;
        }

        if(mbMapUpdated && mpLastKeyFrame)
        {
            const cv::Mat twb1 = mpLastKeyFrame->GetImuPosition();
            const cv::Mat Rwb1 = mpLastKeyFrame->GetImuRotation();
            const cv::Mat Vwb1 = mpLastKeyFrame->GetVelocity();

            const cv::Mat Gz = (cv::Mat_<float>(3,1) << 0,0,-IMU::GRAVITY_VALUE);
            const float t12 = mpImuPreintegratedFromLastKF->dT;

            cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1*mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
            cv::Mat twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
            cv::Mat Vwb2 = Vwb1 + t12*Gz + Rwb1*mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
            mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2); // update mTcw
            mCurrentFrame.mPredRwb = Rwb2.clone();
            mCurrentFrame.mPredtwb = twb2.clone();
            mCurrentFrame.mPredVwb = Vwb2.clone();
            mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
            mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
            return true;
        }
        else if(!mbMapUpdated)
        {
            const cv::Mat twb1 = mLastFrame.GetImuPosition();
            const cv::Mat Rwb1 = mLastFrame.GetImuRotation();
            const cv::Mat Vwb1 = mLastFrame.mVw;
            const cv::Mat Gz = (cv::Mat_<float>(3,1) << 0,0,-IMU::GRAVITY_VALUE);
            const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

            cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
            cv::Mat twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
            cv::Mat Vwb2 = Vwb1 + t12*Gz + Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

            mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);
            mCurrentFrame.mPredRwb = Rwb2.clone();
            mCurrentFrame.mPredtwb = twb2.clone();
            mCurrentFrame.mPredVwb = Vwb2.clone();
            mCurrentFrame.mImuBias =mLastFrame.mImuBias;
            mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
            return true;
        }
        // else
        //     cout << "not IMU prediction!!" << endl;

        return false;
    }


    void Tracking::ComputeGyroBias(const vector<Frame*> &vpFs, float &bwx,  float &bwy, float &bwz)
    {
        const int N = vpFs.size();
        vector<float> vbx;
        vbx.reserve(N);
        vector<float> vby;
        vby.reserve(N);
        vector<float> vbz;
        vbz.reserve(N);

        cv::Mat H = cv::Mat::zeros(3,3,CV_32F);
        cv::Mat grad  = cv::Mat::zeros(3,1,CV_32F);
        for(int i=1;i<N;i++)
        {
            Frame* pF2 = vpFs[i];
            Frame* pF1 = vpFs[i-1];
            cv::Mat VisionR = pF1->GetImuRotation().t()*pF2->GetImuRotation();
            cv::Mat JRg = pF2->mpImuPreintegratedFrame->JRg;
            cv::Mat E = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaRotation().t()*VisionR;
            cv::Mat e = IMU::LogSO3(E);
            assert(fabs(pF2->mTimeStamp-pF1->mTimeStamp-pF2->mpImuPreintegratedFrame->dT)<0.01);

            cv::Mat J = -IMU::InverseRightJacobianSO3(e)*E.t()*JRg;
            grad += J.t()*e;
            H += J.t()*J;
        }

        cv::Mat bg = -H.inv(cv::DECOMP_SVD)*grad;
        bwx = bg.at<float>(0);
        bwy = bg.at<float>(1);
        bwz = bg.at<float>(2);

        for(int i=1;i<N;i++)
        {
            Frame* pF = vpFs[i];
            pF->mImuBias.bwx = bwx;
            pF->mImuBias.bwy = bwy;
            pF->mImuBias.bwz = bwz;
            pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
            pF->mpImuPreintegratedFrame->Reintegrate();
        }
    }

    void Tracking::ComputeVelocitiesAccBias(const vector<Frame*> &vpFs, float &bax,  float &bay, float &baz)
    {
        const int N = vpFs.size();
        const int nVar = 3*N +3; // 3 velocities/frame + acc bias
        const int nEqs = 6*(N-1);

        cv::Mat J(nEqs,nVar,CV_32F,cv::Scalar(0));
        cv::Mat e(nEqs,1,CV_32F,cv::Scalar(0));
        cv::Mat g = (cv::Mat_<float>(3,1)<<0,0,-IMU::GRAVITY_VALUE);

        for(int i=0;i<N-1;i++)
        {
            Frame* pF2 = vpFs[i+1];
            Frame* pF1 = vpFs[i];
            cv::Mat twb1 = pF1->GetImuPosition();
            cv::Mat twb2 = pF2->GetImuPosition();
            cv::Mat Rwb1 = pF1->GetImuRotation();
            cv::Mat dP12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaPosition();
            cv::Mat dV12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaVelocity();
            cv::Mat JP12 = pF2->mpImuPreintegratedFrame->JPa;
            cv::Mat JV12 = pF2->mpImuPreintegratedFrame->JVa;
            float t12 = pF2->mpImuPreintegratedFrame->dT;
            // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
            J.rowRange(6*i,6*i+3).colRange(3*i,3*i+3) += cv::Mat::eye(3,3,CV_32F)*t12;
            J.rowRange(6*i,6*i+3).colRange(3*N,3*N+3) += Rwb1*JP12;
            e.rowRange(6*i,6*i+3) = twb2-twb1-0.5f*g*t12*t12-Rwb1*dP12;
            // Velocity v2=v1+g*t+R1*dV12
            J.rowRange(6*i+3,6*i+6).colRange(3*i,3*i+3) += -cv::Mat::eye(3,3,CV_32F);
            J.rowRange(6*i+3,6*i+6).colRange(3*(i+1),3*(i+1)+3) += cv::Mat::eye(3,3,CV_32F);
            J.rowRange(6*i+3,6*i+6).colRange(3*N,3*N+3) -= Rwb1*JV12;
            e.rowRange(6*i+3,6*i+6) = g*t12+Rwb1*dV12;
        }

        cv::Mat H = J.t()*J;
        cv::Mat B = J.t()*e;
        cv::Mat x(nVar,1,CV_32F);
        cv::solve(H,B,x);

        bax = x.at<float>(3*N);
        bay = x.at<float>(3*N+1);
        baz = x.at<float>(3*N+2);

        for(int i=0;i<N;i++)
        {
            Frame* pF = vpFs[i];
            x.rowRange(3*i,3*i+3).copyTo(pF->mVw);
            if(i>0)
            {
                pF->mImuBias.bax = bax;
                pF->mImuBias.bay = bay;
                pF->mImuBias.baz = baz;
                pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
            }
        }
    }

    // void Tracking::ResetFrameIMU()
    // {
    //     // TODO To implement...
    // }


    /**
             * Pipeline comparasion:
             * 
             * ORB-SLAM 2:
             * 
             * if mState==OK
             *  bOK = TrackReferenceKeyFrame and/or TrackWithMotionModel
             * else
             *  bOK = Relocalization
             * 
             * if bOK
             *  bOK = TrackLocalMap
             * 
             * if bOK
             *  mState = OK
             * else
             *  mState = LOST
             * 
             * ORB-SLAM 3:
             * 
             * if mState == OK
             *  bOK = TrackReferenceKeyFrame and/or TrackWithMotionModel
             * 
             *  if !bOK
             *   // meaning current frame is lost again shortly after relocalization
             *   if current frame id is smaller than the last relocalized frame id + some threshold
             *    mState = LOST
             *   // first lost long after relocalization
             *   else if #keyframe > 10
             *    mState = RECENTLY_LOST
             *   else
             *    mState = LOST
             *  else if mState == RECENTLY_LOST
             *   if mSensor == IMU_RGBD
             *    if imu is initialized
             *     bOK = true
             *     PredictStateIMU
             *    else
             *     bOK = false
             *    if still RECENTLY_LOST after some time threshold
             *     mState = LOST
             *     bOK = false
             *   else
             *    bOK = Relocalization
             *    if !bOK
             *     mState = LOST
             *  else if mState == LOST
             *   if #keyframe < 10
             *     reset
             *   else
             *    create new map // not implemented in our implementation
             * 
             * if bOK
             *   bOK = TrackLocalMap
             * 
             * if bOK
             *  mState = OK
             * else if mState == OK // !bOK && mState, only possible if fail to TrackLocalMap
             *  if mSensor == IMU_RGBD
             *   if imu is not initialized or the imu parameters is not fully optimizated
             *    reset
             *   mState = RECENTLY_LOST
             * else
             *   mState = LOST
             * 
             * ...some more minor changes related to IMU
             * 
             * Our implementation:
             * 
             * if mState == OK or mState == LOST
             *  TrackManhattanFrame
             * 
             **/

    void Tracking::Track() {

        if(mpLocalMapper->mbBadImu)
        {
            cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
            mpSystem->Reset();
            return;
        }
        
        // sanity checks
        if(mState!=NO_IMAGES_YET)
        {   
            if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
            {
                cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
                unique_lock<mutex> lock(mMutexImuQueue);
                mlQueueImuData.clear();
                // CreateMapInAtlas();
                // return;

                // should not happen
                throw std::runtime_error("ERROR: Frame with a timestamp older than previous frame detected!");
            }
            else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+2.0)
            {
                cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
                if(mpMap->IsInertial())
                {

                    if(mpMap->isImuInitialized())
                    {
                        cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                        cout << "Timestamp difference: " << mCurrentFrame.mTimeStamp- mLastFrame.mTimeStamp << endl;
                        // if(!mpMap->GetIniertialBA2())
                        // {
                        //     mpSystem->Reset();
                        // }
                        // else
                        // {
                        //     // CreateMapInAtlas();
                        //     throw std::runtime_error("Timestamp jump detected");
                        // }
                    }
                    else
                    {
                        cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                        mpSystem->Reset();
                    }
                }

                // return;

                // should not happen
                // throw std::runtime_error("Timestamp jump detected");
            }
        }

        if (mSensor == System::IMU_RGBD && mpLastKeyFrame)
            mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

        if (mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;

        // if (mSensor == System::IMU_RGBD  && !mbCreatedMap) // No preintegration only if switching map
        if (mSensor == System::IMU_RGBD)
            PreintegrateIMU();

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        // TODO IMU: Check if there was a big change to the map (e.g. BA)
        mbMapUpdated = false;

        int nCurMapChangeIndex = mpMap->GetMapChangeIndex();
        int nMapChangeIndex = mpMap->GetLastMapChange();
        if(nCurMapChangeIndex>nMapChangeIndex)
        {
            // cout << "Map update detected" << endl;
            mpMap->SetLastMapChange(nCurMapChangeIndex);
            mbMapUpdated = true;
        }

        if (mState == NOT_INITIALIZED) {
            if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_RGBD) {
                StereoInitialization();
                // mpSurfelMapper->InsertKeyFrame(mImGray.clone(), mImDepth.clone(), mCurrentFrame.planeDetector.plane_filter.membershipImg.clone(), mCurrentFrame.mTwc.clone(), 0, true);
            } else
                MonocularInitialization();

            mpFrameDrawer->Update(this);

            if (mState != OK){
                mLastFrame = Frame(mCurrentFrame);
                return;
            }
        } else {
            //Tracking: system is initialized (OK or LOST)
            bool bOK = false;
            bool bManhattan = false;
            mbPredictedWithIMU = false;

            // Initial camera pose estimation using motion model or relocalization
            // if tracking is lost within the current or previous frames
            if (!mbOnlyTracking) {
                
                /**
                 *     if mState == OK || mState == RECENTLY_LOST
                 *         predict state with IMU data
                 *     elif mVelocity is set before:
                 *         predict state with velocity
                 *     else
                 *         set the current state to the last state
                 * 
                 *     try to track with Manhattan
                 *
                 *     if fail to track with Manhattan
                 *         try to track with the original approach
                 * 
                 *     if bOK
                 *         local BA
                 *     
                 *     elif !bOK
                 *         if current frame id is smaller than the last relocalized frame id + some threshold
                 *             mState = LOST
                 *   
                 *         else if #keyframe > 10 (first lost long after relocalization)
                 *             mState = RECENTLY_LOST
                 *         else
                 *             mState = LOST
                 */
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                // Predict the current state with IMU or velocity, provide a bettet prior for searching
                // condition of TranslationWithMotionModel && condition of PredictStateIMU inside TranslationWithMotionModel
                if (!((mVelocity.empty() && !mpMap->isImuInitialized()) || mCurrentFrame.mnId < mnLastRelocFrameId + 2) &&
                    (mpMap->isImuInitialized() &&
                     ((mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU) ||
                      (mState == OK || mState == RECENTLY_LOST))))
                {
                    PredictStateIMU();
                    mbPredictedWithIMU = true;
                }
                else if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                {
                    mCurrentFrame.SetPose(mLastFrame.mTcw);
                }
                else
                {
                    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
                }

                // Searching map elements needs the current pose
                PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);
                pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

                bManhattan = DetectManhattan();

                cout << "bManhattan: " << bManhattan << endl;

                if (bManhattan) {
                    // Translation (only) estimation
                    if ((mVelocity.empty() && !mpMap->isImuInitialized()) || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                        bOK = TranslationEstimation();
                    } else {
                        bOK = TranslationWithMotionModel();

                        if (!bOK) {
                            bOK = TranslationEstimation();
                        }
                    }
                }

                if (bOK) {
                    if (bManhattan)
                        ++manhattanCount;

                    if (fullManhattanFound)
                        ++fullManhattanCount;
                }

                cout << "manhattanCount: " << manhattanCount << endl;
                cout << "fullManhattanCount: " << fullManhattanCount << endl;

                // If can't tracking with Manhattan, use the original way
                if (!bOK) {
                    if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                        bOK = TrackReferenceKeyFrame();
                    } else {
                        bOK = TrackWithMotionModel();
                        if (!bOK) {
                            bOK = TrackReferenceKeyFrame();
                        }
                    }
                }
                
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

                if(mState==OK || bOK) {

                    if (bOK)
                        mState = OK;
                    else if (!bOK)
                    {
                        if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                                mSensor == System::IMU_RGBD)
                        {
                            mState = LOST;
                        }
                        else if(mpMap->KeyFramesInMap()>10) // first lost long after relocalization
                        {
                            cout << "KF in map: " << mpMap->KeyFramesInMap() << endl;
                            mState = RECENTLY_LOST;
                            mTimeStampLost = mCurrentFrame.mTimeStamp;
                            //mCurrentFrame.SetPose(mLastFrame.mTcw);
                        }
                        else
                        {
                            mState = LOST;
                        }
                    }
                }

                // if imu is available
                // predicting state relying only on IMU measurement for a period < 5s
                // then run TrackLocalMap()
                // set to lost if tracking can't be recovered over 5s
                if (mState == RECENTLY_LOST)
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true;
                    if(mSensor == System::IMU_RGBD)
                    {
                        if(mpMap->isImuInitialized()){
                            PredictStateIMU();
                        }    
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                    else
                    {
                        // TODO fix relocalization
                        bOK = Relocalization();
                        if(!bOK)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                } else if (mState == LOST) {
                    Verbose::PrintMess("Tracking is still lost...", Verbose::VERBOSITY_NORMAL);

                    if (mpMap->KeyFramesInMap()<10)
                    {
                        // mpSystem->Reset();
                        // cout << "Reseting current map..." << endl;
                    }
                    else{
                        bOK = Relocalization();
                        // CreateMapInAtlas();
                        cout << "Try to relocalize while lost" << endl;
                    }

                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
                

                if (bOK)
                    bOK = TrackLocalMap();
                // else 
                //     bOK = Relocalization();

            } // NO localization mode

            if(bOK && mState!=OK){
                mpMap->SetImuInitialized(false);   
                mState = OK;
            }
            else if (bOK){
                mState = OK;
            }
            else if (mState == OK)
            {
                if (mSensor == System::IMU_RGBD)
                {
                    Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                    if(!mpMap->isImuInitialized() || !mpMap->GetIniertialBA2())
                    {
                        cout << "IMU is not or recently initialized." << endl;
                        // mpSystem->Reset();
                    }

                    mState=RECENTLY_LOST;
                }
                else
                    mState=LOST; // visual to lost

                if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
                {
                    mTimeStampLost = mCurrentFrame.mTimeStamp;
                }
            }

            // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
            if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) && (mSensor == System::IMU_RGBD) && mpMap->isImuInitialized())
            {
                // TODO check this situation
                Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
                Frame* pF = new Frame(mCurrentFrame);
                pF->mpPrevFrame = new Frame(mLastFrame);

                // Load preintegration
                pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
            }

            if(mpMap->isImuInitialized())
            {
                if(bOK)
                {
                    if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                    {
                        cout << "RESETING FRAME!!!" << endl;
                        // They didn't implement it
                        // ResetFrameIMU();
                    }
                    else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                        mLastBias = mCurrentFrame.mImuBias;
                }
            }

            // Update drawer
            mpFrameDrawer->Update(this);

            // Mark the map points of the current frame if it can be associated with planes of the current frame
            mpMap->FlagMatchedPlanePoints(mCurrentFrame, mfDThRef);

            //Update Planes
            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
                if (pMP) {
                    pMP->UpdateCoefficientsAndPoints(mCurrentFrame, i);
                } else if (!mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mbNewPlane = true;
                }
            }
            
            if (bOK || mState==RECENTLY_LOST) {
                // Update motion model
                if (!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    mVelocity = mCurrentFrame.mTcw * LastTwc;

                } else
                    mVelocity = cv::Mat();

                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                // Clean VO matches
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }
                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    if (pML)
                        if (pML->Observations() < 1) {
                            mCurrentFrame.mvbLineOutlier[i] = false;
                            mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                        }
                }

                // Delete temporal MapPoints
                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
                     lit != lend; lit++) {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }
                for (list<MapLine *>::iterator lit = mlpTemporalLines.begin(), lend = mlpTemporalLines.end();
                     lit != lend; lit++) {
                    MapLine *pML = *lit;
                    delete pML;
                }
                mlpTemporalPoints.clear();
                mlpTemporalLines.clear();

                int referenceIndex = 0;

                double timeDiff = 1e9;
                vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
                for(int i=0; i<vpKFs.size(); i++)
                {
                    double diff = fabs(vpKFs[i]->mTimeStamp - mpReferenceKF->mTimeStamp);
                    if (diff < timeDiff)
                    {
                        referenceIndex = i;
                        timeDiff = diff;
                    }
                }
        
                // Check if we need to insert a new keyframe
                if (bOK || (mState==RECENTLY_LOST && mSensor == System::IMU_RGBD)){
                    bool isKeyFrame = NeedNewKeyFrame();
                    if(isKeyFrame)
                        CreateNewKeyFrame();
                }
            
                // if (isKeyFrame)
                    // mpSurfelMapper->InsertKeyFrame(mImGray.clone(), mImDepth.clone(), mCurrentFrame.planeDetector.plane_filter.membershipImg.clone(), mCurrentFrame.mTwc.clone(), referenceIndex, isKeyFrame);

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    if (mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbLineOutlier[i])
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            if(mState==LOST)
            {
                if(mpMap->KeyFramesInMap()<=5)
                {
                    // mpSystem->Reset();
                    // return;
                }
                if (mSensor == System::IMU_RGBD)
                    if (!mpMap->isImuInitialized())
                    {
                        Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                        // mpSystem->Reset();
                        // return;
                    }

                // CreateMapInAtlas();
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }

        if(mState==OK || mState==RECENTLY_LOST)
        {
            // Store frame pose information to retrieve the complete camera trajectory afterwards.
            if(!mCurrentFrame.mTcw.empty())
            {
                cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
                mlRelativeFramePoses.push_back(Tcr);
                mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
                mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                mlbLost.push_back(mState==LOST);
            }
            else
            {
                // This can happen if tracking is lost
                mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                mlpReferences.push_back(mlpReferences.back());
                mlFrameTimes.push_back(mlFrameTimes.back());
                mlbLost.push_back(mState==LOST);
            }
        }
    }

    void Tracking::StereoInitialization() {
        // TODO: IMU add logic with lines and planes

        // if (mCurrentFrame.N > 500) {
        if (mCurrentFrame.N > 300 || mCurrentFrame.NL > 50) {
            if (mSensor == System::IMU_RGBD)
            {
                if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
                {
                    cout << "not IMU meas" << endl;
                    return;
                }

                if (cv::norm(mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA)<0.5)
                {
                    cout << "not enough acceleration" << endl;
                    return;
                }

                if(mpImuPreintegratedFromLastKF)
                    delete mpImuPreintegratedFromLastKF;

                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            }

            // Set Frame pose to the origin (In case of inertial SLAM to imu)
            if (mSensor == System::IMU_RGBD)
            {
                cv::Mat Rwb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0,3).colRange(0,3).clone();
                cv::Mat twb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0,3).col(3).clone();
                mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, cv::Mat::zeros(3,1,CV_32F));
            }
            else
                mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F)); // Set Frame pose to the origin
            // Create KeyFrame
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

            // Insert KeyFrame in the map
            mpMap->AddKeyFrame(pKFini);

            if (!mpCamera2){
                // Create MapPoints and asscoiate to KeyFrame
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    float z = mCurrentFrame.mvDepth[i];
                    if (z > 0) {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                        pNewMP->AddObservation(pKFini, i);
                        pKFini->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpMap->AddMapPoint(pNewMP);
                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    }
                }

                // Line
                for (int i = 0; i < mCurrentFrame.NL; i++) {

                    pair<float,float> z = mCurrentFrame.mvDepthLine[i];

                    if (z.first > 0 && z.second > 0) {
                        Vector6d line3D = mCurrentFrame.obtain3DLine(i, mImDepth);
                        if (line3D.hasNaN()) {
                            continue;
                        }
                        MapLine *pNewML = new MapLine(line3D, pKFini, mpMap);
                        pNewML->AddObservation(pKFini, i);
                        pKFini->AddMapLine(pNewML, i);
                        pNewML->ComputeDistinctiveDescriptors();
                        pNewML->UpdateAverageDir();
                        mpMap->AddMapLine(pNewML);
                        mCurrentFrame.mvpMapLines[i] = pNewML;
                    }
                }
                
                // Plane
                for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                    cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                    MapPlane *pNewMP = new MapPlane(p3D, pKFini, mpMap);
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPlane(pNewMP, i);
                    pNewMP->UpdateCoefficientsAndPoints();
                    mpMap->AddMapPlane(pNewMP);
                    mCurrentFrame.mvpMapPlanes[i] = pNewMP;
                }
            } else {
                // only consider RGBD but not stereo
                throw logic_error("Not Implemented");
            }

            Verbose::PrintMess("New Map created with " + to_string(mpMap->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
            mpLocalMapper->InsertKeyFrame(pKFini);

            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpMap->GetAllMapPoints();
            mvpLocalMapLines = mpMap->GetAllMapLines();

            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
            mpMap->SetReferenceMapLines(mvpLocalMapLines);

            mpMap->mvpKeyFrameOrigins.push_back(pKFini);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mState = OK;
        }
    }

    void Tracking::MonocularInitialization() {
        int num = 100;
        // 如果单目初始器还没有没创建，则创建单目初始器
        if (!mpInitializer) {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > num) {
                // step 1：得到用于初始化的第一帧，初始化需要两帧
                mInitialFrame = Frame(mCurrentFrame);
                // 记录最近的一帧
                mLastFrame = Frame(mCurrentFrame);
                // mvbPreMatched最大的情况就是当前帧所有的特征点都被匹配上
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

                if (mpInitializer)
                    delete mpInitializer;

                // 由当前帧构造初始化器， sigma:1.0    iterations:200
                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                return;
            }
        } else {
            // Try to initialize
            // step2：如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
            // 如果当前帧特征点太少，重新构造初始器
            // 因此只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
            if ((int) mCurrentFrame.mvKeys.size() <= num) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }

            // Find correspondences
            // step3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
            // mvbPrevMatched为前一帧的特征点，存储了mInitialFrame中哪些点将进行接下来的匹配,类型  std::vector<cv::Point2f> mvbPrevMatched;
            // mvIniMatches存储mInitialFrame, mCurrentFrame之间匹配的特征点，类型为std::vector<int> mvIniMatches; ????
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches,
                                                           100);

            LSDmatcher lmatcher;   //建立线特征之间的匹配
            int lineMatches = lmatcher.SerachForInitialize(mInitialFrame, mCurrentFrame, mvLineMatches);
//        cout << "Tracking::MonocularInitialization(), lineMatches = " << lineMatches << endl;
//
//        cout << "Tracking::MonocularInitialization(), mvLineMatches size = " << mvLineMatches.size() << endl;

            // Check if there are enough correspondences
            // step4：如果初始化的两帧之间的匹配点太少，重新初始化
            if (nmatches < 100) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                return;
            }

            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

            // step5：通过H或者F进行单目初始化，得到两帧之间相对运动，初始化MapPoints
#if 0
                                                                                                                                    if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
    {
        for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
        {
            if(mvIniMatches[i]>=0 && !vbTriangulated[i])
            {
                mvIniMatches[i]=-1;
                nmatches--;
            }
        }

        // Set Frame Poses
        // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
        mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
        cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(Tcw.rowRange(0,3).col(3));
        mCurrentFrame.SetPose(Tcw);

        // step6：将三角化得到的3D点包装成MapPoints
        /// 如果要修改，应该是从这个函数开始
        CreateInitialMapMonocular();
    }
#else
            if (0)//mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated, mvLineMatches, mvLineS3D, mvLineE3D, mvbLineTriangulated))
            {
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);

                // step6：将三角化得到的3D点包装成MapPoints
                /// 如果要修改，应该是从这个函数开始
//            CreateInitialMapMonocular();
                CreateInitialMapMonoWithLine();
            }
#endif
        }
    }

    void Tracking::CreateInitialMapMonocular() {
        // Create KeyFrames
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // Create MapPoints and asscoiate to keyframes
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);

            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            // b.从众多观测到该MapPoint的特征点中挑选出区分度最高的描述子
            pMP->ComputeDistinctiveDescriptors();

            // c.更新该MapPoint的平均观测方向以及观测距离的范围
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            //Add to Map
            mpMap->AddMapPoint(pMP);
        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

        Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

        // Set median depth to 1
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
            cout << "Wrong initialization, reseting..." << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
            if (vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;  //至此，初始化成功
    }

#if 1

/**
* @brief 为单目摄像头三角化生成带有线特征的Map，包括MapPoints和MapLine
*/
    void Tracking::CreateInitialMapMonoWithLine() {
        // step1:创建关键帧，即用于初始化的前两帧
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // step2：将两个关键帧的描述子转为BoW，这里的BoW只有ORB的词袋
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // step3：将关键帧插入到地图，凡是关键帧，都要插入地图
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // step4：将特征点的3D点包装成MapPoints
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            // Create MapPoint
            cv::Mat worldPos(mvIniP3D[i]);

            // step4.1：用3D点构造MapPoint
            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            // step4.2：为该MapPoint添加属性：
            // a.观测到该MapPoint的关键帧
            // b.该MapPoint的描述子
            // c.该MapPoint的平均观测方向和深度范围

            // step4.3：表示该KeyFrame的哪个特征点对应到哪个3D点
            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            // a.表示该MapPoint可以被哪个KeyFrame观测到，以及对应的第几个特征点
            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            // b.从众多观测到该MapPoint的特征点中挑选出区分度最高的描述子
            pMP->UpdateNormalAndDepth();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            // Add to Map
            // step4.4：在地图中添加该MapPoint
            mpMap->AddMapPoint(pMP);
        }

        // step5：将特征线包装成MapLines
        for (size_t i = 0; i < mvLineMatches.size(); i++) {
            if (!mvbLineTriangulated[i])
                continue;

            // Create MapLine
            Vector6d worldPos;
            worldPos << mvLineS3D[i].x, mvLineS3D[i].y, mvLineS3D[i].z, mvLineE3D[i].x, mvLineE3D[i].y, mvLineE3D[i].z;

            //step5.1：用线段的两个端点构造MapLine
            MapLine *pML = new MapLine(worldPos, pKFcur, mpMap);

            //step5.2：为该MapLine添加属性：
            // a.观测到该MapLine的关键帧
            // b.该MapLine的描述子
            // c.该MapLine的平均观测方向和深度范围？

            //step5.3：表示该KeyFrame的哪个特征点可以观测到哪个3D点
            pKFini->AddMapLine(pML, i);
            pKFcur->AddMapLine(pML, i);

            //a.表示该MapLine可以被哪个KeyFrame观测到，以及对应的第几个特征线
            pML->AddObservation(pKFini, i);
            pML->AddObservation(pKFcur, i);

            //b.MapPoint中是选取区分度最高的描述子，pl-slam直接采用前一帧的描述子,这里先按照ORB-SLAM的过程来
            pML->ComputeDistinctiveDescriptors();

            //c.更新该MapLine的平均观测方向以及观测距离的范围
            pML->UpdateAverageDir();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapLines[i] = pML;
            mCurrentFrame.mvbLineOutlier[i] = false;

            // step5.4: Add to Map
            mpMap->AddMapLine(pML);
        }

        // step6：更新关键帧间的连接关系
        // 1.最初是在3D点和关键帧之间建立边，每一个边有一个权重，边的权重是该关键帧与当前关键帧公共3D点的个数
        // 2.加入线特征后，这个关系应该和特征线也有一定的关系，或者就先不加关系，只是单纯的添加线特征

        // step7：全局BA优化，这里需要再进一步修改优化函数，参照OptimizePose函数
        cout << "this Map created with " << mpMap->MapPointsInMap() << " points, and " << mpMap->MapLinesInMap()
             << " lines." << endl;
        //Optimizer::GlobalBundleAdjustemnt(mpMap, 20, true); //true代表使用有线特征的BA

        // step8：将MapPoints的中值深度归一化到1，并归一化两帧之间的变换
        // Q：MapPoints的中值深度归一化为1，MapLine是否也归一化？
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        cout << "medianDepth = " << medianDepth << endl;
        cout << "pKFcur->TrackedMapPoints(1) = " << pKFcur->TrackedMapPoints(1) << endl;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 80) {
            cout << "Wrong initialization, reseting ... " << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale Points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); ++iMP) {
            if (vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        // Scale Line Segments
        vector<MapLine *> vpAllMapLines = pKFini->GetMapLineMatches();
        for (size_t iML = 0; iML < vpAllMapLines.size(); iML++) {
            if (vpAllMapLines[iML]) {
                MapLine *pML = vpAllMapLines[iML];
                pML->SetWorldPos(pML->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mvpLocalMapLines = mpMap->GetAllMapLines();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;
    }

#endif

    void Tracking::CheckReplacedInLastFrame() {
        for (int i = 0; i < mLastFrame.N; i++) {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];

            if (pMP) {
                MapPoint *pRep = pMP->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }

        for (int i = 0; i < mLastFrame.NL; i++) {
            MapLine *pML = mLastFrame.mvpMapLines[i];

            if (pML) {
                MapLine *pReL = pML->GetReplaced();
                if (pReL) {
                    mLastFrame.mvpMapLines[i] = pReL;
                }
            }
        }

//        for (int i = 0; i < mLastFrame.mnPlaneNum; i++) {
//            MapPlane *pMP = mLastFrame.mvpMapPlanes[i];
//
//            if (pMP) {
//                MapPlane *pRep = pMP->GetReplaced();
//                if (pRep) {
//                    mLastFrame.mvpMapPlanes[i] = pRep;
//                }
//            }
//        }
    }

    bool Tracking::DetectManhattan() {

        auto verTh = Config::Get<double>("Plane.MFVerticalThreshold");
        KeyFrame * pKFCandidate = nullptr;
        int maxScore = 0;
        cv::Mat pMFc1, pMFc2, pMFc3, pMFm1, pMFm2, pMFm3;
        fullManhattanFound = false;

        int id1, id2, id3 = -1;

        for (size_t i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            cv::Mat p3Dc1 = mCurrentFrame.mvPlaneCoefficients[i];
            MapPlane *pMP1 = mCurrentFrame.mvpMapPlanes[i];

            if (!pMP1 || pMP1->isBad()) {
                continue;
            }

            // cout << "MF detection id1: " << pMP1->mnId << endl;

            for (size_t j = i + 1; j < mCurrentFrame.mnPlaneNum; j++) {
                cv::Mat p3Dc2 = mCurrentFrame.mvPlaneCoefficients[j];
                MapPlane *pMP2 = mCurrentFrame.mvpMapPlanes[j];

                if (!pMP2 || pMP2->isBad()) {
                    continue;
                }

                // cout << "MF detection id2: " << pMP2->mnId << endl;

                float angle12 = p3Dc1.at<float>(0) * p3Dc2.at<float>(0) +
                              p3Dc1.at<float>(1) * p3Dc2.at<float>(1) +
                              p3Dc1.at<float>(2) * p3Dc2.at<float>(2);

                if (angle12 > verTh || angle12 < -verTh) {
                    continue;
                }

                // cout << "MF detection angle12: " << angle12 << endl;

                for (size_t k = j+1; k < mCurrentFrame.mnPlaneNum; k++) {
                    cv::Mat p3Dc3 = mCurrentFrame.mvPlaneCoefficients[k];
                    MapPlane *pMP3 = mCurrentFrame.mvpMapPlanes[k];

                    if (!pMP3 || pMP3->isBad()) {
                        continue;
                    }

                    // cout << "MF detection id3: " << pMP3->mnId << endl;

                    float angle13 = p3Dc1.at<float>(0) * p3Dc3.at<float>(0) +
                                    p3Dc1.at<float>(1) * p3Dc3.at<float>(1) +
                                    p3Dc1.at<float>(2) * p3Dc3.at<float>(2);

                    float angle23 = p3Dc2.at<float>(0) * p3Dc3.at<float>(0) +
                                    p3Dc2.at<float>(1) * p3Dc3.at<float>(1) +
                                    p3Dc2.at<float>(2) * p3Dc3.at<float>(2);

                    if (angle13 > verTh || angle13 < -verTh || angle23 > verTh || angle23 < -verTh) {
                        continue;
                    }

                    // cout << "MF detection angle13: " << angle13 << endl;
                    // cout << "MF detection angle23: " << angle23 << endl;

                    KeyFrame* pKF = mpMap->GetManhattanObservation(pMP1, pMP2, pMP3);

                    if (!pKF) {
                        continue;
                    }

                    auto idx1 = pMP1->GetIndexInKeyFrame(pKF);
                    auto idx2 = pMP2->GetIndexInKeyFrame(pKF);
                    auto idx3 = pMP3->GetIndexInKeyFrame(pKF);

                    if (idx1 == -1 || idx2 == -1 || idx3 == -1) {
                        continue;
                    }

                    int score = pKF->mvPlanePoints[idx1].size() +
                                pKF->mvPlanePoints[idx2].size() +
                                pKF->mvPlanePoints[idx3].size() +
                                mCurrentFrame.mvPlanePoints[i].size() +
                                mCurrentFrame.mvPlanePoints[j].size() +
                                mCurrentFrame.mvPlanePoints[k].size();

                    if (score > maxScore) {
                        maxScore = score;

                        pKFCandidate = pKF;
                        pMFc1 = p3Dc1;
                        pMFc2 = p3Dc2;
                        pMFc3 = p3Dc3;
                        pMFm1 = pKF->mvPlaneCoefficients[idx1];
                        pMFm2 = pKF->mvPlaneCoefficients[idx2];
                        pMFm3 = pKF->mvPlaneCoefficients[idx3];

                        id1 = pMP1->mnId;
                        id2 = pMP2->mnId;
                        id3 = pMP3->mnId;

                        fullManhattanFound = true;
                        cout << "Full MF detection found!" << endl;
                    }
                }

                KeyFrame* pKF = mpMap->GetPartialManhattanObservation(pMP1, pMP2);

                if (!pKF) {
                    continue;
                }

                auto idx1 = pMP1->GetIndexInKeyFrame(pKF);
                auto idx2 = pMP2->GetIndexInKeyFrame(pKF);

                if (idx1 == -1 || idx2 == -1) {
                    continue;
                }

                int score = pKF->mvPlanePoints[idx1].size() +
                            pKF->mvPlanePoints[idx2].size() +
                            mCurrentFrame.mvPlanePoints[i].size() +
                            mCurrentFrame.mvPlanePoints[j].size();

                if (score > maxScore) {
                    maxScore = score;

                    pKFCandidate = pKF;
                    pMFc1 = p3Dc1;
                    pMFc2 = p3Dc2;
                    pMFm1 = pKF->mvPlaneCoefficients[idx1];
                    pMFm2 = pKF->mvPlaneCoefficients[idx2];

                    id1 = pMP1->mnId;
                    id2 = pMP2->mnId;

                    fullManhattanFound = false;
                    cout << "Partial MF detection found!" << endl;
                }
            }
        }

        if (pKFCandidate==nullptr) {
            return false;
        }

        cout << "Manhattan found!" << endl;

        cout << "Ref MF frame id: " << pKFCandidate->mnFrameId<< endl;

        cout << "Manhattan id1: " << id1 << endl;
        cout << "Manhattan id2: " << id2 << endl;

        if (!fullManhattanFound) {
            cv::Mat pMFc1n = (cv::Mat_<float>(3, 1) << pMFc1.at<float>(0), pMFc1.at<float>(1), pMFc1.at<float>(2));
            cv::Mat pMFc2n = (cv::Mat_<float>(3, 1) << pMFc2.at<float>(0), pMFc2.at<float>(1), pMFc2.at<float>(2));
            pMFc3 = pMFc1n.cross(pMFc2n);

            cv::Mat pMFm1n = (cv::Mat_<float>(3, 1) << pMFm1.at<float>(0), pMFm1.at<float>(1), pMFm1.at<float>(2));
            cv::Mat pMFm2n = (cv::Mat_<float>(3, 1) << pMFm2.at<float>(0), pMFm2.at<float>(1), pMFm2.at<float>(2));
            pMFm3 = pMFm1n.cross(pMFm2n);
        } else {
            cout << "Manhattan id3: " << id3 << endl;
        }

//        cout << "Manhattan pMFc1: " << pMFc1.t() << endl;
//        cout << "Manhattan pMFc2: " << pMFc2.t() << endl;
//        cout << "Manhattan pMFc3: " << pMFc3.t() << endl;
//
//        cout << "Manhattan pMFm1: " << pMFm1.t() << endl;
//        cout << "Manhattan pMFm2: " << pMFm2.t() << endl;
//        cout << "Manhattan pMFm3: " << pMFm3.t() << endl;

        cv::Mat MFc, MFm;
        MFc = cv::Mat::eye(cv::Size(3, 3), CV_32F);
        MFm = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        MFc.at<float>(0, 0) = pMFc1.at<float>(0);
        MFc.at<float>(1, 0) = pMFc1.at<float>(1);
        MFc.at<float>(2, 0) = pMFc1.at<float>(2);
        MFc.at<float>(0, 1) = pMFc2.at<float>(0);
        MFc.at<float>(1, 1) = pMFc2.at<float>(1);
        MFc.at<float>(2, 1) = pMFc2.at<float>(2);
        MFc.at<float>(0, 2) = pMFc3.at<float>(0);
        MFc.at<float>(1, 2) = pMFc3.at<float>(1);
        MFc.at<float>(2, 2) = pMFc3.at<float>(2);

        if (!fullManhattanFound && std::abs(cv::determinant(MFc) + 1) < 0.5) {
            MFc.at<float>(0, 2) = -pMFc3.at<float>(0);
            MFc.at<float>(1, 2) = -pMFc3.at<float>(1);
            MFc.at<float>(2, 2) = -pMFc3.at<float>(2);
        }

        cv::Mat Uc, Wc, VTc;

        cv::SVD::compute(MFc, Wc, Uc, VTc);

        MFc = Uc * VTc;

        MFm.at<float>(0, 0) = pMFm1.at<float>(0);
        MFm.at<float>(1, 0) = pMFm1.at<float>(1);
        MFm.at<float>(2, 0) = pMFm1.at<float>(2);
        MFm.at<float>(0, 1) = pMFm2.at<float>(0);
        MFm.at<float>(1, 1) = pMFm2.at<float>(1);
        MFm.at<float>(2, 1) = pMFm2.at<float>(2);
        MFm.at<float>(0, 2) = pMFm3.at<float>(0);
        MFm.at<float>(1, 2) = pMFm3.at<float>(1);
        MFm.at<float>(2, 2) = pMFm3.at<float>(2);

        if (!fullManhattanFound && std::abs(cv::determinant(MFm) + 1) < 0.5) {
            MFm.at<float>(0, 2) = -pMFm3.at<float>(0);
            MFm.at<float>(1, 2) = -pMFm3.at<float>(1);
            MFm.at<float>(2, 2) = -pMFm3.at<float>(2);
        }

        cv::Mat Um, Wm, VTm;

        cv::SVD::compute(MFm, Wm, Um, VTm);

        MFm = Um * VTm;

        // cout << "MFc: " << MFc << endl;
        // cout << "MFm: " << MFm << endl;

        cv::Mat Rwc = pKFCandidate->GetPoseInverse().rowRange(0,3).colRange(0,3) * MFm * MFc.t();
        manhattanRcw = Rwc.t();

        return true;
    }

    bool Tracking::TranslationEstimation() {
        // Similar to TrackReferenceKeyFrame

        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        vector<MapPoint *> vpMapPointMatches;
        vector<MapLine *> vpMapLineMatches;
        vector<pair<int, int>> vLineMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int initialMatches = nmatches + lmatches + planeMatches;

        cout << "TranslationEstimation: Before: Point Matches: " << nmatches << " , Line Matches:"
             << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
            cout << "TranslationEstimation: Before: Not enough matches" << endl;
            return false;
        }

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.mvpMapLines = vpMapLineMatches;

        manhattanRcw.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));

        // cout << "translation reference,pose before opti" << mCurrentFrame.mTcw << endl;
        Optimizer::TranslationOptimization(&mCurrentFrame);
        // cout << "translation reference,pose after opti" << mCurrentFrame.mTcw << endl;

        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        // Discard outliers
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
//                    nmatches--;

                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;

            }
        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        cout << "TranslationEstimation: After: Matches: " << nmatchesMap << " , Line Matches:"
             << nmatchesLineMap << " , Plane Matches:" << nmatchesPlaneMap << endl;

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        double ratioThresh = Config::Get<double>("MFTrackingThreshold");

        if (nmatchesMap < 3 || nmatchesMap / nmatches < ratioThresh) {
            cout << "TranslationEstimation: After: Not enough matches" << endl;
            return false;
        }

        return true;
    }

    bool Tracking::TranslationWithMotionModel() {
        ORBmatcher matcher(0.9, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        if (mpMap->isImuInitialized() && 
            ((mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU) || 
                (mState == OK ||  mState == RECENTLY_LOST)))
        {
            // Predict ste with IMU if it is initialized and it doesnt need reset
            // Update world to body frame, will be used in TrackLocalMap()

            if (!mbPredictedWithIMU)
                PredictStateIMU();

            // Don't estimate pose by projection (establishing 2D-3D correspondences as the following)
            // Projection be done in TrackLocalMap() so we can safely return here
            // return true;
        }
        else
        {
            mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
        }

        // Project points seen in previous frame
        int th;
        if (mSensor != System::STEREO)
            th = 15;
        else
            th = 7;
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

        fill(mCurrentFrame.mvpMapLines.begin(), mCurrentFrame.mvpMapLines.end(), static_cast<MapLine *>(NULL));
        int lmatches = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

        if (nmatches < 40) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 4 * th, mSensor == System::MONOCULAR);
        }

        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int initialMatches = nmatches + lmatches + planeMatches;

        cout << "TranslationWithMotionModel: Before: Point matches: " << nmatches << " , Line Matches:"
             << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
            cout << "TranslationWithMotionModel: Before: Not enough matches" << endl;
            return false;
        }

        manhattanRcw.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));

        // Optimize frame pose with all matches
        // cout << "translation motion model,pose before opti" << mCurrentFrame.mTcw << endl;
        if (mpMap->isImuInitialized() && mCurrentFrame.mpcpi &&
            ((mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU) || 
                (mState == OK ||  mState == RECENTLY_LOST)))
            {
                if (!mbMapUpdated)
                    Optimizer::TranslationInertialOptimizationLastFrame(&mCurrentFrame);
                else
                    Optimizer::TranslationInertialOptimizationLastKeyFrame(&mCurrentFrame);
            }
        else    
            Optimizer::TranslationOptimization(&mCurrentFrame);
        // cout << "translation motion model,pose after opti" << mCurrentFrame.mTcw << endl;

        // Discard outliers
        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
//                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;
            }

        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        cout << "TranslationWithMotionModel: After: Matches: " << nmatchesMap << " , Line Matches:"
             << nmatchesLineMap << ", Plane Matches:" << nmatchesPlaneMap << endl;

        if (mbOnlyTracking) {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        double ratioThresh = Config::Get<double>("MFTrackingThreshold");

        if (nmatchesMap < 3 || nmatchesMap / nmatches < ratioThresh) {
            cout << "TranslationWithMotionModel: After: Not enough matches" << endl;
            return false;
        }

        return true;
    }

    void Tracking::UpdateLastFrame() {
        // Update pose according to reference keyframe
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();

        mLastFrame.SetPose(Tlr * pRef->GetPose());

        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for (int i = 0; i < mLastFrame.N; i++) {
            float z = mLastFrame.mvDepth[i];
            if (z > 0) {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++) {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1) {
                bCreateNew = true;
            }

            if (bCreateNew) {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

                mLastFrame.mvpMapPoints[i] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            } else {
                nPoints++;
            }

            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }


        // Create "visual odometry" MapLines
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vLineDepthIdx;
        vLineDepthIdx.reserve(mLastFrame.NL);
        int nLines = 0;
        for (int i = 0; i < mLastFrame.NL; i++) {
            pair<float,float> z = mLastFrame.mvDepthLine[i];

            if (z.first > 0 && z.second > 0) {
                bool bCreateNew = false;
                vLineDepthIdx.push_back(make_pair(min(z.first, z.second), i));
                MapLine *pML = mLastFrame.mvpMapLines[i];
                if (!pML)
                    bCreateNew = true;
                else if (pML->Observations() < 1) {
                    bCreateNew = true;
                }
                if (bCreateNew) {
                    Vector6d line3D = mLastFrame.obtain3DLine(i, mImDepth);
                    if (line3D.hasNaN()) {
                        continue;
                    }
                    MapLine *pNewML = new MapLine(line3D, mpMap, &mLastFrame, i);

                    mLastFrame.mvpMapLines[i] = pNewML;

                    mlpTemporalLines.push_back(pNewML);
                    nLines++;
                } else {
                    nLines++;
                }

                if (nLines > 30)
                    break;

            }
        }
    }

    bool Tracking::TrackReferenceKeyFrame() {

        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();
        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        vector<MapPoint *> vpMapPointMatches;
        vector<MapLine *> vpMapLineMatches;
        vector<pair<int, int>> vLineMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        float initialMatches = nmatches + lmatches + planeMatches;

//        cout << "TrackReferenceKeyFrame: Before: Point matches: " << nmatches << " , Line Matches:"
//             << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
//            cout << "TrackReferenceKeyFrame: Before: Not enough matches" << endl;
            return false;
        }

        mCurrentFrame.mvpMapLines = vpMapLineMatches;
        mCurrentFrame.mvpMapPoints = vpMapPointMatches;


//        cout << "tracking reference,pose before opti" << mCurrentFrame.mTcw << endl;
        Optimizer::PoseOptimization(&mCurrentFrame);
//        cout << "tracking reference,pose after opti" << mCurrentFrame.mTcw << endl;

        // Discard outliers

        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        // Line
        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;

            }
        }

        // Plane
        int nDiscardPlane = 0;
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                    nDiscardPlane++;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

//        cout << "TrackReferenceKeyFrame: After: Matches: " << nmatchesMap << " , Line Matches:"
//             << nmatchesLineMap << ", Plane Matches:" << nmatchesPlaneMap << endl;

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

//        if (finalMatches < 10) {
        if (nmatchesMap < 3 || nmatchesMap / nmatches < 0.1) {
//            cout << "TrackReferenceKeyFrame: After: Not enough matches" << endl;
            return false;
        }

        return true;
    }

    bool Tracking::TrackWithMotionModel() {
        ORBmatcher matcher(0.9, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        if (mpMap->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
        {
            // Predict ste with IMU if it is initialized and it doesnt need reset
            // Update world to body frame, will be used in TrackLocalMap()

            if (!mbPredictedWithIMU)
                PredictStateIMU();

            return true;
        }
        else
        {
            mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
        }

        // Project points seen in previous frame
        int th;
        if (mSensor != System::STEREO)
            th = 15;
        else
            th = 7;
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

        fill(mCurrentFrame.mvpMapLines.begin(), mCurrentFrame.mvpMapLines.end(), static_cast<MapLine *>(NULL));
        int lmatches = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

//        vector<MapLine *> vpMapLineMatches;
//        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
//        mCurrentFrame.mvpMapLines = vpMapLineMatches;

        // If few matches, uses a wider window search
        if (nmatches < 40) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
                 static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 4 * th, mSensor == System::MONOCULAR);
        }

        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int initialMatches = nmatches + lmatches + planeMatches;

//        cout << "TrackWithMotionModel: Before: Point matches: " << nmatches << " , Line Matches:"
//             << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
//            cout << "TrackWithMotionModel: Before: Not enough matches" << endl;

            // TODO IMU: check logic in the main loop and the performance difference
            if (mSensor == System::IMU_RGBD)
                return true;
            else
                return false;
        }

        // Optimize frame pose with all matches
//        cout << "tracking motion model,pose before opti" << mCurrentFrame.mTcw << endl;
        Optimizer::PoseOptimization(&mCurrentFrame);
//        cout << "tracking motion model,pose after opti" << mCurrentFrame.mTcw << endl;

        // Discard outliers
        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;
            }

        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

//        cout << "TrackWithMotionModel: After: Matches: " << nmatchesMap << " , Line Matches:"
//             << nmatchesLineMap << ", Plane Matches:" << nmatchesPlaneMap << endl;

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if (mbOnlyTracking) {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

//        if (finalMatches < 10) {
        if (nmatchesMap < 3 || nmatchesMap / nmatches < 0.1) {
//            cout << "TrackWithMotionModel: After: Not enough matches" << endl;

            // TODO IMU: check logic in the main loop and the performance difference
            // if (mSensor == System::IMU_RGBD)
            //     return true;
            // else
                return false;
        }

        return true;
    }

    bool Tracking::TrackLocalMap() {
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        UpdateLocalMap();

        thread threadPoints(&Tracking::SearchLocalPoints, this);
        thread threadLines(&Tracking::SearchLocalLines, this);
        thread threadPlanes(&Tracking::SearchLocalPlanes, this);
        threadPoints.join();
        threadLines.join();
        threadPlanes.join();

        pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int inliers;
        if (!mpMap->isImuInitialized()){
                //cout << "tracking localmap with lines, pose before opti" << endl << mCurrentFrame.mTcw << endl;
                Optimizer::PoseOptimization(&mCurrentFrame);
                // Optimizer::TranslationOptimization(&mCurrentFrame);
                //cout << "tracking localmap with lines, pose after opti" << mCurrentFrame.mTcw << endl;
            }
        else
        {
            if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
            {
                Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
                Optimizer::PoseOptimization(&mCurrentFrame);
            }
            else
            {
                // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
                if(!mbMapUpdated && mnMatchesInliers>30) //  && (mnMatchesInliers>30))
                {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                }
                else
                {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                }
            }
        }

        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (!mCurrentFrame.mvbOutlier[i]) {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking) {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                } else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);

            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (!mCurrentFrame.mvbLineOutlier[i]) {
                    mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                    if (!mbOnlyTracking) {
                        if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                } else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
            }
        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                } else {
                    mCurrentFrame.mvpMapPlanes[i]->IncreaseFound();
                    mnMatchesInliers++;
                }
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

//        cout << "TrackLocalMap: After: Matches: " << mnMatchesInliers << endl;

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 20) {
//            cout << "TrackLocalMap: After: Not enough matches" << endl;
            return false;
        }

        if (mnMatchesInliers < 5) {
//            cout << "TrackLocalMapWithLines: After: Not enough matches" << endl;
            return false;
        } else
            return true;

        // TODO IMU: check performance with and without threshold
        // if (mSensor==System::IMU_RGBD){
        //     if (mnMatchesInliers<5)
        //         return false;
        //     else
        //         return true;
        // }
    }


    bool Tracking::NeedNewKeyFrame() {
        if (!mpLastKeyFrame)
            return true;

        // Frequently insert keyframes if imu is still not initialized
        if(mSensor == System::IMU_RGBD && !mpMap->isImuInitialized())
        {
            if (mSensor == System::IMU_RGBD && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
                return true;
            else
                return false;
        }

        if (mbOnlyTracking)
            return false;

// If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;
        
        // Return false if IMU is initialazing
        if (mpLocalMapper->IsInitializing())
            return false;

        const int nKFs = mpMap->KeyFramesInMap();

// Do not insert keyframes if not enough frames have passed from last relocalisation
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

// Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        // Require less observations if there is less than 3 kf in the map
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

// Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

// Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
// "total matches = matches to map + visual odometry matches"
// Visual odometry matches will become MapPoints if we insert a keyframe.
// This ratio measures how many MapPoints we could create if we insert a keyframe.
        int nMap = 0; //nTrackedClose
        int nTotal = 0;
        int nNonTrackedClose = 0;
        if (mSensor != System::MONOCULAR) {
            for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
                    nTotal++;
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                        nMap++;
                    else
                        nNonTrackedClose++;
                }
            }
        } else {
            // There are no visual odometry matches in the monocular case
            nMap = 1;
            nTotal = 1;
        }

        const float ratioMap = (float) nMap / fmax(1.0f, nTotal);

// Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        if(mpCamera2){
            throw logic_error("Not implemented");
            // thRefRatio = 0.75f;
        }

        float thMapRatio = 0.35f;
        if (mnMatchesInliers > 300)
            thMapRatio = 0.20f;

        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
        //Condition 1c: tracking is weak
        const bool c1c = mSensor != System::MONOCULAR && mSensor != System::IMU_RGBD && (mnMatchesInliers < nRefMatches * 0.25 || ratioMap < 0.3f);
        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        // mnMatchesInliers > 15 is threshold for creating enough new map points
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || ratioMap < thMapRatio) &&
                         mnMatchesInliers > 15);

        // Temporal condition for Inertial cases
        bool c3 = false;
        if(mpLastKeyFrame)
        {
            if (mSensor==System::IMU_RGBD)
            {
                if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                    c3 = true;
            }
        }

        // mCurrentFrame.mbNewPlane is useless here because it will be set to true only after needkeyframe()
        if (((c1a || c1b || c1c) && c2) || c3 || mCurrentFrame.mbNewPlane) {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if (bLocalMappingIdle) {
                return true;
            } else {
                mpLocalMapper->InterruptBA();
                if (mSensor != System::MONOCULAR) {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                        return true;
                    else
                        return false;
                } else
                    return false;
            }
        }

        return false;
    }

    void Tracking::CreateNewKeyFrame() {

        if(mpLocalMapper->IsInitializing())
            return;

        if (!mpLocalMapper->SetNotStop(true))
            return;

        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        if(mpMap->isImuInitialized())
            pKF->bImu = true;
            
        pKF->SetNewBias(mCurrentFrame.mImuBias);
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        if(mpLastKeyFrame)
        {
            pKF->mPrevKF = mpLastKeyFrame;
            mpLastKeyFrame->mNextKF = pKF;
        }
        else
            Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

        // create preintegration from last KF
        if (mSensor == System::IMU_RGBD)
        {
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
        }

        if (mSensor != System::MONOCULAR) {

            mCurrentFrame.UpdatePoseMatrices();

            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            vector<pair<float, int> > vDepthIdx;
            vDepthIdx.reserve(mCurrentFrame.N);

            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty()) {
                sort(vDepthIdx.begin(), vDepthIdx.end());

                int nPoints = 0;
                for (size_t j = 0; j < vDepthIdx.size(); j++) {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }

                    if (bCreateNew) {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                        pNewMP->AddObservation(pKF, i);
                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpMap->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;
                    } else {
                        nPoints++;
                    }

                    if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                        break;
                }
            }

            vector<pair<float, int>> vLineDepthIdx;
            vLineDepthIdx.reserve(mCurrentFrame.NL);

            for (int i = 0; i < mCurrentFrame.NL; i++) {
                pair<float,float> z = mCurrentFrame.mvDepthLine[i];
                if (z.first > 0 && z.second > 0) {
                    vLineDepthIdx.push_back(make_pair(min(z.first, z.second), i));
                }
            }

            if (!vLineDepthIdx.empty()) {
                sort(vLineDepthIdx.begin(),vLineDepthIdx.end());

                int nLines = 0;
                for (size_t j = 0; j < vLineDepthIdx.size(); j++) {
                    int i = vLineDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapLine *pMP = mCurrentFrame.mvpMapLines[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    }

                    if (bCreateNew) {
                        Vector6d line3D = mCurrentFrame.obtain3DLine(i, mImDepth);
                        if (line3D.hasNaN()) {
                            continue;
                        }
                        MapLine *pNewML = new MapLine(line3D, pKF, mpMap);
                        pNewML->AddObservation(pKF, i);
                        pKF->AddMapLine(pNewML, i);
                        pNewML->ComputeDistinctiveDescriptors();
                        pNewML->UpdateAverageDir();
                        mpMap->AddMapLine(pNewML);
                        mCurrentFrame.mvpMapLines[i] = pNewML;
                        nLines++;
                    } else {
                        nLines++;
                    }

                    if (nLines > 30)
                        break;
                }
            }

            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                if (mCurrentFrame.mvpParallelPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
//                if (mCurrentFrame.mvpParallelPlanes[i]) {
                    mCurrentFrame.mvpParallelPlanes[i]->AddParObservation(pKF, i);
                }
                if (mCurrentFrame.mvpVerticalPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
//                if (mCurrentFrame.mvpVerticalPlanes[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i]->AddVerObservation(pKF, i);
                }

                if (mCurrentFrame.mvpMapPlanes[i]) {
                    mCurrentFrame.mvpMapPlanes[i]->AddObservation(pKF, i);
                    continue;
                }

                if (mCurrentFrame.mvbPlaneOutlier[i]) {
//                    mCurrentFrame.mvbPlaneOutlier[i] = false;
                    continue;
                }

//                if (mCurrentFrame.mvpMapPlanes[i] || mCurrentFrame.mvbPlaneOutlier[i]) {
//                    continue;
//                }
//
//                if (mCurrentFrame.mvpParallelPlanes[i]) {
//                    mCurrentFrame.mvpParallelPlanes[i]->AddParObservation(pKF, i);
//                }
//                if (mCurrentFrame.mvpVerticalPlanes[i]) {
//                    mCurrentFrame.mvpVerticalPlanes[i]->AddVerObservation(pKF, i);
//                }

                pKF->SetNotErase();

                cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                MapPlane *pNewMP = new MapPlane(p3D, pKF, mpMap);
                pNewMP->AddObservation(pKF,i);
                pKF->AddMapPlane(pNewMP, i);
                pNewMP->UpdateCoefficientsAndPoints();
                mpMap->AddMapPlane(pNewMP);
            }
//            mpPointCloudMapping->print();
        }

        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);

        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
    }

    void Tracking::SearchLocalPoints() {
// Do not search map points already matched
        for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPoint *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

// Project points in frame and check its visibility
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->isBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
                pMP->IncreaseVisible();
                nToMatch++; //将要match的
            }

            // Only used in frame drawing
            // if(pMP->mbTrackInView)
            // {
            //     mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
            // }
        }

        if (nToMatch > 0) {
            ORBmatcher matcher(0.8);
            int th = 1;
            if (mSensor == System::RGBD || mSensor == System::IMU_RGBD)
                th = 3;

            if(mpMap->isImuInitialized())
            {
                if(mpMap->GetIniertialBA2())
                    th=2;
                else
                    th=3;
            }
            else if(!mpMap->isImuInitialized() && mSensor==System::IMU_RGBD)
            {
                th=10;
            }

            // If the camera has been relocalised recently, perform a coarser search
            if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
                th=5;

            if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
                th=15; // 15

            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);  // TODO IMU: far point
        }
    }

    void Tracking::SearchLocalLines() {
        for (vector<MapLine *>::iterator vit = mCurrentFrame.mvpMapLines.begin(), vend = mCurrentFrame.mvpMapLines.end();
             vit != vend; vit++) {
            MapLine *pML = *vit;
            if (pML) {
                if (pML->isBad()) {
                    *vit = static_cast<MapLine *>(NULL);
                } else {
                    pML->IncreaseVisible();
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    pML->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        for (vector<MapLine *>::iterator vit = mvpLocalMapLines.begin(), vend = mvpLocalMapLines.end();
             vit != vend; vit++) {
            MapLine *pML = *vit;

            if (pML->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pML->isBad())
                continue;

            if (mCurrentFrame.isInFrustum(pML, 0.6)) {
                pML->IncreaseVisible();
                nToMatch++;
            }
        }

        if (nToMatch > 0) {
            LSDmatcher matcher;
            int th = 1;

            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;

            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines, th);
        }
    }

    void Tracking::SearchLocalPlanes() {
        for (vector<MapPlane *>::iterator vit = mCurrentFrame.mvpMapPlanes.begin(), vend = mCurrentFrame.mvpMapPlanes.end();
             vit != vend; vit++) {
            MapPlane *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPlane *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                }
            }
        }
    }


    void Tracking::UpdateLocalMap() {
// This is for visualization
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

// Update
        UpdateLocalKeyFrames();

        UpdateLocalPoints();
        UpdateLocalLines();
    }

    void Tracking::UpdateLocalLines() {
        mvpLocalMapLines.clear();

        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapLine *> vpMLs = pKF->GetMapLineMatches();

            for (vector<MapLine *>::const_iterator itML = vpMLs.begin(), itEndML = vpMLs.end();
                 itML != itEndML; itML++) {
                MapLine *pML = *itML;
                if (!pML)
                    continue;
                if (pML->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pML->isBad()) {
                    mvpLocalMapLines.push_back(pML);
                    pML->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    void Tracking::UpdateLocalPoints() {
        mvpLocalMapPoints.clear();

        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end();
                 itMP != itEndMP; itMP++) {
                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
                else if (pMP->isBad() || pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                else {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }


    void Tracking::UpdateLocalKeyFrames() {
// Each map point vote for the keyframes in which it has been observed
        map<KeyFrame *, int> keyframeCounter;

        if(!mpMap->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
        {
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                {
                    if(!pMP->isBad())
                    {
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        for(map<KeyFrame*, size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                            keyframeCounter[it->first]++;
                    }
                    else
                    {
                        mCurrentFrame.mvpMapPoints[i]=NULL;
                    }
                }
            }
        }
        else
        {
            for(int i=0; i<mLastFrame.N; i++)
            {
                // Using lastframe since current frame has not matches yet
                if(mLastFrame.mvpMapPoints[i])
                {
                    MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                    if(!pMP)
                        continue;
                    if(!pMP->isBad())
                    {
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        for(map<KeyFrame*, size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                            keyframeCounter[it->first]++;
                    }
                    else
                    {
                        // MODIFICATION
                        mLastFrame.mvpMapPoints[i]=NULL;
                    }
                }
            }
        }

//        for (int i = 0; i < mCurrentFrame.NL; i++) {
//            if (mCurrentFrame.mvpMapLines[i]) {
//                MapLine *pML = mCurrentFrame.mvpMapLines[i];
//                if (!pML->isBad()) {
//                    const map<KeyFrame *, size_t> observations = pML->GetObservations();
//                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
//                         it != itend; it++)
//                        keyframeCounter[it->first]++;
//                } else {
//                    mCurrentFrame.mvpMapLines[i] = NULL;
//                }
//            }
//        }
//
//        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
//            if (mCurrentFrame.mvpMapPlanes[i]) {
//                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
//                if (!pMP->isBad()) {
//                    const map<KeyFrame *, size_t> observations = pMP->GetObservations();
//                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
//                         it != itend; it++)
//                        keyframeCounter[it->first]++;
//                } else {
//                    mCurrentFrame.mvpMapPlanes[i] = NULL;
//                }
//            }
//        }

        if (keyframeCounter.empty())
            return;

        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
             it != itEnd; it++) {
            KeyFrame *pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max) {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(it->first);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }


// Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80)
                break;

            KeyFrame *pKF = *itKF;

            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
                 itNeighKF != itEndNeighKF; itNeighKF++) {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad()) {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
                KeyFrame *pChildKF = *sit;
                if (!pChildKF->isBad()) {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            KeyFrame *pParent = pKF->GetParent();
            if (pParent) {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }

        }

        // Add 10 last temporal KFs (mainly for IMU)
        if(mSensor == System::IMU_RGBD && mvpLocalKeyFrames.size()<80)
        {
            //cout << "CurrentKF: " << mCurrentFrame.mnId << endl;
            KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

            const int Nd = 20;
            for(int i=0; i<Nd; i++){
                if (!tempKeyFrame)
                    break;
                //cout << "tempKF: " << tempKeyFrame << endl;
                if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(tempKeyFrame);
                    tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    tempKeyFrame=tempKeyFrame->mPrevKF;
                }
            }
        }

        if (pKFmax) {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    bool Tracking::Relocalization() {
        cout << "Tracking:localization" << endl;
// Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

// Relocalization is performed when tracking is lost
// Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

        cout << "Tracking,vpCandidateKFs" << vpCandidateKFs.size() << endl;
        if (vpCandidateKFs.empty())
            return false;

        const int nKFs = vpCandidateKFs.size();

// We perform first an ORB matching with each candidate
// If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);

        vector<PnPsolver *> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);

        vector<vector<MapPoint *> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0;

        for (int i = 0; i < nKFs; i++) {
            KeyFrame *pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if (nmatches < 15) {
                    vbDiscarded[i] = true;
                    continue;
                } else {
                    PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                    vpPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }
        }

// Alternatively perform some iterations of P4P RANSAC
// Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch) {
            for (int i = 0; i < nKFs; i++) {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                PnPsolver *pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (!Tcw.empty()) {
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++) {
                        if (vbInliers[j]) {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50) {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10,
                                                                      100);

                        if (nadditional + nGood >= 50) {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50) {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3,
                                                                          64);

                                // Final optimization
                                if (nGood + nadditional >= 50) {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                            }
                        }
                    }


                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50) {
                        bMatch = true;
                        break;
                    }
                }
            }

            if (!bMatch) {

            }
        }

        if (!bMatch) {
            return false;
        } else {
            mnLastRelocFrameId = mCurrentFrame.mnId;
            cout << "Relocalized!!" << endl;
            return true;
        }

    }

    void Tracking::Reset() {
        mpViewer->RequestStop();

        cout << "System Reseting" << endl;
        while (!mpViewer->isStopped())
            usleep(3000);

// Reset Local Mapping
        cout << "Reseting Local Mapper...";
        mpLocalMapper->RequestReset();
        cout << " done" << endl;

// // Reset Loop Closing
//         cout << "Reseting Loop Closing...";
//         mpLoopClosing->RequestReset();
//         cout << " done" << endl;

// Clear BoW Database
        cout << "Reseting Database...";
        mpKeyFrameDB->clear();
        cout << " done" << endl;

// Clear Map (this erase MapPoints and KeyFrames)
        mpMap->clear();

        if (mSensor==System::IMU_RGBD)
            mpMap->SetInertialSensor();

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        if (mpInitializer) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        mpViewer->Release();
    }

    void Tracking::ChangeCalibration(const string &strSettingPath) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }

    void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
    {
        // Map * pMap = pCurrentKeyFrame->GetMap();
        list<KeyFrame*>::iterator lRit = mlpReferences.begin();
        list<bool>::iterator lbL = mlbLost.begin();
        for(list<cv::Mat>::iterator lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
        {
            if(*lbL)
                continue;

            KeyFrame* pKF = *lRit;

            while(pKF->isBad())
            {
                pKF = pKF->GetParent();
            }

            // if(pKF->GetMap() == pMap)
            (*lit).rowRange(0,3).col(3)=(*lit).rowRange(0,3).col(3)*s;
        }

        mLastBias = b;

        mpLastKeyFrame = pCurrentKeyFrame;

        mLastFrame.SetNewBias(mLastBias);
        mCurrentFrame.SetNewBias(mLastBias);

        cv::Mat Gz = (cv::Mat_<float>(3,1) << 0, 0, -IMU::GRAVITY_VALUE);

        cv::Mat twb1;
        cv::Mat Rwb1;
        cv::Mat Vwb1;
        float t12;

        while(!mCurrentFrame.imuIsPreintegrated())
        {
            usleep(500);
        }


        if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
        {
            mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                        mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                        mLastFrame.mpLastKeyFrame->GetVelocity());
        }
        else
        {
            twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
            Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
            Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
            t12 = mLastFrame.mpImuPreintegrated->dT;

            mLastFrame.SetImuPoseVelocity(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                        twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                        Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }

        if (mCurrentFrame.mpImuPreintegrated)
        {
            twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
            Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
            Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
            t12 = mCurrentFrame.mpImuPreintegrated->dT;

            mCurrentFrame.SetImuPoseVelocity(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                        twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                        Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }

        mnFirstImuFrameId = mCurrentFrame.mnId;
    }

    void Tracking::InformOnlyTracking(const bool &flag) {
        mbOnlyTracking = flag;
    }

    int Tracking::GetMatchesInliers()
    {
        return mnMatchesInliers;
    }

} //namespace ORB_SLAM
