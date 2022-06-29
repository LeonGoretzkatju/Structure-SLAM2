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



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <time.h>
#include <pcl/io/ply_io.h>
// #include <tinyply.h>

// using namespace tinyply;
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

bool has_suffix(const std::string &str, const std::string &suffix) {
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);
}
namespace ORB_SLAM2
{

Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NORMAL;

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)),mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;
    else if(mSensor==IMU_RGBD)
        cout << "IMU RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }

    // TO DO
    //float resolution = fsSettings["PointCloudMapping.Resolution"];
    //float resolution = 0.01;

    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
    clock_t tStart = clock();
    mpVocabulary = new ORBVocabulary();
    bool bVocLoad  = mpVocabulary->loadFromTextFile(strVocFile);
	//else
	 //   bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
    if (!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Failed to open at: " << strVocFile << endl;
        exit(-1);
    }
    //printf("Vocabulary loaded in %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    cout<<"Vocabulary loaded!"<<endl<<endl;
    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();

    if (mSensor == IMU_RGBD)
        mpMap->SetInertialSensor();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    // TO DO
    //mpPointCloudMapping = make_shared<PointCloudMapping>( resolution );

    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR, mSensor == IMU_RGBD);
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);

    //Initialize the Surfel Mapping thread and launch
    mpSurfelMapper = new SurfelMapping(mpMap, strSettingsFile);
    mptSurfelMapping = new thread(&ORB_SLAM2::SurfelMapping::Run, mpSurfelMapper);

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    ///mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

    //Initialize the Viewer thread and launch
    mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }




    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetSurfelMapper(mpSurfelMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}


cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point>& vImuMeas)
{
    if(mSensor!=RGBD && mSensor!=IMU_RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD nor RGBD-Inertial." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
        }
    }

    if (mSensor == System::IMU_RGBD)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depthmap, timestamp);

    unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}


void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}
    bool System::MapChanged()
    {
        static int n=0;
        int curn = mpMap->GetLastBigChangeIdx();
        if(n<curn)
        {
            n=curn;
            return true;
        }
        else
            return false;
    }

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();

    // pcl::PointCloud<pcl::PointSurfel>::Ptr pointCloud = mpSurfelMapper->Stop();
    // saveSurfels(pointCloud);

    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);

    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished()|| mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }
    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;
        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        //ss<<setprecision(6) <<pKF->mTimeStamp;
        //ss>>filename;
        //cout<<filename<<endl;
        /*
        ofstream Pmatrix("results/"+to_string(pKF->mTimeStamp)+".P");
        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        //转换成P矩阵
        float P_00=R.at<float>(0,0)*pKF->fx + R.at<float>(2,0)*pKF->cx;
        float P_01=R.at<float>(0,1)*pKF->fx+R.at<float>( 2,1 )*pKF->cx;
        float P_02=R.at<float>(0,2)*pKF->fx+R.at<float> ( 2,2 )*pKF->cx;
        float P_03= t.at<float>(0) *pKF->fx+ t.at<float>(2) *pKF->cx;

        float P_10=R.at<float>(1,0)*pKF->fy+R.at<float>( 2,0 )*pKF->cy;
        float P_11=R.at<float>(1,1)*pKF->fy+R.at<float>( 2,1 )*pKF->cy;
        float P_12=R.at<float>(1,2)*pKF->fy+R.at<float> ( 2,2 )*pKF->cy;
        float P_13=t.at<float>(1)*pKF->fy+ t.at<float>(2) *pKF->cy;

        float P_20=R.at<float>(2,0);
        float P_21=R.at<float>(2,1);
        float P_22=R.at<float>(2,2);
        float P_23=t.at<float>(2);

        Pmatrix<< P_00<< " "<<P_01<<" "<<P_02<<" "<<P_03<<endl;
        Pmatrix<< P_10<< " "<<P_11<<" "<<P_12<<" "<<P_13<<endl;
        Pmatrix<< P_20<< " "<<P_21<<" "<<P_22<<" "<<P_23<<endl;
        Pmatrix.close();
        */
    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}
    int System::GetTrackingState()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackingState;
    }

    vector<MapPoint*> System::GetTrackedMapPoints()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackedMapPoints;
    }

    vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackedKeyPointsUn;
    }

    Map* System::getMap() {
        unique_lock<mutex> lock(mMutexState);
        return mpMap;
    }

    struct float2 { float x, y; };
    struct float3 { float x, y, z; };
    struct float17 { float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q; };
    struct uchar3 { uint8_t x, y, z; };
    struct ushort1 { uint16_t x; };
    struct int2 { int x, y; };

    void System::saveSurfels(pcl::PointCloud<pcl::PointSurfel>::Ptr pointCloud) {
        std::filebuf fb_ascii;
        fb_ascii.open("Surfels.ply", std::ios::out);
        std::ostream outstream_ascii(&fb_ascii);
        if (outstream_ascii.fail()) throw std::runtime_error("failed to open Surfels.ply");

        auto &points = pointCloud->points;

        std::vector<float3> vertices, normals;
        std::vector<ushort1> labels;
        std::vector<uchar3> colours;
        std::vector<float2> props;

        for (auto &point : points) {
            if (std::isnan(point.x))
                continue;
            vertices.push_back({point.x, point.y, point.z});
            labels.push_back({1});
            normals.push_back({point.normal_x, point.normal_y, point.normal_z});
            colours.push_back({point.r, point.g, point.b});
            props.push_back({point.confidence, point.radius});
        }

        size_t pointCloudSize = vertices.size();

        std::vector<float17> cameraProp1;
        std::vector<int2> cameraProp2;
        std::vector<float2> cameraProp3;
        cameraProp1.push_back({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0});
        cameraProp2.push_back({(int) pointCloudSize, 1});
        cameraProp3.push_back({0, 0});

//         PlyFile cube_file;

//         cube_file.add_properties_to_element("vertex", { "x", "y", "z" },
//                                             Type::FLOAT32, pointCloudSize, reinterpret_cast<uint8_t*>(vertices.data()), Type::INVALID, 0);

// //        cube_file.add_properties_to_element("vertex", { "label" },
// //                                            Type::UINT16, pointCloudSize, reinterpret_cast<uint8_t*>(labels.data()), Type::INVALID, 0);

//         cube_file.add_properties_to_element("vertex", { "nx", "ny", "nz" },
//                                             Type::FLOAT32, pointCloudSize, reinterpret_cast<uint8_t*>(normals.data()), Type::INVALID, 0);

//         cube_file.add_properties_to_element("vertex", { "red", "green", "blue" },
//                                             Type::UINT8, pointCloudSize, reinterpret_cast<uint8_t*>(colours.data()), Type::INVALID, 0);

//         cube_file.add_properties_to_element("vertex", { "alpha" },
//                                             Type::UINT8, pointCloudSize, reinterpret_cast<uint8_t*>(labels.data()), Type::INVALID, 0);

//         cube_file.add_properties_to_element("vertex", { "quality", "radius" },
//                                             Type::FLOAT32, pointCloudSize, reinterpret_cast<uint8_t*>(props.data()), Type::INVALID, 0);

//         cube_file.add_properties_to_element("camera", { "view_px",
//                                                         "view_py",
//                                                         "view_pz",
//                                                         "x_axisx",
//                                                         "x_axisy",
//                                                         "x_axisz",
//                                                         "y_axisx",
//                                                         "y_axisy",
//                                                         "y_axisz",
//                                                         "z_axisx",
//                                                         "z_axisy",
//                                                         "z_axisz",
//                                                         "focal",
//                                                         "scalex",
//                                                         "scaley",
//                                                         "centerx",
//                                                         "centery" },
//                                             Type::FLOAT32, 1, reinterpret_cast<uint8_t*>(cameraProp1.data()), Type::INVALID, 0);

//         cube_file.add_properties_to_element("camera", { "viewportx",
//                                                         "viewporty" },
//                                             Type::INT32, 1, reinterpret_cast<uint8_t*>(cameraProp2.data()), Type::INVALID, 0);

//         cube_file.add_properties_to_element("camera", { "k1",
//                                                         "k2" },
//                                             Type::FLOAT32, 1, reinterpret_cast<uint8_t*>(cameraProp3.data()), Type::INVALID, 0);

//         cube_file.write(outstream_ascii, false);

    }

} //namespace ORB_SLAM
