/**
 * Deal with RGB-D-Inertial sequences
 *
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>
#include<System.h>
#include<Mesh.h>
#include<MapPlane.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

void LoadAccAndGyro(const string &strAccPath, const string &strGyroPath,
                    vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_inertial path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    vector<cv::Point3f> vAcc, vGyro;
    vector<double> vTimestampsImu;
    int nImu, first_imu = 0;

    string strAssociationFilename = string(argv[4]);
    string pathImu = string(argv[3]) + "/IMU.txt";
    string pathAcc = string(argv[3]) + "/d400_accelerometer.txt";
    cout<<"pathAcc: "<<pathAcc<<endl;

    string pathGyro = string(argv[3]) + "/d400_gyroscope.txt";
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);
    LoadIMU(pathImu, vTimestampsImu, vAcc, vGyro);
    //LoadAccAndGyro(pathAcc, pathGyro, vTimestampsImu, vAcc, vGyro);

    // Check consistency in the number of images, depthmaps, IMU data
    int nImages = vstrImageFilenamesRGB.size();
    nImu = vTimestampsImu.size();

    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }
    else if(nImu<=0)
    {
        cerr << endl << "Failed to load IMU" << endl;
        return 1;
    }

     // Find first imu to be considered, supposing imu measurements start first
    while(vTimestampsImu[first_imu]<=vTimestamps[0])
        first_imu++;
    first_imu--; // first imu measurement to be considered
    //cout<<vTimestampsImu[first_imu]<<", image: "<<vTimestamps[0]<<endl;
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::IMU_RGBD,true);
    ORB_SLAM2::Config::SetParameterFile(argv[2]);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    // Main loop
    cv::Mat imRGB, imD;
    vector<ORB_SLAM2::IMU::Point> vImuMeas;

    for(int ni=0; ni<nImages; ni++)
    {
        cout<<"************ the "<<ni<<"th image"<<endl;
        cout<< vstrImageFilenamesRGB[ni] << endl;
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+ vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+ vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];
     
        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        // Load imu measurements from previous frame
        vImuMeas.clear();

        if(ni>0)
            {
                // cout << "t_cam " << tframe << endl;

                while(vTimestampsImu[first_imu]<=vTimestamps[ni])
                {
                    vImuMeas.push_back(ORB_SLAM2::IMU::Point(vAcc[first_imu].x,vAcc[first_imu].y,vAcc[first_imu].z,
                                                             vGyro[first_imu].x,vGyro[first_imu].y,vGyro[first_imu].z,
                                                             vTimestampsImu[first_imu]));
                    first_imu++;
                }
            }


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image and IMUs to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe, vImuMeas);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        cout<<"track time: "<<T<<", "<<ttrack<<endl;
        if(ttrack<T)
            usleep((T-ttrack)*1e7);
    }
    char bStop;

    cout << "please type 'x', if you want to shutdown the system." << endl;

    while (bStop != 'x'){
        bStop = getchar();
    }
    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}

void LoadAccAndGyro(const string &strAccPath, const string &strGyroPath,
                    vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fAcc;
    ifstream fGyro;
    fAcc.open(strAccPath.c_str());
    fGyro.open(strGyroPath.c_str());
    vTimeStamps.reserve(10300);
    vAcc.reserve(10300);
    vGyro.reserve(10300);

}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{

    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(10300);
    vAcc.reserve(10300);
    vGyro.reserve(10300);

    while(!fImu.eof())
    {
        string s;
        getline(fImu,s);
        if (s[0] == '#')
            continue;

        if(!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);

            vTimeStamps.push_back(data[0]/1e6); // µs 
            vAcc.push_back(cv::Point3f(data[4],data[5],data[6]));
            vGyro.push_back(cv::Point3f(data[1],data[2],data[3]));
        }
    }
}
