//
// Created by raza on 01.10.20.
//

#include "SurfelMapping.h"

namespace ORB_SLAM2 {
    SurfelMapping::SurfelMapping(Map *map, const string &strSettingPath) : mMap(map),
    mbStop(false), drift_free_poses(10) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        int imgWidth = fSettings["Camera.width"];
        int imgHeight = fSettings["Camera.height"];

        float distanceFar = fSettings["Surfel.distanceFar"];
        float distanceNear = fSettings["Surfel.distanceNear"];

        mSurfelFusion.initialize(imgWidth, imgHeight, fx, fy, cx, cy, distanceFar, distanceNear);
    }

    void SurfelMapping::Run() {
        while(true) {
            if (CheckNewKeyFrames()) {
                ProcessNewKeyFrame();
            }

            {
                unique_lock<mutex> lock(mMutexStop);
                if (mbStop) {
                    mbStop = false;
                    break;
                }
            }
        }
    }

    pcl::PointCloud<pcl::PointSurfel>::Ptr SurfelMapping::Stop() {
        unique_lock<mutex> lock(mMutexStop);
        mbStop = true;

        pcl::PointCloud<pcl::PointSurfel>::Ptr pointCloud(new pcl::PointCloud<pcl::PointSurfel>());
        for(int surfel_it = 0, surfel_end = mMap->mvLocalSurfels.size(); surfel_it < surfel_end; surfel_it++)
        {
            if(mMap->mvLocalSurfels[surfel_it].update_times < 5)
                continue;
            pcl::PointSurfel p;
            p.x = mMap->mvLocalSurfels[surfel_it].px;
            p.y = mMap->mvLocalSurfels[surfel_it].py;
            p.z = mMap->mvLocalSurfels[surfel_it].pz;
            p.r = mMap->mvLocalSurfels[surfel_it].r;
            p.g = mMap->mvLocalSurfels[surfel_it].g;
            p.b = mMap->mvLocalSurfels[surfel_it].b;
            p.normal_x = mMap->mvLocalSurfels[surfel_it].nx;
            p.normal_y = mMap->mvLocalSurfels[surfel_it].ny;
            p.normal_z = mMap->mvLocalSurfels[surfel_it].nz;
            p.radius = mMap->mvLocalSurfels[surfel_it].size * 1000;
            p.confidence = mMap->mvLocalSurfels[surfel_it].weight;
//            p.intensity = mMap->mvLocalSurfels[surfel_it].color;
            pointCloud->push_back(p);
        }

        for(int surfel_it = 0, surfel_end = mMap->mvInactiveSurfels.size(); surfel_it < surfel_end; surfel_it++)
        {
//            if(mMap->mvInactiveSurfels[surfel_it].update_times < 5)
//                continue;
            pcl::PointSurfel p;
            p.x = mMap->mvInactiveSurfels[surfel_it].px;
            p.y = mMap->mvInactiveSurfels[surfel_it].py;
            p.z = mMap->mvInactiveSurfels[surfel_it].pz;
            p.r = mMap->mvInactiveSurfels[surfel_it].r;
            p.g = mMap->mvInactiveSurfels[surfel_it].g;
            p.b = mMap->mvInactiveSurfels[surfel_it].b;
            p.normal_x = mMap->mvInactiveSurfels[surfel_it].nx;
            p.normal_y = mMap->mvInactiveSurfels[surfel_it].ny;
            p.normal_z = mMap->mvInactiveSurfels[surfel_it].nz;
            p.radius = mMap->mvInactiveSurfels[surfel_it].size * 1000;
            p.confidence = mMap->mvInactiveSurfels[surfel_it].weight;
//            p.intensity = mMap->mvLocalSurfels[surfel_it].color;
            pointCloud->push_back(p);
        }

        std::vector<ORB_SLAM2::MapPlane*> mapPlanes = mMap->GetAllMapPlanes();
        double radius = 0.1414 * 1000;

        for(auto pMP : mapPlanes) {
            auto &planePoints = pMP->mvPlanePoints->points;
            cv::Mat P3Dw = pMP->GetWorldPos();

            for (auto &planePoint : planePoints) {
                pcl::PointSurfel p;
                p.x = planePoint.x;
                p.y = planePoint.y;
                p.z = planePoint.z;
                p.r = planePoint.r;
                p.g = planePoint.g;
                p.b = planePoint.b;

                p.normal_x = P3Dw.at<float>(0);
                p.normal_y = P3Dw.at<float>(1);
                p.normal_z = P3Dw.at<float>(2);
                p.radius = radius;
                p.confidence = 1;
//                p.intensity = pMP->mRed + pMP->mBlue + pMP->mGreen / 255*3;

                pointCloud->push_back(p);
            }
        }

//        (*pointCloud) += (*inactive_pointcloud);

        return pointCloud;
    }

    void SurfelMapping::InsertKeyFrame(const cv::Mat &imRGB, const cv::Mat &imDepth, const cv::Mat planeMembershipImg, const cv::Mat& pose, const int referenceIndex, const bool isNewKeyFrame) {
        unique_lock<mutex> lock(mMutexNewKFs);
        mlNewKeyFrames.emplace_back(imRGB, imDepth, planeMembershipImg, pose, referenceIndex, isNewKeyFrame);
    }

    bool SurfelMapping::CheckNewKeyFrames() {
        unique_lock<mutex> lock(mMutexNewKFs);
        return (!mlNewKeyFrames.empty());
    }

    void SurfelMapping::ProcessNewKeyFrame() {
        std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat, int, bool> frame;
        {
            unique_lock<mutex> lock(mMutexNewKFs);
            frame = mlNewKeyFrames.front();
            mlNewKeyFrames.pop_front();
        }

        cv::Mat image = std::get<0>(frame);
        cv::Mat depth = std::get<1>(frame);
        cv::Mat planeMembershipImg = std::get<2>(frame);
        cv::Mat pose = std::get<3>(frame);
        int relativeIndex = std::get<4>(frame);
        bool isKeyFrame = std::get<5>(frame);

        if(isKeyFrame)
        {
            PoseElement poseElement;
            int index = poses_database.size();
            if(!poses_database.empty())
            {
                poseElement.linked_pose_index.push_back(relativeIndex);
                poses_database[relativeIndex].linked_pose_index.push_back(index);
            }
            poses_database.push_back(poseElement);
            local_surfels_indexs.insert(index);
        }

        move_add_surfels(relativeIndex);

        Eigen::Matrix4f poseEigen = Eigen::Matrix4f::Zero();
        poseEigen(0,0) = pose.at<float>(0,0);
        poseEigen(1,0) = pose.at<float>(1,0);
        poseEigen(2,0) = pose.at<float>(2,0);
        poseEigen(3,0) = pose.at<float>(3,0);
        poseEigen(0,1) = pose.at<float>(0,1);
        poseEigen(1,1) = pose.at<float>(1,1);
        poseEigen(2,1) = pose.at<float>(2,1);
        poseEigen(3,1) = pose.at<float>(3,1);
        poseEigen(0,2) = pose.at<float>(0,2);
        poseEigen(1,2) = pose.at<float>(1,2);
        poseEigen(2,2) = pose.at<float>(2,2);
        poseEigen(3,2) = pose.at<float>(3,2);
        poseEigen(0,3) = pose.at<float>(0,3);
        poseEigen(1,3) = pose.at<float>(1,3);
        poseEigen(2,3) = pose.at<float>(2,3);
        poseEigen(3,3) = pose.at<float>(3,3);

        fuse_map(image, depth, planeMembershipImg, poseEigen, relativeIndex);
    }

    void SurfelMapping::move_add_surfels(int reference_index)
    {
        // remove inactive surfels
        printf("get inactive surfels for pose %d.\n", reference_index);
        // vector<int> drift_poses;
        vector<int> poses_to_add;
        vector<int> poses_to_remove;
        get_add_remove_poses(reference_index, poses_to_add, poses_to_remove);
        std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
        std::chrono::duration<double> move_pointcloud_time;

        if(poses_to_remove.size() > 0)
        {

            start_time = std::chrono::system_clock::now();
            int added_surfel_num = 0;
            float sum_update_times = 0.0;
            for(int inactive_index : poses_to_remove)
            {
                poses_database[inactive_index].points_begin_index = mMap->mvInactiveSurfels.size();
                poses_database[inactive_index].points_pose_index = pointcloud_pose_index.size();
                pointcloud_pose_index.push_back(inactive_index);
                for(auto &local_surfel : mMap->mvLocalSurfels)
                {
                    if(local_surfel.update_times > 0 && local_surfel.last_update == inactive_index)
                    {
                        poses_database[inactive_index].attached_surfels.push_back(local_surfel);

                        PointType p;
                        p.x = local_surfel.px;
                        p.y = local_surfel.py;
                        p.z = local_surfel.pz;
//                        p.intensity = local_surfel.color;
                        mMap->mvInactiveSurfels.push_back(local_surfel);

                        added_surfel_num += 1;
                        sum_update_times += local_surfel.update_times;

                        // delete the surfel from the local point
                        local_surfel.update_times = 0;
                    }
                }
                printf("remove pose %d from local poses, get %d surfels.\n", inactive_index, poses_database[inactive_index].attached_surfels.size());
                local_surfels_indexs.erase(inactive_index);
            }
            sum_update_times = sum_update_times / added_surfel_num;
            end_time = std::chrono::system_clock::now();
            move_pointcloud_time = end_time - start_time;
            printf("move surfels cost %f ms. the average update times is %f.\n", move_pointcloud_time.count()*1000.0, sum_update_times);
        }
        if(poses_to_add.size() > 0)
        {
            // 1.0 add indexs
            local_surfels_indexs.insert(poses_to_add.begin(), poses_to_add.end());
            // 2.0 add surfels
            // 2.1 remove from inactive_pointcloud
            start_time = std::chrono::system_clock::now();
            std::vector<std::pair<int, int>> remove_info;//first, pointcloud start, pointcloud size, pointcloud pose index
            for(int add_i = 0; add_i < poses_to_add.size(); add_i++)
            {
                int add_index = poses_to_add[add_i];
                int pointcloud_pose_index = poses_database[add_index].points_pose_index;
                remove_info.push_back(std::make_pair(pointcloud_pose_index, add_index));
            }
            std::sort(
                    remove_info.begin(),
                    remove_info.end(),
                    []( const std::pair<int, int >& first, const std::pair<int, int>& second)
                    {
                        return first.first < second.first;
                    }
            );
            int remove_begin_index = remove_info[0].second;
            int remove_points_size = poses_database[remove_begin_index].attached_surfels.size();
            int remove_pose_size = 1;
            for(int remove_i = 1; remove_i <= remove_info.size(); remove_i++)
            {
                bool need_remove = false;
                if(remove_i == remove_info.size())
                    need_remove = true;
                if(remove_i < remove_info.size())
                {
                    if(remove_info[remove_i].first != (remove_info[remove_i-1].first + 1))
                        need_remove = true;
                }
                if(!need_remove)
                {
                    int this_pose_index = remove_info[remove_i].second;
                    remove_points_size += poses_database[this_pose_index].attached_surfels.size();
                    remove_pose_size += 1;
                    continue;
                }

                int remove_end_index = remove_info[remove_i - 1].second;
                printf("remove from pose %d -> %d, has %d points\n", remove_begin_index, remove_end_index, remove_points_size);

                vector<SurfelElement>::iterator begin_ptr;
                vector<SurfelElement>::iterator end_ptr;
                begin_ptr = mMap->mvInactiveSurfels.begin() + poses_database[remove_begin_index].points_begin_index;
                end_ptr = begin_ptr + remove_points_size;
                mMap->mvInactiveSurfels.erase(begin_ptr, end_ptr);

                for(int pi = poses_database[remove_end_index].points_pose_index + 1; pi < pointcloud_pose_index.size(); pi++)
                {
                    poses_database[pointcloud_pose_index[pi]].points_begin_index -= remove_points_size;
                    poses_database[pointcloud_pose_index[pi]].points_pose_index -= remove_pose_size;
                }

                pointcloud_pose_index.erase(
                        pointcloud_pose_index.begin() + poses_database[remove_begin_index].points_pose_index,
                        pointcloud_pose_index.begin() + poses_database[remove_end_index].points_pose_index + 1
                );


                if(remove_i < remove_info.size())
                {
                    remove_begin_index = remove_info[remove_i].second;;
                    remove_points_size = poses_database[remove_begin_index].attached_surfels.size();
                    remove_pose_size = 1;
                }
            }

            // 2.3 add the surfels into local
            for(int pi = 0; pi < poses_to_add.size(); pi++)
            {
                int pose_index = poses_to_add[pi];
                mMap->mvLocalSurfels.insert(
                        mMap->mvLocalSurfels.end(),
                        poses_database[pose_index].attached_surfels.begin(),
                        poses_database[pose_index].attached_surfels.end());
                poses_database[pose_index].attached_surfels.clear();
                poses_database[pose_index].points_begin_index = -1;
                poses_database[pose_index].points_pose_index = -1;
            }
            end_time = std::chrono::system_clock::now();
            move_pointcloud_time = end_time - start_time;
            printf("add surfels cost %f ms.\n", move_pointcloud_time.count()*1000.0);
        }
    }

    void SurfelMapping::get_add_remove_poses(int root_index, vector<int> &pose_to_add, vector<int> &pose_to_remove)
    {
        vector<int> driftfree_poses;
        get_driftfree_poses(root_index, driftfree_poses, drift_free_poses);
        {
            printf("\ndriftfree poses: ");
            for(int i = 0; i < driftfree_poses.size(); i++)
            {
                printf("%d, ", driftfree_poses[i]);
            }
        }
        pose_to_add.clear();
        pose_to_remove.clear();
        // get to add
        for(int i = 0; i < driftfree_poses.size(); i++)
        {
            int temp_pose = driftfree_poses[i];
            if(local_surfels_indexs.find(temp_pose) == local_surfels_indexs.end())
                pose_to_add.push_back(temp_pose);
        }
        {
            printf("\nto add: ");
            for(int i = 0; i < pose_to_add.size(); i++)
            {
                printf("%d, ", pose_to_add[i]);
            }
        }
        // get to remove
        for(auto i = local_surfels_indexs.begin(); i != local_surfels_indexs.end(); i++)
        {
            int temp_pose = *i;
            if( std::find(driftfree_poses.begin(), driftfree_poses.end(), temp_pose) ==  driftfree_poses.end() )
            {
                pose_to_remove.push_back(temp_pose);
            }
        }
        {
            printf("\nto remove: ");
            for(int i = 0; i < pose_to_remove.size(); i++)
            {
                printf("%d, ", pose_to_remove[i]);
            }
            printf("\n");
        }
    }

    void SurfelMapping::get_driftfree_poses(int root_index, vector<int> &driftfree_poses, int driftfree_range)
    {
        if(poses_database.size() < root_index + 1)
        {
            printf("get_driftfree_poses: pose database do not have the root index! This should only happen in initializaion!\n");
            return;
        }
        vector<int> this_level;
        vector<int> next_level;
        this_level.push_back(root_index);
        driftfree_poses.push_back(root_index);
        // get the drift
        for(int i = 1; i < driftfree_range; i++)
        {
            for(auto this_it = this_level.begin(); this_it != this_level.end(); this_it++)
            {
                for(auto linked_it = poses_database[*this_it].linked_pose_index.begin();
                    linked_it != poses_database[*this_it].linked_pose_index.end();
                    linked_it++)
                {
                    bool already_saved = (find(driftfree_poses.begin(), driftfree_poses.end(), *linked_it) != driftfree_poses.end());
                    if(!already_saved)
                    {
                        next_level.push_back(*linked_it);
                        driftfree_poses.push_back(*linked_it);
                    }
                }
            }
            this_level.swap(next_level);
            next_level.clear();
        }
    }

    void SurfelMapping::fuse_map(cv::Mat image, cv::Mat depth, cv::Mat planeMembershipImg, const Eigen::Matrix4f& pose_input, int reference_index)
    {
        printf("fuse surfels with reference index %d and %d surfels!\n", reference_index, mMap->mvLocalSurfels.size());

        vector<SurfelElement> new_surfels;
        mSurfelFusion.fuse_initialize_map(
                reference_index,
                image,
                depth,
                planeMembershipImg,
                pose_input,
                mMap->mvLocalSurfels,
                new_surfels
        );

        // get the deleted surfel index
        vector<int> deleted_index;
        for(int i = 0; i < mMap->mvLocalSurfels.size(); i++)
        {
            if(mMap->mvLocalSurfels[i].update_times == 0)
                deleted_index.push_back(i);
        }

        // add new initialized surfels
        int add_surfel_num = 0;
        for(int i = 0; i < new_surfels.size(); i++)
        {
            if(new_surfels[i].update_times != 0)
            {
                SurfelElement this_surfel = new_surfels[i];
                if(deleted_index.size() > 0)
                {
                    mMap->mvLocalSurfels[deleted_index.back()] = this_surfel;
                    deleted_index.pop_back();
                }
                else
                    mMap->mvLocalSurfels.push_back(this_surfel);
                add_surfel_num += 1;
            }
        }
        // remove deleted surfels
        while(deleted_index.size() > 0)
        {
            mMap->mvLocalSurfels[deleted_index.back()] = mMap->mvLocalSurfels.back();
            deleted_index.pop_back();
            mMap->mvLocalSurfels.pop_back();
        }

        printf("add %d surfels, we now have %d local surfels.\n", add_surfel_num, mMap->mvLocalSurfels.size());
    }
}