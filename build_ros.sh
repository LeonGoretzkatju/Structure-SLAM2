echo "Building ROS nodes"

cd Examples/ROS/ManhattanSLAM
mkdir build
cd build
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/deeprealtech/SLAM/StructureVIO/Examples/ROS
cmake .. -DROS_BUILD_TYPE=Release
make -j10
