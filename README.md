# Object Tracking

## EfficientTAM Setup Guide

### Installation Steps

1. **Create and activate a dedicated Conda environment**:
   ```bash
   conda env create -f etam_conda_env.yml -n etam
   conda activate etam
   ```

2. **Clone the repository**:
   ```bash
   git clone git@github.com:IIIIQIIII/etam_kalman.git etam
   cd etam
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

## UCL-follow-anything Setup

### Installation Steps

1. **Prepare your environment**:
   ```bash
   conda deactivate  # Ensure no conda environment is active
   cd ~  # Return to home directory
   ```

2. **Create workspace and clone repository**:
   ```bash
   mkdir go2_ws && cd go2_ws
   git clone https://github.com/UCL-follow-anything/follow-anything.git src
   ```
   
   This will create the following structure:
   ```
   go2_ws/src
   ```

3. **Set up ROS environment and build Livox driver**:
   ```bash
   source /opt/ros/humble/setup.sh
   cd src/livox_ros_driver2 && ./build.sh humble
   ```

4. **Build the workspace**:
   ```bash
   cd ~/go2_ws && colcon build
   ```

5. **Source the setup file**:
   ```bash
   source install/setup.bash
   ```

6. **Test your installation**:
   ```bash
   ros2 launch go2_config gazebo_mid360.launch.py rviz:=true
   ```
   This will verify that the program is running normally.

## Depth Alignment Package Setup

### Installation Steps

1. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install ros-humble-depth-image-proc ros-humble-cv-bridge ros-humble-image-transport ros-humble-image-transport-plugins
   ```

2. **Clone the repository**:
   ```bash
   cd ~/go2_ws/src
   git clone https://github.com/IIIIQIIII/depth_align_pkg.git
   ```

3. **Build the package**:
   ```bash
   cd ~/go2_ws && colcon build --packages-select depth_align_pkg
   ```

4. **Source the setup file**:
   ```bash
   source install/setup.bash
   ```

## Tracking Package Setup

### Installation Steps

1. **Install required dependencies**:
   ```bash
   sudo apt install -y ros-humble-rclpy ros-humble-std-msgs ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-nav-msgs
   sudo apt install -y ros-humble-cv-bridge
   sudo apt install -y ros-humble-tf2-ros ros-humble-tf-transformations
   ```

2. **Clone the tracking repository**:
   ```bash
   git clone git@github.com:IIIIQIIII/tracking.git tracking
   ```

3. **Build the tracking package**:
   ```bash
   cd ~/go2_ws && python -m colcon build --packages-select tracking
   ```

4. **Source the setup file**:
   ```bash
   source install/setup.bash
   ```

### Running the Tracking System

To run the complete tracking system, open three separate terminals and run each of the following commands:

1. **Terminal 1: Launch Gazebo and depth alignment**:
   ```bash
   ros2 launch tracking gazebo_depth_align.launch.py rviz:=false
   ```

2. **Terminal 2: Run interactive segmentation**:
   ```bash
   ros2 run tracking interactive_segmentation
   ```

3. **Terminal 3: Run histogram-based Kalman tracking**:
   ```bash
   ros2 run tracking hist_kalman_tracking
   ```