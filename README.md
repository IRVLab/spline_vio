# Continuous-Time Spline Visual-Inertial Odometry

## Related Publications
- **Direct Sparse Odometry**, J. Engel, V. Koltun, D. Cremers, In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2018
- **Continuous-Time Spline Visual-Inertial Odometry**, J. Mo and J. Sattar, under review, [arXiv](https://arxiv.org/abs/2109.09035).

## Dependencies
- [ROS](https://www.ros.org/)

- [DSO dependencies](https://github.com/JakobEngel/dso#2-installation) 

## Install
```
cd catkin_ws/src
git clone https://github.com/IRVLab/spline_vio.git
cd ..
catkin_make
```

## Usage
- Use [Kalibr toolbox](https://github.com/ethz-asl/kalibr) to calibrate camera and IMU. 

- Convert camera parameters to [DSO format](https://github.com/JakobEngel/dso#31-dataset-format).

- Create a launch file following the example of [tum.launch](https://github.com/IRVLab/spline_vio/blob/master/launch/tum.launch).

```
roslaunch spline_vio [YOUR_LAUNCH_FILE]
```

- Ctrl-C to terminate the program, the final trajectory (results.txt) will be written to ~/Desktop folder by default.

## Output file
- results.txt: poses of all frames, using the TUM RGB-D / TUM monoVO format ([timestamp x y z qx qy qz qw] of the cameraToWorld transformation).

## Modifications to DSO for this project
- ROS interface: main.cpp
- Predict pose using spine for front-end tracking: FullSystem::trackNewCoarse()
- IMU/Spline state: HessianBlocks.h
- IMU/Spline Jacobians (check [Jacobians.pdf](https://github.com/IRVLab/spline_vio/blob/main/Jacobians.pdf)): HessianBlocks::getImuHi()
- Constraints Jacobians (check [Jacobians.pdf](https://github.com/IRVLab/spline_vio/blob/main/Jacobians.pdf)): EnergyFunctional::getImuHessianCurrentFrame()
- Solve the constraint nonlinear optimization problem: EnergyFunctional::solveSystemF()
