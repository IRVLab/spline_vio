# Continuous-Time Spline Visual-Inertial Odometry (position spline in IMU frame)

## Please refer to the [README](https://github.com/IRVLab/spline_vio/blob/main/README.md) in the main branch for installation and usage.

## Motivation
The original method (main branch) models the position spline in the camera frame, which introduces a system error when transforming acceleration prediction (Section II-D, IMU Synthesis) from camera frame (a(t) in w) to IMU frame (a(t) in i): there should be additional terms related to the relative translation between camera and IMU. However, since this relative translation is usually small (e.g., a few centimeters in EuRoC and TUM-VI datasets), we achieved relatively good results even with this suboptimal formulation, because the additional terms are negligible.

## Improvement
Model the position spline in the IMU frame to eliminate the system error in the original method, see [Modifications.pdf](https://github.com/IRVLab/spline_vio/blob/position_spine_in_imu_frame/Modifications.pdf) for details. However, this is not the only solution; for example, we can also modify the vision system (DSO) to accommodate poses in IMU frame. We opt for this formulation since it requires minimal code modification.

## Results
No significant improvement on EuRoC and TUM-VI datasets (since the distance between camera and IMU is small, the resulting system error is small):
| Method    | EuRoC Machine Hall            | EuRoC Vicon Room        | TUM-VI Room                     |
| --------- | ----------------------------- | ----------------------- | ------------------------------- |
| Original  | 0.066 0.056 0.142 0.131 0.129 | 0.087 x x	0.103 0.111	x | 0.085 0.186	0.114 0.142	x 0.137 |
| Modified  | 0.088 0.074 0.148 0.325 0.117 | 0.084	x x 0.062 0.152	x | 0.076 0.141	0.118 0.132	x 0.080 |