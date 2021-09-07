// Copyright (C) <2020> <Jiawei Mo, Junaed Sattar>

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// This file is modified from <https://github.com/JakobEngel/dso>

#pragma once

#include "IOWrapper/Output3DWrapper.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/NumType.h"
#include "util/settings.h"

#include <math.h>
#include <vector>

namespace dso {
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

template <int b, typename T>
T *allocAligned(int size, std::vector<T *> &rawPtrVec) {
  const int padT = 1 + ((1 << b) / sizeof(T));
  T *ptr = new T[size + padT];
  rawPtrVec.push_back(ptr);
  T *alignedPtr = (T *)((((uintptr_t)(ptr + padT)) >> b) << b);
  return alignedPtr;
}

class CoarseTracker {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CoarseTracker(int w, int h);
  ~CoarseTracker();

  void makeK(CalibHessian *HCalib);

  void setCoarseTrackingRef(std::vector<FrameHessian *> frameHessians);

  void scaleCoarseDepthL0(float scale);

  void debugPlotIDepthMap(float *minID, float *maxID,
                          std::vector<IOWrap::Output3DWrapper *> &wraps);
  void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper *> &wraps);

  bool trackNewestCoarse(FrameHessian *newFrameHessian, SE3 &lastToNew_out,
                         AffLight &aff_g2l_out, int coarsestLvl,
                         Vec5 minResForAbort, Vec5 &lastResiduals,
                         IOWrap::Output3DWrapper *wrap = 0);

  // act as pure ouptut
  int refFrameID;
  FrameHessian *lastRef;
  AffLight lastRef_aff_g2l;
  Vec3 lastFlowIndicators;
  double firstCoarseRMSE;

private:
  void makeCoarseDepthL0(std::vector<FrameHessian *> frameHessians);

  Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH,
               bool plot_img = false);
  void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew,
                 AffLight aff_g2l);

  std::vector<float *> ptrToDelete;

  float *idepth_[PYR_LEVELS];
  float *weight_sums_[PYR_LEVELS];
  float *weight_sums_bak_[PYR_LEVELS];

  Mat33f Ki_[PYR_LEVELS];
  float fx_[PYR_LEVELS];
  float fy_[PYR_LEVELS];
  float cx_[PYR_LEVELS];
  float cy_[PYR_LEVELS];
  int w_[PYR_LEVELS];
  int h_[PYR_LEVELS];

  // pc buffers
  float *pc_u_[PYR_LEVELS];
  float *pc_v_[PYR_LEVELS];
  float *pc_idepth_[PYR_LEVELS];
  float *pc_color_[PYR_LEVELS];
  int pc_n_[PYR_LEVELS];

  // warped buffers
  float *poseBufWarped_idepth;
  float *poseBufWarped_u;
  float *poseBufWarped_v;
  float *poseBufWarped_dx;
  float *poseBufWarped_dy;
  float *poseBufWarped_residual;
  float *poseBufWarped_weight;
  float *poseBufWarped_refColor;
  int poseBufWarped_n;

  FrameHessian *newFrame;
  Accumulator9 poseAcc;
};

class CoarseDistanceMap {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Mat33f K[PYR_LEVELS];
  Mat33f Ki[PYR_LEVELS];

  CoarseDistanceMap(int w, int h);
  ~CoarseDistanceMap();

  void makeDistanceMap(std::vector<FrameHessian *> frameHessians,
                       FrameHessian *frame);

  void makeInlierVotes(std::vector<FrameHessian *> frameHessians);

  void makeK(CalibHessian *HCalib);

  float *fwdWarpedIDDistFinal;

  void addIntoDistFinal(int u, int v);

private:
  int w_[PYR_LEVELS];
  int h_[PYR_LEVELS];

  PointFrameResidual **coarseProjectionGrid;
  int *coarseProjectionGridnum;
  Eigen::Vector2i *bfsList1;
  Eigen::Vector2i *bfsList2;

  void growDistBFS(int bfsNum);
};

} // namespace dso
