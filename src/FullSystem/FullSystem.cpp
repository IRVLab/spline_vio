/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"
#include "ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "PixelSelector.h"
#include "PixelSelector2.h"
#include "ResidualProjections.h"
#include "stdio.h"
#include "util/ImageAndExposure.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

#include <chrono>

namespace dso {
int FrameHessian::instanceCounter = 0;
int PointHessian::instanceCounter = 0;
int CalibHessian::instanceCounter = 0;

FullSystem::FullSystem() {
  selectionMap = new float[wG[0] * hG[0]];

  coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
  coarse_tracker_ = new CoarseTracker(wG[0], hG[0]);
  coarse_tracker_for_new_kf_ = new CoarseTracker(wG[0], hG[0]);
  coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
  pixelSelector = new PixelSelector(wG[0], hG[0]);

  statistics_lastNumOptIts = 0;
  statistics_numDroppedPoints = 0;
  statistics_numActivatedPoints = 0;
  statistics_numCreatedPoints = 0;
  statistics_numForceDroppedResBwd = 0;
  statistics_numForceDroppedResFwd = 0;
  statistics_numMargResFwd = 0;
  statistics_numMargResBwd = 0;

  lastCoarseRMSE.setConstant(100);

  currentMinActDist = 2;
  initialized = false;

  ef = new EnergyFunctional();
  ef->red = &this->treadReduce;

  isLost = false;
  initFailed = false;

  minIdJetVisDebug = -1;
  maxIdJetVisDebug = -1;
  minIdJetVisTracker = -1;
  maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem() {
  delete[] selectionMap;

  for (FrameShell *s : allFrameHistory)
    delete s;

  delete coarseDistanceMap;
  delete coarse_tracker_;
  delete coarse_tracker_for_new_kf_;
  delete coarseInitializer;
  delete pixelSelector;
  delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW,
                                  int originalH) {}

void FullSystem::setGammaFunction(float *BInv) {
  if (BInv == 0)
    return;

  // copy BInv.
  memcpy(HCalib.Binv, BInv, sizeof(float) * 256);

  // invert.
  for (int i = 1; i < 255; i++) {
    // find val, such that Binv[val] = i.
    // I dont care about speed for this, so do it the stupid way.

    for (int s = 1; s < 255; s++) {
      if (BInv[s] <= i && BInv[s + 1] >= i) {
        HCalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
        break;
      }
    }
  }
  HCalib.B[0] = 0;
  HCalib.B[255] = 255;
}

void FullSystem::printResult(std::string file) {
  boost::unique_lock<boost::mutex> lock(trackMutex);
  boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

  int total_opt_tt = 0;
  for (int tt : opt_tt) {
    total_opt_tt += tt;
  }
  printf("Opt tt: %.1f\n", float(total_opt_tt) / opt_tt.size());

  std::ofstream myfile;
  myfile.open(file.c_str());
  myfile << std::setprecision(15);

  // mark existing frameHessians marginalized
  for (FrameHessian *fh : frameHessians) {
    fh->shell->marginalizedAt = 0;
  }

  for (FrameShell *s : allFrameHistory) {
    if (!s->poseValid)
      continue;

    if (setting_onlyLogKFPoses && s->marginalizedAt == s->id)
      continue;

    // scale the translation
    if (setting_enable_imu && s->trackingRef) {
      s->camToTrackingRef.translation() *= s->trackingRef->scale;
      s->camToWorld = s->trackingRef->camToWorld * s->camToTrackingRef;
    }

    myfile << s->timestamp << " " << s->camToWorld.translation().transpose()
           << " " << s->camToWorld.so3().unit_quaternion().x() << " "
           << s->camToWorld.so3().unit_quaternion().y() << " "
           << s->camToWorld.so3().unit_quaternion().z() << " "
           << s->camToWorld.so3().unit_quaternion().w() << "\n";
  }
  myfile.close();
  printf("saved to %s\n", file.c_str());
}

Vec4 FullSystem::trackNewCoarse(FrameHessian *fh) {

  assert(allFrameHistory.size() > 2);
  // set pose initialization.

  for (IOWrap::Output3DWrapper *ow : outputWrapper)
    ow->pushLiveFrame(fh);

  FrameHessian *lastF = coarse_tracker_->lastRef;

  AffLight aff_last_2_l = AffLight(0, 0);

  std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
  FrameShell *slast = allFrameHistory[allFrameHistory.size() - 2];
  FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3];
  SE3 slast_2_sprelast;
  SE3 lastF_2_slast;
  { // lock on global pose consistency!
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
    lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
    aff_last_2_l = slast->aff_g2l;
  }
  SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast.

  SE3 lastF_2_fh_imu;
  if (setting_enable_imu && HCalib.imu_initialized) {
    // imu predicted motion
    double t = slast->timestamp - fh->shell->timestamp;
    Vec3 tsl_fh_2_w = slast->camToWorld.translation() - fh->getSplineTw_c2t(t);
    Mat33 rot_fh_2_w =
        slast->camToWorld.rotationMatrix() * fh->getSplineR_c_t(t).transpose();
    SE3 fh_2_w(rot_fh_2_w, tsl_fh_2_w);
    lastF_2_fh_imu = fh_2_w.inverse() * lastF->shell->camToWorld;
    lastF_2_fh_tries.push_back(lastF_2_fh_imu);
  }
  // assume constant motion.
  lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);
  // assume double motion (frame skipped)
  lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() *
                             lastF_2_slast);
  // assume half motion.
  lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() *
                             lastF_2_slast);
  // assume zero motion.
  lastF_2_fh_tries.push_back(lastF_2_slast);
  // assume zero motion FROM KF.
  lastF_2_fh_tries.push_back(SE3());

  // get last delta-movement.
  auto lastF_2_fh_const = fh_2_slast.inverse() * lastF_2_slast;
  // just try a TON of different initializations (all rotations). In the end,
  // if they don't work they will only be tried on the coarsest level, which
  // is super fast anyway. also, if tracking rails here we loose, so we
  // really, really want to avoid that.
  std::vector<std::vector<float>> rot_signs = {
      {1, 0, 0},   {0, 1, 0},   {0, 0, 1},   {-1, 0, 0},   {0, -1, 0},
      {0, 0, -1},  {1, 1, 0},   {0, 1, 1},   {1, 0, 1},    {-1, 1, 0},
      {0, -1, 1},  {-1, 0, 1},  {1, -1, 0},  {0, 1, -1},   {1, 0, -1},
      {-1, -1, 0}, {0, -1, -1}, {-1, 0, -1}, {-1, -1, -1}, {-1, -1, 1},
      {-1, 1, -1}, {-1, 1, 1},  {1, -1, -1}, {1, -1, 1},   {1, 1, -1},
      {1, 1, 1}};
  for (float rot_delta = 0.02; rot_delta < 0.05; rot_delta += 0.01) {
    for (auto &rs : rot_signs) {
      lastF_2_fh_tries.push_back(
          lastF_2_fh_const *
          SE3(Sophus::Quaterniond(1, rs[0] * rot_delta, rs[1] * rot_delta,
                                  rs[2] * rot_delta),
              Vec3(0, 0, 0)));
    }
  }

  if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
    lastF_2_fh_tries.clear();
    lastF_2_fh_tries.push_back(SE3());
  }

  Vec3 flowVecs = Vec3(100, 100, 100);
  SE3 lastF_2_fh = SE3();
  AffLight aff_g2l = AffLight(0, 0);

  // as long as maxResForImmediateAccept is not reached, I'll continue through
  // the options. I'll keep track of the so-far best achieved residual for each
  // level in achievedRes. If on a coarse level, tracking is WORSE than
  // achievedRes, we will not continue to save time.

  Vec5 achievedRes = Vec5::Constant(NAN);
  bool haveOneGood = false;
  int tryIterations = 0;
  for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
    AffLight aff_g2l_this = aff_last_2_l;
    SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
    Vec5 currentRes;
    bool trackingIsGood = coarse_tracker_->trackNewestCoarse(
        fh, lastF_2_fh_this, aff_g2l_this, pyrLevelsUsed - 1, achievedRes,
        currentRes); // in each level has to be at least as good as the last
                     // try.
    tryIterations++;

    // if (i != 0) {
    //   printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl "
    //          "%d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
    //          i, i, pyrLevelsUsed - 1, aff_g2l_this.a, aff_g2l_this.b,
    //          achievedRes[0], achievedRes[1], achievedRes[2], achievedRes[3],
    //          achievedRes[4], currentRes[0], currentRes[1], currentRes[2],
    //          currentRes[3], currentRes[4]);
    // }

    // do we have a new winner?
    if (trackingIsGood && std::isfinite((float)currentRes[0]) &&
        !(currentRes[0] >= achievedRes[0])) {
      // printf("take over. minRes %f -> %f!\n", achievedRes[0],
      // currentRes[0]);
      flowVecs = coarse_tracker_->lastFlowIndicators;
      aff_g2l = aff_g2l_this;
      lastF_2_fh = lastF_2_fh_this;
      haveOneGood = true;
    }

    // take over achieved res (always).
    if (haveOneGood) {
      for (int i = 0; i < 5; i++) {
        if (!std::isfinite((float)achievedRes[i]) ||
            achievedRes[i] > currentRes[i]) // take over if achievedRes is
                                            // either bigger or NAN.
          achievedRes[i] = currentRes[i];
      }
    }

    if (haveOneGood &&
        achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
      break;
  }

  if (setting_enable_imu && setting_print_imu && HCalib.imu_initialized &&
      tryIterations > 1 &&
      (lastF_2_fh_imu.inverse() * lastF_2_fh).log().norm() > 0.1) {
    printf("IMU motion prediction is bad\n");
    std::cout << "imu " << std::fixed << std::setw(6) << std::setprecision(3)
              << lastF_2_fh_imu.log().transpose() << std::endl;
    std::cout << "dso " << std::fixed << std::setw(6) << std::setprecision(3)
              << lastF_2_fh.log().transpose() << std::endl;
    // while (std::cin.get() != '\n') {
    // }
  }

  if (!haveOneGood) {
    printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope "
           "we may somehow recover.\n");
    flowVecs = Vec3(0, 0, 0);
    aff_g2l = aff_last_2_l;
    lastF_2_fh = lastF_2_fh_tries[0];
  }

  lastCoarseRMSE = achievedRes;

  // no lock required, as fh is not used anywhere yet.
  fh->shell->camToTrackingRef = lastF_2_fh.inverse();
  fh->shell->trackingRef = lastF->shell;
  fh->shell->aff_g2l = aff_g2l;
  fh->shell->camToWorld =
      fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

  if (coarse_tracker_->firstCoarseRMSE < 0)
    coarse_tracker_->firstCoarseRMSE = achievedRes[0];

  if (!setting_debugout_runquiet)
    printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a,
           aff_g2l.b, fh->ab_exposure, achievedRes[0]);

  return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::traceNewCoarse(FrameHessian *fh) {
  boost::unique_lock<boost::mutex> lock(mapMutex);

  int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0,
      trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

  Mat33f K = Mat33f::Identity();
  K(0, 0) = HCalib.fxl();
  K(1, 1) = HCalib.fyl();
  K(0, 2) = HCalib.cxl();
  K(1, 2) = HCalib.cyl();

  for (FrameHessian *host : frameHessians) // go through all active frames
  {

    SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
    Vec3f Kt = K * hostToNew.translation().cast<float>();

    Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure,
                                            host->aff_g2l(), fh->aff_g2l())
                    .cast<float>();

    for (ImmaturePoint *ph : host->immaturePoints) {
      ph->traceOn(fh, KRKi, Kt, aff, &HCalib, false);

      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
        trace_good++;
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
        trace_badcondition++;
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
        trace_oob++;
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
        trace_out++;
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
        trace_skip++;
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
        trace_uninitialized++;
      trace_total++;
    }
  }
  //	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip.
  //%'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%)
  // uninit.\n", 			trace_total, trace_good,
  // 100*trace_good/(float)trace_total, 			trace_skip,
  // 100*trace_skip/(float)trace_total, trace_badcondition,
  // 100*trace_badcondition/(float)trace_total, trace_oob,
  // 100*trace_oob/(float)trace_total, 			trace_out,
  // 100*trace_out/(float)trace_total, 			trace_uninitialized,
  // 100*trace_uninitialized/(float)trace_total);
}

void FullSystem::activatePointsMT_Reductor(
    std::vector<PointHessian *> *optimized,
    std::vector<ImmaturePoint *> *toOptimize, int min, int max, Vec10 *stats,
    int tid) {
  ImmaturePointTemporaryResidual *tr =
      new ImmaturePointTemporaryResidual[frameHessians.size()];
  for (int k = min; k < max; k++) {
    (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
  }
  delete[] tr;
}

void FullSystem::activatePointsMT() {

  if (ef->nPoints < setting_desiredPointDensity * 0.66)
    currentMinActDist -= 0.8;
  if (ef->nPoints < setting_desiredPointDensity * 0.8)
    currentMinActDist -= 0.5;
  else if (ef->nPoints < setting_desiredPointDensity * 0.9)
    currentMinActDist -= 0.2;
  else if (ef->nPoints < setting_desiredPointDensity)
    currentMinActDist -= 0.1;

  if (ef->nPoints > setting_desiredPointDensity * 1.5)
    currentMinActDist += 0.8;
  if (ef->nPoints > setting_desiredPointDensity * 1.3)
    currentMinActDist += 0.5;
  if (ef->nPoints > setting_desiredPointDensity * 1.15)
    currentMinActDist += 0.2;
  if (ef->nPoints > setting_desiredPointDensity)
    currentMinActDist += 0.1;

  if (currentMinActDist < 0)
    currentMinActDist = 0;
  if (currentMinActDist > 4)
    currentMinActDist = 4;

  if (!setting_debugout_runquiet)
    printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
           currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);

  FrameHessian *newestHs = frameHessians.back();

  // make dist map.
  coarseDistanceMap->makeK(&HCalib);
  coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

  // coarse_tracker_->debugPlotDistMap("distMap");

  std::vector<ImmaturePoint *> toOptimize;
  toOptimize.reserve(20000);

  for (FrameHessian *host : frameHessians) // go through all active frames
  {
    if (host == newestHs)
      continue;

    SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi =
        (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() *
         coarseDistanceMap->Ki[0]);
    Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

    for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1) {
      ImmaturePoint *ph = host->immaturePoints[i];
      ph->idxInImmaturePoints = i;

      // delete points that have never been traced successfully, or that are
      // outlier on the last trace.
      if (!std::isfinite(ph->idepth_max) ||
          ph->lastTraceStatus == IPS_OUTLIER) {
        //				immature_invalid_deleted++;
        // remove point.
        delete ph;
        host->immaturePoints[i] = 0;
        continue;
      }

      // can activate only if this is true.
      bool canActivate = (ph->lastTraceStatus == IPS_GOOD ||
                          ph->lastTraceStatus == IPS_SKIPPED ||
                          ph->lastTraceStatus == IPS_BADCONDITION ||
                          ph->lastTraceStatus == IPS_OOB) &&
                         ph->lastTracePixelInterval < 8 &&
                         ph->quality > setting_minTraceQuality &&
                         (ph->idepth_max + ph->idepth_min) > 0;

      // if I cannot activate the point, skip it. Maybe also delete it.
      if (!canActivate) {
        // if point will be out afterwards, delete it instead.
        if (ph->host->flaggedForMarginalization ||
            ph->lastTraceStatus == IPS_OOB) {
          //					immature_notReady_deleted++;
          delete ph;
          host->immaturePoints[i] = 0;
        }
        //				immature_notReady_skipped++;
        continue;
      }

      // see if we need to activate point due to distance map.
      Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) +
                  Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
      int u = ptp[0] / ptp[2] + 0.5f;
      int v = ptp[1] / ptp[2] + 0.5f;

      if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {

        float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] +
                     (ptp[0] - floorf((float)(ptp[0])));

        if (dist >= currentMinActDist * ph->my_type) {
          coarseDistanceMap->addIntoDistFinal(u, v);
          toOptimize.push_back(ph);
        }
      } else {
        delete ph;
        host->immaturePoints[i] = 0;
      }
    }
  }

  //	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip
  //%d)\n", 			(int)toOptimize.size(), immature_deleted,
  // immature_notReady, immature_needMarg, immature_want, immature_margskip);

  std::vector<PointHessian *> optimized;
  optimized.resize(toOptimize.size());

  if (multiThreading)
    treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this,
                                   &optimized, &toOptimize, _1, _2, _3, _4),
                       0, toOptimize.size(), 50);

  else
    activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0,
                              0);

  for (unsigned k = 0; k < toOptimize.size(); k++) {
    PointHessian *newpoint = optimized[k];
    ImmaturePoint *ph = toOptimize[k];

    if (newpoint != 0 && newpoint != (PointHessian *)((long)(-1))) {
      newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;
      newpoint->host->pointHessians.push_back(newpoint);
      ef->insertPoint(newpoint);
      for (PointFrameResidual *r : newpoint->residuals)
        ef->insertResidual(r);
      assert(newpoint->efPoint != 0);
      delete ph;
    } else if (newpoint == (PointHessian *)((long)(-1)) ||
               ph->lastTraceStatus == IPS_OOB) {
      ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
      delete ph;
    } else {
      assert(newpoint == 0 || newpoint == (PointHessian *)((long)(-1)));
    }
  }

  for (FrameHessian *host : frameHessians) {
    for (int i = 0; i < (int)host->immaturePoints.size(); i++) {
      if (host->immaturePoints[i] == 0) {
        host->immaturePoints[i] = host->immaturePoints.back();
        host->immaturePoints.pop_back();
        i--;
      }
    }
  }
}

void FullSystem::activatePointsOldFirst() { assert(false); }

void FullSystem::flagPointsForRemoval() {
  assert(EFIndicesValid);

  std::vector<FrameHessian *> fhsToKeepPoints;
  std::vector<FrameHessian *> fhsToMargPoints;

  // if(setting_margPointVisWindow>0)
  {
    for (int i = ((int)frameHessians.size()) - 1;
         i >= 0 && i >= ((int)frameHessians.size()); i--)
      if (!frameHessians[i]->flaggedForMarginalization)
        fhsToKeepPoints.push_back(frameHessians[i]);

    for (int i = 0; i < (int)frameHessians.size(); i++)
      if (frameHessians[i]->flaggedForMarginalization)
        fhsToMargPoints.push_back(frameHessians[i]);
  }

  // ef->setAdjointsF();
  // ef->setDeltaF(&HCalib);
  int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

  for (FrameHessian *host : frameHessians) // go through all active frames
  {
    for (unsigned int i = 0; i < host->pointHessians.size(); i++) {
      PointHessian *ph = host->pointHessians[i];
      if (ph == 0)
        continue;

      if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) {
        host->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        host->pointHessians[i] = 0;
        flag_nores++;
      } else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) ||
                 host->flaggedForMarginalization) {
        flag_oob++;
        if (ph->isInlierNew()) {
          flag_in++;
          int ngoodRes = 0;
          for (PointFrameResidual *r : ph->residuals) {
            r->resetOOB();
            r->linearize(&HCalib);
            r->efResidual->isLinearized = false;
            r->applyRes(true);
            if (r->efResidual->isActive()) {
              r->efResidual->fixLinearizationF(ef);
              ngoodRes++;
            }
          }
          if (ph->idepth_hessian > setting_minIdepthH_marg) {
            flag_inin++;
            ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
            host->pointHessiansMarginalized.push_back(ph);
          } else {
            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
            host->pointHessiansOut.push_back(ph);
          }

        } else {
          host->pointHessiansOut.push_back(ph);
          ph->efPoint->stateFlag = EFPointStatus::PS_DROP;

          // printf("drop point in frame %d (%d goodRes, %d activeRes)\n",
          // ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
        }

        host->pointHessians[i] = 0;
      }
    }

    for (int i = 0; i < (int)host->pointHessians.size(); i++) {
      if (host->pointHessians[i] == 0) {
        host->pointHessians[i] = host->pointHessians.back();
        host->pointHessians.pop_back();
        i--;
      }
    }
  }
}

void FullSystem::addActiveFrame(const std::vector<Vec7> &new_imu_data,
                                ImageAndExposure *image, int incoming_id) {

  if (isLost)
    return;
  boost::unique_lock<boost::mutex> lock(trackMutex);

  imu_data.insert(imu_data.end(), new_imu_data.begin(), new_imu_data.end());
  if (!initialized && coarseInitializer->frameID < 0) {
    if (imu_data.size() < setting_min_g_imu) {
      return;
    }
  }

  // add into allFrameHistory
  FrameHessian *fh = new FrameHessian();
  FrameShell *shell = new FrameShell();
  shell->camToWorld =
      SE3(); // no lock required, as fh is not used anywhere yet.
  shell->aff_g2l = AffLight(0, 0);
  shell->marginalizedAt = shell->id = allFrameHistory.size();
  shell->timestamp = image->timestamp;
  shell->incoming_id = incoming_id;
  fh->shell = shell;
  fh->setImuData(new_imu_data);
  allFrameHistory.push_back(shell);

  // make Images / derivatives etc.
  fh->ab_exposure = image->exposure_time;
  fh->makeImages(image->image, &HCalib);

  if (!initialized) {
    // use initializer!
    // first frame set. fh is kept by coarseInitializer.
    if (coarseInitializer->frameID < 0) {
      coarseInitializer->setFirst(&HCalib, fh);
      fh->setImuData(imu_data);
      imu_data.clear();
    } else {
      if (coarseInitializer->trackFrame(fh, outputWrapper)) // if SNAPPED
      {
        initializeFromInitializer(fh);
        if (initFailed)
          return;
        lock.unlock();
        deliverTrackedFrame(fh, true);
      } else {
        // if still initializing
        fh->shell->poseValid = false;
        delete fh;
      }
    }
    return;
  } else // do front-end operation.
  {
    // SWAP tracking reference?.
    if (coarse_tracker_for_new_kf_->refFrameID > coarse_tracker_->refFrameID) {
      boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
      CoarseTracker *tmp = coarse_tracker_;
      coarse_tracker_ = coarse_tracker_for_new_kf_;
      coarse_tracker_for_new_kf_ = tmp;
    }

    if (setting_enable_imu && HCalib.imu_initialized) {
      fh->propagateImuState(allFrameHistory[allFrameHistory.size() - 2],
                            coarse_tracker_->lastRef->imu_bias, &HCalib);
    }

    Vec4 tres = trackNewCoarse(fh);
    if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) ||
        !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3])) {
      printf("Initial Tracking failed: LOST!\n");
      isLost = true;
      return;
    }

    if (setting_enable_imu && HCalib.imu_initialized) {
      fh->updateVel(allFrameHistory[allFrameHistory.size() - 2]);
    }

    bool needToMakeKF = false;
    if (setting_keyframesPerSecond > 0) {
      needToMakeKF =
          allFrameHistory.size() == 1 ||
          (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) >
              0.95f / setting_keyframesPerSecond;
    } else {
      Vec2 refToFh = AffLight::fromToVecExposure(
          coarse_tracker_->lastRef->ab_exposure, fh->ab_exposure,
          coarse_tracker_->lastRef_aff_g2l, fh->shell->aff_g2l);

      // BRIGHTNESS CHECK
      needToMakeKF = allFrameHistory.size() == 1 ||
                     setting_kfGlobalWeight * setting_maxShiftWeightT *
                                 sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
                             setting_kfGlobalWeight * setting_maxShiftWeightR *
                                 sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
                             setting_kfGlobalWeight * setting_maxShiftWeightRT *
                                 sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
                             setting_kfGlobalWeight * setting_maxAffineWeight *
                                 fabs(logf((float)refToFh[0])) >
                         1 ||
                     2 * coarse_tracker_->firstCoarseRMSE < tres[0];
    }

    for (IOWrap::Output3DWrapper *ow : outputWrapper)
      ow->publishCamPose(fh->shell, &HCalib);

    lock.unlock();
    deliverTrackedFrame(fh, needToMakeKF);
    return;
  }
}
void FullSystem::deliverTrackedFrame(FrameHessian *fh, bool needKF) {
  if (goStepByStep && lastRefStopID != coarse_tracker_->refFrameID) {
    MinimalImageF3 img(wG[0], hG[0], fh->dI);
    IOWrap::displayImage("frameToTrack", &img);
    while (true) {
      char k = IOWrap::waitKey(0);
      if (k == ' ')
        break;
      handleKey(k);
    }
    lastRefStopID = coarse_tracker_->refFrameID;
  } else
    handleKey(IOWrap::waitKey(1));

  if (needKF)
    makeKeyFrame(fh);
  else
    makeNonKeyFrame(fh);
}

void FullSystem::makeNonKeyFrame(FrameHessian *fh) {
  // needs to be set by mapping thread. no lock required since we are in mapping
  // thread.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    assert(fh->shell->trackingRef != 0);
    fh->shell->camToWorld =
        fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld, fh->shell->aff_g2l);
  }

  traceNewCoarse(fh);
  delete fh;
}

void FullSystem::makeKeyFrame(FrameHessian *fh) {
  // needs to be set by mapping thread
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    assert(fh->shell->trackingRef != 0);
    fh->shell->camToWorld =
        fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld, fh->shell->aff_g2l);
  }

  traceNewCoarse(fh);

  boost::unique_lock<boost::mutex> lock(mapMutex);

  // Flag Frames to be Marginalized.
  flagFramesForMarginalization(fh);

  // add New Frame to Hessian Struct.
  fh->setImuData(imu_data);
  imu_data.clear();
  if (setting_enable_imu && HCalib.imu_initialized) {
    fh->propagateImuState(allKeyFramesHistory.back(),
                          coarse_tracker_->lastRef->imu_bias, &HCalib);
  }
  fh->idx = frameHessians.size();
  frameHessians.push_back(fh);
  fh->frameID = allKeyFramesHistory.size();
  allKeyFramesHistory.push_back(fh->shell);
  ef->insertFrame(fh, &HCalib);

  setPrecalcValues();

  // add new residuals for old points
  int numFwdResAdde = 0;
  for (FrameHessian *fh1 : frameHessians) // go through all active frames
  {
    if (fh1 == fh)
      continue;
    for (PointHessian *ph : fh1->pointHessians) {
      PointFrameResidual *r = new PointFrameResidual(ph, fh1, fh);
      r->setState(ResState::IN);
      ph->residuals.push_back(r);
      ef->insertResidual(r);
      ph->lastResiduals[1] = ph->lastResiduals[0];
      ph->lastResiduals[0] =
          std::pair<PointFrameResidual *, ResState>(r, ResState::IN);
      numFwdResAdde += 1;
    }
  }

  // Activate Points (& flag for marginalization).
  activatePointsMT();
  ef->makeIDX();

  // imu initialization
  if (setting_enable_imu && allKeyFramesHistory.size() == 5) {
    assert(frameHessians.size() == 5);
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    if (!FrameHessian::initializeImu(frameHessians, &HCalib)) {
      initFailed = true;
      return;
    }
    setPrecalcValues();
  }

  // OPTIMIZE ALL
  fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
  auto start = std::chrono::steady_clock::now();
  float rmse = optimize(setting_maxOptIterations);
  auto end = std::chrono::steady_clock::now();
  opt_tt.push_back(
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count());

  // Figure Out if INITIALIZATION FAILED
  if (allKeyFramesHistory.size() <= 4) {
    if (allKeyFramesHistory.size() == 2 &&
        rmse > 20 * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed = true;
    }
    if (allKeyFramesHistory.size() == 3 &&
        rmse > 13 * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed = true;
    }
    if (allKeyFramesHistory.size() == 4 &&
        rmse > 9 * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed = true;
    }
  }

  if (isLost)
    return;

  // REMOVE OUTLIER
  removeOutliers();

  // reset imu states for initialization
  if (setting_enable_imu && allKeyFramesHistory.size() == 5) {
    for (int i = 0; i < frameHessians.size(); i++) {
      frameHessians[i]->setImuStateZero(&HCalib);
      if (i > 0) {
        frameHessians[i]->updateVel(frameHessians[i - 1]->shell);
      }
    }
  }

  {
    boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
    coarse_tracker_for_new_kf_->makeK(&HCalib);
    coarse_tracker_for_new_kf_->setCoarseTrackingRef(frameHessians);

    coarse_tracker_for_new_kf_->debugPlotIDepthMap(
        &minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
    coarse_tracker_for_new_kf_->debugPlotIDepthMapFloat(outputWrapper);
  }

  // debugPlot("post Optimize");

  // (Activate-)Marginalize Points
  flagPointsForRemoval();
  ef->dropPointsF();
  getNullspaces(ef->lastNullspaces_pose, ef->lastNullspaces_scale,
                ef->lastNullspaces_affA, ef->lastNullspaces_affB);
  ef->marginalizePointsF();

  // add new Immature points & new residuals
  makeNewTraces(fh, 0);

  for (IOWrap::Output3DWrapper *ow : outputWrapper) {
    ow->publishGraph(ef->connectivityMap);
    ow->publishKeyframes(frameHessians, false, &HCalib);
  }

  // Marginalize Frames

  for (unsigned int i = 0; i < frameHessians.size(); i++)
    if (frameHessians[i]->flaggedForMarginalization) {
      marginalizeFrame(frameHessians[i]);
      i = 0;
    }

  // while (std::cin.get() != '\n') {
  // }
}

void FullSystem::initializeFromInitializer(FrameHessian *newFrame) {
  boost::unique_lock<boost::mutex> lock(mapMutex);

  // add firstframe.
  FrameHessian *firstFrame = coarseInitializer->firstFrame;
  firstFrame->idx = frameHessians.size();
  frameHessians.push_back(firstFrame);
  firstFrame->frameID = allKeyFramesHistory.size();
  allKeyFramesHistory.push_back(firstFrame->shell);
  ef->insertFrame(firstFrame, &HCalib);
  setPrecalcValues();

  // int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0],
  // hG[0], setting_desiredDensity); int numPointsTotal =
  // pixelSelector->makeMaps(firstFrame->dIp,
  // selectionMap,setting_desiredDensity);

  firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
  firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
  firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);

  float sumID = 1e-5, numID = 1e-5;
  for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
    sumID += coarseInitializer->points[0][i].iR;
    numID++;
  }
  float rescaleFactor = 1 / (sumID / numID);

  // randomly sub-select the points I need.
  float keepPercentage =
      setting_desiredPointDensity / coarseInitializer->numPoints[0];

  if (!setting_debugout_runquiet)
    printf("Initialization: keep %.1f%% (need %d, have %d)!\n",
           100 * keepPercentage, (int)(setting_desiredPointDensity),
           coarseInitializer->numPoints[0]);

  for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
    if (rand() / (float)RAND_MAX > keepPercentage)
      continue;

    Pnt *point = coarseInitializer->points[0] + i;
    ImmaturePoint *pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f,
                                          firstFrame, point->my_type, &HCalib);

    if (!std::isfinite(pt->energyTH)) {
      delete pt;
      continue;
    }

    pt->idepth_max = pt->idepth_min = 1;
    PointHessian *ph = new PointHessian(pt, &HCalib);
    delete pt;
    if (!std::isfinite(ph->energyTH)) {
      delete ph;
      continue;
    }

    ph->setIdepthScaled(point->iR * rescaleFactor);
    ph->setIdepthZero(ph->idepth);
    ph->hasDepthPrior = true;
    ph->setPointStatus(PointHessian::ACTIVE);

    firstFrame->pointHessians.push_back(ph);
    ef->insertPoint(ph);
  }

  SE3 firstToNew = coarseInitializer->thisToNext;
  firstToNew.translation() /= rescaleFactor;

  // align w.r.t. gravity
  Vec3 g_imu = Vec3::Zero(); // gravity in imu frame
  printf("Gravity direction estimated from %d imu data.\n", setting_min_g_imu);
  assert(firstFrame->imu_data.size() >= setting_min_g_imu);
  int imu_count = 0;
  for (const ImuData &d : firstFrame->imu_data) {
    g_imu = g_imu + d.acc;
    imu_count++;
    if (imu_count >= setting_min_g_imu) {
      break;
    }
  }
  g_imu.normalize();

  Vec3 sg0;
  sg0 << 1, 0, 0;
  HCalib.setSgZero(sg0);

  Vec3 g_world = HCalib.getG().normalized();
  Vec3 rot = SO3::hat(g_imu) * g_world;
  double sin_theta = rot.norm();
  double cos_theta = g_imu.dot(g_world);
  Vec3 axis = rot.normalized();
  Mat33 rot_w_i0 = cos_theta * Mat33::Identity() +
                   (1 - cos_theta) * axis * axis.transpose() +
                   sin_theta * SO3::hat(axis);
  Mat33 rot_w_c0 = rot_w_i0 * setting_rot_imu_cam;
  SE3 tfm_w_c0(rot_w_c0, Vec3::Zero());

  // really no lock required, as we are initializing.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    firstFrame->shell->camToWorld = tfm_w_c0;
    firstFrame->shell->aff_g2l = AffLight(0, 0);
    firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld,
                                 firstFrame->shell->aff_g2l);
    firstFrame->shell->trackingRef = 0;
    firstFrame->shell->camToTrackingRef = SE3();

    newFrame->shell->camToWorld = tfm_w_c0 * firstToNew.inverse();
    newFrame->shell->aff_g2l = AffLight(0, 0);
    newFrame->setEvalPT_scaled(newFrame->shell->camToWorld,
                               newFrame->shell->aff_g2l);
    newFrame->shell->trackingRef = firstFrame->shell;
    newFrame->shell->camToTrackingRef = firstToNew.inverse();
  }

  initialized = true;
  printf("INITIALIZE FROM INITIALIZER (%d pts)!\n",
         (int)firstFrame->pointHessians.size());
}

void FullSystem::makeNewTraces(FrameHessian *newFrame, float *gtDepth) {
  pixelSelector->allowFast = true;
  // int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0],
  // hG[0], setting_desiredDensity);
  int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,
                                               setting_desiredImmatureDensity);

  newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
  // fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
  newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
  newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

  for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
    for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
      int i = x + y * wG[0];
      if (selectionMap[i] == 0)
        continue;

      ImmaturePoint *impt =
          new ImmaturePoint(x, y, newFrame, selectionMap[i], &HCalib);
      if (!std::isfinite(impt->energyTH))
        delete impt;
      else
        newFrame->immaturePoints.push_back(impt);
    }
  // printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());
}

void FullSystem::setPrecalcValues() {
  for (FrameHessian *fh : frameHessians) {
    fh->targetPrecalc.resize(frameHessians.size());
    for (unsigned int i = 0; i < frameHessians.size(); i++)
      fh->targetPrecalc[i].set(fh, frameHessians[i], &HCalib);
  }

  ef->setDeltaF(&HCalib);
}

} // namespace dso
