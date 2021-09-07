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

#include "EnergyFunctional.h"
#include "AccumulatedSCHessian.h"
#include "AccumulatedTopHessian.h"
#include "EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;

void EnergyFunctional::setAdjointsF(CalibHessian *HCalib) {

  if (adHost != 0)
    delete[] adHost;
  if (adTarget != 0)
    delete[] adTarget;
  adHost = new Mat88[nFrames * nFrames];
  adTarget = new Mat88[nFrames * nFrames];

  for (int h = 0; h < nFrames; h++)
    for (int t = 0; t < nFrames; t++) {
      FrameHessian *host = frames[h]->data;
      FrameHessian *target = frames[t]->data;

      SE3 worldToTarget = target->get_camToWorld_evalPT().inverse();

      Mat88 AH = Mat88::Identity();
      Mat88 AT = Mat88::Identity();

      AH.topLeftCorner<6, 6>() = worldToTarget.Adj().transpose();
      AT.topLeftCorner<6, 6>() = -worldToTarget.Adj().transpose();

      Vec2f affLL =
          AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure,
                                      host->aff_g2l_0(), target->aff_g2l_0())
              .cast<float>();
      AT(6, 6) = -affLL[0];
      AH(6, 6) = affLL[0];
      AT(7, 7) = -1;
      AH(7, 7) = affLL[0];

      AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
      AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
      AH.block<1, 8>(6, 0) *= SCALE_A;
      AH.block<1, 8>(7, 0) *= SCALE_B;
      AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
      AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
      AT.block<1, 8>(6, 0) *= SCALE_A;
      AT.block<1, 8>(7, 0) *= SCALE_B;

      adHost[h + t * nFrames] = AH;
      adTarget[h + t * nFrames] = AT;
    }
  cPrior = VecC::Constant(setting_initialCalibHessian);

  if (adHostF != 0)
    delete[] adHostF;
  if (adTargetF != 0)
    delete[] adTargetF;
  adHostF = new Mat88f[nFrames * nFrames];
  adTargetF = new Mat88f[nFrames * nFrames];

  for (int h = 0; h < nFrames; h++)
    for (int t = 0; t < nFrames; t++) {
      adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
      adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
    }

  cPriorF = cPrior.cast<float>();

  EFAdjointsValid = true;
}

EnergyFunctional::EnergyFunctional() {
  adHost = 0;
  adTarget = 0;

  red = 0;

  adHostF = 0;
  adTargetF = 0;
  adHTdeltaF = 0;

  nFrames = nResiduals = nPoints = 0;

  if (setting_enable_imu) {
    HM = MatXX::Zero(CPARS + 3, CPARS + 3);
    bM = VecX::Zero(CPARS + 3);
  } else {
    HM = MatXX::Zero(CPARS, CPARS);
    bM = VecX::Zero(CPARS);
  }

  accSSE_top_L = new AccumulatedTopHessianSSE();
  accSSE_top_A = new AccumulatedTopHessianSSE();
  accSSE_bot = new AccumulatedSCHessianSSE();

  resInA = resInL = resInM = 0;
  currentLambda = 0;
}
EnergyFunctional::~EnergyFunctional() {
  for (EFFrame *f : frames) {
    for (EFPoint *p : f->points) {
      for (EFResidual *r : p->residualsAll) {
        r->data->efResidual = 0;
        delete r;
      }
      p->data->efPoint = 0;
      delete p;
    }
    f->data->efFrame = 0;
    delete f;
  }

  if (adHost != 0)
    delete[] adHost;
  if (adTarget != 0)
    delete[] adTarget;

  if (adHostF != 0)
    delete[] adHostF;
  if (adTargetF != 0)
    delete[] adTargetF;
  if (adHTdeltaF != 0)
    delete[] adHTdeltaF;

  delete accSSE_top_L;
  delete accSSE_top_A;
  delete accSSE_bot;
}

void EnergyFunctional::setDeltaF(CalibHessian *HCalib) {
  if (adHTdeltaF != 0)
    delete[] adHTdeltaF;
  adHTdeltaF = new Mat18f[nFrames * nFrames];
  for (int h = 0; h < nFrames; h++)
    for (int t = 0; t < nFrames; t++) {
      int idx = h + t * nFrames;
      adHTdeltaF[idx] = frames[h]
                                ->data->get_state_minus_stateZero()
                                .head<8>()
                                .cast<float>()
                                .transpose() *
                            adHostF[idx] +
                        frames[t]
                                ->data->get_state_minus_stateZero()
                                .head<8>()
                                .cast<float>()
                                .transpose() *
                            adTargetF[idx];
    }

  cDeltaF = HCalib->value_minus_value_zero.cast<float>();
  for (EFFrame *f : frames) {
    f->delta = f->data->get_state_minus_stateZero().head<8>();
    f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>();

    for (EFPoint *p : f->points)
      p->deltaF = p->data->idepth - p->data->idepth_zero;
  }

  EFDeltaValid = true;
}

// accumulates & shifts L.
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT) {
  if (MT) {
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A,
                            nFrames, _1, _2, _3, _4),
                0, 0, 0);
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
                            accSSE_top_A, &allPoints, this, _1, _2, _3, _4),
                0, allPoints.size(), 50);
    accSSE_top_A->stitchDoubleMT(red, H, b, this, false, true);
    resInA = accSSE_top_A->nres[0];
  } else {
    accSSE_top_A->setZero(nFrames);
    for (EFFrame *f : frames)
      for (EFPoint *p : f->points)
        accSSE_top_A->addPoint<0>(p, this);
    accSSE_top_A->stitchDoubleMT(red, H, b, this, false, false);
    resInA = accSSE_top_A->nres[0];
  }
}

// accumulates & shifts L.
void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT) {
  if (MT) {
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L,
                            nFrames, _1, _2, _3, _4),
                0, 0, 0);
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
                            accSSE_top_L, &allPoints, this, _1, _2, _3, _4),
                0, allPoints.size(), 50);
    accSSE_top_L->stitchDoubleMT(red, H, b, this, true, true);
    resInL = accSSE_top_L->nres[0];
  } else {
    accSSE_top_L->setZero(nFrames);
    for (EFFrame *f : frames)
      for (EFPoint *p : f->points)
        accSSE_top_L->addPoint<1>(p, this);
    accSSE_top_L->stitchDoubleMT(red, H, b, this, true, false);
    resInL = accSSE_top_L->nres[0];
  }
}

void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT) {
  if (MT) {
    red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot,
                            nFrames, _1, _2, _3, _4),
                0, 0, 0);
    red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
                            accSSE_bot, &allPoints, true, _1, _2, _3, _4),
                0, allPoints.size(), 50);
    accSSE_bot->stitchDoubleMT(red, H, b, this, true);
  } else {
    accSSE_bot->setZero(nFrames);
    for (EFFrame *f : frames)
      for (EFPoint *p : f->points)
        accSSE_bot->addPoint(p, true);
    accSSE_bot->stitchDoubleMT(red, H, b, this, false);
  }
}

void EnergyFunctional::expandHbtoFitImu(MatXX &H, VecX &b) {
  int dim = CPARS + 3 + 29 * nFrames;
  MatXX He = MatXX::Zero(dim, dim);
  VecX be = VecX::Zero(dim);

  // H: cam - cam
  He.topLeftCorner<CPARS, CPARS>() = H.topLeftCorner<CPARS, CPARS>();
  // b: cam
  be.head(CPARS) = b.head(CPARS);
  for (int i = 0; i < nFrames; i++) {
    int fi = CPARS + 8 * i;
    int fie = CPARS + 3 + 29 * i;
    // H: cam - xi_ab_i
    He.block(0, fie, CPARS, 8) += H.block(0, fi, CPARS, 8);
    He.block(fie, 0, 8, CPARS) += H.block(fi, 0, 8, CPARS);
    // H: xi_ab_i - xi_ab_i
    He.block(fie, fie, 8, 8) += H.block(fi, fi, 8, 8);
    // H: xi_ab_i - xi_ab_j
    for (int j = i + 1; j < nFrames; j++) {
      int fj = CPARS + 8 * j;
      int fje = CPARS + 3 + 29 * j;
      He.block(fie, fje, 8, 8) += H.block(fi, fj, 8, 8);
      He.block(fje, fie, 8, 8) += H.block(fj, fi, 8, 8);
    }
    // b: xi_ab_i
    be.segment<8>(fie) += b.segment<8>(fi);
  }

  H = He.eval();
  b = be.eval();
}

void EnergyFunctional::getImuHessianCurrentFrame(int fi, CalibHessian *HCalib,
                                                 MatXX &H, VecX &b,
                                                 bool &spline_valid,
                                                 MatXX &J_cst, VecX &r_cst,
                                                 bool print) {
  assert(fi > 0);
  Mat33 I33 = Mat33::Identity();

  FrameHessian *cur_fh = frames[fi]->data;
  FrameHessian *prv_fh = frames[fi - 1]->data;
  double tpf = prv_fh->shell->timestamp - cur_fh->shell->timestamp;
  assert(tpf < 0);
  double tpf2 = tpf * tpf;
  int cur_idx = CPARS + 3 + 29 * fi;
  int prv_idx = CPARS + 3 + 29 * (fi - 1);

  /************************* imu bias error ****************************/
  Mat66 tmpH = setting_weight_imu_bias / -tpf;
  tmpH.topLeftCorner<3, 3>() *= (SCALE_BA * SCALE_BA);
  tmpH.bottomRightCorner<3, 3>() *= (SCALE_BG * SCALE_BG);
  H.block(prv_idx + 8, prv_idx + 8, 6, 6) += tmpH;
  H.block(cur_idx + 8, cur_idx + 8, 6, 6) += tmpH;
  H.block(prv_idx + 8, cur_idx + 8, 6, 6) += -tmpH;
  H.block(cur_idx + 8, prv_idx + 8, 6, 6) += -tmpH;
  Vec6 r_imu_bias = cur_fh->imu_bias - prv_fh->imu_bias;
  Vec6 tmpb = (setting_weight_imu_bias / -tpf) * r_imu_bias;
  tmpb.head(3) *= SCALE_BA;
  tmpb.tail(3) *= SCALE_BG;
  b.segment<6>(prv_idx + 8) += -tmpb;
  b.segment<6>(cur_idx + 8) += tmpb;

  spline_valid = (cur_fh->shell->trackingRef == prv_fh->shell) &&
                 (-tpf < setting_maxImuInterval);
  bool vel_valid = fi < (nFrames - 1);
  if (spline_valid) {
    /*********************** spline constraint *************************/
    int dim = CPARS + 3 + 29 * nFrames;
    if (vel_valid) {
      r_cst = VecX::Zero(6);
      J_cst = MatXX::Zero(6, dim);
    } else {
      r_cst = VecX::Zero(3);
      J_cst = MatXX::Zero(3, dim);
    }

    // rotation
    Mat33 rot_c_p_pred = cur_fh->getSplineR_c_t(tpf);
    Mat33 rot_c_p_meas =
        (cur_fh->PRE_camToWorld.inverse() * prv_fh->PRE_camToWorld)
            .rotationMatrix();
    r_cst.segment<3>(0) = SO3(rot_c_p_meas.transpose() * rot_c_p_pred).log();
    Mat33 rot_p_w_evalPT =
        prv_fh->get_camToWorld_evalPT().rotationMatrix().transpose();
    J_cst.block<3, 3>(0, prv_idx + 3) = -SCALE_XI_ROT * rot_p_w_evalPT;
    J_cst.block<3, 3>(0, cur_idx + 3) = SCALE_XI_ROT * rot_p_w_evalPT;
    J_cst.block<3, 3>(0, cur_idx + 14) = SCALE_SL_ROT * tpf * I33;
    J_cst.block<3, 3>(0, cur_idx + 20) = SCALE_SQ_ROT * tpf2 * I33;
    J_cst.block<3, 3>(0, cur_idx + 26) = SCALE_SC_ROT * tpf * tpf2 * I33;

    // velocity
    if (vel_valid) {
      FrameHessian *nxt_fh = frames[fi + 1]->data;
      double tnf = cur_fh->shell->timestamp - nxt_fh->shell->timestamp;
      if ((nxt_fh->shell->trackingRef == cur_fh->shell) &&
          (-tnf < setting_maxImuInterval)) {
        int nxt_idx = CPARS + 3 + 29 * (fi + 1);
        double tnf2 = tnf * tnf;
        Vec3 d_vel_dso = (1 / tpf) * (prv_fh->PRE_camToWorld.translation() -
                                      cur_fh->PRE_camToWorld.translation()) -
                         (1 / tnf) * (cur_fh->PRE_camToWorld.translation() -
                                      nxt_fh->PRE_camToWorld.translation());
        Vec3 d_vel_imu = (tpf * cur_fh->spline_q + tpf2 * cur_fh->spline_c +
                          tnf * nxt_fh->spline_q + 2 * tnf2 * nxt_fh->spline_c)
                             .head(3);
        r_cst.segment<3>(3) = d_vel_imu - d_vel_dso;
        J_cst.block<3, 3>(3, prv_idx) = -SCALE_XI_TRANS / tpf * I33;
        J_cst.block<3, 3>(3, cur_idx) =
            SCALE_XI_TRANS * (1 / tpf + 1 / tnf) * I33;
        J_cst.block<3, 3>(3, nxt_idx) = -SCALE_XI_TRANS / tnf * I33;
        J_cst.block<3, 3>(3, cur_idx + 17) = SCALE_SQ_TRANS * tpf * I33;
        J_cst.block<3, 3>(3, cur_idx + 23) = SCALE_SC_TRANS * tpf2 * I33;
        J_cst.block<3, 3>(3, nxt_idx + 17) = SCALE_SQ_TRANS * tnf * I33;
        J_cst.block<3, 3>(3, nxt_idx + 23) = SCALE_SC_TRANS * 2 * tnf2 * I33;
      }
    }

    /********************** imu dynamics error *************************/
    Vec6 imu_pred_ave = Vec6::Zero();
    Vec6 imu_meas_ave = Vec6::Zero();
    size_t imu_size = cur_fh->imu_data.size();
    int count = 0;
    if (HCalib->scale_trapped) {
      H.block<3, 3>(CPARS, CPARS) += cur_fh->Hss;
      H.block<29, 3>(cur_idx, CPARS) += cur_fh->Hfs;
      H.block<3, 29>(CPARS, cur_idx) += cur_fh->Hfs.transpose();
      H.block<29, 29>(cur_idx, cur_idx) += cur_fh->Hff;
    }
    for (int j = 0; j < imu_size; j++) {
      // predict imu reading from spline
      double tt = cur_fh->imu_data[j].timestamp - cur_fh->shell->timestamp;
      assert(tt <= 0);

      Vec6 imu_pred;
      imu_pred.head(3) = setting_rot_imu_cam *
                         cur_fh->getSplineR_c_t(tt).transpose() *
                         cur_fh->PRE_worldToCam.rotationMatrix() *
                         (HCalib->getScaleScaled() * cur_fh->getSplineAcc(tt) +
                          HCalib->getG());
      imu_pred.tail(3) = setting_rot_imu_cam * cur_fh->getSplineGryo(tt);
      imu_pred += cur_fh->imu_bias;
      Vec6 imu_meas;
      imu_meas.head(3) = cur_fh->imu_data[j].acc;
      imu_meas.tail(3) = cur_fh->imu_data[j].gyro;
      Vec6 r_imu = imu_pred - imu_meas;

      if (HCalib->scale_trapped) {
        b.segment<3>(CPARS) += cur_fh->JsTW[j] * r_imu;
        b.segment<29>(cur_idx) += cur_fh->JfTW[j] * r_imu;
      } else { // not use FEJ when initialization
        Mat36 JsTW;
        Mat296 JfTW;
        Mat33 Hss;
        Mat2929 Hff;
        Mat293 Hfs;
        cur_fh->getImuHi(HCalib, tt, JsTW, JfTW, Hss, Hff, Hfs);
        H.block<3, 3>(CPARS, CPARS) += Hss;
        H.block<29, 3>(cur_idx, CPARS) += Hfs;
        H.block<3, 29>(CPARS, cur_idx) += Hfs.transpose();
        H.block<29, 29>(cur_idx, cur_idx) += Hff;

        b.segment<3>(CPARS) += JsTW * r_imu;
        b.segment<29>(cur_idx) += JfTW * r_imu;
      }

      if (print) {
        imu_pred_ave += imu_pred;
        imu_meas_ave += imu_meas;
        count++;
        if (count >= (imu_size / 5) || j == (imu_size - 1)) {
          imu_pred_ave /= count;
          imu_meas_ave /= count;
          printf("imu (%5.2f): %5.2f (%5.2f) %6.2f (%6.2f) %5.2f (%5.2f) %5.2f "
                 "(%5.2f) %5.2f (%5.2f) %5.2f (%5.2f) \n",
                 tt, imu_pred_ave[0], imu_meas_ave[0], imu_pred_ave[1],
                 imu_meas_ave[1], imu_pred_ave[2], imu_meas_ave[2],
                 imu_pred_ave[3], imu_meas_ave[3], imu_pred_ave[4],
                 imu_meas_ave[4], imu_pred_ave[5], imu_meas_ave[5]);
          imu_pred_ave.setZero();
          imu_meas_ave.setZero();
          count = 0;
        }
      }
    }
  }
  if (print) {
    printf("id: %d ba: %5.2f %5.2f %5.2f bg: %5.2f %5.2f %5.2f ",
           cur_fh->frameID, cur_fh->imu_bias[0], cur_fh->imu_bias[1],
           cur_fh->imu_bias[2], cur_fh->imu_bias[3], cur_fh->imu_bias[4],
           cur_fh->imu_bias[5]);
    if (spline_valid) {
      if (vel_valid) {
        printf("r_rv: %.0e %.0e\n", r_cst.head(3).norm(), r_cst.tail(3).norm());
      } else {
        printf("r_r:  %.0e\n", r_cst.norm());
      }
    } else {
      printf("\n");
    }
  }
}

void EnergyFunctional::getImuHessian(MatXX &H, VecX &b, MatXX &J_cst,
                                     VecX &r_cst, CalibHessian *HCalib,
                                     std::vector<bool> &is_spline_valid,
                                     bool print) {
  if (nFrames == 1)
    return;

  int dim = CPARS + 3 + 29 * nFrames;

  if (print) {
    FrameHessian *fh0 = frames[0]->data;
    printf("id: %d ba: %5.2f %5.2f %5.2f bg: %5.2f %5.2f %5.2f\n", fh0->frameID,
           fh0->imu_bias[0], fh0->imu_bias[1], fh0->imu_bias[2],
           fh0->imu_bias[3], fh0->imu_bias[4], fh0->imu_bias[5]);
  }

  // get H and b
  H = MatXX::Zero(dim, dim);
  b = VecX::Zero(dim);
  std::vector<MatXX> J_cst_vec;
  std::vector<VecX> r_cst_vec;
  is_spline_valid = std::vector<bool>(nFrames, false);
  for (int i = 1; i < nFrames; i++) {
    bool spline_valid;
    MatXX J_cst_i;
    VecX r_cst_i;
    getImuHessianCurrentFrame(i, HCalib, H, b, spline_valid, J_cst_i, r_cst_i,
                              print);
    if (spline_valid) {
      J_cst_vec.push_back(J_cst_i);
      r_cst_vec.push_back(r_cst_i);
      is_spline_valid[i] = true;
    }
  }

  // concatenate constraints
  int dim_cst = 0;
  for (const VecX &r_cst_i : r_cst_vec) {
    dim_cst += r_cst_i.size();
  }
  J_cst = MatXX::Zero(dim_cst, dim);
  r_cst = VecX::Zero(dim_cst);
  int cur_idx = 0;
  for (int i = 0; i < r_cst_vec.size(); i++) {
    int step = r_cst_vec[i].size();
    J_cst.block(cur_idx, 0, step, dim) = J_cst_vec[i];
    r_cst.segment(cur_idx, step) = r_cst_vec[i];
    cur_idx += step;
  }

  if (print) {
    printf("scale: %5.3f (%5.3f); roll_pitch: (%6.3f, %6.3f)\n\n",
           HCalib->getScaleScaled(), SCALE_SCALE * HCalib->sg_zero[0],
           HCalib->sg_scaled[1], HCalib->sg_scaled[2]);
  }
}

void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian *HCalib, bool MT) {
  assert(x.size() == CPARS + nFrames * 8);

  VecXf xF = x.cast<float>();
  HCalib->step = -x.head<CPARS>();

  Mat18f *xAd = new Mat18f[nFrames * nFrames];
  VecCf cstep = xF.head<CPARS>();
  for (EFFrame *h : frames) {
    h->data->step.head<8>() = -x.segment<8>(CPARS + 8 * h->idx);
    h->data->step.tail<2>().setZero();

    for (EFFrame *t : frames)
      xAd[nFrames * h->idx + t->idx] =
          xF.segment<8>(CPARS + 8 * h->idx).transpose() *
              adHostF[h->idx + nFrames * t->idx] +
          xF.segment<8>(CPARS + 8 * t->idx).transpose() *
              adTargetF[h->idx + nFrames * t->idx];
  }

  if (MT)
    red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt, this, cstep,
                            xAd, _1, _2, _3, _4),
                0, allPoints.size(), 50);
  else
    resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

  delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(const VecCf &xc, Mat18f *xAd, int min,
                                       int max, Vec10 *stats, int tid) {
  for (int k = min; k < max; k++) {
    EFPoint *p = allPoints[k];

    int ngoodres = 0;
    for (EFResidual *r : p->residualsAll)
      if (r->isActive())
        ngoodres++;
    if (ngoodres == 0) {
      p->data->step = 0;
      continue;
    }
    float b = p->bdSumF;
    b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

    for (EFResidual *r : p->residualsAll) {
      if (!r->isActive())
        continue;
      b -= xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
    }

    p->data->step = -b * p->HdiF;
    assert(std::isfinite(p->data->step));
  }
}

double EnergyFunctional::calcMEnergyF() {

  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  VecX delta = getStitchedDeltaF();
  return delta.dot(2 * bM + HM * delta);
}

void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10 *stats, int tid) {

  Accumulator11 E;
  E.initialize();
  VecCf dc = cDeltaF;

  for (int i = min; i < max; i++) {
    EFPoint *p = allPoints[i];
    float dd = p->deltaF;

    for (EFResidual *r : p->residualsAll) {
      if (!r->isLinearized || !r->isActive())
        continue;

      Mat18f dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
      RawResidualJacobian *rJ = r->J;

      // compute Jp*delta
      float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>()) +
                           rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd;

      float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>()) +
                           rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd;

      __m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
      __m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
      __m128 delta_a = _mm_set1_ps((float)(dp[6]));
      __m128 delta_b = _mm_set1_ps((float)(dp[7]));

      for (int i = 0; i + 3 < patternNum; i += 4) {
        // PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
        __m128 Jdelta =
            _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx)) + i), Jp_delta_x);
        Jdelta = _mm_add_ps(
            Jdelta,
            _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx + 1)) + i), Jp_delta_y));
        Jdelta = _mm_add_ps(
            Jdelta,
            _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF)) + i), delta_a));
        Jdelta = _mm_add_ps(
            Jdelta,
            _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF + 1)) + i), delta_b));

        __m128 r0 = _mm_load_ps(((float *)&r->res_toZeroF) + i);
        r0 = _mm_add_ps(r0, r0);
        r0 = _mm_add_ps(r0, Jdelta);
        Jdelta = _mm_mul_ps(Jdelta, r0);
        E.updateSSENoShift(Jdelta);
      }
      for (int i = ((patternNum >> 2) << 2); i < patternNum; i++) {
        float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 +
                       rJ->JIdx[1][i] * Jp_delta_y_1 + rJ->JabF[0][i] * dp[6] +
                       rJ->JabF[1][i] * dp[7];
        E.updateSingleNoShift(
            (float)(Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
      }
    }
    E.updateSingle(p->deltaF * p->deltaF * p->priorF);
  }
  E.finish();
  (*stats)[0] += E.A;
}

double EnergyFunctional::calcLEnergyF_MT() {
  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  double E = 0;
  for (EFFrame *f : frames)
    E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

  E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

  red->reduce(
      boost::bind(&EnergyFunctional::calcLEnergyPt, this, _1, _2, _3, _4), 0,
      allPoints.size(), 50);

  return E + red->stats[0];
}

EFResidual *EnergyFunctional::insertResidual(PointFrameResidual *r) {
  EFResidual *efr = new EFResidual(r, r->point->efPoint, r->host->efFrame,
                                   r->target->efFrame);
  efr->idxInAll = r->point->efPoint->residualsAll.size();
  r->point->efPoint->residualsAll.push_back(efr);

  connectivityMap[(((uint64_t)efr->host->frameID) << 32) +
                  ((uint64_t)efr->target->frameID)][0]++;

  nResiduals++;
  r->efResidual = efr;
  return efr;
}

EFFrame *EnergyFunctional::insertFrame(FrameHessian *fh, CalibHessian *HCalib) {
  EFFrame *eff = new EFFrame(fh);
  eff->idx = frames.size();
  frames.push_back(eff);

  nFrames++;
  fh->efFrame = eff;

  int step = 8;
  int ndim = CPARS + 8 * nFrames;
  if (setting_enable_imu) {
    step = 29;
    ndim = CPARS + 3 + 29 * nFrames;
  }
  assert(HM.cols() == ndim - step);
  bM.conservativeResize(ndim);
  HM.conservativeResize(ndim, ndim);
  bM.tail(step).setZero();
  HM.rightCols(step).setZero();
  HM.bottomRows(step).setZero();

  EFIndicesValid = false;
  EFAdjointsValid = false;
  EFDeltaValid = false;

  setAdjointsF(HCalib);
  makeIDX();

  for (EFFrame *fh2 : frames) {
    connectivityMap[(((uint64_t)eff->frameID) << 32) +
                    ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0, 0);
    if (fh2 != eff)
      connectivityMap[(((uint64_t)fh2->frameID) << 32) +
                      ((uint64_t)eff->frameID)] = Eigen::Vector2i(0, 0);
  }

  return eff;
}

EFPoint *EnergyFunctional::insertPoint(PointHessian *ph) {
  EFPoint *efp = new EFPoint(ph, ph->host->efFrame);
  efp->idxInPoints = ph->host->efFrame->points.size();
  ph->host->efFrame->points.push_back(efp);

  nPoints++;
  ph->efPoint = efp;

  EFIndicesValid = false;

  return efp;
}

void EnergyFunctional::dropResidual(EFResidual *r) {
  EFPoint *p = r->point;
  assert(r == p->residualsAll[r->idxInAll]);

  p->residualsAll[r->idxInAll] = p->residualsAll.back();
  p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
  p->residualsAll.pop_back();

  if (r->isActive())
    r->host->data->shell->statistics_goodResOnThis++;
  else
    r->host->data->shell->statistics_outlierResOnThis++;

  connectivityMap[(((uint64_t)r->host->frameID) << 32) +
                  ((uint64_t)r->target->frameID)][0]--;
  nResiduals--;
  r->data->efResidual = 0;
  delete r;
}

void EnergyFunctional::marginalizeFrame(EFFrame *fh, CalibHessian *HCalib) {
  // will not marginalize latest frame
  assert(fh->idx < (nFrames - 1));
  if (setting_enable_imu) {
    // assume imu is initialized before marginalization starts
    assert(HCalib->imu_initialized);
  }
  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);
  assert((int)fh->points.size() == 0);

  //	VecX eigenvaluesPre = HM.eigenvalues().real();
  //	std::sort(eigenvaluesPre.data(),
  // eigenvaluesPre.data()+eigenvaluesPre.size());
  //

  bool spline_valid;
  if (setting_enable_imu) {
    int dim = CPARS + 3 + 29 * nFrames;
    MatXX HM_change = MatXX::Zero(dim, dim);
    VecX bM_change = VecX::Zero(dim);
    MatXX J_cst;
    VecX r_cst;
    VecX delta = getStitchedDeltaF();
    VecX delta2 = VecX::Zero(dim);
    delta2.head(CPARS) = delta.head(CPARS);
    delta2.segment<3>(CPARS) = HCalib->sg - HCalib->sg_zero;
    // connection from fh->idx to fh->idx+1
    getImuHessianCurrentFrame(fh->idx + 1, HCalib, HM_change, bM_change,
                              spline_valid, J_cst, r_cst, false);
    delta2.segment<8>(CPARS + 3 + 29 * (fh->idx + 1)) =
        delta.segment<8>(CPARS + 8 * (fh->idx + 1));
    if (HCalib->scale_trapped) {
      delta2.segment<21>(CPARS + 3 + 29 * (fh->idx + 1) + 8) =
          frames[fh->idx + 1]->data->state_imu -
          frames[fh->idx + 1]->data->state_imu_zero;
    }

    if (fh->idx > 0) {
      // connection from fh->idx-1 to fh->idx
      getImuHessianCurrentFrame(fh->idx, HCalib, HM_change, bM_change,
                                spline_valid, J_cst, r_cst, false);
      delta2.segment<8>(CPARS + 3 + 29 * (fh->idx - 1)) =
          delta.segment<8>(CPARS + 8 * (fh->idx - 1));
      if (HCalib->scale_trapped) {
        delta2.segment<21>(CPARS + 3 + 29 * (fh->idx - 1) + 8) =
            frames[fh->idx - 1]->data->state_imu -
            frames[fh->idx - 1]->data->state_imu_zero;
      }
    } else {
      spline_valid = false;
    }

    bM_change -= HM_change * delta2;
    HM += setting_margWeightFac * HM_change;
    bM += setting_margWeightFac * bM_change;
  }

  int args_count = CPARS + (setting_enable_imu ? 3 : 0);
  int step = setting_enable_imu ? 29 : 8;
  int odim = args_count + nFrames * step; // old dimension
  int ndim = odim - step;                 // new dimension

  int io = args_count + fh->idx * step; // index of frame to move to end
  int ntail = step * (nFrames - fh->idx - 1);
  assert((io + step + ntail) == args_count + nFrames * step);

  VecX bTmp = bM.segment(io, step);
  VecX tailTMP = bM.tail(ntail);
  bM.segment(io, ntail) = tailTMP;
  bM.tail(step) = bTmp;

  MatXX HtmpCol = HM.block(0, io, odim, step);
  MatXX rightColsTmp = HM.rightCols(ntail);
  HM.block(0, io, odim, ntail) = rightColsTmp;
  HM.rightCols(step) = HtmpCol;

  MatXX HtmpRow = HM.block(io, 0, step, odim);
  MatXX botRowsTmp = HM.bottomRows(ntail);
  HM.block(io, 0, ntail, odim) = botRowsTmp;
  HM.bottomRows(step) = HtmpRow;

  // marginalize. First add prior here, instead of to active. ToDo
  HM.block(io + ntail, io + ntail, 8, 8).diagonal() += fh->prior;
  bM.segment<8>(io + ntail) += fh->prior.cwiseProduct(fh->delta_prior);

  //	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";

  // discard spline parts if not constrained
  if (setting_enable_imu && !spline_valid) {
    HM = HM.topLeftCorner(odim - 15, odim - 15).eval();
    bM = bM.head(odim - 15).eval();
    step = 14;
  }

  VecX SVec =
      (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
  VecX SVecI = SVec.cwiseInverse();

  //	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() <<
  //"\n\n"; 	std::cout << std::setprecision(16) << "SVecI: " <<
  // SVecI.transpose()
  //<< "\n\n";

  // scale!
  MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
  VecX bMScaled = SVecI.asDiagonal() * bM;

  // invert bottom part!
  MatXX hpi = HMScaled.bottomRightCorner(step, step);
  hpi = 0.5f * (hpi + hpi);
  hpi = hpi.inverse();
  hpi = 0.5f * (hpi + hpi);
  assert(std::isfinite(hpi(0, 0)));

  // schur-complement!
  MatXX bli = HMScaled.bottomLeftCorner(step, ndim).transpose() * hpi;
  HMScaled.topLeftCorner(ndim, ndim).noalias() -=
      bli * HMScaled.bottomLeftCorner(step, ndim);
  bMScaled.head(ndim).noalias() -= bli * bMScaled.tail(step);

  // unscale!
  HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
  bMScaled = SVec.asDiagonal() * bMScaled;

  // set.
  HM = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) +
              HMScaled.topLeftCorner(ndim, ndim).transpose());
  bM = bMScaled.head(ndim);

  // remove from vector, without changing the order!
  for (unsigned int i = fh->idx; i + 1 < frames.size(); i++) {
    frames[i] = frames[i + 1];
    frames[i]->idx = i;
  }
  frames.pop_back();
  nFrames--;
  fh->data->efFrame = 0;

  // assert(args_count + (int)frames.size() * step == (int)HM.rows());
  // assert(args_count + (int)frames.size() * step == (int)HM.cols());
  // assert(args_count + (int)frames.size() * step == (int)bM.size());
  assert((int)frames.size() == (int)nFrames);

  //	VecX eigenvaluesPost = HM.eigenvalues().real();
  //	std::sort(eigenvaluesPost.data(),
  // eigenvaluesPost.data()+eigenvaluesPost.size());

  //	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

  //	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
  //	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

  EFIndicesValid = false;
  EFAdjointsValid = false;
  EFDeltaValid = false;

  makeIDX();
  delete fh;
}

void EnergyFunctional::marginalizePointsF() {
  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  allPointsToMarg.clear();
  for (EFFrame *f : frames) {
    for (int i = 0; i < (int)f->points.size(); i++) {
      EFPoint *p = f->points[i];
      if (p->stateFlag == EFPointStatus::PS_MARGINALIZE) {
        p->priorF *= setting_idepthFixPriorMargFac;
        for (EFResidual *r : p->residualsAll)
          if (r->isActive())
            connectivityMap[(((uint64_t)r->host->frameID) << 32) +
                            ((uint64_t)r->target->frameID)][1]++;
        allPointsToMarg.push_back(p);
      }
    }
  }

  accSSE_bot->setZero(nFrames);
  accSSE_top_A->setZero(nFrames);
  for (EFPoint *p : allPointsToMarg) {
    accSSE_top_A->addPoint<2>(p, this);
    accSSE_bot->addPoint(p, false);
    removePoint(p);
  }
  MatXX M, Msc;
  VecX Mb, Mbsc;
  accSSE_top_A->stitchDouble(M, Mb, this, false, false);
  accSSE_bot->stitchDouble(Msc, Mbsc, this);

  resInM += accSSE_top_A->nres[0];

  MatXX H = M - Msc;
  VecX b = Mb - Mbsc;

  if (setting_enable_imu) {
    expandHbtoFitImu(H, b);
  }
  HM += setting_margWeightFac * H;
  bM += setting_margWeightFac * b;

  EFIndicesValid = false;
  makeIDX();
}

void EnergyFunctional::dropPointsF() {

  for (EFFrame *f : frames) {
    for (int i = 0; i < (int)f->points.size(); i++) {
      EFPoint *p = f->points[i];
      if (p->stateFlag == EFPointStatus::PS_DROP) {
        removePoint(p);
        i--;
      }
    }
  }

  EFIndicesValid = false;
  makeIDX();
}

void EnergyFunctional::removePoint(EFPoint *p) {
  for (EFResidual *r : p->residualsAll)
    dropResidual(r);

  EFFrame *h = p->host;
  h->points[p->idxInPoints] = h->points.back();
  h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
  h->points.pop_back();

  nPoints--;
  p->data->efPoint = 0;

  EFIndicesValid = false;

  delete p;
}

void EnergyFunctional::orthogonalize(VecX *b, MatXX *H) {
  //	VecX eigenvaluesPre = H.eigenvalues().real();
  //	std::sort(eigenvaluesPre.data(),
  // eigenvaluesPre.data()+eigenvaluesPre.size()); 	std::cout << "EigPre:: "
  // << eigenvaluesPre.transpose() << "\n";

  // decide to which nullspaces to orthogonalize.
  std::vector<VecX> ns;
  ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
  ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
  //	if(setting_affineOptModeA <= 0)
  //		ns.insert(ns.end(), lastNullspaces_affA.begin(),
  // lastNullspaces_affA.end()); 	if(setting_affineOptModeB <= 0)
  //		ns.insert(ns.end(), lastNullspaces_affB.begin(),
  // lastNullspaces_affB.end());

  // make Nullspaces matrix
  MatXX N(ns[0].rows(), ns.size());
  for (unsigned int i = 0; i < ns.size(); i++)
    N.col(i) = ns[i].normalized();

  // compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
  Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

  VecX SNN = svdNN.singularValues();
  double minSv = 1e10, maxSv = 0;
  for (int i = 0; i < SNN.size(); i++) {
    if (SNN[i] < minSv)
      minSv = SNN[i];
    if (SNN[i] > maxSv)
      maxSv = SNN[i];
  }
  for (int i = 0; i < SNN.size(); i++) {
    if (SNN[i] > setting_solverModeDelta * maxSv)
      SNN[i] = 1.0 / SNN[i];
    else
      SNN[i] = 0;
  }

  MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() *
              svdNN.matrixV().transpose();          // [dim] x 9.
  MatXX NNpiT = N * Npi.transpose();                // [dim] x [dim].
  MatXX NNpiTS = 0.5 * (NNpiT + NNpiT.transpose()); // = N * (N' * N)^-1 * N'.

  if (b != 0)
    *b -= NNpiTS * *b;
  if (H != 0)
    *H -= NNpiTS * *H * NNpiTS;

  //	std::cout << std::setprecision(16) << "Orth SV: " <<
  // SNN.reverse().transpose() << "\n";

  //	VecX eigenvaluesPost = H.eigenvalues().real();
  //	std::sort(eigenvaluesPost.data(),
  // eigenvaluesPost.data()+eigenvaluesPost.size()); 	std::cout << "EigPost::
  // " << eigenvaluesPost.transpose() << "\n";
}

void EnergyFunctional::solveSystemF(int iteration, double lambda,
                                    CalibHessian *HCalib) {
  lambda = 1e-5;

  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  MatXX HL_top, HA_top, H_sc;
  VecX bL_top, bA_top, bM_top, b_sc;

  accumulateAF_MT(HA_top, bA_top, multiThreading);

  accumulateLF_MT(HL_top, bL_top, multiThreading);

  accumulateSCF_MT(H_sc, b_sc, multiThreading);

  MatXX HFinal_top = HL_top + HA_top;
  VecX bFinal_top = bL_top + bA_top;

  // VecX eigenvalues = HFinal_top.eigenvalues().real();
  // std::sort(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());
  // std::cout << "dso " << eigenvalues.transpose() << std::endl;

  MatXX J_cst;
  VecX r_cst;
  std::vector<bool> is_spline_valid;
  bool imu_valid = setting_enable_imu && HCalib->imu_initialized;
  int dim = imu_valid ? CPARS + 3 + 29 * nFrames : 8 * nFrames + CPARS;
  if (imu_valid) {
    /************************* get imu H b *****************************/
    MatXX H_imu;
    VecX b_imu;
    getImuHessian(H_imu, b_imu, J_cst, r_cst, HCalib, is_spline_valid, false);

    /************************* add dso H b *****************************/
    expandHbtoFitImu(HFinal_top, bFinal_top);
    HFinal_top += H_imu;
    bFinal_top += b_imu;
  }

  /******************** add marginalized H b *************************/
  if (!setting_enable_imu || HCalib->imu_initialized) {
    VecX delta = getStitchedDeltaF();
    if (setting_enable_imu) {
      VecX delta2 = VecX::Zero(dim);
      delta2.head(CPARS) = delta.head(CPARS);
      delta2.segment<3>(CPARS) = HCalib->sg - HCalib->sg_zero;
      for (int i = 0; i < nFrames; i++) {
        delta2.segment<8>(CPARS + 3 + 29 * i) = delta.segment<8>(CPARS + 8 * i);
        if (HCalib->scale_trapped) {
          delta2.segment<21>(CPARS + 3 + 29 * i + 8) =
              frames[i]->data->state_imu - frames[i]->data->state_imu_zero;
        }
      }
      delta = delta2.eval();
    }
    bM_top = (bM + HM * delta);
    HFinal_top += HM;
    bFinal_top += bM_top;
  } // else: when imu is not initialized, marginalization shouldn't start

  /************************** add SC H b *******************************/
  if (imu_valid) {
    expandHbtoFitImu(H_sc, b_sc);
  }
  for (int i = 0; i < dim; i++)
    HFinal_top(i, i) *= (1 + lambda);
  HFinal_top -= H_sc * (1.0f / (1 + lambda));
  bFinal_top -= b_sc;

  if (imu_valid) {
    /********************* add constraint J r **************************/
    int cdim = r_cst.size();
    HFinal_top.conservativeResize(dim + cdim, dim + cdim);
    HFinal_top.block(0, dim, dim, cdim) = J_cst.transpose();
    HFinal_top.block(dim, 0, cdim, dim) = J_cst;
    HFinal_top.bottomRightCorner(cdim, cdim).setZero();
    bFinal_top.conservativeResize(dim + cdim);
    bFinal_top.tail(cdim) = r_cst;
    dim += cdim;

    /*************** remove unconstraint imu items *********************/
    int vi = CPARS + 1;
    if (HCalib->scale_trapped) {
      vi += 2;
    }
    for (int i = 0; i < nFrames; i++) {
      int fi = CPARS + 3 + 29 * i;
      int vs = is_spline_valid[i] ? 29 : 14;
      HFinal_top.block(0, vi, dim, vs) =
          HFinal_top.block(0, fi, dim, vs).eval();
      HFinal_top.block(vi, 0, vs, dim) =
          HFinal_top.block(fi, 0, vs, dim).eval();
      bFinal_top.segment(vi, vs) = bFinal_top.segment(fi, vs).eval();
      vi += vs;
    }
    int fi = CPARS + 3 + 29 * nFrames;
    HFinal_top.block(0, vi, dim, cdim) =
        HFinal_top.block(0, fi, dim, cdim).eval();
    HFinal_top.block(vi, 0, cdim, dim) =
        HFinal_top.block(fi, 0, cdim, dim).eval();
    bFinal_top.segment(vi, cdim) = bFinal_top.segment(fi, cdim).eval();
    HFinal_top = HFinal_top.topLeftCorner(vi + cdim, vi + cdim).eval();
    bFinal_top = bFinal_top.head(vi + cdim).eval();

    // eigenvalues = HFinal_top.topLeftCorner(vi, vi).eigenvalues().real();
    // std::sort(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());
    // std::cout << "imu " << eigenvalues.transpose() << std::endl << std::endl;
    // if (HCalib->scale_trapped) {
    //   exit(1);
    // }
  }

  /************************ solve system *****************************/
  VecX SVecI = (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10))
                   .cwiseSqrt()
                   .cwiseInverse();
  MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
  VecX x = SVecI.asDiagonal() *
           HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);

  if (imu_valid) {
    VecX x_dso = VecX::Zero(CPARS + 8 * nFrames);
    x_dso.head(CPARS) = x.head(CPARS);
    HCalib->sg_step.setZero();
    HCalib->sg_step[0] = -x(CPARS);
    int vi = CPARS + 1;
    if (HCalib->scale_trapped) {
      HCalib->sg_step.tail(2) = -x.segment<2>(CPARS + 1);
      vi += 2;
    }
    for (int i = 0; i < nFrames; i++) {
      x_dso.segment<8>(CPARS + i * 8) = x.segment<8>(vi);
      vi += 8;
      frames[i]->data->step_imu.setZero();
      frames[i]->data->step_imu.head(6) = -x.segment<6>(vi);
      vi += 6;
      if (is_spline_valid[i]) {
        frames[i]->data->step_imu.tail(15) = -x.segment<15>(vi);
        vi += 15;
      }
    }
    x = x_dso.eval();
  }

  // if (iteration >= 2) {
  //   VecX xOld = x;
  //   orthogonalize(&x, 0);
  // }

  lastX = x;

  // resubstituteF(x, HCalib);
  currentLambda = lambda;
  resubstituteF_MT(x, HCalib, multiThreading);
  currentLambda = 0;
}

void EnergyFunctional::makeIDX() {
  for (unsigned int idx = 0; idx < frames.size(); idx++)
    frames[idx]->idx = idx;

  allPoints.clear();

  for (EFFrame *f : frames)
    for (EFPoint *p : f->points) {
      allPoints.push_back(p);
      for (EFResidual *r : p->residualsAll) {
        r->hostIDX = r->host->idx;
        r->targetIDX = r->target->idx;
      }
    }

  EFIndicesValid = true;
}

VecX EnergyFunctional::getStitchedDeltaF() const {
  VecX d = VecX(CPARS + nFrames * 8);
  d.head<CPARS>() = cDeltaF.cast<double>();
  for (int h = 0; h < nFrames; h++)
    d.segment<8>(CPARS + 8 * h) = frames[h]->delta;
  return d;
}

} // namespace dso
