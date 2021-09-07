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

#include "HessianBlocks.h"
#include "ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "util/FrameShell.h"

namespace dso {

PointHessian::PointHessian(const ImmaturePoint *const rawPoint,
                           CalibHessian *HCalib) {
  instanceCounter++;
  host = rawPoint->host;
  hasDepthPrior = false;

  idepth_hessian = 0;
  maxRelBaseline = 0;
  numGoodResiduals = 0;

  // set static values & initialization.
  u = rawPoint->u;
  v = rawPoint->v;
  assert(std::isfinite(rawPoint->idepth_max));
  // idepth_init = rawPoint->idepth_GT;

  my_type = rawPoint->my_type;

  setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min) * 0.5);
  setPointStatus(PointHessian::INACTIVE);

  int n = patternNum;
  memcpy(color, rawPoint->color, sizeof(float) * n);
  memcpy(weights, rawPoint->weights, sizeof(float) * n);
  energyTH = rawPoint->energyTH;

  efPoint = 0;
}

void PointHessian::release() {
  for (unsigned int i = 0; i < residuals.size(); i++)
    delete residuals[i];
  residuals.clear();
}

void FrameHessian::setStateZero(const Vec10 &state_zero) {
  assert(state_zero.head<6>().squaredNorm() < 1e-20);

  this->state_zero = state_zero;

  for (int i = 0; i < 6; i++) {
    Vec6 eps;
    eps.setZero();
    eps[i] = 1e-3;
    SE3 EepsP = Sophus::SE3::exp(eps);
    SE3 EepsM = Sophus::SE3::exp(-eps);
    SE3 c2w_leftEps_P_x0 =
        (get_camToWorld_evalPT() * EepsP) * get_camToWorld_evalPT().inverse();
    SE3 c2w_leftEps_M_x0 =
        (get_camToWorld_evalPT() * EepsM) * get_camToWorld_evalPT().inverse();
    nullspaces_pose.col(i) =
        (c2w_leftEps_P_x0.log() - c2w_leftEps_M_x0.log()) / (2e-3);
  }
  // nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
  // nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

  // scale change
  SE3 c2w_leftEps_P_x0 = (get_camToWorld_evalPT());
  c2w_leftEps_P_x0.translation() *= 1.00001;
  c2w_leftEps_P_x0 = c2w_leftEps_P_x0 * get_camToWorld_evalPT().inverse();
  SE3 c2w_leftEps_M_x0 = (get_camToWorld_evalPT());
  c2w_leftEps_M_x0.translation() /= 1.00001;
  c2w_leftEps_M_x0 = c2w_leftEps_M_x0 * get_camToWorld_evalPT().inverse();
  nullspaces_scale = (c2w_leftEps_P_x0.log() - c2w_leftEps_M_x0.log()) / (2e-3);

  nullspaces_affine.setZero();
  nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
  assert(ab_exposure > 0);
  nullspaces_affine.topRightCorner<2, 1>() =
      Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
};

void FrameHessian::release() {
  // DELETE POINT
  // DELETE RESIDUAL
  for (unsigned int i = 0; i < pointHessians.size(); i++)
    delete pointHessians[i];
  for (unsigned int i = 0; i < pointHessiansMarginalized.size(); i++)
    delete pointHessiansMarginalized[i];
  for (unsigned int i = 0; i < pointHessiansOut.size(); i++)
    delete pointHessiansOut[i];
  for (unsigned int i = 0; i < immaturePoints.size(); i++)
    delete immaturePoints[i];

  pointHessians.clear();
  pointHessiansMarginalized.clear();
  pointHessiansOut.clear();
  immaturePoints.clear();
}

void FrameHessian::makeImages(float *color, CalibHessian *HCalib) {

  for (int i = 0; i < pyrLevelsUsed; i++) {
    dIp[i] = new Eigen::Vector3f[wG[i] * hG[i]];
    absSquaredGrad[i] = new float[wG[i] * hG[i]];
  }
  dI = dIp[0];

  // make d0
  int w = wG[0];
  int h = hG[0];
  for (int i = 0; i < w * h; i++)
    dI[i][0] = color[i];

  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    int wl = wG[lvl], hl = hG[lvl];
    Eigen::Vector3f *dI_l = dIp[lvl];

    float *dabs_l = absSquaredGrad[lvl];
    if (lvl > 0) {
      int lvlm1 = lvl - 1;
      int wlm1 = wG[lvlm1];
      Eigen::Vector3f *dI_lm = dIp[lvlm1];

      for (int y = 0; y < hl; y++)
        for (int x = 0; x < wl; x++) {
          dI_l[x + y * wl][0] =
              0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
                       dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
                       dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                       dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
        }
    }

    for (int idx = wl; idx < wl * (hl - 1); idx++) {
      float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
      float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

      if (!std::isfinite(dx))
        dx = 0;
      if (!std::isfinite(dy))
        dy = 0;

      dI_l[idx][1] = dx;
      dI_l[idx][2] = dy;

      dabs_l[idx] = dx * dx + dy * dy;

      if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0) {
        float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
        dabs_l[idx] *= gw * gw; // convert to gradient of original color space
                                // (before removing response).
      }
    }
  }
}

void FrameHessian::getImuHi(CalibHessian *HCalib, double tt, Mat36 &JsTW,
                            Mat296 &JfTW, Mat33 &Hss, Mat2929 &Hff,
                            Mat293 &Hfs) {
  assert(tt <= 0);
  double tt2 = tt * tt;

  double scale_scaled = HCalib->getScaleScaled(HCalib->scale_trapped);
  Vec3 spline_acc = getSplineAcc(tt, HCalib->scale_trapped);
  Vec3 acc_w = scale_scaled * spline_acc + HCalib->getG(HCalib->scale_trapped);
  Mat33 rot_t_w = getSplineR_c_t(tt, HCalib->scale_trapped).transpose() *
                  get_camToWorld_evalPT().rotationMatrix().transpose();
  Mat33 rot_i_w = setting_rot_imu_cam * rot_t_w;
  Mat33 R_acc_t_hat = setting_rot_imu_cam * SO3::hat(rot_t_w * acc_w);

  Eigen::Matrix<double, 6, 3> Js;
  Js.setZero();
  // scale
  Js.block<3, 1>(0, 0) = SCALE_SCALE * rot_i_w * spline_acc;

  Eigen::Matrix<double, 6, 29> Jf;
  Jf.setZero();
  // acc
  Jf.block<3, 3>(0, 8) = SCALE_BA * Mat33::Identity(); // bias_a
  Jf.block<3, 3>(0, 14) = SCALE_SL_ROT * R_acc_t_hat * tt;
  Jf.block<3, 3>(0, 20) = SCALE_SQ_ROT * R_acc_t_hat * tt2;
  Jf.block<3, 3>(0, 26) = SCALE_SC_ROT * R_acc_t_hat * tt * tt2;
  Jf.block<3, 3>(0, 17) = SCALE_SQ_TRANS * rot_i_w * 2 * scale_scaled;
  Jf.block<3, 3>(0, 23) = SCALE_SC_TRANS * rot_i_w * 6 * tt * scale_scaled;

  // gyro
  Jf.block<3, 3>(3, 11) = SCALE_BG * Mat33::Identity(); // bias_g
  Jf.block<3, 3>(3, 14) = SCALE_SL_ROT * setting_rot_imu_cam;
  Jf.block<3, 3>(3, 20) = SCALE_SQ_ROT * setting_rot_imu_cam * 2 * tt;
  Jf.block<3, 3>(3, 26) = SCALE_SC_ROT * setting_rot_imu_cam * 3 * tt2;

  // do not adjust gravity and dso parts when scale has not been trapped
  if (HCalib->scale_trapped) {
    // gravity
    double sr, cr, sp, cp;
    HCalib->getGSinCos(sr, cr, sp, cp, true);
    Eigen::Matrix<double, 3, 2> J_rp;
    J_rp.col(0) << -sp * sr, -cr, -cp * sr;
    J_rp.col(1) << cp * cr, 0, -sp * cr;
    J_rp *= setting_g_norm;
    Js.block<3, 2>(0, 1) = SCALE_G * rot_i_w * J_rp;

    // acc w.r.t. dso rotation
    Jf.block<3, 3>(0, 3) = SCALE_XI_ROT * rot_i_w * SO3::hat(acc_w);
  }

  JsTW = Js.transpose() * setting_weight_imu;
  JfTW = Jf.transpose() * setting_weight_imu;
  Hss = JsTW * Js;
  Hff = JfTW * Jf;
  Hfs = JfTW * Js;
}

void FrameHessian::setImuStateZero(CalibHessian *HCalib) {
  state_imu_zero = state_imu;

  if (HCalib->scale_trapped) {
    JsTW.clear();
    JfTW.clear();
    JsTW.reserve(imu_data.size());
    JfTW.reserve(imu_data.size());
    Hss.setZero();
    Hff.setZero();
    Hfs.setZero();
    for (int i = 0; i < imu_data.size(); i++) {
      double tt = imu_data[i].timestamp - shell->timestamp;
      Mat36 JsTWi;
      Mat296 JfTWi;
      Mat33 Hssi;
      Mat2929 Hffi;
      Mat293 Hfsi;
      getImuHi(HCalib, tt, JsTWi, JfTWi, Hssi, Hffi, Hfsi);
      JsTW.push_back(JsTWi);
      JfTW.push_back(JfTWi);
      Hss += Hssi;
      Hff += Hffi;
      Hfs += Hfsi;
    }
  }
}

bool FrameHessian::initializeImu(
    const std::vector<FrameHessian *> &frame_hessians, CalibHessian *HCalib) {
  assert(frame_hessians.size() == 5);
  FrameHessian *base_frame = frame_hessians.back();

  // calculate the spline
  Mat33 A = Mat33::Zero();
  Eigen::Matrix<double, 3, 6> b = Eigen::Matrix<double, 3, 6>::Zero();
  for (int i = 0; i < 3; i++) {
    FrameHessian *cur_frame = frame_hessians[i + 1];
    A(i, 0) = cur_frame->shell->timestamp - base_frame->shell->timestamp;
    // printf("t_diff %f\n", A(i, 0));
    A(i, 1) = A(i, 0) * A(i, 0);
    A(i, 2) = A(i, 1) * A(i, 0);
    b.row(i) =
        (base_frame->shell->camToWorld.inverse() * cur_frame->shell->camToWorld)
            .log();
    b.row(i).head(3) = cur_frame->shell->camToWorld.translation() -
                       base_frame->shell->camToWorld.translation();
  }
  Eigen::Matrix<double, 3, 6> x = A.inverse() * b;
  Vec6 l0 = x.row(0);
  Vec6 q0 = x.row(1);
  Vec6 c0 = x.row(2);

  for (auto fh : frame_hessians) {
    // calculate spline with time offset
    double t0 = fh->shell->timestamp - base_frame->shell->timestamp;
    Vec6 vel = l0 + 2 * q0 * t0 + 3 * c0 * t0 * t0;
    fh->shell->velInWorld = vel.head(3);
    fh->spline_l_rot = vel.tail(3);
    fh->spline_q = q0 + 3 * c0 * t0;
    fh->spline_c = c0;
  }

  std::vector<ImuData> all_imu_data;
  for (int i = 2; i < 5; i++) {
    all_imu_data.insert(all_imu_data.end(), frame_hessians[i]->imu_data.begin(),
                        frame_hessians[i]->imu_data.end());
  }

  // gyro bias
  Vec3 gyro_bias = Vec3::Zero();
  for (size_t i = 0; i < all_imu_data.size(); i++) {
    double t = all_imu_data[i].timestamp - base_frame->shell->timestamp;
    Vec3 gyro_pred = setting_rot_imu_cam * base_frame->getSplineGryo(t);
    // std::cout << gyro_pred.transpose() << " "
    //           << all_imu_data[i].gyro.transpose() << std::endl;
    gyro_bias += (all_imu_data[i].gyro - gyro_pred);
  }
  gyro_bias /= all_imu_data.size();
  for (auto fh : frame_hessians) {
    fh->imu_bias.tail(3) = gyro_bias;
  }

  // scale and acc bias
  // MatXX A_s_ba = MatXX::Zero(3 * all_imu_data.size(), 4);
  VecX A_s_ba = VecX::Zero(3 * all_imu_data.size());
  VecX b_s_ba = VecX::Zero(3 * all_imu_data.size());
  for (size_t i = 0; i < all_imu_data.size(); i++) {
    double t = all_imu_data[i].timestamp - base_frame->shell->timestamp;
    Mat33 rot_ti_w = setting_rot_imu_cam *
                     base_frame->getSplineR_c_t(t).transpose() *
                     base_frame->PRE_worldToCam.rotationMatrix();
    Vec3 acc_pred = rot_ti_w * base_frame->getSplineAcc(t);
    // A_s_ba.row(3 * i + 0) << acc_pred(0), 1, 0, 0;
    // A_s_ba.row(3 * i + 1) << acc_pred(1), 0, 1, 0;
    // A_s_ba.row(3 * i + 2) << acc_pred(2), 0, 0, 1;
    A_s_ba.segment<3>(3 * i) = acc_pred;
    Vec3 acc_meas = all_imu_data[i].acc - rot_ti_w * HCalib->getG(true);
    b_s_ba.segment<3>(3 * i) = acc_meas;
    // printf("Acc: %6.3f(%6.3f) %6.3f(%6.3f) %6.3f(%6.3f)\n", acc_pred(0),
    //        acc_meas(0), acc_pred(1), acc_meas(1), acc_pred(2), acc_meas(2));
  }
  // Vec4 s_ba =
  //     (A_s_ba.transpose() * A_s_ba).inverse() * A_s_ba.transpose() * b_s_ba;
  double scale = A_s_ba.dot(b_s_ba) / A_s_ba.dot(A_s_ba);

  if (setting_print_imu) {
    printf("Init: scale: %.2f; ba: 0, 0, 0; bg: %.2f, %.2f, %.2f\n\n", scale,
           gyro_bias(0), gyro_bias(1), gyro_bias(2));
  }

  if (scale < 0) {
    printf("initializeImu failed\n");
    return false;
  }

  for (auto fh : frame_hessians) {
    fh->imu_bias.head(3).setConstant(0.0);
  }

  Vec3 sg0;
  sg0 << scale, 0, 0;
  HCalib->setSgZero(sg0);
  HCalib->imu_initialized = true;

  for (auto fh : frame_hessians) {
    fh->setImuStateScaled(fh->getImuStateScaled());
    fh->setImuStateZero(HCalib);
  }
  return true;
}

void FrameHessian::propagateImuState(FrameShell *last_shell,
                                     const Vec6 &last_imu_bias,
                                     CalibHessian *HCalib) {
  imu_bias = last_imu_bias;

  double imu_ts = last_shell->timestamp;
  Mat33 imu_rot_w_ti = last_shell->camToWorld.rotationMatrix(); // ToDo
  MatXX Aa = MatXX::Zero(imu_data.size(), 3);
  MatXX ba = MatXX::Zero(imu_data.size(), 3);
  MatXX Ag = MatXX::Zero(imu_data.size(), 3);
  MatXX bg = MatXX::Zero(imu_data.size(), 3);
  for (int i = 0; i < imu_data.size(); i++) {
    double dt = imu_data[i].timestamp - imu_ts;
    assert(dt >= 0);
    imu_ts = imu_data[i].timestamp;

    double t = imu_data[i].timestamp - shell->timestamp;
    Vec3 unbias_acc = imu_data[i].acc - imu_bias.head(3);
    Vec3 unbias_gyro = imu_data[i].gyro - imu_bias.tail(3);

    // integrate gyro data
    imu_rot_w_ti = imu_rot_w_ti * SO3::exp(unbias_gyro * dt).matrix();

    // acc
    Aa.row(i) << 0, 2 * HCalib->getScaleScaled(),
        6 * t * HCalib->getScaleScaled();
    ba.row(i) = imu_rot_w_ti * setting_rot_imu_cam.transpose() * unbias_acc -
                HCalib->getG();
    // gyro
    Ag.row(i) << 1, 2 * t, 3 * t * t;
    bg.row(i) = setting_rot_imu_cam.transpose() * unbias_gyro;
  }

  Mat33 xa = (Aa.transpose() * Aa).inverse() * Aa.transpose() * ba;
  Mat33 xg = (Ag.transpose() * Ag).inverse() * Ag.transpose() * bg;
  spline_q.head(3) = xa.row(1);
  spline_c.head(3) = xa.row(2);
  spline_l_rot = xg.row(0);
  spline_q.tail(3) = xg.row(1);
  spline_c.tail(3) = xg.row(2);
  setImuStateScaled(getImuStateScaled());
  setImuStateZero(HCalib);

  // calculate current velocity
  double t = last_shell->timestamp - shell->timestamp;
  shell->velInWorld = last_shell->velInWorld -
                      (2 * t * spline_q + 3 * t * t * spline_c).head(3);
}

void FrameHessian::updateVel(FrameShell *last_shell) {
  double t = last_shell->timestamp - shell->timestamp;
  Vec3 tsl_diff =
      last_shell->camToWorld.translation() - shell->camToWorld.translation();
  shell->velInWorld =
      tsl_diff / t - t * spline_q.head(3) - t * t * spline_q.head(3);
}

void CalibHessian::tryTrapScale() {
  sg_zero[0] = sg[0];

  scale_queue(scale_queue_i) = sg[0];
  scale_queue_i = (scale_queue_i + 1) % 10;

  double var =
      1.0 / 9.0 * SCALE_SCALE * SCALE_SCALE *
      (scale_queue - Vec10::Constant(scale_queue.mean())).squaredNorm();
  if (var < setting_scale_trap_thres) {
    scale_trapped = true;
    sg_zero[0] = scale_queue.mean();
    if (setting_print_imu) {
      printf("scale trapped to %8.5f\n", SCALE_SCALE * sg_zero[0]);
      std::cout << SCALE_SCALE * scale_queue.transpose() << std::endl;
    }
  }
}

void FrameFramePrecalc::set(FrameHessian *host, FrameHessian *target,
                            CalibHessian *HCalib) {
  this->host = host;
  this->target = target;

  SE3 leftToLeft_0 =
      target->get_camToWorld_evalPT().inverse() * host->get_camToWorld_evalPT();
  PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
  PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();

  SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
  PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
  PRE_tTll = (leftToLeft.translation()).cast<float>();
  distanceLL = leftToLeft.translation().norm();

  Mat33f K = Mat33f::Zero();
  K(0, 0) = HCalib->fxl();
  K(1, 1) = HCalib->fyl();
  K(0, 2) = HCalib->cxl();
  K(1, 2) = HCalib->cyl();
  K(2, 2) = 1;
  PRE_KRKiTll = K * PRE_RTll * K.inverse();
  PRE_RKiTll = PRE_RTll * K.inverse();
  PRE_KtTll = K * PRE_tTll;

  PRE_aff_mode =
      AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure,
                                  host->aff_g2l(), target->aff_g2l())
          .cast<float>();
  PRE_b0_mode = host->aff_g2l_0().b;
}

} // namespace dso
