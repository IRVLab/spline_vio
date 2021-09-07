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

#pragma once
#define MAX_ACTIVE_FRAMES 100

#include "util/globalCalib.h"
#include "vector"

#include "Residuals.h"
#include "util/FrameShell.h"
#include "util/ImageAndExposure.h"
#include "util/NumType.h"
#include <fstream>
#include <iostream>

namespace dso {

inline Vec2 affFromTo(const Vec2 &from,
                      const Vec2 &to) // contains affine parameters as XtoWorld.
{
  return Vec2(from[0] / to[0], (from[1] - to[1]) / to[0]);
}

struct FrameHessian;
struct PointHessian;

class ImmaturePoint;

class EFFrame;
class EFPoint;

#define SCALE_IDEPTH 1.0f // scales internal value to idepth.
#define SCALE_XI_ROT 1.0f
#define SCALE_XI_TRANS 0.5f
#define SCALE_F 50.0f
#define SCALE_C 50.0f
#define SCALE_W 1.0f
#define SCALE_A 10.0f
#define SCALE_B 1000.0f

#define SCALE_IDEPTH_INVERSE (1.0f / SCALE_IDEPTH)
#define SCALE_XI_ROT_INVERSE (1.0f / SCALE_XI_ROT)
#define SCALE_XI_TRANS_INVERSE (1.0f / SCALE_XI_TRANS)
#define SCALE_F_INVERSE (1.0f / SCALE_F)
#define SCALE_C_INVERSE (1.0f / SCALE_C)
#define SCALE_W_INVERSE (1.0f / SCALE_W)
#define SCALE_A_INVERSE (1.0f / SCALE_A)
#define SCALE_B_INVERSE (1.0f / SCALE_B)

#define SCALE_SCALE 200.0f
#define SCALE_G 1000.0f
#define SCALE_SL_ROT 100.0f
#define SCALE_SQ_TRANS 1000.0f
#define SCALE_SQ_ROT 1000.0f
#define SCALE_SC_TRANS 1000.0f
#define SCALE_SC_ROT 1000.0f
#define SCALE_BA 100.0f
#define SCALE_BG 1.0f

#define SCALE_SCALE_INVERSE (1.0f / SCALE_SCALE)
#define SCALE_G_INVERSE (1.0f / SCALE_G)
#define SCALE_SL_ROT_INVERSE (1.0f / SCALE_SL_ROT)
#define SCALE_SQ_TRANS_INVERSE (1.0f / SCALE_SQ_TRANS)
#define SCALE_SQ_ROT_INVERSE (1.0f / SCALE_SQ_ROT)
#define SCALE_SC_TRANS_INVERSE (1.0f / SCALE_SC_TRANS)
#define SCALE_SC_ROT_INVERSE (1.0f / SCALE_SC_ROT)
#define SCALE_BA_INVERSE (1.0f / SCALE_BA)
#define SCALE_BG_INVERSE (1.0f / SCALE_BG)

typedef Eigen::Matrix<double, 3, 6> Mat36;
typedef Eigen::Matrix<double, 29, 3> Mat293;
typedef Eigen::Matrix<double, 29, 6> Mat296;
typedef Eigen::Matrix<double, 29, 29> Mat2929;

struct ImuData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  double timestamp;
  Vec3 acc;
  Vec3 gyro;

  ImuData(const Vec7 &imu_data) {
    timestamp = imu_data[0];
    acc = imu_data.segment<3>(1);
    gyro = imu_data.tail(3);
  }
};

struct FrameFramePrecalc {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  // static values
  static int instanceCounter;
  FrameHessian *host;   // defines row
  FrameHessian *target; // defines column

  // precalc values
  Mat33f PRE_RTll;
  Mat33f PRE_KRKiTll;
  Mat33f PRE_RKiTll;
  Mat33f PRE_RTll_0;

  Vec2f PRE_aff_mode;
  float PRE_b0_mode;

  Vec3f PRE_tTll;
  Vec3f PRE_KtTll;
  Vec3f PRE_tTll_0;

  float distanceLL;

  inline ~FrameFramePrecalc() {}
  inline FrameFramePrecalc() { host = target = 0; }
  void set(FrameHessian *host, FrameHessian *target, CalibHessian *HCalib);
};

struct FrameHessian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EFFrame *efFrame;

  // constant info & pre-calculated values
  // DepthImageWrap* frame;
  FrameShell *shell;

  Eigen::Vector3f *dI; // trace, fine tracking. Used for direction select (not
                       // for gradient histograms etc.)
  Eigen::Vector3f *
      dIp[PYR_LEVELS]; // coarse tracking / coarse initializer. NAN in [0] only.
  float *absSquaredGrad[PYR_LEVELS]; // only used for pixel select (histograms
                                     // etc.). no NAN.

  int frameID; // incremental ID for keyframes only!
  static int instanceCounter;
  int idx;

  // Photometric Calibration Stuff
  float frameEnergyTH; // set dynamically depending on tracking residual
  float ab_exposure;

  bool flaggedForMarginalization;

  std::vector<PointHessian *> pointHessians; // contains all ACTIVE points.
  std::vector<PointHessian *>
      pointHessiansMarginalized; // contains all MARGINALIZED points (= fully
                                 // marginalized, usually because point went
                                 // OOB.)
  std::vector<PointHessian *>
      pointHessiansOut; // contains all OUTLIER points (= discarded.).
  std::vector<ImmaturePoint *>
      immaturePoints; // contains all OUTLIER points (= discarded.).

  Mat66 nullspaces_pose;
  Mat42 nullspaces_affine;
  Vec6 nullspaces_scale;

  // variable info.
  SE3 camToWorld_evalPT;
  Vec10 state_zero;
  Vec10 state_scaled;
  Vec10 state; // [0-5: camToWorld-leftEps. 6-7: a,b]
  Vec10 step;
  Vec10 state_backup;

  EIGEN_STRONG_INLINE const SE3 &get_camToWorld_evalPT() const {
    return camToWorld_evalPT;
  }
  EIGEN_STRONG_INLINE const Vec10 &get_state_zero() const { return state_zero; }
  EIGEN_STRONG_INLINE const Vec10 &get_state() const { return state; }
  EIGEN_STRONG_INLINE const Vec10 &get_state_scaled() const {
    return state_scaled;
  }
  EIGEN_STRONG_INLINE const Vec10 get_state_minus_stateZero() const {
    return get_state() - get_state_zero();
  }

  // precalc values
  SE3 PRE_camToWorld;
  SE3 PRE_worldToCam;
  std::vector<FrameFramePrecalc, Eigen::aligned_allocator<FrameFramePrecalc>>
      targetPrecalc;
  MinimalImageB3 *debugImage;

  inline Vec6 c2w_leftEps() const { return get_state_scaled().head<6>(); }
  inline AffLight aff_g2l() const {
    return AffLight(get_state_scaled()[6], get_state_scaled()[7]);
  }
  inline AffLight aff_g2l_0() const {
    return AffLight(get_state_zero()[6] * SCALE_A,
                    get_state_zero()[7] * SCALE_B);
  }

  void setStateZero(const Vec10 &state_zero);
  inline void setState(const Vec10 &state) {

    this->state = state;
    state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
    state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
    state_scaled[6] = SCALE_A * state[6];
    state_scaled[7] = SCALE_B * state[7];
    state_scaled[8] = SCALE_A * state[8];
    state_scaled[9] = SCALE_B * state[9];

    PRE_camToWorld = SE3::exp(c2w_leftEps()) * get_camToWorld_evalPT();
    PRE_worldToCam = PRE_camToWorld.inverse();
    // setCurrentNullspace();
  };
  inline void setStateScaled(const Vec10 &state_scaled) {

    this->state_scaled = state_scaled;
    state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
    state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
    state[6] = SCALE_A_INVERSE * state_scaled[6];
    state[7] = SCALE_B_INVERSE * state_scaled[7];
    state[8] = SCALE_A_INVERSE * state_scaled[8];
    state[9] = SCALE_B_INVERSE * state_scaled[9];

    PRE_camToWorld = SE3::exp(c2w_leftEps()) * get_camToWorld_evalPT();
    PRE_worldToCam = PRE_camToWorld.inverse();
    // setCurrentNullspace();
  };
  inline void setEvalPT(const SE3 &camToWorld_evalPT, const Vec10 &state) {

    this->camToWorld_evalPT = camToWorld_evalPT;
    setState(state);
    setStateZero(state);
  };

  inline void setEvalPT_scaled(const SE3 &camToWorld_evalPT,
                               const AffLight &aff_g2l) {
    Vec10 initial_state = Vec10::Zero();
    initial_state[6] = aff_g2l.a;
    initial_state[7] = aff_g2l.b;
    this->camToWorld_evalPT = camToWorld_evalPT;
    setStateScaled(initial_state);
    setStateZero(this->get_state());
  };

  void release();

  inline ~FrameHessian() {
    assert(efFrame == 0);
    release();
    instanceCounter--;
    for (int i = 0; i < pyrLevelsUsed; i++) {
      delete[] dIp[i];
      delete[] absSquaredGrad[i];
    }

    if (debugImage != 0)
      delete debugImage;
  };
  inline FrameHessian()
      : imu_bias(state_imu_scaled.segment<6>(0)),
        spline_l_rot(state_imu_scaled.segment<3>(6)),
        spline_q(state_imu_scaled.segment<6>(9)),
        spline_c(state_imu_scaled.segment<6>(15)) {
    instanceCounter++;
    flaggedForMarginalization = false;
    frameID = -1;
    efFrame = 0;
    frameEnergyTH = 8 * 8 * patternNum;

    debugImage = 0;
  };

  void makeImages(float *color, CalibHessian *HCalib);

  inline Vec10 getPrior() {
    Vec10 p = Vec10::Zero();
    if (frameID == 0) {
      p.head<3>() = Vec3::Constant(setting_initialTransPrior);
      p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);

      p[6] = setting_initialAffAPrior;
      p[7] = setting_initialAffBPrior;
    } else {
      if (setting_affineOptModeA < 0)
        p[6] = setting_initialAffAPrior;
      else
        p[6] = setting_affineOptModeA;

      if (setting_affineOptModeB < 0)
        p[7] = setting_initialAffBPrior;
      else
        p[7] = setting_affineOptModeB;
    }
    p[8] = setting_initialAffAPrior;
    p[9] = setting_initialAffBPrior;
    return p;
  }

  inline Vec10 getPriorZero() { return Vec10::Zero(); }

  // IMU state: cubic spline (3x6) and imu bias (6)
  Vec21 state_imu;
  Vec21 state_imu_zero;
  Vec21 state_imu_scaled;
  Vec21 state_imu_backup;
  Vec21 step_imu;
  Eigen::Ref<Vec6> imu_bias;
  Eigen::Ref<Vec3> spline_l_rot; // linear rot
  Eigen::Ref<Vec6> spline_q;     // quadratic
  Eigen::Ref<Vec6> spline_c;     // cubic

  std::vector<ImuData> imu_data;

  // cached Jacobians/Hessians
  std::vector<Mat36, Eigen::aligned_allocator<Mat36>> JsTW;
  std::vector<Mat296, Eigen::aligned_allocator<Mat296>> JfTW;
  Mat33 Hss;
  Mat2929 Hff;
  Mat293 Hfs;

  inline void setImuData(const std::vector<Vec7> imu_data_vec7) {
    imu_data.clear();
    imu_data.reserve(imu_data_vec7.size());
    for (const Vec7 &data : imu_data_vec7) {
      imu_data.push_back(data);
    }
  }

  EIGEN_STRONG_INLINE const Vec21 &getImuState() const { return state_imu; }
  EIGEN_STRONG_INLINE const Vec21 &getImuStateScaled() const {
    return state_imu_scaled;
  }

  inline void setImuState(const Vec21 &new_state_imu) {
    state_imu = new_state_imu;
    state_imu_scaled.segment<3>(0) = SCALE_BA * state_imu.segment<3>(0);
    state_imu_scaled.segment<3>(3) = SCALE_BG * state_imu.segment<3>(3);
    state_imu_scaled.segment<3>(6) = SCALE_SL_ROT * state_imu.segment<3>(6);
    state_imu_scaled.segment<3>(9) = SCALE_SQ_TRANS * state_imu.segment<3>(9);
    state_imu_scaled.segment<3>(12) = SCALE_SQ_ROT * state_imu.segment<3>(12);
    state_imu_scaled.segment<3>(15) = SCALE_SC_TRANS * state_imu.segment<3>(15);
    state_imu_scaled.segment<3>(18) = SCALE_SC_ROT * state_imu.segment<3>(18);
  }

  inline void setImuStateScaled(const Vec21 &new_state_imu_scaled) {
    state_imu_scaled = new_state_imu_scaled;
    state_imu.segment<3>(0) = SCALE_BA_INVERSE * state_imu_scaled.segment<3>(0);
    state_imu.segment<3>(3) = SCALE_BG_INVERSE * state_imu_scaled.segment<3>(3);
    state_imu.segment<3>(6) =
        SCALE_SL_ROT_INVERSE * state_imu_scaled.segment<3>(6);
    state_imu.segment<3>(9) =
        SCALE_SQ_TRANS_INVERSE * state_imu_scaled.segment<3>(9);
    state_imu.segment<3>(12) =
        SCALE_SQ_ROT_INVERSE * state_imu_scaled.segment<3>(12);
    state_imu.segment<3>(15) =
        SCALE_SC_TRANS_INVERSE * state_imu_scaled.segment<3>(15);
    state_imu.segment<3>(18) =
        SCALE_SC_ROT_INVERSE * state_imu_scaled.segment<3>(18);
  }

  inline Vec3 getSplineAcc(double t, bool use_state_zero = false) const {
    Vec3 acc;
    if (!use_state_zero) {
      acc = (2 * spline_q + 6 * t * spline_c).head(3);
    } else {
      acc = 2 * SCALE_SQ_TRANS * state_imu_zero.segment<3>(9) +
            6 * t * SCALE_SC_TRANS * state_imu_zero.segment<3>(15);
    }
    return acc;
  }

  inline Vec3 getSplineGryo(double t) const {
    return spline_l_rot + (2 * t * spline_q + 3 * t * t * spline_c).tail(3);
  }

  inline Vec3 getSplineTw_c2t(double t) const {
    double t2 = t * t;
    return t * shell->velInWorld + (t2 * spline_q + t * t2 * spline_c).head(3);
  }

  inline Mat33 getSplineR_c_t(double t, bool use_state_zero = false) const {
    double t2 = t * t;
    Vec3 so3;
    if (!use_state_zero) {
      so3 = t * spline_l_rot + (t2 * spline_q + t * t2 * spline_c).tail(3);
    } else {
      so3 = t * SCALE_SL_ROT * state_imu_zero.segment<3>(6) +
            t2 * SCALE_SQ_ROT * state_imu_zero.segment<3>(12) +
            t * t2 * SCALE_SC_ROT * state_imu_zero.segment<3>(18);
    }
    return SO3::exp(so3).matrix();
  }

  void getImuHi(CalibHessian *HCalib, double tt, Mat36 &JsTW, Mat296 &JfTW,
                Mat33 &Hss, Mat2929 &Hff, Mat293 &Hfs);

  void setImuStateZero(CalibHessian *HCalib);

  static bool initializeImu(const std::vector<FrameHessian *> &frame_hessians,
                            CalibHessian *HCalib);

  void propagateImuState(FrameShell *last_shell, const Vec6 &last_imu_bias,
                         CalibHessian *HCalib);

  void updateVel(FrameShell *last_shell);
};

struct CalibHessian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  static int instanceCounter;

  VecC value_zero;
  VecC value_scaled;
  VecCf value_scaledf;
  VecCf value_scaledi;
  VecC value;
  VecC step;
  VecC value_backup;
  VecC value_minus_value_zero;

  bool imu_initialized;
  bool scale_trapped;
  Vec3 sg; // 0: scale; 1-2: roll pitch
  Vec3 sg_scaled;
  Vec3 sg_zero;
  Vec3 sg_backup;
  Vec3 sg_step;

  // for scale trapping
  Vec10 scale_queue;
  int scale_queue_i;

  inline ~CalibHessian() { instanceCounter--; }
  inline CalibHessian() {

    VecC initial_value = VecC::Zero();
    initial_value[0] = fxG[0];
    initial_value[1] = fyG[0];
    initial_value[2] = cxG[0];
    initial_value[3] = cyG[0];

    setValueScaled(initial_value);
    value_zero = value;
    value_minus_value_zero.setZero();

    instanceCounter++;
    for (int i = 0; i < 256; i++)
      Binv[i] = B[i] = i; // set gamma function to identity

    imu_initialized = false;
    scale_trapped = false;

    scale_queue = Vec10::LinSpaced(10, -10, -100);
    scale_queue_i = 0;
  };

  // normal mode: use the optimized parameters everywhere!
  inline float &fxl() { return value_scaledf[0]; }
  inline float &fyl() { return value_scaledf[1]; }
  inline float &cxl() { return value_scaledf[2]; }
  inline float &cyl() { return value_scaledf[3]; }
  inline float &fxli() { return value_scaledi[0]; }
  inline float &fyli() { return value_scaledi[1]; }
  inline float &cxli() { return value_scaledi[2]; }
  inline float &cyli() { return value_scaledi[3]; }

  inline void setValue(const VecC &value) {
    // [0-3: Kl, 4-7: Kr, 8-12: l2r]
    this->value = value;
    value_scaled[0] = SCALE_F * value[0];
    value_scaled[1] = SCALE_F * value[1];
    value_scaled[2] = SCALE_C * value[2];
    value_scaled[3] = SCALE_C * value[3];

    this->value_scaledf = this->value_scaled.cast<float>();
    this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
    this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
    this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
    this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];
    this->value_minus_value_zero = this->value - this->value_zero;
  };

  inline void setValueScaled(const VecC &value_scaled) {
    this->value_scaled = value_scaled;
    this->value_scaledf = this->value_scaled.cast<float>();
    value[0] = SCALE_F_INVERSE * value_scaled[0];
    value[1] = SCALE_F_INVERSE * value_scaled[1];
    value[2] = SCALE_C_INVERSE * value_scaled[2];
    value[3] = SCALE_C_INVERSE * value_scaled[3];

    this->value_minus_value_zero = this->value - this->value_zero;
    this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
    this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
    this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
    this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];
  };

  float Binv[256];
  float B[256];

  EIGEN_STRONG_INLINE float getBGradOnly(float color) {
    int c = color + 0.5f;
    if (c < 5)
      c = 5;
    if (c > 250)
      c = 250;
    return B[c + 1] - B[c];
  }

  EIGEN_STRONG_INLINE float getBInvGradOnly(float color) {
    int c = color + 0.5f;
    if (c < 5)
      c = 5;
    if (c > 250)
      c = 250;
    return Binv[c + 1] - Binv[c];
  }

  inline void setSgZero(const Vec3 &sg_scaled0) {
    sg_scaled = sg_scaled0;
    sg = sg_scaled;
    sg[0] *= SCALE_SCALE_INVERSE;
    sg.tail(2) *= SCALE_G_INVERSE;
    sg_zero = sg;
  }

  inline void setSg(const Vec3 &new_sg) {
    sg = new_sg;
    sg_scaled = sg;
    sg_scaled[0] *= SCALE_SCALE;
    sg_scaled.tail(2) *= SCALE_G;
  }

  inline double getScaleScaled(bool use_state_zero = false) {
    return use_state_zero ? (sg_zero[0] * SCALE_SCALE) : sg_scaled[0];
  }

  inline void getGSinCos(double &sr, double &cr, double &sp, double &cp,
                         bool use_state_zero = false) {
    Vec2 rp =
        use_state_zero ? (sg_zero.tail(2) * SCALE_G).eval() : sg_scaled.tail(2);
    sr = std::sin(rp[0]);
    cr = std::cos(rp[0]);
    sp = std::sin(rp[1]);
    cp = std::cos(rp[1]);
  }

  inline Vec3 getG(bool use_state_zero = false) {
    double sr, cr, sp, cp;
    getGSinCos(sr, cr, sp, cp, use_state_zero);
    Vec3 G;
    G << sp * cr, -sr, cp * cr;
    return setting_g_norm * G;
  }

  void tryTrapScale();
};

// hessian component associated with one point.
struct PointHessian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  static int instanceCounter;
  EFPoint *efPoint;

  // static values
  float color[MAX_RES_PER_POINT];   // colors in host frame
  float weights[MAX_RES_PER_POINT]; // host-weights for respective residuals.

  float u, v;
  int idx;
  float energyTH;
  FrameHessian *host;
  bool hasDepthPrior;

  float my_type;

  float idepth_scaled;
  float idepth_zero_scaled;
  float idepth_zero;
  float idepth;
  float step;
  float idepth_backup;

  float nullspaces_scale;
  float idepth_hessian;
  float maxRelBaseline;
  int numGoodResiduals;

  enum PtStatus { ACTIVE = 0, INACTIVE, OUTLIER, OOB, MARGINALIZED };
  PtStatus status;

  inline void setPointStatus(PtStatus s) { status = s; }

  inline void setIdepth(float idepth) {
    this->idepth = idepth;
    this->idepth_scaled = SCALE_IDEPTH * idepth;
  }
  inline void setIdepthScaled(float idepth_scaled) {
    this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
    this->idepth_scaled = idepth_scaled;
  }
  inline void setIdepthZero(float idepth) {
    idepth_zero = idepth;
    idepth_zero_scaled = SCALE_IDEPTH * idepth;
    nullspaces_scale = -(idepth * 1.001 - idepth / 1.001) * 500;
  }

  std::vector<PointFrameResidual *>
      residuals; // only contains good residuals (not OOB and not OUTLIER).
                 // Arbitrary order.
  std::pair<PointFrameResidual *, ResState>
      lastResiduals[2]; // contains information about residuals to the last two
                        // (!) frames. ([0] = latest, [1] = the one before).

  void release();
  PointHessian(const ImmaturePoint *const rawPoint, CalibHessian *HCalib);
  inline ~PointHessian() {
    assert(efPoint == 0);
    release();
    instanceCounter--;
  }

  inline bool isOOB(const std::vector<FrameHessian *> &toKeep,
                    const std::vector<FrameHessian *> &toMarg) const {

    int visInToMarg = 0;
    for (PointFrameResidual *r : residuals) {
      if (r->state_state != ResState::IN)
        continue;
      for (FrameHessian *k : toMarg)
        if (r->target == k)
          visInToMarg++;
    }
    if ((int)residuals.size() >= setting_minGoodActiveResForMarg &&
        numGoodResiduals > setting_minGoodResForMarg + 10 &&
        (int)residuals.size() - visInToMarg < setting_minGoodActiveResForMarg)
      return true;

    if (lastResiduals[0].second == ResState::OOB)
      return true;
    if (residuals.size() < 2)
      return false;
    if (lastResiduals[0].second == ResState::OUTLIER &&
        lastResiduals[1].second == ResState::OUTLIER)
      return true;
    return false;
  }

  inline bool isInlierNew() {
    return (int)residuals.size() >= setting_minGoodActiveResForMarg &&
           numGoodResiduals >= setting_minGoodResForMarg;
  }
};

} // namespace dso
