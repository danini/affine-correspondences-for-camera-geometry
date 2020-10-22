// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include "model.h"
#include <opencv2/core.hpp>
#include <Eigen/Eigen>
#include <iostream>

namespace gcransac
{
	namespace preemption
	{
		template<size_t _UsingAffineFrames,
			typename _ModelEstimator>
			class EssentialMatrixUncertaintyBasedPreemption
		{
		protected:
			Eigen::Matrix<double, 6, 16> covariance;

			// Compute the derivatives
			Eigen::Matrix<double, 6, 16> A = Eigen::Matrix<double, 6, 16>::Zero();
			Eigen::Matrix<double, 6, 6> B = Eigen::Matrix<double, 6, 6>::Zero();

			double trace_threshold,
				trace_of_last_run;

		public:
			EssentialMatrixUncertaintyBasedPreemption(const double trace_threshold_) :
				trace_threshold(trace_threshold_)
			{
			}

			void initialize(const cv::Mat &points_) {}

			const double &getTraceOfLastRun()
			{
				return trace_of_last_run;
			}

			static constexpr bool providesScore()
			{
				return false;
			}

			static constexpr bool definedForPoints()
			{
				return false;
			}

			static constexpr bool definedForAffinities()
			{
				return true;
			}

			bool verifyModel(
				const gcransac::Model &model_,
				const _ModelEstimator &estimator_, // The model estimator
				const double &threshold_,
				const size_t &iteration_number_,
				const Score &best_score_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t sample_number_,
				std::vector<size_t> &inliers_,
				Score &score_)
			{
				double trace = 0.0;

				if constexpr (_UsingAffineFrames && definedForAffinities())
					verifyAffineModel(
						model_,
						points_,
						minimal_sample_,
						sample_number_,
						trace);
				else
				{
					if constexpr (definedForPoints())
						verifyPointModel(
							model_,
							points_,
							minimal_sample_,
							sample_number_,
							trace);
					else
						return true;
				}

				trace_of_last_run = trace;
				return trace < trace_threshold;
			}

		protected:
			OLGA_INLINE bool verifyAffineModel(
				gcransac::Model model_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t sample_number_,
				double &trace_)
			{
				const double *points_ptr =
					reinterpret_cast<double *>(points_.data);
				Eigen::Matrix3d E =
					model_.descriptor.block<3, 3>(0, 0).normalized();

				cv::Mat cvE(3, 3, CV_64F, E.data());

				cv::Mat cvR1, cvR2, cvt;
				cv::decomposeEssentialMat(cvE, cvR1, cvR2, cvt);
				Eigen::Matrix3d R1 = Eigen::Map<Eigen::Matrix3d>(reinterpret_cast<double *>(cvR1.data), 3, 3);
				Eigen::Matrix3d R2 = Eigen::Map<Eigen::Matrix3d>(reinterpret_cast<double *>(cvR2.data), 3, 3);
				Eigen::Vector3d t = Eigen::Map<Eigen::Vector3d>(reinterpret_cast<double *>(cvt.data), 3, 1);
				t.normalize();
				int cidx = 0;
				Eigen::Vector3d aa;

				std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> combinations =
				{ std::make_pair(R1, t) ,
				std::make_pair(R2, t),
				std::make_pair(R1, -t),
				std::make_pair(R2, -t) };

				for (int i = 0; i < 4; ++i)
				{
					Eigen::Vector3d &t = combinations[i].second;
					std::vector<int> sign_s(2), sign_r(2);
					for (size_t i = 0; i < 2; ++i)
					{
						const size_t idx = minimal_sample_[i] * points_.cols;
						const double &x1 = points_ptr[idx],
							&y1 = points_ptr[idx + 1],
							&x2 = points_ptr[idx + 2],
							&y2 = points_ptr[idx + 3];

						Eigen::Vector3d
							ui(x1, y1, 1);
						Eigen::Vector3d
							vi(x2, y2, 1);
						Eigen::Vector3d
							wi = combinations[i].first * vi;
						Eigen::Vector3d
							m = (t.cross(ui)).cross(t);

						Eigen::Matrix3d mm;
						mm << t, m, ui.cross(wi);

						sign_s[i] = mm.determinant() < 0 ? -1 : 1;
						sign_r[i] = sign_s[i] * (m.dot(wi) < 0 ? -1 : 1);
					}

					Eigen::Matrix3d Rd = combinations[i].first - combinations[i].first.transpose();
					Eigen::Vector3d axis(Rd(2, 1), Rd(0, 2), Rd(1, 0));

					axis = axis * (1 / sqrt(axis.dot(axis)));
					aa = acos(0.5*(combinations[i].first.trace() - 1)) * axis;

					if ((sign_s[0] + sign_s[1]) / 2.0 > 0 && (sign_r[0] + sign_r[1]) / 2.0 > 0)
					{
						cidx = i;
						break;
					}
				}

				const size_t cols = points_.cols;
				const size_t &idx1 = minimal_sample_[0] * cols,
					&idx2 = minimal_sample_[1] * cols;

				const double &x11 = points_ptr[idx1],
					&y11 = points_ptr[idx1 + 1],
					&x12 = points_ptr[idx1 + 2],
					&y12 = points_ptr[idx1 + 3],
					&a111 = points_ptr[idx1 + 4],
					&a112 = points_ptr[idx1 + 5],
					&a121 = points_ptr[idx1 + 6],
					&a122 = points_ptr[idx1 + 7],
					&x21 = points_ptr[idx2],
					&y21 = points_ptr[idx2 + 1],
					&x22 = points_ptr[idx2 + 2],
					&y22 = points_ptr[idx2 + 3],
					&a211 = points_ptr[idx2 + 4],
					&a212 = points_ptr[idx2 + 5],
					&a221 = points_ptr[idx2 + 6],
					&a222 = points_ptr[idx2 + 7];

				// Compute the derivatives
				Eigen::Matrix<double, 6, 16> A;
				Eigen::Matrix<double, 6, 6> B;

				derivMeasurementsAffine(x11, y11, x21, y21,
					x12, y12, x22, y22,
					a111, a112, a121, a122,
					a211, a212, a221, a222,
					aa(0), aa(1), aa(2), combinations[cidx].second(0), combinations[cidx].second(1), combinations[cidx].second(2),
					A);

				derivParamsAffine(x11, y11, x21, y21,
					x12, y12, x22, y22,
					a111, a112, a121, a122,
					a211, a212, a221, a222,
					aa(0), aa(1), aa(2), combinations[cidx].second(0), combinations[cidx].second(1), combinations[cidx].second(2),
					B);

				Eigen::Matrix<double, 6, 16> x =
					(B.transpose() * B).llt().solve(B.transpose() * A);

				trace_ = 1e-14 * (x.row(0).dot(x.row(0)) +
					x.row(1).dot(x.row(1)) +
					x.row(2).dot(x.row(2)) +
					x.row(3).dot(x.row(3)) +
					x.row(4).dot(x.row(4)) +
					x.row(5).dot(x.row(5)));

				return true;
			}

			OLGA_INLINE bool verifyPointModel(
				const gcransac::Model &model_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t &sample_number_,
				double &trace_)
			{
				// normalize F
				Eigen::Matrix3d F =
					model_.descriptor.block<3, 3>(0, 0).normalized();

				const double *points_ptr =
					reinterpret_cast<double *>(points_.data);

				const size_t cols = points_.cols;
				const size_t &idx1 = minimal_sample_[0] * cols,
					&idx2 = minimal_sample_[1] * cols,
					&idx3 = minimal_sample_[2] * cols,
					&idx4 = minimal_sample_[3] * cols,
					&idx5 = minimal_sample_[4] * cols,
					&idx6 = minimal_sample_[5] * cols,
					&idx7 = minimal_sample_[6] * cols;

				const double &x11 = points_ptr[idx1],
					&y11 = points_ptr[idx1 + 1],
					&x12 = points_ptr[idx1 + 2],
					&y12 = points_ptr[idx1 + 3],
					&x21 = points_ptr[idx2],
					&y21 = points_ptr[idx2 + 1],
					&x22 = points_ptr[idx2 + 2],
					&y22 = points_ptr[idx2 + 3],
					&x31 = points_ptr[idx3],
					&y31 = points_ptr[idx3 + 1],
					&x32 = points_ptr[idx3 + 2],
					&y32 = points_ptr[idx3 + 3],
					&x41 = points_ptr[idx4],
					&y41 = points_ptr[idx4 + 1],
					&x42 = points_ptr[idx4 + 2],
					&y42 = points_ptr[idx4 + 3],
					&x51 = points_ptr[idx5],
					&y51 = points_ptr[idx5 + 1],
					&x52 = points_ptr[idx5 + 2],
					&y52 = points_ptr[idx5 + 3],
					&x61 = points_ptr[idx6],
					&y61 = points_ptr[idx6 + 1],
					&x62 = points_ptr[idx6 + 2],
					&y62 = points_ptr[idx6 + 3],
					&x71 = points_ptr[idx7],
					&y71 = points_ptr[idx7 + 1],
					&x72 = points_ptr[idx7 + 2],
					&y72 = points_ptr[idx7 + 3];

				// Compute the derivatives
				Eigen::Matrix<double, 9, 28> A;
				Eigen::Matrix<double, 9, 9> B;

				derivMeasurements(x11, y11, x21, y21, x31, y31, x41, y41, x51, y51, x61, y61, x71, y71,
					x12, y12, x22, y22, x32, y32, x42, y42, x52, y52, x62, y62, x72, y72,
					F(0, 0), F(1, 0), F(2, 0), F(0, 1), F(1, 1), F(2, 1), F(0, 2), F(1, 2),
					A);

				derivParams(x11, y11, x21, y21, x31, y31, x41, y41, x51, y51, x61, y61, x71, y71,
					x12, y12, x22, y22, x32, y32, x42, y42, x52, y52, x62, y62, x72, y72,
					F(0, 0), F(1, 0), F(2, 0), F(0, 1), F(1, 1), F(2, 1), F(0, 2), F(1, 2), F(2, 2),
					B);

				Eigen::Matrix<double, 9, 28> x =
					(B.transpose() * B).llt().solve(B.transpose() * A);

				trace_ = 1e-13 * (x.row(0).dot(x.row(0)) +
					x.row(1).dot(x.row(1)) +
					x.row(2).dot(x.row(2)) +
					x.row(3).dot(x.row(3)) +
					x.row(4).dot(x.row(4)) +
					x.row(5).dot(x.row(5)) +
					x.row(6).dot(x.row(6)) +
					x.row(7).dot(x.row(7)) +
					x.row(8).dot(x.row(8)));

				return true;
			}

			OLGA_INLINE void derivMeasurements(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u13,
				const double &v13,
				const double &u14,
				const double &v14,
				const double &u15,
				const double &v15,
				const double &u16,
				const double &v16,
				const double &u17,
				const double &v17,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &u23,
				const double &v23,
				const double &u24,
				const double &v24,
				const double &u25,
				const double &v25,
				const double &u26,
				const double &v26,
				const double &u27,
				const double &v27,
				const double &aa1,
				const double &aa2,
				const double &aa3,
				const double &b1,
				const double &b2,
				const double &b3,
				const double &f11,
				const double &f21,
				const double &f31,
				const double &f12,
				const double &f22,
				const double &f32,
				const double &f13,
				const double &f23,
				const double &f33,
				Eigen::Matrix<double, 6, 16> &A)
			{
				A <<
					f11 * u21 + f21 * v21 + f31, f12 * u21 + f22 * v21 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u11 + f12 * v11 + f13, f21 * u11 + f22 * v11 + f23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, f11 * u22 + f21 * v22 + f31, f12 * u22 + f22 * v22 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u12 + f12 * v12 + f13, f21 * u12 + f22 * v12 + f23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, f11 * u23 + f21 * v23 + f31, f12 * u23 + f22 * v23 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u13 + f12 * v13 + f13, f21 * u13 + f22 * v13 + f23, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, f11 * u24 + f21 * v24 + f31, f12 * u24 + f22 * v24 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u14 + f12 * v14 + f13, f21 * u14 + f22 * v14 + f23, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, f11 * u25 + f21 * v25 + f31, f12 * u25 + f22 * v25 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u15 + f12 * v15 + f13, f21 * u15 + f22 * v15 + f23, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u26 + f21 * v26 + f31, f12 * u26 + f22 * v26 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u16 + f12 * v16 + f13, f21 * u16 + f22 * v16 + f23, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u27 + f21 * v27 + f31, f12 * u27 + f22 * v27 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u17 + f12 * v17 + f13, f21 * u17 + f22 * v17 + f23,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
			}

			OLGA_INLINE void derivParams(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u13,
				const double &v13,
				const double &u14,
				const double &v14,
				const double &u15,
				const double &v15,
				const double &u16,
				const double &v16,
				const double &u17,
				const double &v17,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &u23,
				const double &v23,
				const double &u24,
				const double &v24,
				const double &u25,
				const double &v25,
				const double &u26,
				const double &v26,
				const double &u27,
				const double &v27,
				const double &aa1,
				const double &aa2,
				const double &aa3,
				const double &b1,
				const double &b2,
				const double &b3,
				const double &f11,
				const double &f21,
				const double &f31,
				const double &f12,
				const double &f22,
				const double &f32,
				const double &f13,
				const double &f23,
				const double &f33,
				Eigen::Matrix<double, 9, 9> &B)
			{
				const double
					f11sqr = f11 * f11,
					f12sqr = f12 * f12,
					f13sqr = f13 * f12,
					f21sqr = f21 * f21,
					f22sqr = f22 * f22,
					f23sqr = f23 * f23,
					f31sqr = f31 * f31,
					f32sqr = f32 * f32,
					f33sqr = f33 * f33;

				const double
					s1 = std::pow(f11sqr + f12sqr + f13sqr + f21sqr + f22sqr + f23sqr + f31sqr + f32sqr + f33sqr, -0.1e1 / 0.2e1);

				B <<
					u11 * u21, u11 * v21, u11, u21 * v11, v11 * v21, v11, u21, v21, 1,
					u12 * u22, u12 * v22, u12, u22 * v12, v12 * v22, v12, u22, v22, 1,
					u13 * u23, u13 * v23, u13, u23 * v13, v13 * v23, v13, u23, v23, 1,
					u14 * u24, u14 * v24, u14, u24 * v14, v14 * v24, v14, u24, v24, 1,
					u15 * u25, u15 * v25, u15, u25 * v15, v15 * v25, v15, u25, v25, 1,
					u16 * u26, u16 * v26, u16, u26 * v16, v16 * v26, v16, u26, v26, 1,
					u17 * u27, u17 * v27, u17, u27 * v17, v17 * v27, v17, u27, v27, 1,
					f22 * f33 - f23 * f32, -f12 * f33 + f13 * f32, f12 * f23 - f13 * f22, -f21 * f33 + f23 * f31, f11 * f33 - f13 * f31, -f11 * f23 + f13 * f21, f21 * f32 - f22 * f31, -f11 * f32 + f12 * f31, f11 * f22 - f12 * f21,
					s1 * f11, s1 * f21, s1 * f31, s1 * f12, s1 * f22, s1 * f32, s1 * f13, s1 * f23, s1 * f33;
			}

			OLGA_INLINE void derivMeasurementsAffine(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &a1_1,
				const double &a2_1,
				const double &a3_1,
				const double &a4_1,
				const double &a1_2,
				const double &a2_2,
				const double &a3_2,
				const double &a4_2,
				const double &aa1,
				const double &aa2,
				const double &aa3,
				const double &b1,
				const double &b2,
				const double &b3,
				Eigen::Matrix<double, 6, 16> &A)
			{
				const double sqraa1 = aa1 * aa1,
					sqraa2 = aa2 * aa2,
					sqraa3 = aa3 * aa3;

				const double s2 =
					sqraa1 + sqraa2 + sqraa3;
				const double s1 =
					std::pow(s2, -0.3e1 / 0.2e1);
				const double s3 =
					sqrt(s2);
				const double s4 = b1 * v21 - b2 * u21,
					s5 = -b3 * u21 + b1,
					s6 = aa1 * u11 + aa2 * v11 + aa3,
					s7 = -aa2 * b3 + aa3 * b2,
					s8 = b3 * u21 - b1,
					s9 = aa1 * u12 + aa2 * v12 + aa3,
					s10 = -aa1 * b3 + aa3 * b1,
					s11 = -v21 * b3 + b2,
					s12 = std::pow(s2, -0.5e1 / 0.2e1);

				A << ((((s8)* aa2 + aa3 * (s4)) * aa1 - (sqraa2 + sqraa3) * (s11)) * cos(s3) + s3 * ((s4)* aa2 + aa3 * (s5)) * sin(s3) + aa1 * ((s5)* aa2 + (v21 * b3 - b2) * aa1 - aa3 * (s4))) / (s2), ((((s11)* aa1 + aa3 * (s4)) * aa2 + (sqraa1 + sqraa3) * (s5)) * cos(s3) - s3 * ((s4)* aa1 - aa3 * (s11)) * sin(s3) + ((s5)* aa2 + (v21 * b3 - b2) * aa1 - aa3 * (s4)) * aa2) / (s2), 0, 0, ((sqraa2 * b2 + ((-b2 * v11 + b3) * aa3 + b3 * aa1 * u11) * aa2 - sqraa3 * b3 * v11 - aa1 * aa3 * b2 * u11 + (-b3 * v11 + b2) * sqraa1) * cos(s3) + ((b2 * v11 + b3) * aa1 - b2 * u11 * aa2 - aa3 * b3 * u11) * s3 * sin(s3) + (s6) * (s7)) / (s2), ((-sqraa1 * b1 + (-b3 * v11 * aa2 + aa3 * (b1 * u11 - b3)) * aa1 + sqraa3 * b3 * u11 + aa3 * b1 * v11 * aa2 - (-b3 * u11 + b1) * sqraa2) * cos(s3) - (b1 * v11 * aa1 + aa3 * b3 * v11 - (b1 * u11 + b3) * aa2) * s3 * sin(s3) - (s10) * (s6)) / (s2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, ((((b3 * u22 - b1) * aa2 + aa3 * (b1 * v22 - b2 * u22)) * aa1 - (sqraa2 + sqraa3) * (-b3 * v22 + b2)) * cos(s3) + s3 * ((b1 * v22 - b2 * u22) * aa2 + aa3 * (-b3 * u22 + b1)) * sin(s3) + aa1 * ((-b3 * u22 + b1) * aa2 + (b3 * v22 - b2) * aa1 - aa3 * (b1 * v22 - b2 * u22))) / (s2), ((((-b3 * v22 + b2) * aa1 + aa3 * (b1 * v22 - b2 * u22)) * aa2 + (sqraa1 + sqraa3) * (-b3 * u22 + b1)) * cos(s3) - s3 * ((b1 * v22 - b2 * u22) * aa1 - aa3 * (-b3 * v22 + b2)) * sin(s3) + ((-b3 * u22 + b1) * aa2 + (b3 * v22 - b2) * aa1 - aa3 * (b1 * v22 - b2 * u22)) * aa2) / (s2), 0, 0, ((sqraa2 * b2 + ((-b2 * v12 + b3) * aa3 + b3 * aa1 * u12) * aa2 - sqraa3 * b3 * v12 - aa1 * aa3 * b2 * u12 + (-b3 * v12 + b2) * sqraa1) * cos(s3) + ((b2 * v12 + b3) * aa1 - b2 * u12 * aa2 - aa3 * b3 * u12) * s3 * sin(s3) + (s9) * (s7)) / (s2), ((-sqraa1 * b1 + (-b3 * v12 * aa2 + aa3 * (b1 * u12 - b3)) * aa1 + sqraa3 * b3 * u12 + aa3 * b1 * v12 * aa2 - (-b3 * u12 + b1) * sqraa2) * cos(s3) - (b1 * v12 * aa1 + aa3 * b3 * v12 - (b1 * u12 + b3) * aa2) * s3 * sin(s3) - (s10) * (s9)) / (s2), 0, 0, 0, 0, 0, 0, 0, 0,
					s1 * ((s2) * ((-aa2 * b2 - aa3 * b3) * a1_1 + a3_1 * aa2 * b1) * sin(s3) + s3 * ((-aa1 * (s7)* a1_1 + (aa1 * aa3 * b1 + b3 * (sqraa2 + sqraa3)) * a3_1) * cos(s3) - aa1 * ((aa2 * b3 - aa3 * b2) * a1_1 + (s10)* a3_1))), s1 * (-((aa1 * b1 + aa3 * b3) * a3_1 - a1_1 * aa1 * b2) * (s2)* sin(s3) + s3 * ((aa2 * (s10)* a3_1 - a1_1 * (sqraa1 * b3 + aa2 * aa3 * b2 + sqraa3 * b3)) * cos(s3) - ((s10)* a3_1 - a1_1 * (s7)) * aa2)), 0, 0, -s1 * ((s2) * (aa2 * b2 + aa3 * b3) * sin(s3) + aa1 * s3 * (cos(s3) - 0.1e1) * (s7)), s1 * ((aa1 * aa3 * b1 + b3 * (sqraa2 + sqraa3)) * s3 * cos(s3) + b1 * aa2 * (s2)* sin(s3) - aa1 * s3 * (s10)), 0, 0, (((b2 * v11 + b3) * aa1 - u11 * (aa2 * b2 + aa3 * b3)) * (s2)* sin(s3) + s3 * (((-b3 * v11 + b2) * sqraa1 - u11 * (s7)* aa1 + (-aa3 * v11 + aa2) * (aa2 * b2 + aa3 * b3)) * cos(s3) + (s6) * (s7))) * s1, 0, -((s2) * (b1 * v11 * aa1 + (-b1 * u11 - b3) * aa2 + aa3 * b3 * v11) * sin(s3) + s3 * ((sqraa1 * b1 + (b3 * v11 * aa2 - aa3 * (b1 * u11 - b3)) * aa1 + (-b3 * u11 + b1) * sqraa2 - aa3 * b1 * v11 * aa2 - sqraa3 * b3 * u11) * cos(s3) + (s10) * (s6))) * s1, 0, 0, 0, 0, 0,
					s1 * ((s2) * ((-aa2 * b2 - aa3 * b3) * a2_1 + a4_1 * aa2 * b1) * sin(s3) + s3 * ((-aa1 * (s7)* a2_1 + (aa1 * aa3 * b1 + b3 * (sqraa2 + sqraa3)) * a4_1) * cos(s3) - aa1 * ((aa2 * b3 - aa3 * b2) * a2_1 + (s10)* a4_1))), s1 * (-((aa1 * b1 + aa3 * b3) * a4_1 - a2_1 * aa1 * b2) * (s2)* sin(s3) + s3 * ((aa2 * (s10)* a4_1 - a2_1 * (sqraa1 * b3 + aa2 * aa3 * b2 + sqraa3 * b3)) * cos(s3) - ((s10)* a4_1 - a2_1 * (s7)) * aa2)), 0, 0, s1 * (-(aa2 * aa3 * b2 + b3 * (sqraa1 + sqraa3)) * s3 * cos(s3) + aa1 * b2 * (s2)* sin(s3) + aa2 * s3 * (s7)), -s1 * ((s2) * (aa1 * b1 + aa3 * b3) * sin(s3) - aa2 * s3 * (cos(s3) - 0.1e1) * (s10)), 0, 0, 0, (((b2 * v11 + b3) * aa1 - u11 * (aa2 * b2 + aa3 * b3)) * (s2)* sin(s3) + s3 * (((-b3 * v11 + b2) * sqraa1 - u11 * (s7)* aa1 + (-aa3 * v11 + aa2) * (aa2 * b2 + aa3 * b3)) * cos(s3) + (s6) * (s7))) * s1, 0, -((s2) * (b1 * v11 * aa1 + (-b1 * u11 - b3) * aa2 + aa3 * b3 * v11) * sin(s3) + s3 * ((sqraa1 * b1 + (b3 * v11 * aa2 - aa3 * (b1 * u11 - b3)) * aa1 + (-b3 * u11 + b1) * sqraa2 - aa3 * b1 * v11 * aa2 - sqraa3 * b3 * u11) * cos(s3) + (s10) * (s6))) * s1, 0, 0, 0, 0,
					0, 0, s1 * ((s2) * ((-aa2 * b2 - aa3 * b3) * a1_2 + a3_2 * aa2 * b1) * sin(s3) + s3 * ((-aa1 * (s7)* a1_2 + (aa1 * aa3 * b1 + b3 * (sqraa2 + sqraa3)) * a3_2) * cos(s3) - aa1 * ((aa2 * b3 - aa3 * b2) * a1_2 + (s10)* a3_2))), s1 * (-((aa1 * b1 + aa3 * b3) * a3_2 - a1_2 * aa1 * b2) * (s2)* sin(s3) + s3 * ((aa2 * (s10)* a3_2 - a1_2 * (sqraa1 * b3 + aa2 * aa3 * b2 + sqraa3 * b3)) * cos(s3) - ((s10)* a3_2 - a1_2 * (s7)) * aa2)), 0, 0, -s1 * ((s2) * (aa2 * b2 + aa3 * b3) * sin(s3) + aa1 * s3 * (cos(s3) - 0.1e1) * (s7)), s1 * ((aa1 * aa3 * b1 + b3 * (sqraa2 + sqraa3)) * s3 * cos(s3) + b1 * aa2 * (s2)* sin(s3) - aa1 * s3 * (s10)), 0, 0, 0, 0, (((b2 * v12 + b3) * aa1 - u12 * (aa2 * b2 + aa3 * b3)) * (s2)* sin(s3) + s3 * (((-b3 * v12 + b2) * sqraa1 - u12 * (s7)* aa1 + (-aa3 * v12 + aa2) * (aa2 * b2 + aa3 * b3)) * cos(s3) + (s9) * (s7))) * s1, 0, -((b1 * v12 * aa1 + (-b1 * u12 - b3) * aa2 + aa3 * b3 * v12) * (s2)* sin(s3) + s3 * ((sqraa1 * b1 + (b3 * v12 * aa2 - aa3 * (b1 * u12 - b3)) * aa1 + (-b3 * u12 + b1) * sqraa2 - aa3 * b1 * v12 * aa2 - sqraa3 * b3 * u12) * cos(s3) + (s10) * (s9))) * s1, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
			}

			OLGA_INLINE void derivParamsAffine(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &a1_1,
				const double &a2_1,
				const double &a3_1,
				const double &a4_1,
				const double &a1_2,
				const double &a2_2,
				const double &a3_2,
				const double &a4_2,
				const double &aa1,
				const double &aa2,
				const double &aa3,
				const double &b1,
				const double &b2,
				const double &b3,
				Eigen::Matrix<double, 6, 6> &B)
			{
				const double sqraa1 = aa1 * aa1,
					sqraa2 = aa2 * aa2,
					sqraa3 = aa3 * aa3;

				const double s2 =
					sqraa1 + sqraa2 + sqraa3;
				const double s1 =
					std::pow(s2, -0.3e1 / 0.2e1);
				const double s3 =
					sqrt(s2);
				const double s4 = b1 * v21 - b2 * u21,
					s5 = -b3 * u21 + b1,
					s6 = aa1 * u11 + aa2 * v11 + aa3,
					s7 = -aa2 * b3 + aa3 * b2,
					s8 = b3 * u21 - b1,
					s9 = aa1 * u12 + aa2 * v12 + aa3,
					s10 = -aa1 * b3 + aa3 * b1,
					s11 = -v21 * b3 + b2,
					s12 = std::pow(s2, -0.5e1 / 0.2e1);

				const double k1 = aa1 * aa1 * aa1,
					k2 = aa2 * aa2 * aa2,
					k3 = aa3 * aa3 * aa3,
					k5 = aa3 * aa3 * aa3 * aa3,
					k6 = aa2 * aa2 * aa2 * aa2,
					k7 = aa1 * aa1 * aa1 * aa1,
					k4 = b3 * u11 * u21 + v21 * b3 * v11 - b1 * u11 - b2 * v11,
					k8 = v21 * b3 - b2,
					k9 = std::pow(b1 * b1 + b2 * b2 + b3 * b3, -0.1e1 / 0.2e1);

				B << ((-v21 * sqraa2 + (aa3 * v11 * v21 - aa1 * u11 - aa3) * aa2 + sqraa3 * v11 + aa1 * aa3 * u11 * v21 + (-v21 + v11) * sqraa1) * cos(s3) - s3 * (aa1 * v11 * v21 - v21 * u11 * aa2 - aa3 * u11 + aa1) * sin(s3) + (-aa3 * v21 + aa2) * (s6)) / (s2), ((u21 * sqraa1 + (-aa3 * u11 * u21 + aa2 * v11 + aa3) * aa1 - sqraa3 * u11 - aa3 * v11 * u21 * aa2 - sqraa2 * (u11 - u21)) * cos(s3) + s3 * (v11 * u21 * aa1 - aa2 * u11 * u21 + aa3 * v11 - aa2) * sin(s3) - (-aa3 * u21 + aa1) * (s6)) / (s2), ((-v11 * u21 * sqraa1 + ((u11 * u21 - v11 * v21) * aa2 - aa3 * v21) * aa1 + v21 * u11 * sqraa2 + aa3 * u21 * aa2 + sqraa3 * (u11 * v21 - u21 * v11)) * cos(s3) + (u21 * aa1 + v21 * aa2 - aa3 * (u11 * u21 + v11 * v21)) * s3 * sin(s3) + (aa1 * v21 - aa2 * u21) * (s6)) / (s2), -s12 * (-(s2) * ((s4 - v11 * (s5)) * k1 + ((v11 * (k8)+(s5)* u11) * aa2 - aa3 * ((s4)* u11 - v21 * b3 + b2)) * sqraa1 + (((s11)* u11 + s4) * sqraa2 + ((-b1 * v21 * v11 + b2 * u21 * v11 - b3 * u21 + b1) * aa3 + (-b1 * v21 + b2 * u21) * u11 - v21 * b3 + b2) * aa2 - aa3 * (aa3 * ((k8)* u11 + v11 * (s5)) + (s5)* u11 - v11 * (k8))) * aa1 - (sqraa2 + sqraa3) * (b1 * v21 * v11 - b2 * u21 * v11 - b3 * u21 + b1)) * sin(s3) + (((b1 * v21 * v11 - b2 * u21 * v11 - b3 * u21 + b1) * k7 + (((-b1 * v21 + b2 * u21) * u11 - v21 * b3 + b2) * aa2 - aa3 * ((s5)* u11 - v11 * (k8))) * k1 + ((b1 * v21 * v11 - b2 * u21 * v11 - b3 * u21 + b1) * sqraa2 + ((s8)* u11 - v11 * (k8)) * aa2 + aa3 * ((b1 * v21 * v11 - b2 * u21 * v11 - b3 * u21 + b1) * aa3 + (s4)* u11 - v21 * b3 + b2)) * sqraa1 + (((-b1 * v21 + b2 * u21) * u11 - v21 * b3 + b2) * k2 + (((s8)* u11 + v11 * (k8)) * aa3 + (0.2e1 * v21 * b3 - 0.2e1 * b2) * u11 - 0.2e1 * v11 * (s5)) * sqraa2 - aa3 * (((s4)* u11 + k8) * aa3 - 0.2e1 * b1 * v21 * v11 + 0.2e1 * b2 * u21 * v11 - 0.2e1 * b3 * u21 + 0.2e1 * b1) * aa2 + 0.2e1 * sqraa3 * (((b3 * u21 / 0.2e1 - b1 / 0.2e1) * u11 + v11 * (k8) / 0.2e1) * aa3 + (k8)* u11 + s4)) * aa1 - (((s8)* u11 - v11 * (k8)) * aa2 + aa3 * ((s4)* u11 - v21 * b3 + b2)) * (sqraa2 + sqraa3)) * cos(s3) + ((v11 * (k8)+(s5)* u11) * aa2 - aa3 * ((s4)* u11 - v21 * b3 + b2)) * sqraa1 + (((-0.2e1 * v21 * b3 + 0.2e1 * b2) * u11 + 0.2e1 * v11 * (s5)) * sqraa2 - 0.2e1 * aa3 * (b1 * v21 * v11 - b2 * u21 * v11 + s8) * aa2 - 0.2e1 * sqraa3 * ((k8)* u11 + s4)) * aa1 + (((s8)* u11 - v11 * (k8)) * aa2 + aa3 * ((s4)* u11 - v21 * b3 + b2)) * (sqraa2 + sqraa3)) * s3), -(-(s2) * ((-b2 * u21 + b1 * v21 - (k8)* u11) * k2 + ((v11 * (k8)+(s5)* u11) * aa1 - aa3 * ((s4)* v11 + s8)) * sqraa2 + (((s8)* v11 + s4) * sqraa1 + ((-b1 * v21 * u11 + b2 * u11 * u21 + k8) * aa3 + (s4)* v11 - b3 * u21 + b1) * aa1 - aa3 * (aa3 * ((k8)* u11 + v11 * (s5)) + (s11)* v11 + (s5)* u11)) * aa2 + (sqraa1 + sqraa3) * (b1 * v21 * u11 - b2 * u11 * u21 + k8)) * sin(s3) + (((-b1 * v21 * u11 + b2 * u11 * u21 - v21 * b3 + b2) * k6 + (((s4)* v11 - b3 * u21 + b1) * aa1 - aa3 * ((s11)* v11 + (s5)* u11)) * k2 + ((-b1 * v21 * u11 + b2 * u11 * u21 - v21 * b3 + b2) * sqraa1 + ((s11)* v11 - (s5)* u11) * aa1 - aa3 * ((b1 * v21 * u11 - b2 * u11 * u21 + k8) * aa3 + (-b1 * v21 + b2 * u21) * v11 - b3 * u21 + b1)) * sqraa2 + (((s4)* v11 - b3 * u21 + b1) * k1 + ((-(s5)* u11 + v11 * (k8)) * aa3 + (-0.2e1 * b3 * u21 + 0.2e1 * b1) * v11 - 0.2e1 * (k8)* u11) * sqraa1 + aa3 * (((s4)* v11 - b3 * u21 + b1) * aa3 + 0.2e1 * b1 * v21 * u11 - 0.2e1 * b2 * u11 * u21 - 0.2e1 * v21 * b3 + 0.2e1 * b2) * aa1 + 0.2e1 * sqraa3 * (((v21 * b3 / 0.2e1 - b2 / 0.2e1) * v11 - (s5)* u11 / 0.2e1) * aa3 + v11 * (s5)+s4)) * aa2 - (((s11)* v11 - (s5)* u11) * aa1 + aa3 * ((s4)* v11 + s8)) * (sqraa1 + sqraa3)) * cos(s3) + ((v11 * (k8)+(s5)* u11) * aa1 - aa3 * ((s4)* v11 + s8)) * sqraa2 + (((0.2e1 * b3 * u21 - 0.2e1 * b1) * v11 + 0.2e1 * (k8)* u11) * sqraa1 - 0.2e1 * aa3 * (b1 * v21 * u11 - b2 * u11 * u21 - v21 * b3 + b2) * aa1 - 0.2e1 * sqraa3 * (v11 * (s5)+s4)) * aa2 + (((s11)* v11 - (s5)* u11) * aa1 + aa3 * ((s4)* v11 + s8)) * (sqraa1 + sqraa3)) * s3) * s12, -(-((-b3 * u11 * v21 + b3 * u21 * v11 - b1 * v11 + b2 * u11) * k3 + ((b2 * u11 * u21 + (-b1 * u11 + b3) * v21 - b2) * aa1 - ((-b2 * v11 + b3) * u21 + b1 * (v11 * v21 - 0.1e1)) * aa2) * sqraa3 + (((b3 * v11 - b2) * u21 + b1 * (v21 - v11)) * sqraa1 + ((-b3 * u11 * u21 + v21 * b3 * v11 + b1 * u11 - b2 * v11) * aa2 + (-b2 * v11 - b3) * u21 + b1 * (v11 * v21 + 0.1e1)) * aa1 + ((-b2 * u21 + (-b3 * u11 + b1) * v21 + b2 * u11) * aa2 + b2 * u11 * u21 + (-b1 * u11 - b3) * v21 + b2) * aa2) * aa3 + (sqraa1 + sqraa2) * (-b3 * u11 * u21 - v21 * b3 * v11 + b1 * u11 + b2 * v11)) * (s2)* sin(s3) + s3 * (((k4)* k5 + (((-b2 * v11 - b3) * u21 + b1 * (v11 * v21 + 0.1e1)) * aa1 - (-b2 * u11 * u21 + (b1 * u11 + b3) * v21 - b2) * aa2) * k3 + ((k4)* sqraa1 + (-b2 * u11 * u21 + (b1 * u11 - b3) * v21 + b2) * aa1 + ((k4)* aa2 + (-b2 * v11 + b3) * u21 + b1 * (v11 * v21 - 0.1e1)) * aa2) * sqraa3 + (((-b2 * v11 - b3) * u21 + b1 * (v11 * v21 + 0.1e1)) * k1 + ((b2 * u11 * u21 + (-b1 * u11 - b3) * v21 + b2) * aa2 + 0.2e1 * b2 * u21 + (-0.2e1 * b3 * u11 - 0.2e1 * b1) * v21 + 0.2e1 * b2 * u11) * sqraa1 + (((-b2 * v11 - b3) * u21 + b1 * (v11 * v21 + 0.1e1)) * aa2 - 0.2e1 * v21 * b3 * v11 + 0.2e1 * b3 * u11 * u21 - 0.2e1 * b1 * u11 + 0.2e1 * b2 * v11) * aa2 * aa1 - ((-b2 * u11 * u21 + (b1 * u11 + b3) * v21 - b2) * aa2 + (-0.2e1 * b3 * v11 - 0.2e1 * b2) * u21 + 0.2e1 * b1 * (v21 + v11)) * sqraa2) * aa3 - (sqraa1 + sqraa2) * ((-b2 * u11 * u21 + (b1 * u11 - b3) * v21 + b2) * aa1 + ((-b2 * v11 + b3) * u21 + b1 * (v11 * v21 - 0.1e1)) * aa2)) * cos(s3) + ((b2 * u11 * u21 + (-b1 * u11 + b3) * v21 - b2) * aa1 - ((-b2 * v11 + b3) * u21 + b1 * (v11 * v21 - 0.1e1)) * aa2) * sqraa3 + ((-0.2e1 * b2 * u21 + (0.2e1 * b3 * u11 + 0.2e1 * b1) * v21 - 0.2e1 * b2 * u11) * sqraa1 + 0.2e1 * (-b3 * u11 * u21 + v21 * b3 * v11 + b1 * u11 - b2 * v11) * aa2 * aa1 + 0.2e1 * ((-b3 * v11 - b2) * u21 + b1 * (v21 + v11)) * sqraa2) * aa3 + (sqraa1 + sqraa2) * ((-b2 * u11 * u21 + (b1 * u11 - b3) * v21 + b2) * aa1 + ((-b2 * v11 + b3) * u21 + b1 * (v11 * v21 - 0.1e1)) * aa2))) * s12,
					((-v22 * sqraa2 + (aa3 * v12 * v22 - aa1 * u12 - aa3) * aa2 + sqraa3 * v12 + aa1 * aa3 * u12 * v22 + (-v22 + v12) * sqraa1) * cos(s3) - s3 * (aa1 * v12 * v22 - v22 * u12 * aa2 - aa3 * u12 + aa1) * sin(s3) + (-aa3 * v22 + aa2) * (s9)) / (s2), ((u22 * sqraa1 + (-aa3 * u12 * u22 + aa2 * v12 + aa3) * aa1 - sqraa3 * u12 - aa3 * v12 * u22 * aa2 - sqraa2 * (u12 - u22)) * cos(s3) + s3 * (v12 * u22 * aa1 - aa2 * u12 * u22 + aa3 * v12 - aa2) * sin(s3) - (-aa3 * u22 + aa1) * (s9)) / (s2), ((-v12 * u22 * sqraa1 + ((u12 * u22 - v12 * v22) * aa2 - aa3 * v22) * aa1 + v22 * u12 * sqraa2 + aa3 * u22 * aa2 + sqraa3 * (u12 * v22 - u22 * v12)) * cos(s3) + (u22 * aa1 + v22 * aa2 - aa3 * (u12 * u22 + v12 * v22)) * s3 * sin(s3) + (aa1 * v22 - aa2 * u22) * (s9)) / (s2), -s12 * (-(s2) * ((b1 * v22 - b2 * u22 - v12 * (-b3 * u22 + b1)) * k1 + ((v12 * (b3 * v22 - b2) + (-b3 * u22 + b1) * u12) * aa2 - aa3 * ((b1 * v22 - b2 * u22) * u12 - b3 * v22 + b2)) * sqraa1 + (((-b3 * v22 + b2) * u12 + b1 * v22 - b2 * u22) * sqraa2 + ((-b1 * v22 * v12 + b2 * u22 * v12 - b3 * u22 + b1) * aa3 + (-b1 * v22 + b2 * u22) * u12 - b3 * v22 + b2) * aa2 - aa3 * (aa3 * ((b3 * v22 - b2) * u12 + v12 * (-b3 * u22 + b1)) + (-b3 * u22 + b1) * u12 - v12 * (b3 * v22 - b2))) * aa1 - (sqraa2 + sqraa3) * (b1 * v22 * v12 - b2 * u22 * v12 - b3 * u22 + b1)) * sin(s3) + (((b1 * v22 * v12 - b2 * u22 * v12 - b3 * u22 + b1) * k7 + (((-b1 * v22 + b2 * u22) * u12 - b3 * v22 + b2) * aa2 - aa3 * ((-b3 * u22 + b1) * u12 - v12 * (b3 * v22 - b2))) * k1 + ((b1 * v22 * v12 - b2 * u22 * v12 - b3 * u22 + b1) * sqraa2 + ((b3 * u22 - b1) * u12 - v12 * (b3 * v22 - b2)) * aa2 + aa3 * ((b1 * v22 * v12 - b2 * u22 * v12 - b3 * u22 + b1) * aa3 + (b1 * v22 - b2 * u22) * u12 - b3 * v22 + b2)) * sqraa1 + (((-b1 * v22 + b2 * u22) * u12 - b3 * v22 + b2) * k2 + (((b3 * u22 - b1) * u12 + v12 * (b3 * v22 - b2)) * aa3 + (0.2e1 * b3 * v22 - 0.2e1 * b2) * u12 - 0.2e1 * v12 * (-b3 * u22 + b1)) * sqraa2 - aa3 * (((b1 * v22 - b2 * u22) * u12 + b3 * v22 - b2) * aa3 - 0.2e1 * b1 * v22 * v12 + 0.2e1 * b2 * u22 * v12 - 0.2e1 * b3 * u22 + 0.2e1 * b1) * aa2 + 0.2e1 * sqraa3 * (((b3 * u22 / 0.2e1 - b1 / 0.2e1) * u12 + v12 * (b3 * v22 - b2) / 0.2e1) * aa3 + (b3 * v22 - b2) * u12 + b1 * v22 - b2 * u22)) * aa1 - (((b3 * u22 - b1) * u12 - v12 * (b3 * v22 - b2)) * aa2 + aa3 * ((b1 * v22 - b2 * u22) * u12 - b3 * v22 + b2)) * (sqraa2 + sqraa3)) * cos(s3) + ((v12 * (b3 * v22 - b2) + (-b3 * u22 + b1) * u12) * aa2 - aa3 * ((b1 * v22 - b2 * u22) * u12 - b3 * v22 + b2)) * sqraa1 + (((-0.2e1 * b3 * v22 + 0.2e1 * b2) * u12 + 0.2e1 * v12 * (-b3 * u22 + b1)) * sqraa2 - 0.2e1 * aa3 * (b1 * v22 * v12 - b2 * u22 * v12 + b3 * u22 - b1) * aa2 - 0.2e1 * sqraa3 * ((b3 * v22 - b2) * u12 + b1 * v22 - b2 * u22)) * aa1 + (((b3 * u22 - b1) * u12 - v12 * (b3 * v22 - b2)) * aa2 + aa3 * ((b1 * v22 - b2 * u22) * u12 - b3 * v22 + b2)) * (sqraa2 + sqraa3)) * s3), -(-(s2) * ((-b2 * u22 + b1 * v22 - (b3 * v22 - b2) * u12) * k2 + ((v12 * (b3 * v22 - b2) + (-b3 * u22 + b1) * u12) * aa1 - aa3 * ((b1 * v22 - b2 * u22) * v12 + b3 * u22 - b1)) * sqraa2 + (((b3 * u22 - b1) * v12 + b1 * v22 - b2 * u22) * sqraa1 + ((-b1 * v22 * u12 + b2 * u12 * u22 + b3 * v22 - b2) * aa3 + (b1 * v22 - b2 * u22) * v12 - b3 * u22 + b1) * aa1 - aa3 * (aa3 * ((b3 * v22 - b2) * u12 + v12 * (-b3 * u22 + b1)) + (-b3 * v22 + b2) * v12 + (-b3 * u22 + b1) * u12)) * aa2 + (sqraa1 + sqraa3) * (b1 * v22 * u12 - b2 * u12 * u22 + b3 * v22 - b2)) * sin(s3) + (((-b1 * v22 * u12 + b2 * u12 * u22 - b3 * v22 + b2) * k6 + (((b1 * v22 - b2 * u22) * v12 - b3 * u22 + b1) * aa1 - aa3 * ((-b3 * v22 + b2) * v12 + (-b3 * u22 + b1) * u12)) * k2 + ((-b1 * v22 * u12 + b2 * u12 * u22 - b3 * v22 + b2) * sqraa1 + ((-b3 * v22 + b2) * v12 - (-b3 * u22 + b1) * u12) * aa1 - aa3 * ((b1 * v22 * u12 - b2 * u12 * u22 + b3 * v22 - b2) * aa3 + (-b1 * v22 + b2 * u22) * v12 - b3 * u22 + b1)) * sqraa2 + (((b1 * v22 - b2 * u22) * v12 - b3 * u22 + b1) * k1 + ((-(-b3 * u22 + b1) * u12 + v12 * (b3 * v22 - b2)) * aa3 + (-0.2e1 * b3 * u22 + 0.2e1 * b1) * v12 - 0.2e1 * (b3 * v22 - b2) * u12) * sqraa1 + aa3 * (((b1 * v22 - b2 * u22) * v12 - b3 * u22 + b1) * aa3 + 0.2e1 * b1 * v22 * u12 - 0.2e1 * b2 * u12 * u22 - 0.2e1 * b3 * v22 + 0.2e1 * b2) * aa1 + 0.2e1 * sqraa3 * (((b3 * v22 / 0.2e1 - b2 / 0.2e1) * v12 - (-b3 * u22 + b1) * u12 / 0.2e1) * aa3 + v12 * (-b3 * u22 + b1) + b1 * v22 - b2 * u22)) * aa2 - (((-b3 * v22 + b2) * v12 - (-b3 * u22 + b1) * u12) * aa1 + aa3 * ((b1 * v22 - b2 * u22) * v12 + b3 * u22 - b1)) * (sqraa1 + sqraa3)) * cos(s3) + ((v12 * (b3 * v22 - b2) + (-b3 * u22 + b1) * u12) * aa1 - aa3 * ((b1 * v22 - b2 * u22) * v12 + b3 * u22 - b1)) * sqraa2 + (((0.2e1 * b3 * u22 - 0.2e1 * b1) * v12 + 0.2e1 * (b3 * v22 - b2) * u12) * sqraa1 - 0.2e1 * aa3 * (b1 * v22 * u12 - b2 * u12 * u22 - b3 * v22 + b2) * aa1 - 0.2e1 * sqraa3 * (v12 * (-b3 * u22 + b1) + b1 * v22 - b2 * u22)) * aa2 + (((-b3 * v22 + b2) * v12 - (-b3 * u22 + b1) * u12) * aa1 + aa3 * ((b1 * v22 - b2 * u22) * v12 + b3 * u22 - b1)) * (sqraa1 + sqraa3)) * s3) * s12, -(-((-b3 * u12 * v22 + b3 * u22 * v12 - b1 * v12 + b2 * u12) * k3 + ((b2 * u12 * u22 + (-b1 * u12 + b3) * v22 - b2) * aa1 - ((-b2 * v12 + b3) * u22 + b1 * (v12 * v22 - 0.1e1)) * aa2) * sqraa3 + (((b3 * v12 - b2) * u22 + b1 * (v22 - v12)) * sqraa1 + ((-b3 * u12 * u22 + v22 * b3 * v12 + b1 * u12 - b2 * v12) * aa2 + (-b2 * v12 - b3) * u22 + b1 * (v12 * v22 + 0.1e1)) * aa1 + ((-b2 * u22 + (-b3 * u12 + b1) * v22 + b2 * u12) * aa2 + b2 * u12 * u22 + (-b1 * u12 - b3) * v22 + b2) * aa2) * aa3 + (sqraa1 + sqraa2) * (-b3 * u12 * u22 - v22 * b3 * v12 + b1 * u12 + b2 * v12)) * (s2)* sin(s3) + s3 * (((b3 * u12 * u22 + v22 * b3 * v12 - b1 * u12 - b2 * v12) * k5 + (((-b2 * v12 - b3) * u22 + b1 * (v12 * v22 + 0.1e1)) * aa1 - (-b2 * u12 * u22 + (b1 * u12 + b3) * v22 - b2) * aa2) * k3 + ((b3 * u12 * u22 + v22 * b3 * v12 - b1 * u12 - b2 * v12) * sqraa1 + (-b2 * u12 * u22 + (b1 * u12 - b3) * v22 + b2) * aa1 + ((b3 * u12 * u22 + v22 * b3 * v12 - b1 * u12 - b2 * v12) * aa2 + (-b2 * v12 + b3) * u22 + b1 * (v12 * v22 - 0.1e1)) * aa2) * sqraa3 + (((-b2 * v12 - b3) * u22 + b1 * (v12 * v22 + 0.1e1)) * k1 + ((b2 * u12 * u22 + (-b1 * u12 - b3) * v22 + b2) * aa2 + 0.2e1 * b2 * u22 + (-0.2e1 * b3 * u12 - 0.2e1 * b1) * v22 + 0.2e1 * b2 * u12) * sqraa1 + (((-b2 * v12 - b3) * u22 + b1 * (v12 * v22 + 0.1e1)) * aa2 - 0.2e1 * v22 * b3 * v12 + 0.2e1 * b3 * u12 * u22 - 0.2e1 * b1 * u12 + 0.2e1 * b2 * v12) * aa2 * aa1 - ((-b2 * u12 * u22 + (b1 * u12 + b3) * v22 - b2) * aa2 + (-0.2e1 * b3 * v12 - 0.2e1 * b2) * u22 + 0.2e1 * b1 * (v22 + v12)) * sqraa2) * aa3 - (sqraa1 + sqraa2) * ((-b2 * u12 * u22 + (b1 * u12 - b3) * v22 + b2) * aa1 + ((-b2 * v12 + b3) * u22 + b1 * (v12 * v22 - 0.1e1)) * aa2)) * cos(s3) + ((b2 * u12 * u22 + (-b1 * u12 + b3) * v22 - b2) * aa1 - ((-b2 * v12 + b3) * u22 + b1 * (v12 * v22 - 0.1e1)) * aa2) * sqraa3 + ((-0.2e1 * b2 * u22 + (0.2e1 * b3 * u12 + 0.2e1 * b1) * v22 - 0.2e1 * b2 * u12) * sqraa1 + 0.2e1 * (-b3 * u12 * u22 + v22 * b3 * v12 + b1 * u12 - b2 * v12) * aa2 * aa1 + 0.2e1 * ((-b3 * v12 - b2) * u22 + b1 * (v22 + v12)) * sqraa2) * aa3 + (sqraa1 + sqraa2) * ((-b2 * u12 * u22 + (b1 * u12 - b3) * v22 + b2) * aa1 + ((-b2 * v12 + b3) * u22 + b1 * (v12 * v22 - 0.1e1)) * aa2))) * s12,
					-(((aa1 * v11 - aa2 * u11) * a3_1 - v21 * aa2 - aa3) * (s2)* sin(s3) + (((sqraa1 - aa1 * aa3 * u11 + aa2 * (-aa3 * v11 + aa2)) * a3_1 + aa1 * (-aa3 * v21 + aa2)) * cos(s3) + aa3 * (s6)* a3_1 - aa1 * (-aa3 * v21 + aa2)) * s3) * s1, ((s2) * ((aa1 * v11 - aa2 * u11) * a1_1 - aa2 * u21) * sin(s3) + s3 * (((sqraa1 - aa1 * aa3 * u11 + aa2 * (-aa3 * v11 + aa2)) * a1_1 - aa1 * aa3 * u21 - sqraa2 - sqraa3) * cos(s3) + aa3 * (s6)* a1_1 + aa1 * aa3 * u21 - sqraa1)) * s1, s1 * ((s2) * (a1_1 * aa1 + a3_1 * aa2 - aa3 * (a1_1 * u11 + a3_1 * v11 + u21)) * sin(s3) - s3 * ((v11 * a1_1 * sqraa1 + ((-a1_1 * u11 + a3_1 * v11 - u21) * aa2 + aa3 * a3_1) * aa1 + (-a3_1 * u11 - v21) * sqraa2 - aa3 * a1_1 * aa2 - sqraa3 * (-a1_1 * v11 + a3_1 * u11 + v21)) * cos(s3) + (-a3_1 * u11 - v21) * sqraa1 + ((a1_1 * u11 - a3_1 * v11 + u21) * aa2 - aa3 * a3_1) * aa1 + (aa2 * v11 + aa3) * a1_1 * aa2)), s12 * (-(s2) * ((-a1_1 * b3 * v11 + b2 * a1_1 - b1 * a3_1) * k1 + (aa3 * (-b2 * u11 * a1_1 + b1 * a3_1 * u11 - a3_1 * b3 + s4) - ((-a1_1 * u11 + a3_1 * v11 - u21) * b3 + b1) * aa2) * sqraa1 + (sqraa3 * ((-a1_1 * v11 + a3_1 * u11 + v21) * b3 - b2) + ((a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa2 + (-a1_1 * u11 - a3_1 * v11 - u21) * b3 + b1) * aa3 + (((a3_1 * u11 + v21) * b3 - b1 * a3_1 + b2 * (a1_1 - 0.1e1)) * aa2 + b1 * a3_1 * u11 - b2 * u11 * a1_1 + b1 * v21 + a3_1 * b3 - b2 * u21) * aa2) * aa1 + (-a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * (sqraa2 + sqraa3)) * sin(s3) + (((a1_1 * b3 - v11 * (-b2 * a1_1 + b1 * a3_1)) * k7 + (aa3 * ((-a1_1 * u11 - a3_1 * v11 - u21) * b3 + b1) + (-b2 * u11 * a1_1 + b1 * a3_1 * u11 + a3_1 * b3 + s4) * aa2) * k1 + ((a1_1 * b3 - v11 * (-b2 * a1_1 + b1 * a3_1)) * sqraa3 + (b2 * u11 * a1_1 - b1 * a3_1 * u11 + a3_1 * b3 - b1 * v21 + b2 * u21) * aa3 - aa2 * ((-a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa2 + (a1_1 * u11 - a3_1 * v11 + u21) * b3 - b1)) * sqraa1 + (((-a1_1 * u11 - a3_1 * v11 - u21) * b3 + b1) * k3 + ((-b2 * u11 * a1_1 + b1 * a3_1 * u11 + a3_1 * b3 + s4) * aa2 + (-0.2e1 * a3_1 * u11 - 0.2e1 * v21) * b3 - 0.2e1 * b1 * a3_1 + 0.2e1 * b2 * (a1_1 + 0.1e1)) * sqraa3 - 0.2e1 * (((a1_1 * u11 / 0.2e1 + a3_1 * v11 / 0.2e1 + u21 / 0.2e1) * b3 - b1 / 0.2e1) * aa2 + a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa2 * aa3 + ((-b2 * u11 * a1_1 + b1 * a3_1 * u11 + a3_1 * b3 + s4) * aa2 + (-0.2e1 * a1_1 * v11 - 0.2e1 * a3_1 * u11 - 0.2e1 * v21) * b3 + 0.2e1 * b2) * sqraa2) * aa1 + (aa3 * (-b2 * u11 * a1_1 + b1 * a3_1 * u11 - a3_1 * b3 + s4) - ((-a1_1 * u11 + a3_1 * v11 - u21) * b3 + b1) * aa2) * (sqraa2 + sqraa3)) * cos(s3) + (aa3 * (-b2 * u11 * a1_1 + b1 * a3_1 * u11 - a3_1 * b3 + s4) - ((-a1_1 * u11 + a3_1 * v11 - u21) * b3 + b1) * aa2) * sqraa1 + (((0.2e1 * a3_1 * u11 + 0.2e1 * v21) * b3 + 0.2e1 * b1 * a3_1 - 0.2e1 * b2 * (a1_1 + 0.1e1)) * sqraa3 + 0.2e1 * aa3 * (a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa2 + 0.2e1 * ((a1_1 * v11 + a3_1 * u11 + v21) * b3 - b2) * sqraa2) * aa1 - (aa3 * (-b2 * u11 * a1_1 + b1 * a3_1 * u11 - a3_1 * b3 + s4) - ((-a1_1 * u11 + a3_1 * v11 - u21) * b3 + b1) * aa2) * (sqraa2 + sqraa3)) * s3), s12 * ((s2) * (((-a3_1 * u11 - v21) * b3 + b1 * a3_1 - b2 * a1_1 + b2) * k2 + (((-a1_1 * u11 + a3_1 * v11 - u21) * b3 + b1) * aa1 - aa3 * (a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1))) * sqraa2 + ((a1_1 * b3 * v11 - b2 * a1_1 + b1 * a3_1) * sqraa1 + ((b2 * u11 * a1_1 - b1 * a3_1 * u11 + a3_1 * b3 - b1 * v21 + b2 * u21) * aa3 - a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa1 - aa3 * (aa3 * ((-a1_1 * v11 + a3_1 * u11 + v21) * b3 - b2) + (-a1_1 * u11 - a3_1 * v11 - u21) * b3 + b1)) * aa2 + (sqraa1 + sqraa3) * (-b2 * u11 * a1_1 + b1 * a3_1 * u11 + a3_1 * b3 + s4)) * sin(s3) + s3 * (((-b2 * u11 * a1_1 + b1 * a3_1 * u11 + a3_1 * b3 + s4) * k6 + ((a1_1 * b3 - v11 * (-b2 * a1_1 + b1 * a3_1)) * aa1 + aa3 * ((-a1_1 * u11 - a3_1 * v11 - u21) * b3 + b1)) * k2 + ((-b2 * u11 * a1_1 + b1 * a3_1 * u11 + a3_1 * b3 + s4) * sqraa1 + ((-a1_1 * u11 + a3_1 * v11 - u21) * b3 + b1) * aa1 + aa3 * ((-b2 * u11 * a1_1 + b1 * a3_1 * u11 + a3_1 * b3 + s4) * aa3 - a1_1 * b3 - v11 * (-b2 * a1_1 + b1 * a3_1))) * sqraa2 + ((a1_1 * b3 - v11 * (-b2 * a1_1 + b1 * a3_1)) * k1 + (aa3 * ((-a1_1 * u11 - a3_1 * v11 - u21) * b3 + b1) + (0.2e1 * a1_1 * v11 + 0.2e1 * a3_1 * u11 + 0.2e1 * v21) * b3 - 0.2e1 * b2) * sqraa1 - 0.2e1 * aa3 * ((-a1_1 * b3 / 0.2e1 + v11 * (-b2 * a1_1 + b1 * a3_1) / 0.2e1) * aa3 + b1 * a3_1 * u11 - b2 * u11 * a1_1 + b1 * v21 - a3_1 * b3 - b2 * u21) * aa1 - 0.2e1 * sqraa3 * (((a1_1 * u11 / 0.2e1 + a3_1 * v11 / 0.2e1 + u21 / 0.2e1) * b3 - b1 / 0.2e1) * aa3 - a1_1 * b3 * v11 + b1 * a3_1 - b2 * a1_1)) * aa2 + (sqraa1 + sqraa3) * (((a1_1 * u11 - a3_1 * v11 + u21) * b3 - b1) * aa1 + aa3 * (a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)))) * cos(s3) + (((a1_1 * u11 - a3_1 * v11 + u21) * b3 - b1) * aa1 + aa3 * (a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1))) * sqraa2 + (((-0.2e1 * a1_1 * v11 - 0.2e1 * a3_1 * u11 - 0.2e1 * v21) * b3 + 0.2e1 * b2) * sqraa1 + 0.2e1 * aa3 * (-b2 * u11 * a1_1 + b1 * a3_1 * u11 - a3_1 * b3 + s4) * aa1 + 0.2e1 * sqraa3 * (-a1_1 * b3 * v11 - b2 * a1_1 + b1 * a3_1)) * aa2 - (sqraa1 + sqraa3) * (((a1_1 * u11 - a3_1 * v11 + u21) * b3 - b1) * aa1 + aa3 * (a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1))))), s12 * (-(s2) * (((-a1_1 * v11 + a3_1 * u11 + v21) * b3 - b2) * k3 + ((-b2 * u11 * a1_1 + b1 * a3_1 * u11 - a3_1 * b3 + s4) * aa1 + (a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa2) * sqraa3 + ((-a1_1 * b3 * v11 + b2 * a1_1 - b1 * a3_1) * sqraa1 + (((a1_1 * u11 - a3_1 * v11 + u21) * b3 - b1) * aa2 + a1_1 * b3 - v11 * (-b2 * a1_1 + b1 * a3_1)) * aa1 + (((a3_1 * u11 + v21) * b3 - b1 * a3_1 + b2 * a1_1 - b2) * aa2 + b1 * a3_1 * u11 - b2 * u11 * a1_1 + b1 * v21 + a3_1 * b3 - b2 * u21) * aa2) * aa3 - (sqraa1 + sqraa2) * ((-a1_1 * u11 - a3_1 * v11 - u21) * b3 + b1)) * sin(s3) + s3 * ((((-a1_1 * u11 - a3_1 * v11 - u21) * b3 + b1) * k5 + ((a1_1 * b3 - v11 * (-b2 * a1_1 + b1 * a3_1)) * aa1 + (-b2 * u11 * a1_1 + b1 * a3_1 * u11 + a3_1 * b3 + s4) * aa2) * k3 + (((-a1_1 * u11 - a3_1 * v11 - u21) * b3 + b1) * sqraa1 + (b2 * u11 * a1_1 - b1 * a3_1 * u11 + a3_1 * b3 - b1 * v21 + b2 * u21) * aa1 - (((a1_1 * u11 + a3_1 * v11 + u21) * b3 - b1) * aa2 + a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa2) * sqraa3 + ((a1_1 * b3 - v11 * (-b2 * a1_1 + b1 * a3_1)) * k1 + ((-b2 * u11 * a1_1 + b1 * a3_1 * u11 + a3_1 * b3 + s4) * aa2 + (0.2e1 * a3_1 * u11 + 0.2e1 * v21) * b3 + 0.2e1 * b1 * a3_1 - 0.2e1 * b2 * a1_1 - 0.2e1 * b2) * sqraa1 - ((-a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa2 + (0.2e1 * a1_1 * u11 - 0.2e1 * a3_1 * v11 + 0.2e1 * u21) * b3 - 0.2e1 * b1) * aa2 * aa1 + sqraa2 * ((-b2 * u11 * a1_1 + b1 * a3_1 * u11 + a3_1 * b3 + s4) * aa2 - 0.2e1 * a1_1 * b3 * v11 + 0.2e1 * b1 * a3_1 - 0.2e1 * b2 * a1_1)) * aa3 + (sqraa1 + sqraa2) * ((-b2 * u11 * a1_1 + b1 * a3_1 * u11 - a3_1 * b3 + s4) * aa1 + (a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa2)) * cos(s3) + ((-b2 * u11 * a1_1 + b1 * a3_1 * u11 - a3_1 * b3 + s4) * aa1 + (a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa2) * sqraa3 + (((-0.2e1 * a3_1 * u11 - 0.2e1 * v21) * b3 - 0.2e1 * b1 * a3_1 + 0.2e1 * b2 * a1_1 + 0.2e1 * b2) * sqraa1 - 0.2e1 * ((-a1_1 * u11 + a3_1 * v11 - u21) * b3 + b1) * aa2 * aa1 - 0.2e1 * sqraa2 * (-a1_1 * b3 * v11 - b2 * a1_1 + b1 * a3_1)) * aa3 - (sqraa1 + sqraa2) * ((-b2 * u11 * a1_1 + b1 * a3_1 * u11 - a3_1 * b3 + s4) * aa1 + (a1_1 * b3 + v11 * (-b2 * a1_1 + b1 * a3_1)) * aa2))),
					-((s2) * ((aa1 * v11 - aa2 * u11) * a4_1 + aa1 * v21) * sin(s3) + s3 * (((sqraa1 - aa1 * aa3 * u11 + aa2 * (-aa3 * v11 + aa2)) * a4_1 - aa2 * aa3 * v21 - sqraa1 - sqraa3) * cos(s3) + aa3 * (s6)* a4_1 + aa2 * aa3 * v21 - sqraa2)) * s1, ((s2) * ((aa1 * v11 - aa2 * u11) * a2_1 + u21 * aa1 + aa3) * sin(s3) + (((sqraa1 - aa1 * aa3 * u11 + aa2 * (-aa3 * v11 + aa2)) * a2_1 + aa2 * (-aa3 * u21 + aa1)) * cos(s3) + aa3 * (s6)* a2_1 - aa2 * (-aa3 * u21 + aa1)) * s3) * s1, s1 * ((a2_1 * aa1 + a4_1 * aa2 - aa3 * (a2_1 * u11 + a4_1 * v11 + v21)) * (s2)* sin(s3) - s3 * (((a2_1 * v11 + u21) * sqraa1 + ((-a2_1 * u11 + a4_1 * v11 + v21) * aa2 + aa3 * a4_1) * aa1 - u11 * a4_1 * sqraa2 - aa3 * a2_1 * aa2 - sqraa3 * (-a2_1 * v11 + a4_1 * u11 - u21)) * cos(s3) - a4_1 * sqraa1 * u11 + ((a2_1 * u11 - a4_1 * v11 - v21) * aa2 - aa3 * a4_1) * aa1 + aa2 * ((a2_1 * v11 + u21) * aa2 + aa3 * a2_1))), -((((-a2_1 * v11 - u21) * b3 - b1 * a4_1 + b2 * a2_1 + b1) * k1 + (((a2_1 * u11 - a4_1 * v11 - v21) * b3 + b2) * aa2 + aa3 * (-a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1))) * sqraa1 + ((a4_1 * b3 * u11 + b2 * a2_1 - b1 * a4_1) * sqraa2 + (aa3 * (-b2 * v11 * a2_1 + b1 * a4_1 * v11 + a2_1 * b3 + s4) + a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1)) * aa2 + aa3 * (aa3 * ((-a2_1 * v11 + a4_1 * u11 - u21) * b3 + b1) + (-a2_1 * u11 - a4_1 * v11 - v21) * b3 + b2)) * aa1 + (sqraa2 + sqraa3) * (-b2 * v11 * a2_1 + b1 * a4_1 * v11 - a2_1 * b3 + s4)) * (s2)* sin(s3) + (((-b2 * v11 * a2_1 + b1 * a4_1 * v11 - a2_1 * b3 + s4) * k7 + ((-a4_1 * b3 - u11 * (-b2 * a2_1 + b1 * a4_1)) * aa2 + aa3 * ((a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2)) * k1 + ((-b2 * v11 * a2_1 + b1 * a4_1 * v11 - a2_1 * b3 + s4) * sqraa2 + ((a2_1 * u11 - a4_1 * v11 - v21) * b3 + b2) * aa2 + aa3 * ((-b2 * v11 * a2_1 + b1 * a4_1 * v11 - a2_1 * b3 + s4) * aa3 - a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1))) * sqraa1 + ((-a4_1 * b3 - u11 * (-b2 * a2_1 + b1 * a4_1)) * k2 + (aa3 * ((a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) + (0.2e1 * a2_1 * v11 + 0.2e1 * a4_1 * u11 + 0.2e1 * u21) * b3 - 0.2e1 * b1) * sqraa2 + 0.2e1 * aa3 * ((-a4_1 * b3 / 0.2e1 - u11 * (-b2 * a2_1 + b1 * a4_1) / 0.2e1) * aa3 + b1 * a4_1 * v11 - b2 * v11 * a2_1 + b1 * v21 + a2_1 * b3 - b2 * u21) * aa2 + 0.2e1 * sqraa3 * (((a2_1 * u11 / 0.2e1 + a4_1 * v11 / 0.2e1 + v21 / 0.2e1) * b3 - b2 / 0.2e1) * aa3 + a4_1 * b3 * u11 + b1 * a4_1 - b2 * a2_1)) * aa1 - (((a2_1 * u11 - a4_1 * v11 - v21) * b3 + b2) * aa2 + aa3 * (-a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1))) * (sqraa2 + sqraa3)) * cos(s3) + (((-a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * aa2 - aa3 * (-a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1))) * sqraa1 + (((-0.2e1 * a2_1 * v11 - 0.2e1 * a4_1 * u11 - 0.2e1 * u21) * b3 + 0.2e1 * b1) * sqraa2 - 0.2e1 * aa3 * (-b2 * v11 * a2_1 + b1 * a4_1 * v11 + a2_1 * b3 + s4) * aa2 - 0.2e1 * sqraa3 * (a4_1 * b3 * u11 - b2 * a2_1 + b1 * a4_1)) * aa1 + (((a2_1 * u11 - a4_1 * v11 - v21) * b3 + b2) * aa2 + aa3 * (-a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1))) * (sqraa2 + sqraa3)) * s3) * s12, -s12 * (-((-a4_1 * b3 * u11 - b2 * a2_1 + b1 * a4_1) * k2 + ((b2 * v11 * a2_1 - b1 * a4_1 * v11 - a2_1 * b3 - b1 * v21 + b2 * u21) * aa3 + ((-a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * aa1) * sqraa2 + (((a2_1 * v11 - a4_1 * u11 + u21) * b3 - b1) * sqraa3 + ((a4_1 * b3 - u11 * (-b2 * a2_1 + b1 * a4_1)) * aa1 + (a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * aa3 + (((a2_1 * v11 + u21) * b3 - b2 * a2_1 + b1 * (a4_1 - 0.1e1)) * aa1 + b1 * a4_1 * v11 - b2 * v11 * a2_1 + b1 * v21 - a2_1 * b3 - b2 * u21) * aa1) * aa2 + (sqraa1 + sqraa3) * (a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1))) * (s2)* sin(s3) + (((-a4_1 * b3 - u11 * (-b2 * a2_1 + b1 * a4_1)) * k6 + (aa3 * ((a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) + (-b2 * v11 * a2_1 + b1 * a4_1 * v11 - a2_1 * b3 + s4) * aa1) * k2 + ((-a4_1 * b3 - u11 * (-b2 * a2_1 + b1 * a4_1)) * sqraa3 + aa3 * (-b2 * v11 * a2_1 + b1 * a4_1 * v11 + a2_1 * b3 + s4) - aa1 * ((a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1)) * aa1 + (-a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2)) * sqraa2 + (((a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * k3 + ((-b2 * v11 * a2_1 + b1 * a4_1 * v11 - a2_1 * b3 + s4) * aa1 + (-0.2e1 * a2_1 * v11 - 0.2e1 * u21) * b3 - 0.2e1 * b2 * a2_1 + 0.2e1 * b1 * (a4_1 + 0.1e1)) * sqraa3 + 0.2e1 * aa1 * (((a2_1 * u11 / 0.2e1 + a4_1 * v11 / 0.2e1 + v21 / 0.2e1) * b3 - b2 / 0.2e1) * aa1 - a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1)) * aa3 + ((-b2 * v11 * a2_1 + b1 * a4_1 * v11 - a2_1 * b3 + s4) * aa1 + (-0.2e1 * a2_1 * v11 - 0.2e1 * a4_1 * u11 - 0.2e1 * u21) * b3 + 0.2e1 * b1) * sqraa1) * aa2 - (sqraa1 + sqraa3) * (aa3 * (-b2 * v11 * a2_1 + b1 * a4_1 * v11 + a2_1 * b3 + s4) - ((-a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * aa1)) * cos(s3) + ((b2 * v11 * a2_1 - b1 * a4_1 * v11 - a2_1 * b3 - b1 * v21 + b2 * u21) * aa3 + ((-a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * aa1) * sqraa2 + (((0.2e1 * a2_1 * v11 + 0.2e1 * u21) * b3 + 0.2e1 * b2 * a2_1 - 0.2e1 * b1 * (a4_1 + 0.1e1)) * sqraa3 - 0.2e1 * aa1 * (-a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1)) * aa3 - 0.2e1 * sqraa1 * ((-a2_1 * v11 - a4_1 * u11 - u21) * b3 + b1)) * aa2 + (sqraa1 + sqraa3) * (aa3 * (-b2 * v11 * a2_1 + b1 * a4_1 * v11 + a2_1 * b3 + s4) - ((-a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * aa1)) * s3), -s12 * (-(((a2_1 * v11 - a4_1 * u11 + u21) * b3 - b1) * k3 + ((b2 * v11 * a2_1 - b1 * a4_1 * v11 - a2_1 * b3 - b1 * v21 + b2 * u21) * aa2 - (-a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1)) * aa1) * sqraa3 + ((-a4_1 * b3 * u11 - b2 * a2_1 + b1 * a4_1) * sqraa2 + (((-a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * aa1 - a4_1 * b3 - u11 * (-b2 * a2_1 + b1 * a4_1)) * aa2 + (((a2_1 * v11 + u21) * b3 - b2 * a2_1 + b1 * (a4_1 - 0.1e1)) * aa1 + b1 * a4_1 * v11 - b2 * v11 * a2_1 + b1 * v21 - a2_1 * b3 - b2 * u21) * aa1) * aa3 - ((a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * (sqraa1 + sqraa2)) * (s2)* sin(s3) + s3 * ((((a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * k5 + ((-a4_1 * b3 - u11 * (-b2 * a2_1 + b1 * a4_1)) * aa2 + (-b2 * v11 * a2_1 + b1 * a4_1 * v11 - a2_1 * b3 + s4) * aa1) * k3 + (((a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * sqraa2 + (-b2 * v11 * a2_1 + b1 * a4_1 * v11 + a2_1 * b3 + s4) * aa2 + aa1 * (((a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * aa1 - a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1))) * sqraa3 + ((-a4_1 * b3 - u11 * (-b2 * a2_1 + b1 * a4_1)) * k2 + ((-b2 * v11 * a2_1 + b1 * a4_1 * v11 - a2_1 * b3 + s4) * aa1 + (0.2e1 * a2_1 * v11 + 0.2e1 * u21) * b3 + 0.2e1 * b2 * a2_1 - 0.2e1 * b1 * (a4_1 + 0.1e1)) * sqraa2 - aa1 * ((a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1)) * aa1 + (-0.2e1 * a2_1 * u11 + 0.2e1 * a4_1 * v11 + 0.2e1 * v21) * b3 - 0.2e1 * b2) * aa2 + ((-b2 * v11 * a2_1 + b1 * a4_1 * v11 - a2_1 * b3 + s4) * aa1 - 0.2e1 * a4_1 * b3 * u11 - 0.2e1 * b1 * a4_1 + 0.2e1 * b2 * a2_1) * sqraa1) * aa3 - (sqraa1 + sqraa2) * ((-b2 * v11 * a2_1 + b1 * a4_1 * v11 + a2_1 * b3 + s4) * aa2 + (-a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1)) * aa1)) * cos(s3) + ((b2 * v11 * a2_1 - b1 * a4_1 * v11 - a2_1 * b3 - b1 * v21 + b2 * u21) * aa2 - (-a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1)) * aa1) * sqraa3 + (((-0.2e1 * a2_1 * v11 - 0.2e1 * u21) * b3 - 0.2e1 * b2 * a2_1 + 0.2e1 * b1 * (a4_1 + 0.1e1)) * sqraa2 + 0.2e1 * ((-a2_1 * u11 + a4_1 * v11 + v21) * b3 - b2) * aa1 * aa2 + 0.2e1 * sqraa1 * (a4_1 * b3 * u11 - b2 * a2_1 + b1 * a4_1)) * aa3 + (sqraa1 + sqraa2) * ((-b2 * v11 * a2_1 + b1 * a4_1 * v11 + a2_1 * b3 + s4) * aa2 + (-a4_1 * b3 + u11 * (-b2 * a2_1 + b1 * a4_1)) * aa1))),
					-(((aa1 * v12 - aa2 * u12) * a3_2 - v22 * aa2 - aa3) * (s2)* sin(s3) + (((sqraa1 - aa1 * aa3 * u12 + aa2 * (-aa3 * v12 + aa2)) * a3_2 + aa1 * (-aa3 * v22 + aa2)) * cos(s3) + aa3 * (s9)* a3_2 - aa1 * (-aa3 * v22 + aa2)) * s3) * s1, ((s2) * ((aa1 * v12 - aa2 * u12) * a1_2 - aa2 * u22) * sin(s3) + s3 * (((sqraa1 - aa1 * aa3 * u12 + aa2 * (-aa3 * v12 + aa2)) * a1_2 - aa1 * aa3 * u22 - sqraa2 - sqraa3) * cos(s3) + aa3 * (s9)* a1_2 + aa1 * aa3 * u22 - sqraa1)) * s1, s1 * ((s2) * (a1_2 * aa1 + a3_2 * aa2 - aa3 * (a1_2 * u12 + a3_2 * v12 + u22)) * sin(s3) - s3 * ((v12 * a1_2 * sqraa1 + ((-a1_2 * u12 + a3_2 * v12 - u22) * aa2 + aa3 * a3_2) * aa1 + (-a3_2 * u12 - v22) * sqraa2 - aa3 * a1_2 * aa2 - sqraa3 * (-a1_2 * v12 + a3_2 * u12 + v22)) * cos(s3) + (-a3_2 * u12 - v22) * sqraa1 + ((a1_2 * u12 - a3_2 * v12 + u22) * aa2 - aa3 * a3_2) * aa1 + (aa2 * v12 + aa3) * a1_2 * aa2)), s12 * (-(s2) * ((-a1_2 * b3 * v12 + b2 * a1_2 - b1 * a3_2) * k1 + (aa3 * (-b2 * u12 * a1_2 + b1 * a3_2 * u12 - a3_2 * b3 + b1 * v22 - b2 * u22) - ((-a1_2 * u12 + a3_2 * v12 - u22) * b3 + b1) * aa2) * sqraa1 + (sqraa3 * ((-a1_2 * v12 + a3_2 * u12 + v22) * b3 - b2) + ((a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa2 + (-a1_2 * u12 - a3_2 * v12 - u22) * b3 + b1) * aa3 + (((a3_2 * u12 + v22) * b3 - b1 * a3_2 + b2 * (a1_2 - 0.1e1)) * aa2 + b1 * a3_2 * u12 - b2 * u12 * a1_2 + b1 * v22 + a3_2 * b3 - b2 * u22) * aa2) * aa1 + (-a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * (sqraa2 + sqraa3)) * sin(s3) + (((a1_2 * b3 - v12 * (-b2 * a1_2 + b1 * a3_2)) * k7 + (aa3 * ((-a1_2 * u12 - a3_2 * v12 - u22) * b3 + b1) + (-b2 * u12 * a1_2 + b1 * a3_2 * u12 + a3_2 * b3 + b1 * v22 - b2 * u22) * aa2) * k1 + ((a1_2 * b3 - v12 * (-b2 * a1_2 + b1 * a3_2)) * sqraa3 + (b2 * u12 * a1_2 - b1 * a3_2 * u12 + a3_2 * b3 - b1 * v22 + b2 * u22) * aa3 - aa2 * ((-a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa2 + (a1_2 * u12 - a3_2 * v12 + u22) * b3 - b1)) * sqraa1 + (((-a1_2 * u12 - a3_2 * v12 - u22) * b3 + b1) * k3 + ((-b2 * u12 * a1_2 + b1 * a3_2 * u12 + a3_2 * b3 + b1 * v22 - b2 * u22) * aa2 + (-0.2e1 * a3_2 * u12 - 0.2e1 * v22) * b3 - 0.2e1 * b1 * a3_2 + 0.2e1 * b2 * (a1_2 + 0.1e1)) * sqraa3 - 0.2e1 * (((a1_2 * u12 / 0.2e1 + a3_2 * v12 / 0.2e1 + u22 / 0.2e1) * b3 - b1 / 0.2e1) * aa2 + a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa2 * aa3 + ((-b2 * u12 * a1_2 + b1 * a3_2 * u12 + a3_2 * b3 + b1 * v22 - b2 * u22) * aa2 + (-0.2e1 * a1_2 * v12 - 0.2e1 * a3_2 * u12 - 0.2e1 * v22) * b3 + 0.2e1 * b2) * sqraa2) * aa1 + (aa3 * (-b2 * u12 * a1_2 + b1 * a3_2 * u12 - a3_2 * b3 + b1 * v22 - b2 * u22) - ((-a1_2 * u12 + a3_2 * v12 - u22) * b3 + b1) * aa2) * (sqraa2 + sqraa3)) * cos(s3) + (aa3 * (-b2 * u12 * a1_2 + b1 * a3_2 * u12 - a3_2 * b3 + b1 * v22 - b2 * u22) - ((-a1_2 * u12 + a3_2 * v12 - u22) * b3 + b1) * aa2) * sqraa1 + (((0.2e1 * a3_2 * u12 + 0.2e1 * v22) * b3 + 0.2e1 * b1 * a3_2 - 0.2e1 * b2 * (a1_2 + 0.1e1)) * sqraa3 + 0.2e1 * aa3 * (a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa2 + 0.2e1 * ((a1_2 * v12 + a3_2 * u12 + v22) * b3 - b2) * sqraa2) * aa1 - (aa3 * (-b2 * u12 * a1_2 + b1 * a3_2 * u12 - a3_2 * b3 + b1 * v22 - b2 * u22) - ((-a1_2 * u12 + a3_2 * v12 - u22) * b3 + b1) * aa2) * (sqraa2 + sqraa3)) * s3), s12 * ((s2) * (((-a3_2 * u12 - v22) * b3 + b1 * a3_2 - b2 * a1_2 + b2) * k2 + (((-a1_2 * u12 + a3_2 * v12 - u22) * b3 + b1) * aa1 - aa3 * (a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2))) * sqraa2 + ((a1_2 * b3 * v12 - b2 * a1_2 + b1 * a3_2) * sqraa1 + ((b2 * u12 * a1_2 - b1 * a3_2 * u12 + a3_2 * b3 - b1 * v22 + b2 * u22) * aa3 - a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa1 - aa3 * (aa3 * ((-a1_2 * v12 + a3_2 * u12 + v22) * b3 - b2) + (-a1_2 * u12 - a3_2 * v12 - u22) * b3 + b1)) * aa2 + (sqraa1 + sqraa3) * (-b2 * u12 * a1_2 + b1 * a3_2 * u12 + a3_2 * b3 + b1 * v22 - b2 * u22)) * sin(s3) + s3 * (((-b2 * u12 * a1_2 + b1 * a3_2 * u12 + a3_2 * b3 + b1 * v22 - b2 * u22) * k6 + ((a1_2 * b3 - v12 * (-b2 * a1_2 + b1 * a3_2)) * aa1 + aa3 * ((-a1_2 * u12 - a3_2 * v12 - u22) * b3 + b1)) * k2 + ((-b2 * u12 * a1_2 + b1 * a3_2 * u12 + a3_2 * b3 + b1 * v22 - b2 * u22) * sqraa1 + ((-a1_2 * u12 + a3_2 * v12 - u22) * b3 + b1) * aa1 + aa3 * ((-b2 * u12 * a1_2 + b1 * a3_2 * u12 + a3_2 * b3 + b1 * v22 - b2 * u22) * aa3 - a1_2 * b3 - v12 * (-b2 * a1_2 + b1 * a3_2))) * sqraa2 + ((a1_2 * b3 - v12 * (-b2 * a1_2 + b1 * a3_2)) * k1 + (aa3 * ((-a1_2 * u12 - a3_2 * v12 - u22) * b3 + b1) + (0.2e1 * a1_2 * v12 + 0.2e1 * a3_2 * u12 + 0.2e1 * v22) * b3 - 0.2e1 * b2) * sqraa1 - 0.2e1 * aa3 * ((-a1_2 * b3 / 0.2e1 + v12 * (-b2 * a1_2 + b1 * a3_2) / 0.2e1) * aa3 + b1 * a3_2 * u12 - b2 * u12 * a1_2 + b1 * v22 - a3_2 * b3 - b2 * u22) * aa1 - 0.2e1 * sqraa3 * (((a1_2 * u12 / 0.2e1 + a3_2 * v12 / 0.2e1 + u22 / 0.2e1) * b3 - b1 / 0.2e1) * aa3 - a1_2 * b3 * v12 + b1 * a3_2 - b2 * a1_2)) * aa2 + (sqraa1 + sqraa3) * (((a1_2 * u12 - a3_2 * v12 + u22) * b3 - b1) * aa1 + aa3 * (a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)))) * cos(s3) + (((a1_2 * u12 - a3_2 * v12 + u22) * b3 - b1) * aa1 + aa3 * (a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2))) * sqraa2 + (((-0.2e1 * a1_2 * v12 - 0.2e1 * a3_2 * u12 - 0.2e1 * v22) * b3 + 0.2e1 * b2) * sqraa1 + 0.2e1 * aa3 * (-b2 * u12 * a1_2 + b1 * a3_2 * u12 - a3_2 * b3 + b1 * v22 - b2 * u22) * aa1 + 0.2e1 * sqraa3 * (-a1_2 * b3 * v12 - b2 * a1_2 + b1 * a3_2)) * aa2 - (sqraa1 + sqraa3) * (((a1_2 * u12 - a3_2 * v12 + u22) * b3 - b1) * aa1 + aa3 * (a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2))))), s12 * (-(s2) * (((-a1_2 * v12 + a3_2 * u12 + v22) * b3 - b2) * k3 + ((-b2 * u12 * a1_2 + b1 * a3_2 * u12 - a3_2 * b3 + b1 * v22 - b2 * u22) * aa1 + (a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa2) * sqraa3 + ((-a1_2 * b3 * v12 + b2 * a1_2 - b1 * a3_2) * sqraa1 + (((a1_2 * u12 - a3_2 * v12 + u22) * b3 - b1) * aa2 + a1_2 * b3 - v12 * (-b2 * a1_2 + b1 * a3_2)) * aa1 + (((a3_2 * u12 + v22) * b3 - b1 * a3_2 + b2 * a1_2 - b2) * aa2 + b1 * a3_2 * u12 - b2 * u12 * a1_2 + b1 * v22 + a3_2 * b3 - b2 * u22) * aa2) * aa3 - (sqraa1 + sqraa2) * ((-a1_2 * u12 - a3_2 * v12 - u22) * b3 + b1)) * sin(s3) + s3 * ((((-a1_2 * u12 - a3_2 * v12 - u22) * b3 + b1) * k5 + ((a1_2 * b3 - v12 * (-b2 * a1_2 + b1 * a3_2)) * aa1 + (-b2 * u12 * a1_2 + b1 * a3_2 * u12 + a3_2 * b3 + b1 * v22 - b2 * u22) * aa2) * k3 + (((-a1_2 * u12 - a3_2 * v12 - u22) * b3 + b1) * sqraa1 + (b2 * u12 * a1_2 - b1 * a3_2 * u12 + a3_2 * b3 - b1 * v22 + b2 * u22) * aa1 - (((a1_2 * u12 + a3_2 * v12 + u22) * b3 - b1) * aa2 + a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa2) * sqraa3 + ((a1_2 * b3 - v12 * (-b2 * a1_2 + b1 * a3_2)) * k1 + ((-b2 * u12 * a1_2 + b1 * a3_2 * u12 + a3_2 * b3 + b1 * v22 - b2 * u22) * aa2 + (0.2e1 * a3_2 * u12 + 0.2e1 * v22) * b3 + 0.2e1 * b1 * a3_2 - 0.2e1 * b2 * a1_2 - 0.2e1 * b2) * sqraa1 - ((-a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa2 + (0.2e1 * a1_2 * u12 - 0.2e1 * a3_2 * v12 + 0.2e1 * u22) * b3 - 0.2e1 * b1) * aa2 * aa1 + sqraa2 * ((-b2 * u12 * a1_2 + b1 * a3_2 * u12 + a3_2 * b3 + b1 * v22 - b2 * u22) * aa2 - 0.2e1 * a1_2 * b3 * v12 + 0.2e1 * b1 * a3_2 - 0.2e1 * b2 * a1_2)) * aa3 + (sqraa1 + sqraa2) * ((-b2 * u12 * a1_2 + b1 * a3_2 * u12 - a3_2 * b3 + b1 * v22 - b2 * u22) * aa1 + (a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa2)) * cos(s3) + ((-b2 * u12 * a1_2 + b1 * a3_2 * u12 - a3_2 * b3 + b1 * v22 - b2 * u22) * aa1 + (a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa2) * sqraa3 + (((-0.2e1 * a3_2 * u12 - 0.2e1 * v22) * b3 - 0.2e1 * b1 * a3_2 + 0.2e1 * b2 * a1_2 + 0.2e1 * b2) * sqraa1 - 0.2e1 * ((-a1_2 * u12 + a3_2 * v12 - u22) * b3 + b1) * aa2 * aa1 - 0.2e1 * sqraa2 * (-a1_2 * b3 * v12 - b2 * a1_2 + b1 * a3_2)) * aa3 - (sqraa1 + sqraa2) * ((-b2 * u12 * a1_2 + b1 * a3_2 * u12 - a3_2 * b3 + b1 * v22 - b2 * u22) * aa1 + (a1_2 * b3 + v12 * (-b2 * a1_2 + b1 * a3_2)) * aa2))),
					k9 * b1, k9 * b2, k9 * b3, 0, 0, 0;
			}
		};

		template<size_t _UsingAffineFrames,
			typename _ModelEstimator>
			class FundamentalMatrixUncertaintyBasedPreemption
		{
		protected:
			Eigen::Matrix<double, 16, 16> covariance;

			// Compute the derivatives
			Eigen::Matrix<double, 8, 24> A = Eigen::Matrix<double, 8, 24>::Zero();
			Eigen::Matrix<double, 8, 8> B = Eigen::Matrix<double, 8, 8>::Zero();

			double trace_threshold,
				trace_of_last_run;

		public:
			FundamentalMatrixUncertaintyBasedPreemption(const double trace_threshold_) :
				trace_threshold(trace_threshold_)
			{
			}

			void initialize(const cv::Mat &points_) {}

			const double &getTraceOfLastRun()
			{
				return trace_of_last_run;
			}

			bool verifyModel(
				const gcransac::Model &model_,
				const _ModelEstimator &estimator_, // The model estimator
				const double &threshold_,
				const size_t &iteration_number_,
				const Score &best_score_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t sample_number_,
				std::vector<size_t> &inliers_,
				Score &score_)
			{
				double trace = 0.0;

				// Normalize points
				constexpr double scale = 1.0 / 1000.0;
				static const Eigen::Matrix3d normalizing_transform = scale * Eigen::Matrix3d::Identity();
				static const Eigen::Matrix3d normalizing_transform_inverse = normalizing_transform.inverse();
				cv::Mat normalized_points(sample_number_, points_.cols, points_.type());
				size_t *sample = new size_t[sample_number_];
				for (size_t point_idx = 0; point_idx < sample_number_; ++point_idx)
				{
					points_.row(minimal_sample_[point_idx]).copyTo(normalized_points.row(point_idx));
					normalized_points.at<double>(point_idx, 0) *= scale;
					normalized_points.at<double>(point_idx, 1) *= scale;
					normalized_points.at<double>(point_idx, 2) *= scale;
					normalized_points.at<double>(point_idx, 3) *= scale;
					sample[point_idx] = point_idx;
				}

				gcransac::Model normalized_model;
				normalized_model.descriptor =
					normalizing_transform_inverse.transpose() *
					model_.descriptor *
					normalizing_transform_inverse;

				if constexpr (_UsingAffineFrames)
					verifyAffineModel(
						normalized_model,
						normalized_points,
						sample,
						sample_number_,
						trace);
				else
					verifyPointModel(
						normalized_model,
						normalized_points,
						sample,
						sample_number_,
						trace);

				delete[] sample;
				trace_of_last_run = trace;
				return trace < trace_threshold;
			}

		protected:
			OLGA_INLINE bool verifyAffineModel(
				gcransac::Model model_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t sample_number_,
				double &trace_)
			{
				Eigen::Matrix3d E =
					model_.descriptor.block<3, 3>(0, 0).normalized();

				const double *points_ptr =
					reinterpret_cast<double *>(points_.data);

				const size_t cols = points_.cols;
				const size_t &idx1 = minimal_sample_[0] * cols,
					&idx2 = minimal_sample_[1] * cols,
					&idx3 = minimal_sample_[2] * cols;

				const double &x11 = points_ptr[idx1],
					&y11 = points_ptr[idx1 + 1],
					&x12 = points_ptr[idx1 + 2],
					&y12 = points_ptr[idx1 + 3],
					&a111 = points_ptr[idx1 + 4],
					&a112 = points_ptr[idx1 + 5],
					&a121 = points_ptr[idx1 + 6],
					&a122 = points_ptr[idx1 + 7],
					&x21 = points_ptr[idx2],
					&y21 = points_ptr[idx2 + 1],
					&x22 = points_ptr[idx2 + 2],
					&y22 = points_ptr[idx2 + 3],
					&a211 = points_ptr[idx2 + 4],
					&a212 = points_ptr[idx2 + 5],
					&a221 = points_ptr[idx2 + 6],
					&a222 = points_ptr[idx2 + 7],
					&x31 = points_ptr[idx3],
					&y31 = points_ptr[idx3 + 1],
					&x32 = points_ptr[idx3 + 2],
					&y32 = points_ptr[idx3 + 3];

				// Compute the derivatives
				Eigen::Matrix<double, 9, 24> A;
				Eigen::Matrix<double, 9, 9> B;

				derivMeasurementsAffine(x11, y11, x21, y21, x31, y31,
					x12, y12, x22, y22, x32, y32,
					a111, a112, a121, a122,
					a211, a212, a221, a222,
					E(0, 0), E(1, 0), E(2, 0), E(0, 1), E(1, 1), E(2, 1), E(0, 2), E(1, 2),
					A);

				derivParamsAffine(x11, y11, x21, y21, x31, y31,
					x12, y12, x22, y22, x32, y32,
					a111, a112, a121, a122,
					a211, a212, a221, a222,
					E(0, 0), E(1, 0), E(2, 0), E(0, 1), E(1, 1), E(2, 1), E(0, 2), E(1, 2), E(2, 2),
					B);

				Eigen::Matrix<double, 9, 24> x =
					(B.transpose() * B).llt().solve(B.transpose() * A);

				trace_ = 1e-14 * (x.row(0).dot(x.row(0)) +
					x.row(1).dot(x.row(1)) +
					x.row(2).dot(x.row(2)) +
					x.row(3).dot(x.row(3)) +
					x.row(4).dot(x.row(4)) +
					x.row(5).dot(x.row(5)) +
					x.row(6).dot(x.row(6)) +
					x.row(7).dot(x.row(7)) +
					x.row(8).dot(x.row(8)));

				return true;
			}

			OLGA_INLINE bool verifyPointModel(
				const gcransac::Model &model_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t &sample_number_,
				double &trace_)
			{
				// normalize F
				Eigen::Matrix3d E =
					model_.descriptor.block<3, 3>(0, 0).normalized();

				const double *points_ptr =
					reinterpret_cast<double *>(points_.data);

				const size_t cols = points_.cols;
				const size_t &idx1 = minimal_sample_[0] * cols,
					&idx2 = minimal_sample_[1] * cols,
					&idx3 = minimal_sample_[2] * cols,
					&idx4 = minimal_sample_[3] * cols,
					&idx5 = minimal_sample_[4] * cols,
					&idx6 = minimal_sample_[5] * cols,
					&idx7 = minimal_sample_[6] * cols;

				const double &x11 = points_ptr[idx1],
					&y11 = points_ptr[idx1 + 1],
					&x12 = points_ptr[idx1 + 2],
					&y12 = points_ptr[idx1 + 3],
					&x21 = points_ptr[idx2],
					&y21 = points_ptr[idx2 + 1],
					&x22 = points_ptr[idx2 + 2],
					&y22 = points_ptr[idx2 + 3],
					&x31 = points_ptr[idx3],
					&y31 = points_ptr[idx3 + 1],
					&x32 = points_ptr[idx3 + 2],
					&y32 = points_ptr[idx3 + 3],
					&x41 = points_ptr[idx4],
					&y41 = points_ptr[idx4 + 1],
					&x42 = points_ptr[idx4 + 2],
					&y42 = points_ptr[idx4 + 3],
					&x51 = points_ptr[idx5],
					&y51 = points_ptr[idx5 + 1],
					&x52 = points_ptr[idx5 + 2],
					&y52 = points_ptr[idx5 + 3],
					&x61 = points_ptr[idx6],
					&y61 = points_ptr[idx6 + 1],
					&x62 = points_ptr[idx6 + 2],
					&y62 = points_ptr[idx6 + 3],
					&x71 = points_ptr[idx7],
					&y71 = points_ptr[idx7 + 1],
					&x72 = points_ptr[idx7 + 2],
					&y72 = points_ptr[idx7 + 3];

				// Compute the derivatives
				Eigen::Matrix<double, 9, 28> A;
				Eigen::Matrix<double, 9, 9> B;

				derivMeasurements(x11, y11, x21, y21, x31, y31, x41, y41, x51, y51, x61, y61, x71, y71,
					x12, y12, x22, y22, x32, y32, x42, y42, x52, y52, x62, y62, x72, y72,
					E(0, 0), E(1, 0), E(2, 0), E(0, 1), E(1, 1), E(2, 1), E(0, 2), E(1, 2),
					A);

				derivParams(x11, y11, x21, y21, x31, y31, x41, y41, x51, y51, x61, y61, x71, y71,
					x12, y12, x22, y22, x32, y32, x42, y42, x52, y52, x62, y62, x72, y72,
					E(0, 0), E(1, 0), E(2, 0), E(0, 1), E(1, 1), E(2, 1), E(0, 2), E(1, 2), E(2, 2),
					B);

				Eigen::Matrix<double, 9, 28> x =
					(B.transpose() * B).llt().solve(B.transpose() * A);

				trace_ = 1e-13 * (x.row(0).dot(x.row(0)) +
					x.row(1).dot(x.row(1)) +
					x.row(2).dot(x.row(2)) +
					x.row(3).dot(x.row(3)) +
					x.row(4).dot(x.row(4)) +
					x.row(5).dot(x.row(5)) +
					x.row(6).dot(x.row(6)) +
					x.row(7).dot(x.row(7)) +
					x.row(8).dot(x.row(8)));

				return true;
			}

			OLGA_INLINE void derivMeasurements(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u13,
				const double &v13,
				const double &u14,
				const double &v14,
				const double &u15,
				const double &v15,
				const double &u16,
				const double &v16,
				const double &u17,
				const double &v17,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &u23,
				const double &v23,
				const double &u24,
				const double &v24,
				const double &u25,
				const double &v25,
				const double &u26,
				const double &v26,
				const double &u27,
				const double &v27,
				const double &f11,
				const double &f21,
				const double &f31,
				const double &f12,
				const double &f22,
				const double &f32,
				const double &f13,
				const double &f23,
				Eigen::Matrix<double, 9, 28> &A)
			{
				A <<
					f11 * u21 + f21 * v21 + f31, f12 * u21 + f22 * v21 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u11 + f12 * v11 + f13, f21 * u11 + f22 * v11 + f23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, f11 * u22 + f21 * v22 + f31, f12 * u22 + f22 * v22 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u12 + f12 * v12 + f13, f21 * u12 + f22 * v12 + f23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, f11 * u23 + f21 * v23 + f31, f12 * u23 + f22 * v23 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u13 + f12 * v13 + f13, f21 * u13 + f22 * v13 + f23, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, f11 * u24 + f21 * v24 + f31, f12 * u24 + f22 * v24 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u14 + f12 * v14 + f13, f21 * u14 + f22 * v14 + f23, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, f11 * u25 + f21 * v25 + f31, f12 * u25 + f22 * v25 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u15 + f12 * v15 + f13, f21 * u15 + f22 * v15 + f23, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u26 + f21 * v26 + f31, f12 * u26 + f22 * v26 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u16 + f12 * v16 + f13, f21 * u16 + f22 * v16 + f23, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u27 + f21 * v27 + f31, f12 * u27 + f22 * v27 + f32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, f11 * u17 + f12 * v17 + f13, f21 * u17 + f22 * v17 + f23,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
			}

			OLGA_INLINE void derivParams(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u13,
				const double &v13,
				const double &u14,
				const double &v14,
				const double &u15,
				const double &v15,
				const double &u16,
				const double &v16,
				const double &u17,
				const double &v17,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &u23,
				const double &v23,
				const double &u24,
				const double &v24,
				const double &u25,
				const double &v25,
				const double &u26,
				const double &v26,
				const double &u27,
				const double &v27,
				const double &f11,
				const double &f21,
				const double &f31,
				const double &f12,
				const double &f22,
				const double &f32,
				const double &f13,
				const double &f23,
				const double &f33,
				Eigen::Matrix<double, 9, 9> &B)
			{
				const double
					f11sqr = f11 * f11,
					f12sqr = f12 * f12,
					f13sqr = f13 * f12,
					f21sqr = f21 * f21,
					f22sqr = f22 * f22,
					f23sqr = f23 * f23,
					f31sqr = f31 * f31,
					f32sqr = f32 * f32,
					f33sqr = f33 * f33;

				const double
					s1 = std::pow(f11sqr + f12sqr + f13sqr + f21sqr + f22sqr + f23sqr + f31sqr + f32sqr + f33sqr, -0.1e1 / 0.2e1);

				B <<
					u11 * u21, u11 * v21, u11, u21 * v11, v11 * v21, v11, u21, v21, 1,
					u12 * u22, u12 * v22, u12, u22 * v12, v12 * v22, v12, u22, v22, 1,
					u13 * u23, u13 * v23, u13, u23 * v13, v13 * v23, v13, u23, v23, 1,
					u14 * u24, u14 * v24, u14, u24 * v14, v14 * v24, v14, u24, v24, 1,
					u15 * u25, u15 * v25, u15, u25 * v15, v15 * v25, v15, u25, v25, 1,
					u16 * u26, u16 * v26, u16, u26 * v16, v16 * v26, v16, u26, v26, 1,
					u17 * u27, u17 * v27, u17, u27 * v17, v17 * v27, v17, u27, v27, 1,
					f22 * f33 - f23 * f32, -f12 * f33 + f13 * f32, f12 * f23 - f13 * f22, -f21 * f33 + f23 * f31, f11 * f33 - f13 * f31, -f11 * f23 + f13 * f21, f21 * f32 - f22 * f31, -f11 * f32 + f12 * f31, f11 * f22 - f12 * f21,
					s1 * f11, s1 * f21, s1 * f31, s1 * f12, s1 * f22, s1 * f32, s1 * f13, s1 * f23, s1 * f33;
			}

			OLGA_INLINE void derivMeasurementsAffine(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u13,
				const double &v13,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &u23,
				const double &v23,
				const double &a1_1,
				const double &a2_1,
				const double &a3_1,
				const double &a4_1,
				const double &a1_2,
				const double &a2_2,
				const double &a3_2,
				const double &a4_2,
				const double &f1,
				const double &f4,
				const double &f7,
				const double &f2,
				const double &f5,
				const double &f8,
				const double &f3,
				const double &f6,
				Eigen::Matrix<double, 9, 24> &A)
			{
				A <<
					f1 * u21 + f4 * v21 + f7, f2 * u21 + f5 * v21 + f8, 0, 0, 0, 0, u11 * f1 + v11 * f2 + f3, u11 * f4 + v11 * f5 + f6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, f1 * u22 + f4 * v22 + f7, f2 * u22 + f5 * v22 + f8, 0, 0, 0, 0, u12 * f1 + v12 * f2 + f3, u12 * f4 + v12 * f5 + f6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, f1 * u23 + f4 * v23 + f7, f2 * u23 + f5 * v23 + f8, 0, 0, 0, 0, f1 * u13 + f2 * v13 + f3, f4 * u13 + f5 * v13 + f6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					a1_1 * f1 + a3_1 * f4, a1_1 * f2 + a3_1 * f5, 0, 0, 0, 0, f1, f4, 0, 0, 0, 0, u11 * f1 + v11 * f2 + f3, 0, u11 * f4 + v11 * f5 + f6, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					a2_1 * f1 + a4_1 * f4, a2_1 * f2 + a4_1 * f5, 0, 0, 0, 0, f2, f5, 0, 0, 0, 0, 0, u11 * f1 + v11 * f2 + f3, 0, u11 * f4 + v11 * f5 + f6, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, a1_2 * f1 + a3_2 * f4, a1_2 * f2 + a3_2 * f5, 0, 0, 0, 0, f1, f4, 0, 0, 0, 0, 0, 0, u12 * f1 + v12 * f2 + f3, 0, u12 * f4 + v12 * f5 + f6, 0, 0, 0, 0, 0,
					0, 0, a2_2 * f1 + a4_2 * f4, a2_2 * f2 + a4_2 * f5, 0, 0, 0, 0, f2, f5, 0, 0, 0, 0, 0, 0, 0, u12 * f1 + v12 * f2 + f3, 0, u12 * f4 + v12 * f5 + f6, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
			}

			OLGA_INLINE void derivParamsAffine(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u13,
				const double &v13,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &u23,
				const double &v23,
				const double &a1_1,
				const double &a2_1,
				const double &a3_1,
				const double &a4_1,
				const double &a1_2,
				const double &a2_2,
				const double &a3_2,
				const double &a4_2,
				const double &f1,
				const double &f4,
				const double &f7,
				const double &f2,
				const double &f5,
				const double &f8,
				const double &f3,
				const double &f6,
				const double &f9,
				Eigen::Matrix<double, 9, 9> &B)
			{
				const double f1sqr = f1 * f1,
					f2sqr = f2 * f2,
					f3sqr = f3 * f3,
					f4sqr = f4 * f4,
					f5sqr = f5 * f5,
					f6sqr = f6 * f6,
					f7sqr = f7 * f7,
					f8sqr = f8 * f8,
					f9sqr = f9 * f9;

				const double
					s1 = std::pow(f1sqr + f2sqr + f3sqr + f4sqr + f5sqr + f6sqr + f7sqr + f8sqr + f9sqr, -0.1e1 / 0.2e1);

				B <<
					u11 * u21, u11 * v21, u11, u21 * v11, v11 * v21, v11, u21, v21, 1,
					u12 * u22, u12 * v22, u12, u22 * v12, v12 * v22, v12, u22, v22, 1,
					u13 * u23, u13 * v23, u13, u23 * v13, v13 * v23, v13, u23, v23, 1,
					a1_1 * u11 + u21, a3_1 * u11 + v21, 1, a1_1 * v11, a3_1 * v11, 0, a1_1, a3_1, 0,
					a2_1 * u11, a4_1 * u11, 0, a2_1 * v11 + u21, a4_1 * v11 + v21, 1, a2_1, a4_1, 0,
					a1_2 * u12 + u22, a3_2 * u12 + v22, 1, a1_2 * v12, a3_2 * v12, 0, a1_2, a3_2, 0,
					a2_2 * u12, a4_2 * u12, 0, a2_2 * v12 + u22, a4_2 * v12 + v22, 1, a2_2, a4_2, 0,
					f5 * f9 - f6 * f8, -f2 * f9 + f3 * f8, f2 * f6 - f3 * f5, -f4 * f9 + f6 * f7, f1 * f9 - f3 * f7, -f1 * f6 + f3 * f4, f4 * f8 - f5 * f7, -f1 * f8 + f2 * f7, f1 * f5 - f2 * f4,
					s1 * f1, s1 * f4, s1 * f7, s1 * f2, s1 * f5, s1 * f8, s1 * f3, s1 * f6, s1 * f9;
			}
		};

		template<size_t _UsingAffineFrames,
			typename _ModelEstimator>
			class HomographyUncertaintyBasedPreemption
		{
		protected:
			Eigen::Matrix<double, 16, 16> covariance;

			// Compute the derivatives
			Eigen::Matrix<double, 9, 16> A = Eigen::Matrix<double, 9, 16>::Zero();
			Eigen::Matrix<double, 9, 9> B = Eigen::Matrix<double, 9, 9>::Zero();
			size_t sample_number;

			double trace_threshold,
				trace_of_last_run;

		public:
			HomographyUncertaintyBasedPreemption(const double trace_threshold_) :
				trace_threshold(trace_threshold_),
				sample_number(0),
				A(Eigen::Matrix<double, 9, 16>::Zero()),
				B(Eigen::Matrix<double, 9, 9>::Zero())
			{
				B(0, 6) = 1;
				B(1, 7) = 1;
				B(2, 6) = 1;
				B(3, 7) = 1;
				B(4, 6) = 1;
				B(5, 7) = 1;
				B(6, 6) = 1;
				B(7, 7) = 1;
			}

			void initialize(const cv::Mat &points_) {}

			static constexpr bool isProvidingScore() { return false; }
			static constexpr const char *getName() { return "uncertainty-based filtering"; }

			const double &getTraceOfLastRun()
			{
				return trace_of_last_run;
			}

			bool verifyModel(
				const gcransac::Model &model_,
				const _ModelEstimator &estimator_, // The model estimator
				const double &threshold_,
				const size_t &iteration_number_,
				const Score &best_score_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t sample_number_,
				std::vector<size_t> &inliers_,
				Score &score_)
			{
				double trace;

				if constexpr (_UsingAffineFrames)
					verifyAffineModel(
						model_,
						points_,
						minimal_sample_,
						sample_number_,
						trace);
				else
					verifyPointModel(
						model_,
						points_,
						minimal_sample_,
						sample_number_,
						trace);

				trace_of_last_run = trace;
				return trace < trace_threshold;
			}

		protected:

			OLGA_INLINE bool verifyAffineModel(
				gcransac::Model model_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t sample_number_,
				double &trace_)
			{
				const Eigen::Matrix3d H =
					model_.descriptor.block<3, 3>(0, 0).normalized();

				const double *points_ptr =
					reinterpret_cast<double *>(points_.data);

				const size_t cols = points_.cols;
				const size_t &idx1 = minimal_sample_[0] * cols,
					&idx2 = minimal_sample_[1] * cols;

				const double &x11 = points_ptr[idx1],
					&y11 = points_ptr[idx1 + 1],
					&x12 = points_ptr[idx1 + 2],
					&y12 = points_ptr[idx1 + 3],
					&a111 = points_ptr[idx1 + 4],
					&a112 = points_ptr[idx1 + 5],
					&a121 = points_ptr[idx1 + 6],
					&a122 = points_ptr[idx1 + 7],
					&x21 = points_ptr[idx2],
					&y21 = points_ptr[idx2 + 1],
					&x22 = points_ptr[idx2 + 2],
					&y22 = points_ptr[idx2 + 3],
					&a211 = points_ptr[idx2 + 4],
					&a212 = points_ptr[idx2 + 5],
					&a221 = points_ptr[idx2 + 6],
					&a222 = points_ptr[idx2 + 7];

				// Compute the derivatives
				Eigen::Matrix<double, 9, 16> A;
				Eigen::Matrix<double, 9, 9> B;

				deriv_measurements_affine(x11, y11, x21, y21,
					x12, y12, x22, y22,
					a111, a112, a121, a122,
					a211, a212, a221, a222,
					H(0, 0), H(1, 0), H(2, 0), H(0, 1), H(1, 1), H(2, 1), H(0, 2), H(1, 2), H(2, 2),
					A);

				deriv_params_affine(x11, y11, x21, y21,
					x12, y12, x22, y22,
					a111, a112, a121, a122,
					a211, a212, a221, a222,
					H(0, 0), H(1, 0), H(2, 0), H(0, 1), H(1, 1), H(2, 1), H(0, 2), H(1, 2), H(2, 2),
					B);

				Eigen::Matrix<double, 9, 16> x =
					(B.transpose() * B).llt().solve(B.transpose() * A);

				trace_ = 1e-8 * (x.row(0).dot(x.row(0)) +
					x.row(1).dot(x.row(1)) +
					x.row(2).dot(x.row(2)) +
					x.row(3).dot(x.row(3)) +
					x.row(4).dot(x.row(4)) +
					x.row(5).dot(x.row(5)) +
					x.row(6).dot(x.row(6)) +
					x.row(7).dot(x.row(7)) +
					x.row(8).dot(x.row(8)));

				return true;
			}

			OLGA_INLINE bool verifyPointModel(
				const gcransac::Model &model_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t &sample_number_,
				double &trace_)
			{
				// TODO: check if H has already been divided by H(2,2)
				const Eigen::MatrixXd &H = model_.descriptor;

				const double *points_ptr =
					reinterpret_cast<double *>(points_.data);

				const size_t &cols = points_.cols;
				const size_t &idx1 = minimal_sample_[0] * cols,
					&idx2 = minimal_sample_[1] * cols,
					&idx3 = minimal_sample_[2] * cols,
					&idx4 = minimal_sample_[3] * cols;

				const double &x11 = points_ptr[idx1],
					&y11 = points_ptr[idx1 + 1],
					&x12 = points_ptr[idx1 + 2],
					&y12 = points_ptr[idx1 + 3],
					&x21 = points_ptr[idx2],
					&y21 = points_ptr[idx2 + 1],
					&x22 = points_ptr[idx2 + 2],
					&y22 = points_ptr[idx2 + 3],
					&x31 = points_ptr[idx3],
					&y31 = points_ptr[idx3 + 1],
					&x32 = points_ptr[idx3 + 2],
					&y32 = points_ptr[idx3 + 3],
					&x41 = points_ptr[idx4],
					&y41 = points_ptr[idx4 + 1],
					&x42 = points_ptr[idx4 + 2],
					&y42 = points_ptr[idx4 + 3];

				deriv_measurements(x11, y11, x21, y21, x31, y31, x41, y41,
					x12, y12, x22, y22, x32, y32, x42, y42,
					H(0, 0), H(1, 0), H(2, 0), H(0, 1), H(1, 1), H(2, 1), H(0, 2), H(1, 2), H(2, 2),
					A);

				deriv_params(x11, y11, x21, y21, x31, y31, x41, y41,
					x12, y12, x22, y22, x32, y32, x42, y42,
					H(0, 0), H(1, 0), H(2, 0), H(0, 1), H(1, 1), H(2, 1), H(0, 2), H(1, 2), H(2, 2),
					B);

				Eigen::Matrix<double, 9, 16> x =
					(B.transpose() * B).llt().solve(B.transpose() * A);

				trace_ = 0.0000000001 * (x.row(0).dot(x.row(0)) +
					x.row(1).dot(x.row(1)) +
					x.row(2).dot(x.row(2)) +
					x.row(3).dot(x.row(3)) +
					x.row(4).dot(x.row(4)) +
					x.row(5).dot(x.row(5)) +
					x.row(6).dot(x.row(6)) +
					x.row(7).dot(x.row(7)));
				return true;
			}

			OLGA_INLINE void deriv_measurements_affine(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &a11_1,
				const double &a12_1,
				const double &a21_1,
				const double &a22_1,
				const double &a11_2,
				const double &a12_2,
				const double &a21_2,
				const double &a22_2,
				const double &h11,
				const double &h21,
				const double &h31,
				const double &h12,
				const double &h22,
				const double &h32,
				const double &h13,
				const double &h23,
				const double &h33,
				Eigen::Matrix<double, 9, 16> &A)
			{
				A << -h31 * u21 + h11, -h32 * u21 + h12, 0, 0, -u11 * h31 - v11 * h32 - h33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					-h31 * v21 + h21, -h32 * v21 + h22, 0, 0, 0, -u11 * h31 - v11 * h32 - h33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, -h31 * u22 + h11, -h32 * u22 + h12, 0, 0, -u12 * h31 - v12 * h32 - h33, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, -h31 * v22 + h21, -h32 * v22 + h22, 0, 0, 0, -u12 * h31 - v12 * h32 - h33, 0, 0, 0, 0, 0, 0, 0, 0,
					-a11_1 * h31, -a11_1 * h32, 0, 0, -h31, 0, 0, 0, -u11 * h31 - v11 * h32 - h33, 0, 0, 0, 0, 0, 0, 0,
					-a12_1 * h31, -a12_1 * h32, 0, 0, -h32, 0, 0, 0, 0, -u11 * h31 - v11 * h32 - h33, 0, 0, 0, 0, 0, 0,
					0, 0, -a11_2 * h31, -a11_2 * h32, 0, 0, -h31, 0, 0, 0, 0, 0, -u12 * h31 - v12 * h32 - h33, 0, 0, 0,
					0, 0, -a12_2 * h31, -a12_2 * h32, 0, 0, -h32, 0, 0, 0, 0, 0, 0, -u12 * h31 - v12 * h32 - h33, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
			}


			OLGA_INLINE void deriv_params_affine(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &a11_1,
				const double &a12_1,
				const double &a21_1,
				const double &a22_1,
				const double &a11_2,
				const double &a12_2,
				const double &a21_2,
				const double &a22_2,
				const double &h11,
				const double &h21,
				const double &h31,
				const double &h12,
				const double &h22,
				const double &h32,
				const double &h13,
				const double &h23,
				const double &h33,
				Eigen::Matrix<double, 9, 9> &B)
			{
				const double s =
					std::pow(h11 * h11 + h12 * h12 + h13 * h13 + h21 * h21 + h22 * h22 + h23 * h23 + h31 * h31 + h32 * h32 + h33 * h33, -0.1e1 / 0.2e1);

				B <<
					u11, 0, -u11 * u21, v11, 0, -u21 * v11, 1, 0, -u21,
					0, u11, -u11 * v21, 0, v11, -v11 * v21, 0, 1, -v21,
					u12, 0, -u12 * u22, v12, 0, -u22 * v12, 1, 0, -u22,
					0, u12, -u12 * v22, 0, v12, -v12 * v22, 0, 1, -v22,
					1, 0, -a11_1 * u11 - u21, 0, 0, -a11_1 * v11, 0, 0, -a11_1,
					0, 0, -a12_1 * u11, 1, 0, -a12_1 * v11 - u21, 0, 0, -a12_1,
					1, 0, -a11_2 * u12 - u22, 0, 0, -a11_2 * v12, 0, 0, -a11_2,
					0, 0, -a12_2 * u12, 1, 0, -a12_2 * v12 - u22, 0, 0, -a12_2,
					s * h11, s * h21, s * h31, s * h12, s * h22, s * h32, s * h13, s * h23, s * h33;
			}

			OLGA_INLINE void deriv_measurements(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u13,
				const double &v13,
				const double &u14,
				const double &v14,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &u23,
				const double &v23,
				const double &u24,
				const double &v24,
				const double &h11,
				const double &h21,
				const double &h31,
				const double &h12,
				const double &h22,
				const double &h32,
				const double &h13,
				const double &h23,
				const double &h33,
				Eigen::Matrix<double, 9, 16> &A)
			{
				const double
					h32v11 = h32 * v11,
					h32v12 = h32 * v12,
					h32v13 = h32 * v13,
					h32v14 = h32 * v14,
					h31u11 = h31 * u11,
					h31u12 = h31 * u12,
					h31u13 = h31 * u13,
					h31u14 = h31 * u14;
				const double
					h31u13_h32v13 = -h31u13 - h32v13 - 1,
					h31u14_h32v14 = -h31u14 - h32v14 - 1,
					h31u12_h32v12 = -h31u12 - h32v12 - 1,
					h31u11_h32v11 = -h31u11 - h32v11 - 1;

				A(0, 0) = -h31 * u21 + h11;
				A(0, 1) = -h32 * u21 + h12;
				A(0, 8) = h31u11_h32v11 - h33;

				A(1, 0) = -h31 * v21 + h21;
				A(1, 1) = -h32 * v21 + h22;
				A(1, 9) = h31u11_h32v11 - h33;

				A(2, 2) = -h31 * u22 + h11;
				A(2, 3) = -h32 * u22 + h12;
				A(2, 10) = h31u12_h32v12 - h33;

				A(3, 2) = -h31 * v22 + h21;
				A(3, 3) = -h32 * v22 + h22;
				A(3, 11) = h31u12_h32v12 - h33;

				A(4, 4) = -h31 * u23 + h11;
				A(4, 5) = -h32 * u23 + h12;
				A(4, 12) = h31u13_h32v13 - h33;

				A(5, 4) = -h31 * v23 + h21;
				A(5, 5) = -h32 * v23 + h22;
				A(5, 13) = h31u13_h32v13 - h33;

				A(6, 6) = -h31 * u24 + h11;
				A(6, 7) = -h32 * u24 + h12;
				A(6, 14) = h31u14_h32v14 - h33;

				A(7, 6) = -h31 * v24 + h21;
				A(7, 7) = -h32 * v24 + h22;
				A(7, 15) = h31u14_h32v14 - h33;

			}

			OLGA_INLINE void deriv_params(
				const double &u11,
				const double &v11,
				const double &u12,
				const double &v12,
				const double &u13,
				const double &v13,
				const double &u14,
				const double &v14,
				const double &u21,
				const double &v21,
				const double &u22,
				const double &v22,
				const double &u23,
				const double &v23,
				const double &u24,
				const double &v24,
				const double &h11,
				const double &h21,
				const double &h31,
				const double &h12,
				const double &h22,
				const double &h32,
				const double &h13,
				const double &h23,
				const double &h33,
				Eigen::Matrix<double, 9, 9> &B)
			{

				const double s =
					std::pow(h11 * h11 + h12 * h12 + h13 * h13 + h21 * h21 + h22 * h22 + h23 * h23 + h31 * h31 + h32 * h32 + h33 * h33, -0.1e1 / 0.2e1);

				B(0, 0) = u11;
				B(0, 2) = -u11 * u21;
				B(0, 3) = v11;
				B(0, 5) = -u21 * v11;
				B(0, 8) = -u21;

				B(1, 1) = u11;
				B(1, 2) = -u11 * v21;
				B(1, 4) = v11;
				B(1, 5) = -v11 * v21;
				B(1, 8) = -v21;

				B(2, 0) = u12;
				B(2, 2) = -u12 * u22;
				B(2, 3) = v12;
				B(2, 5) = -u22 * v12;
				B(2, 8) = -u22;

				B(3, 1) = u12;
				B(3, 2) = -u12 * v22;
				B(3, 4) = v12;
				B(3, 5) = -v12 * v22;
				B(3, 8) = -v22;

				B(4, 0) = u13;
				B(4, 2) = -u13 * u23;
				B(4, 3) = v13;
				B(4, 5) = -u23 * v13;
				B(4, 8) = -u23;

				B(5, 1) = u13;
				B(5, 2) = -u13 * v23;
				B(5, 4) = v13;
				B(5, 5) = -v13 * v23;
				B(5, 8) = -v23;

				B(6, 0) = u14;
				B(6, 2) = -u14 * u24;
				B(6, 3) = v14;
				B(6, 5) = -u24 * v14;
				B(6, 8) = -u24;

				B(7, 1) = u14;
				B(7, 2) = -u14 * v24;
				B(7, 4) = v14;
				B(7, 5) = -v14 * v24;
				B(7, 8) = -v24;

				B(8, 0) = s * h11;
				B(8, 1) = s * h21;
				B(8, 2) = s * h31;
				B(8, 3) = s * h12;
				B(8, 4) = s * h22;
				B(8, 5) = s * h32;
				B(8, 6) = s * h13;
				B(8, 7) = s * h23;
				B(8, 8) = s * h33;
			}
		};

	}
}