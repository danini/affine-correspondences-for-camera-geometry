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

#include <vector>
#include "homography_estimator.h"
#include "solver_homography_two_affinities.h"
#include "solver_fundamental_matrix_three_affine.h"
#include "solver_essential_matrix_two_affine.h"

namespace gcransac
{
	namespace estimator
	{
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class AffinityBasedHomographyEstimator :
			public gcransac::estimator::RobustHomographyEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>
		{
		public:
			using gcransac::estimator::RobustHomographyEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::isValidSample;

			AffinityBasedHomographyEstimator() :
				gcransac::estimator::RobustHomographyEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>()
			{}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			bool isValidSample(
				const cv::Mat& data_, // All data points
				const size_t *sample_) const // The indices of the selected points
			{
				// The size of a minimal sample
				constexpr size_t kSampleSize = sampleSize();
				double *a, *b, *c, *d;

				a = reinterpret_cast<double *>(data_.row(sample_[0]).data);
				b = reinterpret_cast<double *>(data_.row(sample_[1]).data);

				// Transform imagined points by the affine matrix
				c = new double[4];
				d = new double[4];

				Eigen::Vector2d pt11, pt12, pt21, pt22;
				pt11 << 1, 0;
				pt21 << 0, 1;

				Eigen::Matrix2d A;
				A << a[4], a[5],
					a[6], a[7];

				pt12 = A * pt11;
				pt22 = A * pt21;

				c[0] = 1;
				c[1] = 0;
				c[2] = pt12(0);
				c[3] = pt12(1);

				d[0] = 1;
				d[1] = 0;
				d[2] = pt22(0);
				d[3] = pt22(1);

				// Check oriented constraints
				Eigen::Vector3d p, q;

				cross_product(p, a, b, 1);
				cross_product(q, a + 2, b + 2, 1);

				if ((p[0] * c[0] + p[1] * c[1] + p[2])*(q[0] * c[2] + q[1] * c[3] + q[2]) < 0)
					return false;
				if ((p[0] * d[0] + p[1] * d[1] + p[2])*(q[0] * d[2] + q[1] * d[3] + q[2]) < 0)
					return false;

				cross_product(p, c, d, 1);
				cross_product(q, c + 2, d + 2, 1);

				if ((p[0] * a[0] + p[1] * a[1] + p[2])*(q[0] * a[2] + q[1] * a[3] + q[2]) < 0)
					return false;
				if ((p[0] * b[0] + p[1] * b[1] + p[2])*(q[0] * b[2] + q[1] * b[3] + q[2]) < 0)
					return false;

				if (kSampleSize == 2)
				{
					delete[] c;
					delete[] d;
				}

				return true;
			}
		};

		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class AffinityBasedFundamentalMatrixEstimator :
			public gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>
		{
		public:
			AffinityBasedFundamentalMatrixEstimator(
				const double minimum_inlier_ratio_in_validity_check_ = 0.1,
				const bool apply_degensac_ = false,
				const double degensac_homography_threshold_ = 3.0) :
				gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>(
					minimum_inlier_ratio_in_validity_check_,
					apply_degensac_,
					degensac_homography_threshold_)
			{}
		};

		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class AffinityBasedEssentialMatrixEstimator :
			public gcransac::estimator::EssentialMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>
		{
		public:
			AffinityBasedEssentialMatrixEstimator(
				Eigen::Matrix3d intrinsics_src_, // The intrinsic parameters of the source camera
				Eigen::Matrix3d intrinsics_dst_,  // The intrinsic parameters of the destination camera
				const double minimum_inlier_ratio_in_validity_check_ = 0.5,
				const double point_ratio_for_selecting_from_multiple_models_ = 0.05) :
				gcransac::estimator::EssentialMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>(
					intrinsics_src_,
					intrinsics_dst_,
					minimum_inlier_ratio_in_validity_check_,
					point_ratio_for_selecting_from_multiple_models_)
			{}
		};
	}

	namespace utils
	{
		// The default estimator for homography fitting from affine correspondences
		typedef gcransac::estimator::AffinityBasedHomographyEstimator<gcransac::estimator::solver::HomographyTwoAffinitySolver, // The solver used for fitting a model to a minimal sample
			gcransac::estimator::solver::HomographyFourPointSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultAffinityBasedHomographyEstimator;

		// The default estimator for fundamental matrix fitting from affine correspondences
		typedef gcransac::estimator::AffinityBasedFundamentalMatrixEstimator<gcransac::estimator::solver::FundamentalMatrixThreeAffineSolver, // The solver used for fitting a model to a minimal sample
			gcransac::estimator::solver::FundamentalMatrixEightPointSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultAffinityBasedFundamentalMatrixEstimator;

		// The default estimator for essential matrix fitting from affine correspondences
		typedef gcransac::estimator::AffinityBasedEssentialMatrixEstimator<gcransac::estimator::solver::EssentialMatrixTwoAffineSolver, // The solver used for fitting a model to a minimal sample
			gcransac::estimator::solver::FundamentalMatrixEightPointSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultAffinityBasedEssentialMatrixEstimator;

	}
}