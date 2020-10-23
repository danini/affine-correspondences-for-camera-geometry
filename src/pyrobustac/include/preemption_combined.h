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
#include <random>
#include <algorithm>
#include <iomanip>

#define LOG_ETA_0 log(0.05)

namespace gcransac
{
	namespace preemption
	{
		template <size_t _UseAffines,
			typename _ModelEstimator,
			typename _UncertaintyCalculator>
			class CombinedPreemptiveVerfication
		{
		protected:
			preemption::SPRTPreemptiveVerfication<_ModelEstimator> *preemption_sprt;
			_UncertaintyCalculator *preemption_uncertainty;
			const double uncertainty_sigma;

		public:
			static constexpr bool isProvidingScore() { return true; }
			static constexpr const char *getName() { return "combined"; }

			~CombinedPreemptiveVerfication() {
				delete preemption_sprt;
				delete preemption_uncertainty;
			}

			CombinedPreemptiveVerfication(
				const double trace_threshold_ = 1e-3,
				const double uncertainty_sigma_ = 1e-3) :
				uncertainty_sigma(uncertainty_sigma_)
			{
				preemption_uncertainty =
					new _UncertaintyCalculator(trace_threshold_);
			}

			void initialize(const cv::Mat &points_,
				const _ModelEstimator &estimator_)
			{
				preemption_sprt = 
					new preemption::SPRTPreemptiveVerfication<_ModelEstimator>(points_, estimator_);
			}

			static constexpr bool providesScore()
			{
				return true;
			}

			bool verifyModel(const gcransac::Model &model_,
				const _ModelEstimator &estimator_, // The model estimator
				const double &threshold_,
				const size_t &iterationNumber_,
				const Score &bestScore_,
				const cv::Mat &points_,
				const size_t *minimalSample_,
				const size_t sampleNumber_,
				std::vector<size_t> &inliers_,
				Score &score_)
			{
				bool validModel = true;
                std::vector<size_t> temp_inliers;
                Score temp_score;
				validModel = preemption_uncertainty->verifyModel(model_,
					estimator_,
					threshold_,
					iterationNumber_,
					bestScore_,
					points_,
					minimalSample_,
					sampleNumber_,
					temp_inliers,
                    temp_score);

				if (!validModel)
					return false;

				const double &kUncertaintyTrace =
					preemption_uncertainty->getTraceOfLastRun();

				constexpr double kLambda = 0.0005;
				const double kUncertaintyProbability =
					exp(-kLambda * kUncertaintyTrace);

				return preemption_sprt->verifyModel(model_,
					estimator_,
					threshold_,
					iterationNumber_,
					bestScore_,
					points_,
					minimalSample_,
					sampleNumber_,
					inliers_,
					score_);
			}

		};
	}
}
