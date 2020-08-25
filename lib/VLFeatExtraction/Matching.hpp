#pragma once

#include <Eigen/Eigen>

#include "Options.hpp"
#include "Features.hpp"
#include "Descriptors.hpp"

namespace VlFeatExtraction {

  typedef struct {
    int point2D_idx1;
    int point2D_idx2;
    double score;
  } FeatureMatch;

  typedef std::vector<FeatureMatch> FeatureMatches;

  Eigen::MatrixXi ComputeSiftDistanceMatrix(
    const FeatureKeypoints* keypoints1, const FeatureKeypoints* keypoints2,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    const std::function<bool(float, float, float, float)>& guided_filter);

  size_t FindBestMatchesOneWay(const Eigen::MatrixXi& dists,
    const float max_ratio, const float max_distance,
    std::vector<int>* matches,
	  std::vector<float>* ratios);

  void FindBestMatches(const Eigen::MatrixXi& dists, const float max_ratio,
    const float max_distance, const bool cross_check,
    FeatureMatches* matches);

  void MatchSiftFeaturesCPU(const SiftMatchingOptions& match_options,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    FeatureMatches* matches);

}