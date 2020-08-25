#pragma once
#include <Eigen/Eigen>

namespace VlFeatExtraction {

  typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FeatureDescriptors;

  Eigen::MatrixXf L2NormalizeFeatureDescriptors(
    const Eigen::MatrixXf& descriptors);

  Eigen::MatrixXf L1RootNormalizeFeatureDescriptors(
    const Eigen::MatrixXf& descriptors);

  FeatureDescriptors FeatureDescriptorsToUnsignedByte(
    const Eigen::MatrixXf& descriptors);

  FeatureDescriptors TransformVLFeatToUBCFeatureDescriptors(
    const FeatureDescriptors& vlfeat_descriptors);

}