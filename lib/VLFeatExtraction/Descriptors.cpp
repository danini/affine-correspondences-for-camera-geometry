#include "Descriptors.hpp"
#include <array>

template <typename T1, typename T2>
T2 TruncateCast(const T1 value) {
  return std::min(
    static_cast<T1>(std::numeric_limits<T2>::max()),
    std::max(static_cast<T1>(std::numeric_limits<T2>::min()), value));
}

Eigen::MatrixXf VlFeatExtraction::L2NormalizeFeatureDescriptors(const Eigen::MatrixXf& descriptors) {
  return descriptors.rowwise().normalized();
}

Eigen::MatrixXf VlFeatExtraction::L1RootNormalizeFeatureDescriptors(const Eigen::MatrixXf& descriptors) {
  Eigen::MatrixXf descriptors_normalized(descriptors.rows(),
    descriptors.cols());
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    const float norm = descriptors.row(r).lpNorm<1>();
    descriptors_normalized.row(r) = descriptors.row(r) / norm;
    descriptors_normalized.row(r) =
      descriptors_normalized.row(r).array().sqrt();
  }
  return descriptors_normalized;
}

VlFeatExtraction::FeatureDescriptors VlFeatExtraction::FeatureDescriptorsToUnsignedByte(const Eigen::MatrixXf& descriptors) {
  FeatureDescriptors descriptors_unsigned_byte(descriptors.rows(),
    descriptors.cols());
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::MatrixXf::Index c = 0; c < descriptors.cols(); ++c) {
      const float scaled_value = std::round(512.0f * descriptors(r, c));
      descriptors_unsigned_byte(r, c) =
        TruncateCast<float, uint8_t>(scaled_value);
    }
  }
  return descriptors_unsigned_byte;
}

VlFeatExtraction::FeatureDescriptors VlFeatExtraction::TransformVLFeatToUBCFeatureDescriptors(const FeatureDescriptors& vlfeat_descriptors) {
  FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(),
    vlfeat_descriptors.cols());
  const std::array<int, 8> q{ { 0, 7, 6, 5, 4, 3, 2, 1 } };
  for (FeatureDescriptors::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k) {
          ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
            vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
        }
      }
    }
  }
  return ubc_descriptors;
}
