#pragma once

#include <string>

namespace VlFeatExtraction {
  
struct SiftExtractionOptions {
  // Number of threads for feature extraction.
  int num_threads = -1;

  // Whether to use the GPU for feature extraction.
  bool use_gpu = true;

  // Index of the GPU used for feature extraction. For multi-GPU extraction,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum image size, otherwise image will be down-scaled.
  int max_image_size = 3200;

  // Maximum number of features to detect, keeping larger-scale features.
  int max_num_features = 8192;

  // First octave in the pyramid, i.e. -1 upsamples the image by one level.
  int first_octave = -1;

  // Number of octaves.
  int num_octaves = 4;

  // Number of levels per octave.
  int octave_resolution = 3;

  // Peak threshold for detection.
  double peak_threshold = 0.02 / octave_resolution;

  // Edge threshold for detection.
  double edge_threshold = 10.0;

  // Estimate affine shape of SIFT features in the form of oriented ellipses as
  // opposed to original SIFT which estimates oriented disks.
  bool estimate_affine_shape = false;

  // Maximum number of orientations per keypoint if not estimate_affine_shape.
  int max_num_orientations = 2;

  // Fix the orientation to 0 for upright features.
  bool upright = false;

  // Whether to adapt the feature detection depending on the image darkness.
  // Note that this feature is only available in the OpenGL SiftGPU version.
  bool darkness_adaptivity = false;

  // Domain-size pooling parameters. Domain-size pooling computes an average
  // SIFT descriptor across multiple scales around the detected scale. This was
  // proposed in "Domain-Size Pooling in Local Descriptors and Network
  // Architectures", J. Dong and S. Soatto, CVPR 2015. This has been shown to
  // outperform other SIFT variants and learned descriptors in "Comparative
  // Evaluation of Hand-Crafted and Learned Local Features", Schönberger,
  // Hardmeier, Sattler, Pollefeys, CVPR 2016.
  bool domain_size_pooling = false;
  double dsp_min_scale = 1.0 / 6.0;
  double dsp_max_scale = 3.0;
  int dsp_num_scales = 10;

  enum class Normalization {
    // L1-normalizes each descriptor followed by element-wise square rooting.
    // This normalization is usually better than standard L2-normalization.
    // See "Three things everyone should know to improve object retrieval",
    // Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
    L1_ROOT,
    // Each vector is L2-normalized.
    L2,
  };
  Normalization normalization = Normalization::L1_ROOT;

  //bool Check() const; // TODO
};

struct SiftMatchingOptions {
  // Number of threads for feature matching and geometric verification.
  int num_threads = -1;

  // Whether to use the GPU for feature matching.
  bool use_gpu = true;

  // Index of the GPU used for feature matching. For multi-GPU matching,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum distance ratio between first and second best match.
  double max_ratio = 0.8;

  // Maximum distance to best match.
  double max_distance = 0.7;

  // Whether to enable cross checking in matching.
  bool cross_check = true;

  // Maximum number of matches.
  int max_num_matches = 32768;

  // Maximum epipolar error in pixels for geometric verification.
  double max_error = 4.0;

  // Confidence threshold for geometric verification.
  double confidence = 0.999;

  // Minimum/maximum number of RANSAC iterations. Note that this option
  // overrules the min_inlier_ratio option.
  int min_num_trials = 30;
  int max_num_trials = 10000;

  // A priori assumed minimum inlier ratio, which determines the maximum
  // number of iterations.
  double min_inlier_ratio = 0.25;

  // Minimum number of inliers for an image pair to be considered as
  // geometrically verified.
  int min_num_inliers = 15;

  // Whether to attempt to estimate multiple geometric models per image pair.
  bool multiple_models = false;

  // Whether to perform guided matching, if geometric verification succeeds.
  bool guided_matching = false;

  //bool Check() const; // TODO
};

}