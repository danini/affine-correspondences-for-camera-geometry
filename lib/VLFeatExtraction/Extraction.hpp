#pragma once

#include "Options.hpp"
#include "Features.hpp"
#include "Descriptors.hpp"

#include <VLFeat/covdet.h>
#include <VLFeat/sift.h>

#include <chrono>
#include <iostream>

namespace VlFeatExtraction {

  inline
    bool extract(const float* imgFloat, int cols, int rows, const SiftExtractionOptions& options, FeatureKeypoints* keypoints, FeatureDescriptors* descriptors) {

    if (!imgFloat)
      return false;

    // create a detector object
    std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
      vl_covdet_new(VL_COVDET_METHOD_DOG), // NOTE: https://knowyourmeme.com/memes/yes-this-is-dog
      &vl_covdet_delete);

    if (!covdet) {
      return false;
    }

    // set various parameters (optional)
    vl_covdet_set_first_octave(covdet.get(), options.first_octave);
    vl_covdet_set_octave_resolution(covdet.get(), options.octave_resolution);
    vl_covdet_set_peak_threshold(covdet.get(), options.peak_threshold);
    vl_covdet_set_edge_threshold(covdet.get(), options.edge_threshold);

    // process the image and run the detector
    //vl_covdet_put_image(covdet.get(), imgFloat.ptr<float>(), imgFloat.cols, imgFloat.rows);
    vl_covdet_put_image(covdet.get(), imgFloat, cols, rows);

    vl_covdet_detect(covdet.get(), options.max_num_features);

    if (!options.upright) {
      if (options.estimate_affine_shape) {
        vl_covdet_extract_affine_shape(covdet.get());
        vl_covdet_extract_orientations(covdet.get()); // NOTE: IVAN MOD
      }
      else {
        vl_covdet_extract_orientations(covdet.get());
      }
    }

    const int num_features = vl_covdet_get_num_features(covdet.get());
    VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

    // Sort features according to detected octave and scale.
    std::sort(
      features, features + num_features,
      [](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
        if (feature1.o == feature2.o) {
          return feature1.s > feature2.s;
        }
        else {
          return feature1.o > feature2.o;
        }
      });

    const size_t max_num_features = static_cast<size_t>(options.max_num_features);

    const int kMaxOctaveResolution = 1000;

    keypoints->reserve(std::min<size_t>(num_features, max_num_features));

    // Copy detected keypoints and clamp when maximum number of features reached.
    int prev_octave_scale_idx = std::numeric_limits<int>::max();
    for (int i = 0; i < num_features; ++i) {
      FeatureKeypoint keypoint;
      keypoint.x = features[i].frame.x + 0.5;
      keypoint.y = features[i].frame.y + 0.5;
      keypoint.a11 = features[i].frame.a11;
      keypoint.a12 = features[i].frame.a12;
      keypoint.a21 = features[i].frame.a21;
      keypoint.a22 = features[i].frame.a22;
      keypoints->push_back(keypoint);

      const int octave_scale_idx =
        features[i].o * kMaxOctaveResolution + features[i].s;
      //assert(octave_scale_idx < prev_octave_scale_idx);

      if (octave_scale_idx != prev_octave_scale_idx &&
        keypoints->size() >= max_num_features) {
        break;
      }

      prev_octave_scale_idx = octave_scale_idx;
    }

    // Compute the descriptors for the detected keypoints->
    if (descriptors != nullptr) {
      descriptors->resize(keypoints->size(), 128);

      const size_t kPatchResolution = 15;
      const size_t kPatchSide = 2 * kPatchResolution + 1;
      const double kPatchRelativeExtent = 7.5;
      const double kPatchRelativeSmoothing = 1;
      const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
      const double kSigma =
        kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

      std::vector<float> patch(kPatchSide * kPatchSide);
      std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

      double dsp_min_scale = 1;
      double dsp_scale_step = 0;
      int dsp_num_scales = 1;
      if (options.domain_size_pooling) {
        dsp_min_scale = options.dsp_min_scale;
        dsp_scale_step = (options.dsp_max_scale - options.dsp_min_scale) /
          options.dsp_num_scales;
        dsp_num_scales = options.dsp_num_scales;
      }

      Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor>
        scaled_descriptors(dsp_num_scales, 128);

      std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
        vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
      if (!sift) {
        return false;
      }

      vl_sift_set_magnif(sift.get(), 3.0);

      for (size_t i = 0; i < keypoints->size(); ++i) {
        for (int s = 0; s < dsp_num_scales; ++s) {
          const double dsp_scale = dsp_min_scale + s * dsp_scale_step;

          VlFrameOrientedEllipse scaled_frame = features[i].frame;
          scaled_frame.a11 *= dsp_scale;
          scaled_frame.a12 *= dsp_scale;
          scaled_frame.a21 *= dsp_scale;
          scaled_frame.a22 *= dsp_scale;

          vl_covdet_extract_patch_for_frame(
            covdet.get(), patch.data(), kPatchResolution, kPatchRelativeExtent,
            kPatchRelativeSmoothing, scaled_frame);

          vl_imgradient_polar_f(patchXY.data(), patchXY.data() + 1, 2,
            2 * kPatchSide, patch.data(), kPatchSide,
            kPatchSide, kPatchSide);

          vl_sift_calc_raw_descriptor(sift.get(), patchXY.data(),
            scaled_descriptors.row(s).data(),
            kPatchSide, kPatchSide, kPatchResolution,
            kPatchResolution, kSigma, 0);
        }

        Eigen::Matrix<float, 1, 128> descriptor;
        if (options.domain_size_pooling) {
          descriptor = scaled_descriptors.colwise().mean();
        }
        else {
          descriptor = scaled_descriptors;
        }

        if (options.normalization == SiftExtractionOptions::Normalization::L2) {
          descriptor = L2NormalizeFeatureDescriptors(descriptor);
        }
        else if (options.normalization ==
          SiftExtractionOptions::Normalization::L1_ROOT) {
          descriptor = L1RootNormalizeFeatureDescriptors(descriptor);
        }
        else {
          //LOG(FATAL) << "Normalization type not supported";
          std::cerr << "Normalization type not supported";
          return false;
        }

        descriptors->row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
      }

      *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);

    }

    return true;
  }


}