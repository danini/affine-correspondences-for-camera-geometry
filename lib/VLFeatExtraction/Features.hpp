#pragma once

#include <vector>

namespace VlFeatExtraction {

  struct FeatureKeypoint {
    FeatureKeypoint();
    FeatureKeypoint(const float x, const float y);
    FeatureKeypoint(const float x, const float y, const float scale,
      const float orientation);
    FeatureKeypoint(const float x, const float y, const float a11,
      const float a12, const float a21, const float a22);

    static FeatureKeypoint FromParameters(const float x, const float y,
      const float scale_x,
      const float scale_y,
      const float orientation,
      const float shear);

    // Rescale the feature location and shape size by the given scale factor.
    void Rescale(const float scale);
    void Rescale(const float scale_x, const float scale_y);

    // Compute similarity shape parameters from affine shape.
    float ComputeScale() const;
    float ComputeScaleX() const;
    float ComputeScaleY() const;
    float ComputeOrientation() const;
    float ComputeShear() const;

    // Location of the feature, with the origin at the upper left image corner,
    // i.e. the upper left pixel has the coordinate (0.5, 0.5).
    float x;
    float y;

    // Affine shape of the feature.
    float a11;
    float a12;
    float a21;
    float a22;
  };

  typedef std::vector<FeatureKeypoint> FeatureKeypoints;

}