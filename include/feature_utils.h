#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include <VlFeatExtraction/Features.cpp>
#include <VlFeatExtraction/Extraction.cpp>
#include <VlFeatExtraction/Matching.cpp>
#include <VlFeatExtraction/Descriptors.cpp>

#include <VLFeat/covdet.h>
#include <VLFeat/sift.h>

using namespace VlFeatExtraction;

namespace ACExtraction
{
	bool extractFeatures(
		const float* imgFloat,
		int cols,
		int rows,
		const SiftExtractionOptions& options,
		FeatureKeypoints* keypoints,
		FeatureDescriptors* descriptors)
	{
		if (!imgFloat)
			return false;

		// create a detector object
		std::unique_ptr<VlCovDet, void(*)(VlCovDet*)> covdet(
			vl_covdet_new(VL_COVDET_METHOD_DOG),
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
			assert(octave_scale_idx < prev_octave_scale_idx);

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

			std::unique_ptr<VlSiftFilt, void(*)(VlSiftFilt*)> sift(
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
					std::cerr << "Normalization type not supported";
					return false;
				}

				descriptors->row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
			}

			*descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);

		}

		return true;
	}

	bool getAffineCorrespondences(
		const std::string &image_1_,
		const std::string &image_2_,
		const std::string &correspondence_path_,
		cv::Mat &matches_,
		double &extraction_time_,
		const bool order_matches_ = true,
		const bool display_matches_ = false)
	{
		extraction_time_ = 0;
		std::ifstream infile(correspondence_path_);

		if (infile.is_open())
		{
			double x1, y1, x2, y2, a11, a12, a21, a22;
			matches_.create(0, 8, CV_64F);
			cv::Mat row(1, 8, CV_64F);

			while (infile >> x1 >> y1 >> x2 >> y2 >>
				a11 >> a12 >> a21 >> a22)
			{
				row.at<double>(0) = x1;
				row.at<double>(1) = y1;
				row.at<double>(2) = x2;
				row.at<double>(3) = y2;
				row.at<double>(4) = a11;
				row.at<double>(5) = a12;
				row.at<double>(6) = a21;
				row.at<double>(7) = a22;
				matches_.push_back(row);
			}

			infile.close();
			return true;
		}

		FeatureKeypoints fkp_left, fkp_right;
		FeatureDescriptors dsc_left, dsc_right;

		SiftExtractionOptions options_f;
		SiftMatchingOptions options_m;

		options_f.estimate_affine_shape = true;
		options_f.domain_size_pooling = true;

		// Read Images
		cv::Mat im_left_mat = cv::imread(image_1_, cv::IMREAD_GRAYSCALE);
		cv::Mat im_right_mat = cv::imread(image_2_, cv::IMREAD_GRAYSCALE);

		if (im_left_mat.empty()) {
			std::cerr << "Failed to read image" << std::endl;
			return false;
		}

		if (im_right_mat.empty()) {
			std::cerr << "Failed to read image" << std::endl;
			return false;
		}

		cv::Mat im_left_matF;
		im_left_mat.convertTo(im_left_matF, CV_32F, 1.0 / 255.0);

		cv::Mat im_right_matF;
		im_right_mat.convertTo(im_right_matF, CV_32F, 1.0 / 255.0);

		// Perform extraction
		bool success1 = extract(im_left_matF.ptr<float>(), im_left_matF.cols, im_left_matF.rows, options_f, &fkp_left, &dsc_left);
		bool success2 = extract(im_right_matF.ptr<float>(), im_right_matF.cols, im_right_matF.rows, options_f, &fkp_right, &dsc_right);

		if (success1 && success2) {

			FeatureMatches matches;

			// Perform matching
			auto t_start = std::chrono::high_resolution_clock::now();
			MatchSiftFeaturesCPU(options_m, dsc_left, dsc_right, &matches);
			auto t_end = std::chrono::high_resolution_clock::now();
			extraction_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();

			if (order_matches_)
			{
				sort(matches.begin(), matches.end(),
					[](const auto & a, const auto & b) -> bool
				{
					return a.score > b.score;
				});
			}

			// Save matches
			{
				matches_.create(matches.size(), 8, CV_64F);
				double *matches_data = reinterpret_cast<double *>(matches_.data);

				for (auto& match : matches) {
					const auto&[left_p, right_p] = std::make_pair(match.point2D_idx1, match.point2D_idx2);

					const auto& left = fkp_left[left_p];
					const auto& right = fkp_right[right_p];

					*(matches_data++) = left.x;
					*(matches_data++) = left.y;
					*(matches_data++) = right.x;
					*(matches_data++) = right.y;

					Eigen::Matrix2d A1, A2, A;
					A1 << left.a11, left.a12,
						left.a21, left.a22;
					A2 << right.a11, right.a12,
						right.a21, right.a22;
					A = A2 * A1.inverse();

					*(matches_data++) = A(0, 0);
					*(matches_data++) = A(0, 1);
					*(matches_data++) = A(1, 0);
					*(matches_data++) = A(1, 1);
				}
			}

			// Display matches using OpenCV
			if (display_matches_)
			{
				std::vector<cv::DMatch> matches_to_draw;
				std::vector< cv::KeyPoint > keypoints_Object, keypoints_Scene; // Keypoints

				auto convertToCV = [](std::vector< cv::KeyPoint >& keypointsCV, const FeatureKeypoints& keypoints) {
					keypointsCV.clear();
					keypointsCV.reserve(keypoints.size());
					for (const auto& kp : keypoints) {
						keypointsCV.emplace_back(kp.x, kp.y, kp.ComputeOrientation());
					}
				};

				convertToCV(keypoints_Object, fkp_right);
				convertToCV(keypoints_Scene, fkp_left);

				matches_to_draw.reserve(matches.size());

				// Iterate through the matches from descriptor
				for (unsigned int i = 0; i < matches.size(); i++)
				{
					cv::DMatch v;
					v.trainIdx = matches[i].point2D_idx1;
					v.queryIdx = matches[i].point2D_idx2;
					// This is for all matches
					matches_to_draw.push_back(v);
				}
				auto image = cv::imread(image_1_);
				auto walls = cv::imread(image_2_);

				cv::Mat output = cv::Mat::zeros(std::max(image.rows, walls.rows), image.cols + walls.cols, image.type());
				using namespace std;
				cv::drawMatches(image, keypoints_Object, walls, keypoints_Scene, matches_to_draw, output);// , cv::Scalar::all(-1), cv::Scalar::all(-1), vector<vector<char> >(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

				cv::imshow("OpenCV_view", output);
				cv::waitKey(0);
			}
		}
		else {
			std::cerr << "failed to extract features" << std::endl;
			return false;
		}

		std::ofstream outfile(correspondence_path_);

		if (outfile.is_open())
		{
			double x1, y1, x2, y2, a11, a12, a21, a22;
			double *matches_ptr = reinterpret_cast<double *>(matches_.data);
			for (int r = 0; r < matches_.rows; ++r)
			{
				for (int c = 0; c < matches_.cols; ++c)
					outfile << *(matches_ptr++) << " ";
				outfile << "\n";
			}

			outfile.close();
		}

		return true;
	}
}