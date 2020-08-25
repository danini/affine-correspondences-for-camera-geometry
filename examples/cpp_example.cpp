#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <mutex>
#include <memory>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include "feature_utils.h"
#include "utils.h"

#include "preemption_sprt.h"
#include "preemption_uncertainty_based.h"
#include "preemption_combined.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "flann_neighborhood_graph.h"
#include "essential_estimator.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "types.h"
#include "model.h"
#include "affine_estimators.h"

// A function to test affine and point-based fundamental matrix estimation
void testFundamentalMatrixFitting(
	const std::string &imageSourcePath_, // The path where the source image is to be found
	const std::string &imageDestinationPath_, // The path where the destination image is to be found
	const std::string &correspondencePath_, // The path where the correspondneces are to be found or to be saved
	const std::string &imageSourceIntrinsicsPath_, // The path where the intrinsic parameters of the source image are to be found
	const std::string &imageDestinationIntrinsicsPath_, // The path where the intrinsic parameters of the destination image are to be found
	const std::string &groundTruthPosePath_, // The path where the ground truth pose is to be found
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold used for robust fitting
	const double &ransacConfidence_, // The requiring confidence in the results
	const double &spatialCoherenceWeight_); // The weight of the spatial coherence term in GC-RANSAC

// A function to test affine and point-based essential matrix estimation
void testEssentialMatrixFitting(
	const std::string &imageSourcePath_, // The path where the source image is to be found
	const std::string &imageDestinationPath_, // The path where the destination image is to be found
	const std::string &correspondencePath_, // The path where the correspondneces are to be found or to be saved
	const std::string &imageSourceIntrinsicsPath_, // The path where the intrinsic parameters of the source image are to be found
	const std::string &imageDestinationIntrinsicsPath_, // The path where the intrinsic parameters of the destination image are to be found
	const std::string &groundTruthPosePath_, // The path where the ground truth pose is to be found
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold used for robust fitting
	const double &ransacConfidence_, // The requiring confidence in the results
	const double &spatialCoherenceWeight_); // The weight of the spatial coherence term in GC-RANSAC

// A function to test affine and point-based homography estimation
void testHomographyFitting(
	const std::string &imageSourcePath_, // The path where the source image is to be found
	const std::string &imageDestinationPath_, // The path where the destination image is to be found
	const std::string &correspondencePath_, // The path where the correspondneces are to be found or to be saved
	const std::string &groundTruthHomographyPath_, // The path where the ground truth homography is to be found
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold used for robust fitting
	const double &ransacConfidence_, // The requiring confidence in the results
	const double &spatialCoherenceWeight_); // The weight of the spatial coherence term in GC-RANSAC

// A function to estimate the fundamental matrix given a set of data points
// and an estimator.
template <typename _Estimator>
void estimateFundamentalMatrix(
	const cv::Mat &matches_, // The correspondences
	const std::string &sourceImagePath_, // The path of the source image
	const std::string &destinationImagePath_, // The path of the destination image
	Eigen::Matrix3d &fundamentalMatrix_, // The estimated fundamental matrix
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold
	const double &ransacConfidence_, // The confidence required in the results
	const double &spatialCoherenceWeight_, // The weight of the spatial coherence term in GC-RANSAC
	const bool &visualizeResults_ = true); // A flag determining if the resulting inliers should be drawn or not 

// A function to estimate the essential matrix given a set of data points
// and an estimator.
template <typename _Estimator>
void estimateEssentialMatrix(
	const cv::Mat &matches_, // The correspondences
	const std::string &sourceImagePath_, // The path of the source image
	const std::string &destinationImagePath_, // The path of the destination image
	const Eigen::Matrix3d &instrinsicsSource_,
	const Eigen::Matrix3d &instrinsicsDestination_,
	Eigen::Matrix3d &essentialMatrix_, // The estimated essential matrix
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold
	const double &ransacConfidence_, // The confidence required in the results
	const double &spatialCoherenceWeight_, // The weight of the spatial coherence term in GC-RANSAC
	const bool &visualizeResults_ = true); // A flag determining if the resulting inliers should be drawn or not 

// A function to estimate the homography given a set of data points
// and an estimator.
template <typename _Estimator>
void estimateHomography(
	const cv::Mat &matches_, // The correspondences
	const std::string &sourceImagePath_, // The path of the source image
	const std::string &destinationImagePath_, // The path of the destination image
	Eigen::Matrix3d &homography_, // The estimated homography matrix
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold
	const double &ransacConfidence_, // The confidence required in the results
	const double &spatialCoherenceWeight_, // The weight of the spatial coherence term in GC-RANSAC
	const bool &visualizeResults_ = true); // A flag determining if the resulting inliers should be drawn or not

void normalizeCorrespondences(const cv::Mat &points_,
	const Eigen::Matrix3d &intrinsics_src_,
	const Eigen::Matrix3d &intrinsics_dst_,
	cv::Mat &normalized_points_);

// Calculating the error of the estimated homography matrices
void calculateHomographyError(
	const cv::Size &sourceImageSize, // The size of the source image
	const cv::Size &destinationImageSize, // The size of the destination image
	const cv::Mat &matches_, // The affine correspondences
	Eigen::Matrix3d &homography_, // The estimated homography matrix
	Eigen::Matrix3d &groundTruthHomography_, // The ground truth homography matrix
	double &error_); // The error of the estimated homography matrix

// Calculating the error of the estimated essential matrices given the ground truth
// relative pose between the source and destination cameras.
void calculatePoseError(
	Eigen::Matrix3d &essentialMatrix_, // The essential mtrix from which the error is to be calculated
	const Eigen::Matrix3d &intrinsicsSource_, // The intrinsic parameters of the source camera
	const Eigen::Matrix3d &intrinsicsDestination_, //The intrinsic parameters of the destination camera
	Eigen::Matrix<double, 3, 4> &groundTruthRelativePose_, // The ground truth pose
	double &rotation_error_, // The rotation error in degrees 
	double &translation_error_); // The translation error in degrees

double rotationError(
	const cv::Mat &rotation_1_,
	const cv::Mat &rotation_2_);

double translationError(
	const cv::Mat &translation_1_,
	const cv::Mat &translation_2_);

template<typename T, typename LabelType>
void drawMatches(
	const cv::Mat &points_,
	const std::vector<LabelType>& labeling_,
	const cv::Mat& image1_,
	const cv::Mat& image2_,
	cv::Mat& out_image_);

int main(int argc, const char* argv[])
{
	const double 
		// The inlier-outlier threshold used for homography estimation
		kInlierOutlierThresholdHomography = 5.0, 
		// The inlier-outlier threshold used for fundamental matrix estimation
		kInlierOutlierThresholdFundamentalMatrix = 0.75,
		// The inlier-outlier threshold used for essential matrix estimation
		kInlierOutlierThresholdEssentialMatrix = 0.75,
		// The weight of the spatial coherence term used for all problems
		kSpatialCoherenceWeight = 0.975,
		// The required RANSAC confidence used for all problems
		kRansacConfidence = 0.999;

	// Testing homography estimation
	testHomographyFitting(
		"data/graf1.png", // The path of the source image 
		"data/graf2.png", // The path of the destination image 
		"data/graf_1_2.txt", // The path where the correspondences will be saved or loaded from
		"data/homography_graf_1_2.txt", // The path where the ground truth homography is stored
		kInlierOutlierThresholdHomography, // The inlier-outlier threshold used for the robust estimation
		kRansacConfidence, // The required RANSAC confidence in the results
		kSpatialCoherenceWeight); // The spatial coherence weight in GC-RANSAC

	// Testing fundamental matrix estimation
	testFundamentalMatrixFitting(
		"data/00046350_1877865767.jpg", // The path of the source image 
		"data/00325388_2471426448.jpg", // The path of the destination image 
		"data/00046350_1877865767_00325388_2471426448.txt", // The path where the correspondences will be saved or loaded from
		"data/00046350_1877865767.K", // The path of the source camera's intrinsic parameters
		"data/00325388_2471426448.K", // The path of the destination camera's intrinsic parameters
		"data/pose_00046350_1877865767_00325388_2471426448.txt", // The path where the ground truth pose is stored
		kInlierOutlierThresholdFundamentalMatrix, // The inlier-outlier threshold used for the robust estimation
		kRansacConfidence, // The required RANSAC confidence in the results
		kSpatialCoherenceWeight); // The spatial coherence weight in GC-RANSAC

	// Testing essential matrix estimation
	testEssentialMatrixFitting(
		"data/00046350_1877865767.jpg", // The path of the source image 
		"data/00325388_2471426448.jpg", // The path of the destination image 
		"data/00046350_1877865767_00325388_2471426448.txt", // The path where the correspondences will be saved or loaded from
		"data/00046350_1877865767.K", // The path of the source camera's intrinsic parameters
		"data/00325388_2471426448.K", // The path of the destination camera's intrinsic parameters
		"data/pose_00046350_1877865767_00325388_2471426448.txt", // The path where the ground truth pose is stored
		kInlierOutlierThresholdEssentialMatrix, // The inlier-outlier threshold used for the robust estimation
		kRansacConfidence, // The required RANSAC confidence in the results
		kSpatialCoherenceWeight); // The spatial coherence weight in GC-RANSAC

	return 0;
}

// Calculating the error of the estimated essential matrices given the ground truth
// relative pose between the source and destination cameras.
void calculatePoseError(
	Eigen::Matrix3d &essentialMatrix_, // The essential mtrix from which the error is to be calculated
	const Eigen::Matrix3d &intrinsicsSource_, // The intrinsic parameters of the source camera
	const Eigen::Matrix3d &intrinsicsDestination_, //The intrinsic parameters of the destination camera
	Eigen::Matrix<double, 3, 4> &groundTruthRelativePose_, // The ground truth pose
	double &rotation_error_, // The rotation error in degrees 
	double &translation_error_) // The translation error in degrees
{
	// Converting the Eigen matrices to OpenCV Mats.
	cv::Mat cvGroundTruthPose(4, 3, CV_64FC1, groundTruthRelativePose_.data());
	cv::Mat cvEssentialMatrix(3, 3, CV_64FC1, essentialMatrix_.data());

	// The matrices have to be transposed due to the column-wise order in Eigen
	cvEssentialMatrix = cvEssentialMatrix.t();
	cvGroundTruthPose = cvGroundTruthPose.t();

	// Decomposing the essential matrix to rotations and translation
	cv::Mat rotation1, rotation2, translation;
	cv::decomposeEssentialMat(cvEssentialMatrix, rotation1, rotation2, translation);

	// Calculating the error of the rotation matrices
	rotation_error_ =
		MIN(rotationError(cvGroundTruthPose(cv::Rect(0, 0, 3, 3)), rotation1),
			rotationError(cvGroundTruthPose(cv::Rect(0, 0, 3, 3)), rotation2));

	// Calculating the error of the translations
	translation_error_ =
		MIN(translationError(cvGroundTruthPose(cv::Rect(3, 0, 1, 3)), translation),
			translationError(cvGroundTruthPose(cv::Rect(3, 0, 1, 3)), -translation));
}

// Calculating the error of the estimated homography matrices
void calculateHomographyError(
	const cv::Size &sourceImageSize, // The size of the source image
	const cv::Size &destinationImageSize, // The size of the destination image
	const cv::Mat &matches_, // The affine correspondences
	Eigen::Matrix3d &homography_, // The estimated homography matrix
	Eigen::Matrix3d &groundTruthHomography_, // The ground truth homography matrix
	double &error_) // The error of the estimated homography matrix
{
	// Converting the Eigen matrices to OpenCV Mats.
	cv::Mat cvGroundTruthHomography(3, 3, CV_64FC1, groundTruthHomography_.data());
	cv::Mat cvHomography(3, 3, CV_64FC1, homography_.data());

	// Calculating the visibility mask by transforming the source image by the 
	// ground truth homography matrix to the destination image and, then,
	// back by the inverse homography matrix. Finally, the pixels
	// which can be seen from both images are kept. 
	cv::Mat mask1 = cv::Mat::ones(sourceImageSize, CV_64F);
	cv::Mat mask1in2;
	cv::warpPerspective(mask1, mask1in2, cvGroundTruthHomography, destinationImageSize);
	cv::Mat mask1inback;
	cv::warpPerspective(mask1in2, mask1inback, cvGroundTruthHomography.inv(), sourceImageSize);

	for (size_t x = 0; x < sourceImageSize.width; x += 1.0)
		for (size_t y = 0; y < sourceImageSize.height; y += 1.0)
			if (mask1inback.at<double>(y, x) > 0)
				mask1inback.at<double>(y, x) = 1;

	std::vector<cv::Point2d> coords;
	coords.reserve(sourceImageSize.width * sourceImageSize.height);
	for (size_t x = 0; x < sourceImageSize.width; x += 1.0)
		for (size_t y = 0; y < sourceImageSize.height; y += 1.0)
			coords.emplace_back(cv::Point2d(x, y));

	std::vector<cv::Point2d> xy_rep_gt,
		xy_rep_estimated;
	cv::perspectiveTransform(coords, xy_rep_gt, cvGroundTruthHomography);
	cv::perspectiveTransform(coords, xy_rep_estimated, cvHomography);

	double summed_errors = 0.0;
	size_t summed_locations = 0;
	for (size_t point_idx = 0; point_idx < coords.size(); ++point_idx)
	{
		const double &x1 = coords[point_idx].x,
			&y1 = coords[point_idx].y;

		if (mask1inback.at<double>(y1, x1) <= std::numeric_limits<double>::epsilon())
			continue;

		const cv::Point2d diff =
			xy_rep_gt[point_idx] - xy_rep_estimated[point_idx];
		const double error =
			cv::norm(diff);

		summed_errors += error;
		++summed_locations;
	}

	error_ = summed_errors / summed_locations;
}

double rotationError(
	const cv::Mat &rotation_1_,
	const cv::Mat &rotation_2_)
{
	cv::Mat R2R1 =
		rotation_2_ * rotation_1_.t();

	const double cos_angle =
		std::max(std::min(1.0, 0.5 * (cv::trace(R2R1).val[0] - 1.0)), -1.0);
	const double angle =
		180.0 / M_PI * std::acos(cos_angle);
	return angle;
}

double translationError(
	const cv::Mat &translation_1_,
	const cv::Mat &translation_2_)
{
	cv::Mat t1 = translation_1_ / cv::norm(translation_1_);
	cv::Mat t2 = translation_2_ / cv::norm(translation_2_);

	const double cos_angle = t1.dot(t2);
	const double angle = std::acos(std::max(std::min(cos_angle, 1.0), -1.0));
	return angle * 180 / M_PI;
}

// A function to test affine and point-based homography estimation
void testHomographyFitting(
	const std::string &imageSourcePath_, // The path where the source image is to be found
	const std::string &imageDestinationPath_, // The path where the destination image is to be found
	const std::string &correspondencePath_, // The path where the correspondneces are to be found or to be saved
	const std::string &groundTruthHomographyPath_, // The path where the ground truth homography is to be found
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold used for robust fitting
	const double &ransacConfidence_, // The requiring confidence in the results
	const double &spatialCoherenceWeight_) // The weight of the spatial coherence term in GC-RANSAC
{
	// Extract affine correspondences
	cv::Mat matches; // The extracted matches
	double extractionTime; // The feature extraction time

	// Loading the ground truth pose.
	// It is used only to calculate the error of the fundamental matrix
	printf("Loading the ground truth homography matrix from '%s'\n", groundTruthHomographyPath_.c_str());
	Eigen::Matrix<double, 3, 3> groundTruthHomography;
	if (!gcransac::utils::loadMatrix<double, 3, 3>(groundTruthHomographyPath_, groundTruthHomography))
	{
		fprintf(stderr, "The ground truth homography is not loaded correctly.\n");
		return;
	}

	// Load the images to get their size for the error calculation
	cv::Mat image1 = cv::imread(imageSourcePath_);
	cv::Mat image2 = cv::imread(imageDestinationPath_);
	const cv::Size kSourceImageSize = image1.size();
	const cv::Size kDestinationImageSize = image2.size();
	image1.release();
	image2.release();

	// A function obtaining the affine correspondences
	printf("Obtaining affine correspondences for images '%s' and '%s'\n", imageSourcePath_.c_str(), imageDestinationPath_.c_str());
	ACExtraction::getAffineCorrespondences(
		imageSourcePath_, // The path of the source image
		imageDestinationPath_, // The path of the destination image
		correspondencePath_, // The path from where the correspondence will be read or saved to.
		matches, // The extracted matches
		extractionTime); // The feature extraction time

	printf("Number of matches found = %d\n", matches.rows);
	printf("Feature matching time = %f\n", extractionTime);

	// Check if enough correspondences are found.
	// The should be more correspondences than two times the minimal
	// sample size.
	constexpr size_t kMinimumCorrespondenceNumber =
		2 * gcransac::utils::DefaultHomographyEstimator::sampleSize();
	if (matches.rows < kMinimumCorrespondenceNumber)
	{
		fprintf(stderr, "Not enough correspondences are found (%d < %d).\n",
			matches.rows,
			kMinimumCorrespondenceNumber);
		return;
	}

	printf("----------------------------------------------------------------------------\n");
	printf("Estimating the homography using affine correspondences.\n");

	// Estimating fundamental matrix from affine correspondences
	Eigen::Matrix3d homographyAC;
	estimateHomography<gcransac::utils::DefaultAffinityBasedHomographyEstimator>(
		matches, // The affine correspondences
		imageSourcePath_, // The path to the source image
		imageDestinationPath_, // The path to the destination image
		homographyAC, // The estimated fundamental matrix
		inlierOutlierThreshold_, // The maximum inlier-outlier threshold of GC-RANSAC
		spatialCoherenceWeight_, // The drawing threshold used for selecting correspondences to be drawn
		ransacConfidence_); // The confidence

	double errorAC = std::numeric_limits<double>::max();

	calculateHomographyError(
		kSourceImageSize, // Size of the source image
		kDestinationImageSize, // Size of the destination image
		matches, // The affine correspondences
		homographyAC, // The estimated fundamental matrix
		groundTruthHomography, // The ground truth pose
		errorAC); // The rotation error in degrees

	printf("Error = %f px\n", errorAC);
	printf("Press a key to continue...\n");
	cv::waitKey(0);

	printf("----------------------------------------------------------------------------\n");
	printf("Estimating the homography using point correspondences.\n");

	Eigen::Matrix3d homographyPC;
	estimateHomography<gcransac::utils::DefaultHomographyEstimator>(
		matches(cv::Rect(0, 0, 4, matches.rows)), // The affine correspondences
		imageSourcePath_, // The intrinsic camera parameters of the source image
		imageDestinationPath_, // The intrinsic camera parameters of the destination image
		homographyPC, // The estimated fundamental matrix
		inlierOutlierThreshold_, // The maximum inlier-outlier threshold of GC-RANSAC
		spatialCoherenceWeight_, // The drawing threshold used for selecting correspondences to be drawn
		ransacConfidence_); // The confidence

	double errorPC = std::numeric_limits<double>::max();

	calculateHomographyError(
		kSourceImageSize, // Size of the source image
		kDestinationImageSize, // Size of the destination image
		matches, // The affine correspondences
		homographyPC, // The estimated fundamental matrix
		groundTruthHomography, // The ground truth pose
		errorPC); // The rotation error in degrees

	printf("Error = %f px\n", errorPC);
	printf("Press a key to continue...\n");
	cv::waitKey(0);
}

// A function to test affine and point-based fundamental matrix estimation
void testFundamentalMatrixFitting(
	const std::string &imageSourcePath_, // The path where the source image is to be found
	const std::string &imageDestinationPath_, // The path where the destination image is to be found
	const std::string &correspondencePath_, // The path where the correspondneces are to be found or to be saved
	const std::string &imageSourceIntrinsicsPath_, // The path where the intrinsic parameters of the source image are to be found
	const std::string &imageDestinationIntrinsicsPath_, // The path where the intrinsic parameters of the destination image are to be found
	const std::string &groundTruthPosePath_, // The path where the ground truth pose is to be found
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold used for robust fitting
	const double &ransacConfidence_, // The requiring confidence in the results
	const double &spatialCoherenceWeight_) // The weight of the spatial coherence term in GC-RANSAC
{
	// Extract affine correspondences
	cv::Mat matches; // The extracted matches
	double extractionTime; // The feature extraction time
	Eigen::Matrix3d intrinsicsSource, // The intrinsic camere matrix of the source image
		intrinsicsDestination; // The intrinsic camere matrix of the destination image

	// Loading the intrinsic camera matrices.
	// They are used only to calculate the error of the fundamental matrix
	if (!gcransac::utils::loadMatrix<double, 3, 3>(imageSourceIntrinsicsPath_, intrinsicsSource) ||
		!gcransac::utils::loadMatrix<double, 3, 3>(imageDestinationIntrinsicsPath_, intrinsicsDestination))
	{
		fprintf(stderr, "The intrinsic camera matrices are not loaded correctly.\n");
		return;
	}

	// Loading the ground truth pose.
	// It is used only to calculate the error of the fundamental matrix
	Eigen::Matrix<double, 3, 4> groundTruthPose;
	if (!gcransac::utils::loadMatrix<double, 3, 4>(groundTruthPosePath_, groundTruthPose))
	{
		fprintf(stderr, "The ground truth pose is not loaded correctly.\n");
		return;
	}

	// A function obtaining the affine correspondences
	ACExtraction::getAffineCorrespondences(
		imageSourcePath_, // The path of the source image
		imageDestinationPath_, // The path of the destination image
		correspondencePath_, // The path from where the correspondence will be read or saved to.
		matches, // The extracted matches
		extractionTime); // The feature extraction time

	printf("Number of matches found = %d\n", matches.rows);
	printf("Feature matching time = %f\n", extractionTime);

	// Check if enough correspondences are found.
	// The should be more correspondences than two times the minimal
	// sample size.
	constexpr size_t kMinimumCorrespondenceNumber =
		2 * gcransac::utils::DefaultFundamentalMatrixEstimator::sampleSize();
	if (matches.rows < kMinimumCorrespondenceNumber)
	{
		fprintf(stderr, "Not enough correspondences are found (%d < %d).\n",
			matches.rows,
			kMinimumCorrespondenceNumber);
		return;
	}

	printf("----------------------------------------------------------------------------\n");
	printf("Estimating the fundamental matrix using affine correspondences.\n");

	// Estimating fundamental matrix from affine correspondences
	Eigen::Matrix3d fundamentalMatrixAC;
	estimateFundamentalMatrix<gcransac::utils::DefaultAffinityBasedFundamentalMatrixEstimator>(
		matches, // The affine correspondences
		imageSourcePath_, // The path to the source image
		imageDestinationPath_, // The path to the destination image
		fundamentalMatrixAC, // The estimated fundamental matrix
		inlierOutlierThreshold_, // The maximum inlier-outlier threshold of GC-RANSAC
		spatialCoherenceWeight_, // The drawing threshold used for selecting correspondences to be drawn
		ransacConfidence_); // The confidence

	double rotationErrorAC = std::numeric_limits<double>::max(),
		translationErrorAC = std::numeric_limits<double>::max();

	Eigen::Matrix3d essentialMatrixAC =
		intrinsicsDestination.transpose() * fundamentalMatrixAC * intrinsicsSource;

	calculatePoseError(essentialMatrixAC, // The estimated fundamental matrix
		intrinsicsSource, // The intrinsic camera parameters of the source image
		intrinsicsDestination, // The intrinsic camera parameters of the destination image
		groundTruthPose, // The ground truth pose
		rotationErrorAC, // The rotation error in degrees
		translationErrorAC); // The translation error in degrees

	printf("Rotation error = %f\370\n", rotationErrorAC);
	printf("Translation error = %f\370\n", translationErrorAC);
	printf("Press a key to continue...\n");
	cv::waitKey(0);

	printf("----------------------------------------------------------------------------\n");
	printf("Estimating the fundamental matrix using point correspondences.\n");

	Eigen::Matrix3d fundamentalMatrixPC;
	estimateFundamentalMatrix<gcransac::utils::DefaultFundamentalMatrixEstimator>(
		matches, // The affine correspondences
		imageSourcePath_, // The intrinsic camera parameters of the source image
		imageDestinationPath_, // The intrinsic camera parameters of the destination image
		fundamentalMatrixPC, // The estimated fundamental matrix
		inlierOutlierThreshold_, // The maximum inlier-outlier threshold of GC-RANSAC
		spatialCoherenceWeight_, // The drawing threshold used for selecting correspondences to be drawn
		ransacConfidence_); // The confidence

	double rotationErrorPC = std::numeric_limits<double>::max(),
		translationErrorPC = std::numeric_limits<double>::max();

	Eigen::Matrix3d essentialMatrixPC =
		intrinsicsDestination.transpose() * fundamentalMatrixPC * intrinsicsSource;

	calculatePoseError(essentialMatrixPC, // The estimated fundamental matrix
		intrinsicsSource, // The intrinsic camera parameters of the source image
		intrinsicsDestination, // The intrinsic camera parameters of the destination image
		groundTruthPose, // The ground truth pose
		rotationErrorPC, // The rotation error in degrees
		translationErrorPC); // The translation error in degrees

	printf("Rotation error = %f\370\n", rotationErrorPC);
	printf("Translation error = %f\370\n", translationErrorPC);
	printf("Press a key to continue...\n");
	cv::waitKey(0);
}

// A function to test affine and point-based essential matrix estimation
void testEssentialMatrixFitting(
	const std::string &imageSourcePath_, // The path where the source image is to be found
	const std::string &imageDestinationPath_, // The path where the destination image is to be found
	const std::string &correspondencePath_, // The path where the correspondneces are to be found or to be saved
	const std::string &imageSourceIntrinsicsPath_, // The path where the intrinsic parameters of the source image are to be found
	const std::string &imageDestinationIntrinsicsPath_, // The path where the intrinsic parameters of the destination image are to be found
	const std::string &groundTruthPosePath_, // The path where the ground truth pose is to be found
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold used for robust fitting
	const double &ransacConfidence_, // The requiring confidence in the results
	const double &spatialCoherenceWeight_) // The weight of the spatial coherence term in GC-RANSAC
{
	// Extract affine correspondences
	cv::Mat matches; // The extracted matches
	double extractionTime; // The feature extraction time
	Eigen::Matrix3d instrinsicsSource, // The intrinsic camere matrix of the source image
		instrinsicsDestination; // The intrinsic camere matrix of the destination image

	// Loading the intrinsic camera matrices.
	if (!gcransac::utils::loadMatrix<double, 3, 3>(imageSourceIntrinsicsPath_, instrinsicsSource) ||
		!gcransac::utils::loadMatrix<double, 3, 3>(imageDestinationIntrinsicsPath_, instrinsicsDestination))
	{
		fprintf(stderr, "The intrinsic camera matrices are not loaded correctly.\n");
		return;
	}

	// Loading the ground truth pose.
	// It is used only to calculate the error of the essential matrix
	Eigen::Matrix<double, 3, 4> groundTruthPose;
	if (!gcransac::utils::loadMatrix<double, 3, 4>(groundTruthPosePath_, groundTruthPose))
	{
		fprintf(stderr, "The ground truth pose is not loaded correctly.\n");
		return;
	}

	// A function obtaining the affine correspondences
	ACExtraction::getAffineCorrespondences(
		imageSourcePath_, // The path of the source image
		imageDestinationPath_, // The path of the destination image
		correspondencePath_, // The path from where the correspondence will be read or saved to.
		matches, // The extracted matches
		extractionTime); // The feature extraction time

	printf("Number of matches found = %d\n", matches.rows);
	printf("Feature matching time = %f\n", extractionTime);

	// Check if enough correspondences are found.
	// The should be more correspondences than two times the minimal
	// sample size.
	constexpr size_t kMinimumCorrespondenceNumber =
		2 * gcransac::utils::DefaultEssentialMatrixEstimator::sampleSize();
	if (matches.rows < kMinimumCorrespondenceNumber)
	{
		fprintf(stderr, "Not enough correspondences are found (%d < %d).\n",
			matches.rows,
			kMinimumCorrespondenceNumber);
		return;
	}

	printf("----------------------------------------------------------------------------\n");
	printf("Estimating the essential matrix using affine correspondences.\n");

	// Estimating fundamental matrix from affine correspondences
	Eigen::Matrix3d essentialMatrixAC;
	estimateEssentialMatrix<gcransac::utils::DefaultAffinityBasedEssentialMatrixEstimator>(
		matches, // The affine correspondences
		imageSourcePath_, // The path to the source image
		imageDestinationPath_, // The path to the destination image
		instrinsicsSource, // The intrinsic camera parameters of the source image
		instrinsicsDestination, // The intrinsic camera parameters of the destination image
		essentialMatrixAC, // The estimated fundamental matrix
		inlierOutlierThreshold_, // The maximum inlier-outlier threshold of GC-RANSAC
		spatialCoherenceWeight_, // The drawing threshold used for selecting correspondences to be drawn
		ransacConfidence_); // The confidence

	double rotationErrorAC = std::numeric_limits<double>::max(),
		translationErrorAC = std::numeric_limits<double>::max();

	calculatePoseError(essentialMatrixAC, // The estimated fundamental matrix
		instrinsicsSource, // The intrinsic camera parameters of the source image
		instrinsicsDestination, // The intrinsic camera parameters of the destination image
		groundTruthPose, // The ground truth pose
		rotationErrorAC, // The rotation error in degrees
		translationErrorAC); // The translation error in degrees

	printf("Rotation error = %f\370\n", rotationErrorAC);
	printf("Translation error = %f\370\n", translationErrorAC);
	printf("Press a key to continue...\n");
	cv::waitKey(0);

	printf("----------------------------------------------------------------------------\n");
	printf("Estimating the essential matrix using point correspondences.\n");

	Eigen::Matrix3d essentialMatrixPC;
	estimateEssentialMatrix<gcransac::utils::DefaultEssentialMatrixEstimator>(
		matches, // The affine correspondences
		imageSourcePath_, // The intrinsic camera parameters of the source image
		imageDestinationPath_, // The intrinsic camera parameters of the destination image
		instrinsicsSource, // The intrinsic camera parameters of the source image
		instrinsicsDestination, // The intrinsic camera parameters of the destination image
		essentialMatrixPC, // The estimated fundamental matrix
		inlierOutlierThreshold_, // The maximum inlier-outlier threshold of GC-RANSAC
		spatialCoherenceWeight_, // The drawing threshold used for selecting correspondences to be drawn
		ransacConfidence_); // The confidence

	double rotationErrorPC = std::numeric_limits<double>::max(),
		translationErrorPC = std::numeric_limits<double>::max();

	calculatePoseError(essentialMatrixPC, // The estimated fundamental matrix
		instrinsicsSource, // The intrinsic camera parameters of the source image
		instrinsicsDestination, // The intrinsic camera parameters of the destination image
		groundTruthPose, // The ground truth pose
		rotationErrorPC, // The rotation error in degrees
		translationErrorPC); // The translation error in degrees

	printf("Rotation error = %f\370\n", rotationErrorPC);
	printf("Translation error = %f\370\n", translationErrorPC);
	printf("Press a key to continue...\n");
	cv::waitKey(0);
}

template <typename _Estimator>
void estimateFundamentalMatrix(
	const cv::Mat &matches_, // The correspondences
	const std::string &sourceImagePath_, // The path of the source image
	const std::string &destinationImagePath_, // The path of the destination image
	Eigen::Matrix3d &fundamentalMatrix_, // The estimated fundamental matrix
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold
	const double &ransacConfidence_, // The confidence required in the results
	const double &spatialCoherenceWeight_, // The weight of the spatial coherence term in GC-RANSAC
	const bool &visualizeResults_)  // A flag determining if the resulting inliers should be drawn or not 
{
	_Estimator estimator(0.5, false); // The robust homography estimator class containing the function for the fitting and residual calculation
	gcransac::FundamentalMatrix model; // The estimated model

	// The number of points in the datasets
	const size_t pointNumber = matches_.rows; // The number of points in the scene

	if (pointNumber < estimator.sampleSize()) // If there are no points, return
	{
		fprintf(stderr, "There are not enough points for estimation a fundamental matrix (%d < %d)\n", pointNumber, estimator.sampleSize());
		return;
	}

	// Initialize the neighborhood used in Graph-cut RANSAC
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&matches_(cv::Rect(0, 0, 4, pointNumber)), // All data points
		20.0); // The radius of the neighborhood ball for determining the neighborhoods.
	
	// Deciding if affine correspondences are used based on the template parameter
	constexpr size_t usingAffineCorrespondences =
		std::is_same<_Estimator, gcransac::utils::DefaultAffinityBasedFundamentalMatrixEstimator>() ? 1 : 0;

	// Defining the combined (SPRT + uncertainty propagation) preemption type based on 
	// whether affine or point correspondences are used
	typedef gcransac::preemption::CombinedPreemptiveVerfication<usingAffineCorrespondences,
		_Estimator, // The solver used for fitting a model to a non-minimal sample
		gcransac::preemption::FundamentalMatrixUncertaintyBasedPreemption<usingAffineCorrespondences, _Estimator>> // The type used for uncertainty propagation
		CombinedPreemptiveVerification;

	// Initializing the preemptive verification object
	CombinedPreemptiveVerification preemptiveVerification;
	preemptiveVerification.initialize(
		matches_,
		estimator);

	// Initialize the sampler used for selecting minimal samples
	gcransac::sampler::ProsacSampler mainSampler(&matches_, _Estimator::sampleSize());
	gcransac::sampler::UniformSampler localOptimizationSampler(&matches_);

	gcransac::GCRANSAC<_Estimator,
		gcransac::neighborhood::FlannNeighborhoodGraph,
		gcransac::MSACScoringFunction<_Estimator>,
		CombinedPreemptiveVerification> gcransac;
	gcransac.settings.threshold = inlierOutlierThreshold_; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.0; // The weight of the spatial coherence term
	gcransac.settings.confidence = ransacConfidence_; // The required confidence in the results
	gcransac.settings.max_iteration_number = 1e4; // The maximum number of iterations

	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();
	gcransac.run(matches_,
		estimator,
		&mainSampler,
		&localOptimizationSampler,
		&neighborhood,
		model,
		preemptiveVerification);
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsedSeconds = end - start;
	std::time_t endTime = std::chrono::system_clock::to_time_t(end);
	
	const size_t &iterationNumber = 
		gcransac.getRansacStatistics().iteration_number; // Number of iterations required

	printf("Iterations: %d\n", iterationNumber);
	printf("Elapsed time: %f secs\n", elapsedSeconds.count());

	// Storing the estimated fundamental matrix
	fundamentalMatrix_ = model.descriptor;

	// Visualization part.
	// Inliers are selected using threshold and the estimated model. 
	// This part is not necessary and is only for visualization purposes. 
	if (visualizeResults_)
	{
		// Load the images of the current test scene
		cv::Mat image1 = cv::imread(sourceImagePath_);
		cv::Mat image2 = cv::imread(destinationImagePath_);

		// The labeling implied by the estimated model and the drawing threshold
		const std::vector<size_t> &inliers = gcransac.getRansacStatistics().inliers;
		std::vector<int> labels(pointNumber, 0);
		for (const size_t &inlierIdx : inliers)
			labels[inlierIdx] = 1;
		printf("Inlier number: %d\n", inliers.size());

		cv::Mat outImage;

		// Draw the matches to the images
		drawMatches<double, int>(matches_, // All points 
			labels, // The labeling obtained by OpenCV
			image1, // The source image
			image2, // The destination image
			outImage); // The image with the matches drawn

		// Show the matches
		std::string window_name = "Visualization";
		gcransac::utils::showImage(outImage, // The image with the matches drawn
			window_name, // The name of the window
			1600, // The width of the window
			900, // The height of the window
			false); 
		outImage.release(); // Clean up the memory

		// Clean up the memory occupied by the images
		image1.release();
		image2.release();
	}
}
	
template <typename _Estimator>
void estimateEssentialMatrix(
	const cv::Mat &matches_, // The correspondences
	const std::string &sourceImagePath_, // The path of the source image
	const std::string &destinationImagePath_, // The path of the destination image
	const Eigen::Matrix3d &instrinsicsSource_, // The intrinsic parameters of the source camera
	const Eigen::Matrix3d &instrinsicsDestination_, // The intrinsic parameters of the destination camera
	Eigen::Matrix3d &essentialMatrix_, // The estimated fundamental matrix
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold
	const double &ransacConfidence_, // The confidence required in the results
	const double &spatialCoherenceWeight_, // The weight of the spatial coherence term in GC-RANSAC
	const bool &visualizeResults_) // A flag determining if the resulting inliers should be drawn or not
{
	_Estimator estimator(instrinsicsSource_, instrinsicsDestination_); // The robust essential matrix estimator class containing the function for the fitting and residual calculation
	gcransac::EssentialMatrix model; // The estimated model

	// The number of points in the datasets
	const size_t kPointNumber = matches_.rows; // The number of points in the scene

	if (kPointNumber < estimator.sampleSize()) // If there are no points, return
	{
		fprintf(stderr, "There are not enough points for estimation a fundamental matrix (%d < %d)\n", kPointNumber, estimator.sampleSize());
		return;
	}

	// Normalize the correspondences by the intrinsic camera matrices
	cv::Mat normalizedMatches(matches_.size(), matches_.type());
	Eigen::Vector3d sourcePoint, destinationPoint;
	Eigen::Matrix3d affinity;
	const Eigen::Matrix3d kInstrinsicsSourceInverse = instrinsicsSource_.inverse(),
		kInstrinsicsDestinationInverse = instrinsicsDestination_.inverse();
	for (size_t pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
	{
		sourcePoint <<
			matches_.at<double>(pointIdx, 0),
			matches_.at<double>(pointIdx, 1),
			1;

		destinationPoint <<
			matches_.at<double>(pointIdx, 2),
			matches_.at<double>(pointIdx, 3),
			1;

		affinity <<
			matches_.at<double>(pointIdx, 4), matches_.at<double>(pointIdx, 5), 0,
			matches_.at<double>(pointIdx, 6), matches_.at<double>(pointIdx, 7), 0,
			0, 0, 1;

		sourcePoint = kInstrinsicsSourceInverse * sourcePoint;
		destinationPoint = kInstrinsicsDestinationInverse * destinationPoint;
		//affinity = kInstrinsicsDestinationInverse * affinity * instrinsicsSource_;

		normalizedMatches.at<double>(pointIdx, 0) = sourcePoint(0);
		normalizedMatches.at<double>(pointIdx, 1) = sourcePoint(1);
		normalizedMatches.at<double>(pointIdx, 2) = destinationPoint(0);
		normalizedMatches.at<double>(pointIdx, 3) = destinationPoint(1);
		normalizedMatches.at<double>(pointIdx, 4) = affinity(0, 0);
		normalizedMatches.at<double>(pointIdx, 5) = affinity(0, 1);
		normalizedMatches.at<double>(pointIdx, 6) = affinity(1, 0);
		normalizedMatches.at<double>(pointIdx, 7) = affinity(1, 1);
	}

	// Normalize the threshold
	const double kAverageDiagonal = (instrinsicsSource_(0, 0) + instrinsicsSource_(1, 1) +
		instrinsicsDestination_(0, 0) + instrinsicsDestination_(1, 1)) / 4.0;
	const double kNormalizedInlierOutlierThreshold =
		inlierOutlierThreshold_ / kAverageDiagonal;

	// Initialize the neighborhood used in Graph-cut RANSAC
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&matches_(cv::Rect(0, 0, 4, kPointNumber)), // All data points
		20.0); // The radius of the neighborhood ball for determining the neighborhoods.
	
	// Deciding if affine correspondences are used based on the template parameter
	constexpr size_t usingAffineCorrespondences =
		std::is_same<_Estimator, gcransac::utils::DefaultAffinityBasedEssentialMatrixEstimator>() ? 1 : 0;

	// Defining the combined (SPRT + uncertainty propagation) preemption type based on 
	// whether affine or point correspondences are used
	typedef gcransac::preemption::CombinedPreemptiveVerfication<usingAffineCorrespondences,
		_Estimator, // The solver used for fitting a model to a non-minimal sample
		gcransac::preemption::EssentialMatrixUncertaintyBasedPreemption<usingAffineCorrespondences, _Estimator>> // The type used for uncertainty propagation
		CombinedPreemptiveVerification;

	// Initializing the preemptive verification object
	CombinedPreemptiveVerification preemptiveVerification;
	preemptiveVerification.initialize(
		matches_,
		estimator);

	// Initialize the sampler used for selecting minimal samples
	gcransac::sampler::ProsacSampler mainSampler(&matches_, _Estimator::sampleSize());
	gcransac::sampler::UniformSampler localOptimizationSampler(&matches_);

	gcransac::GCRANSAC<_Estimator,
		gcransac::neighborhood::FlannNeighborhoodGraph,
		gcransac::MSACScoringFunction<_Estimator>,
		CombinedPreemptiveVerification> gcransac;
	gcransac.settings.threshold = kNormalizedInlierOutlierThreshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.0; // The weight of the spatial coherence term
	gcransac.settings.confidence = ransacConfidence_; // The required confidence in the results
	gcransac.settings.max_iteration_number = 1e4; // The maximum number of iterations
	
	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();
	gcransac.run(normalizedMatches,
		estimator,
		&mainSampler,
		&localOptimizationSampler,
		&neighborhood,
		model,
		preemptiveVerification);
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsedSeconds = end - start;
	std::time_t endTime = std::chrono::system_clock::to_time_t(end);
	
	const size_t &iterationNumber =
		gcransac.getRansacStatistics().iteration_number; // Number of iterations required

	printf("Iterations: %d\n", iterationNumber);
	printf("Elapsed time: %f secs\n", elapsedSeconds.count());

	// Storing the estimated essential matrix
	essentialMatrix_ = model.descriptor;

	// Visualization part.
	// Inliers are selected using threshold and the estimated model. 
	// This part is not necessary and is only for visualization purposes. 
	if (visualizeResults_)
	{
		// Load the images of the current test scene
		cv::Mat image1 = cv::imread(sourceImagePath_);
		cv::Mat image2 = cv::imread(destinationImagePath_);

		// The labeling implied by the estimated model and the drawing threshold
		const std::vector<size_t> &inliers = gcransac.getRansacStatistics().inliers;
		std::vector<int> labels(kPointNumber, 0);
		for (const size_t &inlierIdx : inliers)
			labels[inlierIdx] = 1;
		printf("Inlier number: %d\n", inliers.size());

		cv::Mat outImage;

		// Draw the matches to the images
		drawMatches<double, int>(matches_, // All points 
			labels, // The labeling obtained by OpenCV
			image1, // The source image
			image2, // The destination image
			outImage); // The image with the matches drawn

		// Show the matches
		std::string window_name = "Visualization";
		gcransac::utils::showImage(outImage, // The image with the matches drawn
			window_name, // The name of the window
			1600, // The width of the window
			900, // The height of the window
			false); 
		outImage.release(); // Clean up the memory

		// Clean up the memory occupied by the images
		image1.release();
		image2.release();
	}
}

template <typename _Estimator>
void estimateHomography(
	const cv::Mat &matches_, // The correspondences
	const std::string &sourceImagePath_, // The path of the source image
	const std::string &destinationImagePath_, // The path of the destination image
	Eigen::Matrix3d &homography_, // The estimated fundamental matrix
	const double &inlierOutlierThreshold_, // The inlier-outlier threshold
	const double &ransacConfidence_, // The confidence required in the results
	const double &spatialCoherenceWeight_, // The weight of the spatial coherence term in GC-RANSAC
	const bool &visualizeResults_) // A flag determining if the resulting inliers should be drawn or not
{
	_Estimator estimator; // The robust essential matrix estimator class containing the function for the fitting and residual calculation
	gcransac::Homography model; // The estimated model

	// The number of points in the datasets
	const size_t pointNumber = matches_.rows; // The number of points in the scene

	if (pointNumber < estimator.sampleSize()) // If there are no points, return
	{
		fprintf(stderr, "There are not enough points for estimation a homography (%d < %d)\n", pointNumber, estimator.sampleSize());
		return;
	}

	// Initialize the neighborhood used in Graph-cut RANSAC
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&matches_(cv::Rect(0, 0, 4, pointNumber)), // All data points
		20.0); // The radius of the neighborhood ball for determining the neighborhoods.

	// Initialize the sampler used for selecting minimal samples
	gcransac::sampler::ProsacSampler mainSampler(&matches_, _Estimator::sampleSize());
	gcransac::sampler::UniformSampler localOptimizationSampler(&matches_);

	// Deciding if affine correspondences are used based on the template parameter
	constexpr size_t usingAffineCorrespondences =
		std::is_same<_Estimator, gcransac::utils::DefaultAffinityBasedHomographyEstimator>() ? 1 : 0;

	// Defining the combined (SPRT + uncertainty propagation) preemption type based on 
	// whether affine or point correspondences are used
	typedef gcransac::preemption::CombinedPreemptiveVerfication<usingAffineCorrespondences,
		_Estimator, // The solver used for fitting a model to a non-minimal sample
		gcransac::preemption::HomographyUncertaintyBasedPreemption<usingAffineCorrespondences, _Estimator>> // The type used for uncertainty propagation
		CombinedPreemptiveVerification;

	// Initializing the preemptive verification object
	CombinedPreemptiveVerification preemptiveVerification;
	preemptiveVerification.initialize(
		matches_, 
		estimator);

	gcransac::GCRANSAC<_Estimator,
		gcransac::neighborhood::FlannNeighborhoodGraph,
		gcransac::MSACScoringFunction<_Estimator>,
		CombinedPreemptiveVerification> gcransac;
	gcransac.settings.threshold = inlierOutlierThreshold_; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.0; // The weight of the spatial coherence term
	gcransac.settings.confidence = ransacConfidence_; // The required confidence in the results
	gcransac.settings.max_iteration_number = 1e4; // The maximum number of iterations
	
	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();
	gcransac.run(matches_,
		estimator,
		&mainSampler,
		&localOptimizationSampler,
		&neighborhood,
		model,
		preemptiveVerification);
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsedSeconds = end - start;
	std::time_t endTime = std::chrono::system_clock::to_time_t(end);

	const size_t &iterationNumber =
		gcransac.getRansacStatistics().iteration_number; // Number of iterations required

	printf("Iterations: %d\n", iterationNumber);
	printf("Elapsed time: %f secs\n", elapsedSeconds.count());

	// Storing the estimated essential matrix
	homography_ = model.descriptor;

	// Visualization part.
	// Inliers are selected using threshold and the estimated model. 
	// This part is not necessary and is only for visualization purposes. 
	if (visualizeResults_)
	{
		// Load the images of the current test scene
		cv::Mat image1 = cv::imread(sourceImagePath_);
		cv::Mat image2 = cv::imread(destinationImagePath_);

		// The labeling implied by the estimated model and the drawing threshold
		const std::vector<size_t> &inliers = gcransac.getRansacStatistics().inliers;
		std::vector<int> labels(pointNumber, 0);
		for (const size_t &inlierIdx : inliers)
			labels[inlierIdx] = 1;
		printf("Inlier number: %d\n", inliers.size());

		cv::Mat outImage;

		// Draw the matches to the images
		drawMatches<double, int>(matches_, // All points 
			labels, // The labeling obtained by OpenCV
			image1, // The source image
			image2, // The destination image
			outImage); // The image with the matches drawn

		// Show the matches
		std::string window_name = "Visualization";
		gcransac::utils::showImage(outImage, // The image with the matches drawn
			window_name, // The name of the window
			1600, // The width of the window
			900, // The height of the window
			false);
		outImage.release(); // Clean up the memory

		// Clean up the memory occupied by the images
		image1.release();
		image2.release();
	}
}

void normalizeCorrespondences(const cv::Mat &points_,
	const Eigen::Matrix3d &intrinsics_src_,
	const Eigen::Matrix3d &intrinsics_dst_,
	cv::Mat &normalized_points_)
{
	const Eigen::Matrix3d inverse_intrinsics_src = intrinsics_src_.inverse(),
		inverse_intrinsics_dst = intrinsics_dst_.inverse();

	// Most likely, this is not the fastest solution, but it does
	// not affect the speed of Graph-cut RANSAC, so not a crucial part of
	// this example.
	double x0, y0, x1, y1;
	for (auto r = 0; r < points_.rows; ++r)
	{
		Eigen::Vector3d point_src,
			point_dst,
			normalized_point_src,
			normalized_point_dst;

		x0 = points_.at<double>(r, 0);
		y0 = points_.at<double>(r, 1);
		x1 = points_.at<double>(r, 2);
		y1 = points_.at<double>(r, 3);

		point_src << x0, y0, 1.0; // Homogeneous point in the first image
		point_dst << x1, y1, 1.0; // Homogeneous point in the second image

		// Normalized homogeneous point in the first image
		normalized_point_src =
			inverse_intrinsics_src * point_src;
		// Normalized homogeneous point in the second image
		normalized_point_dst =
			inverse_intrinsics_dst * point_dst;

		// The second four columns contain the normalized coordinates.
		normalized_points_.at<double>(r, 0) = normalized_point_src(0);
		normalized_points_.at<double>(r, 1) = normalized_point_src(1);
		normalized_points_.at<double>(r, 2) = normalized_point_dst(0);
		normalized_points_.at<double>(r, 3) = normalized_point_dst(1);
	}
}

template<typename T, typename LabelType>
void drawMatches(
	const cv::Mat &points_,
	const std::vector<LabelType>& labeling_,
	const cv::Mat& image1_,
	const cv::Mat& image2_,
	cv::Mat& out_image_)
{
	const size_t N = points_.rows;
	std::vector< cv::KeyPoint > keypoints1, keypoints2;
	std::vector< cv::DMatch > matches;

	keypoints1.reserve(N);
	keypoints2.reserve(N);
	matches.reserve(N);

	// Collect the points which has label 1 (i.e. inlier)
	for (auto pt_idx = 0; pt_idx < N; ++pt_idx)
	{
		if (!labeling_[pt_idx])
			continue;

		const T x1 = points_.at<T>(pt_idx, 0);
		const T y1 = points_.at<T>(pt_idx, 1);
		const T x2 = points_.at<T>(pt_idx, 2);
		const T y2 = points_.at<T>(pt_idx, 3);
		const size_t n = keypoints1.size();

		keypoints1.emplace_back(
			cv::KeyPoint(cv::Point_<T>(x1, y1), 0));
		keypoints2.emplace_back(
			cv::KeyPoint(cv::Point_<T>(x2, y2), 0));
		matches.emplace_back(cv::DMatch(static_cast<int>(n), static_cast<int>(n), 0));
	}

	// Draw the matches using OpenCV's built-in function
	cv::drawMatches(image1_,
		keypoints1,
		image2_,
		keypoints2,
		matches,
		out_image_);
}