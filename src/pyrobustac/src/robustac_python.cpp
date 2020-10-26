#include "robustac_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Eigen>
#include "feature_utils.h"

#include "GCRANSAC.h"
#include "flann_neighborhood_graph.h"
#include "grid_neighborhood_graph.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "essential_estimator.h"

#include "preemption_sprt.h"
#include "preemption_uncertainty_based.h"
#include "preemption_combined.h"

#include "affine_estimators.h"

#include "solver_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_eight_point.h"
#include "solver_homography_four_point.h"
#include "solver_essential_matrix_five_point_stewenius.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

using namespace gcransac;

void extractACs_(
	const std::string &imageSourcePath_,
	const std::string &imageDestinationPath_,
	std::vector<double> &srcPts_,
	std::vector<double> &dstPts_,
	std::vector<double> &affines_)
{
	// Load the images to get their size for the error calculation
	cv::Mat image1 = cv::imread(imageSourcePath_);
	cv::Mat image2 = cv::imread(imageDestinationPath_);
	const cv::Size kSourceImageSize = image1.size();
	const cv::Size kDestinationImageSize = image2.size();
	image1.release();
	image2.release();

	cv::Mat matches;
	double extractionTime;

	// A function obtaining the affine correspondences
	printf("Obtaining affine correspondences for images '%s' and '%s'\n", imageSourcePath_.c_str(), imageDestinationPath_.c_str());
	ACExtraction::getAffineCorrespondences(
		imageSourcePath_, // The path of the source image
		imageDestinationPath_, // The path of the destination image
		"", // The path from where the correspondence will be read or saved to.
		matches, // The extracted matches
		extractionTime, // The feature extraction time
		true,
		false,
		false); 

	printf("%d correspondences are found in %f secs.\n", matches.rows, extractionTime);

	image1.release();
	image2.release();

	srcPts_.reserve(2 * matches.rows);
	dstPts_.reserve(2 * matches.rows);
	affines_.reserve(4 * matches.rows);

	for (size_t pointIdx = 0; pointIdx < matches.rows; ++pointIdx)
	{
		srcPts_.emplace_back(matches.at<double>(pointIdx, 0));
		srcPts_.emplace_back(matches.at<double>(pointIdx, 1));

		dstPts_.emplace_back(matches.at<double>(pointIdx, 2));
		dstPts_.emplace_back(matches.at<double>(pointIdx, 3));

		affines_.emplace_back(matches.at<double>(pointIdx, 4));
		affines_.emplace_back(matches.at<double>(pointIdx, 5));
		affines_.emplace_back(matches.at<double>(pointIdx, 6));
		affines_.emplace_back(matches.at<double>(pointIdx, 7));
	}
}

int findFundamentalMat_(
	const std::vector<double>& sourcePoints_,
	const std::vector<double>& destinationPoints_,
	const std::vector<double>& affinities_,
	std::vector<bool>& inlierMask_,
	std::vector<double>& fundamentalMatrix_,
	int sourceImageHeight_,
	int sourceImageWidth_,
	int destinationImageHeight_,
	int destinationImageWidth_,
	double spatialCoherenceWeight_,
	double inlierOutlierThreshold_,
	double ransacConfidence_,
	int max_iters)
{
	typedef gcransac::utils::DefaultAffinityBasedFundamentalMatrixEstimator Estimator;
	const size_t kCellNumberInNeighborhoodGraph = 8;

	const size_t pointNumber = sourcePoints_.size() / 2; // The number of points in the scene
	cv::Mat points(pointNumber, 8, CV_64F);
	int iterations = 0;
	for (int i = 0; i < pointNumber; ++i)
	{
		points.at<double>(i, 0) = sourcePoints_[2 * i];
		points.at<double>(i, 1) = sourcePoints_[2 * i + 1];
		points.at<double>(i, 2) = destinationPoints_[2 * i];
		points.at<double>(i, 3) = destinationPoints_[2 * i + 1];
		points.at<double>(i, 4) = affinities_[4 * i];
		points.at<double>(i, 5) = affinities_[4 * i + 1];
		points.at<double>(i, 6) = affinities_[4 * i + 2];
		points.at<double>(i, 7) = affinities_[4 * i + 3];
	}

	neighborhood::GridNeighborhoodGraph neighborhoodGraph(&points,
		sourceImageWidth_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		sourceImageHeight_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		destinationImageWidth_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		destinationImageHeight_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		kCellNumberInNeighborhoodGraph);

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhoodGraph.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	};

	Estimator estimator;
	FundamentalMatrix model;

	if (pointNumber < estimator.sampleSize()) // If there are no points, return
	{
		fprintf(stderr, "There are not enough points for estimation a homography (%d < %d)\n", pointNumber, estimator.sampleSize());
		return 0;
	}

	// Initialize the sampler used for selecting minimal samples
	gcransac::sampler::ProsacSampler mainSampler(&points,
		Estimator::sampleSize());
	gcransac::sampler::UniformSampler localOptimizationSampler(&points);

	// Deciding if affine correspondences are used based on the template parameter
	constexpr size_t kUsingAffineCorrespondences = 1;

	// Defining the combined (SPRT + uncertainty propagation) preemption type based on 
	// whether affine or point correspondences are used
	typedef gcransac::preemption::CombinedPreemptiveVerfication<kUsingAffineCorrespondences,
		Estimator, // The solver used for fitting a model to a non-minimal sample
		gcransac::preemption::FundamentalMatrixUncertaintyBasedPreemption<kUsingAffineCorrespondences, Estimator>> // The type used for uncertainty propagation
		CombinedPreemptiveVerification;

	// Initializing the preemptive verification object
	CombinedPreemptiveVerification preemptiveVerification;
	preemptiveVerification.initialize(
		points,
		estimator);

	gcransac::GCRANSAC<Estimator,
		gcransac::neighborhood::GridNeighborhoodGraph,
		gcransac::MSACScoringFunction<Estimator>,
		CombinedPreemptiveVerification> gcransac;
	gcransac.settings.threshold = inlierOutlierThreshold_; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatialCoherenceWeight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = ransacConfidence_; // The required confidence in the results
	gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations

	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();
	gcransac.run(points,
		estimator,
		&mainSampler,
		&localOptimizationSampler,
		&neighborhoodGraph,
		model,
		preemptiveVerification);
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsedSeconds = end - start;
	std::time_t endTime = std::chrono::system_clock::to_time_t(end);

	const size_t &iterationNumber =
		gcransac.getRansacStatistics().iteration_number; // Number of iterations required

	printf("Iterations: %d\n", iterationNumber);
	printf("Elapsed time: %f secs\n", elapsedSeconds.count());

	const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));

	fundamentalMatrix_.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			fundamentalMatrix_[i * 3 + j] = model.descriptor(i, j);
		}
	}

	inlierMask_.resize(pointNumber);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < pointNumber; ++pt_idx) {
		inlierMask_[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inlierMask_[statistics.inliers[pt_idx]] = 1;
	}

	return num_inliers;
}

int findEssentialMat_(
	const std::vector<double>& sourcePoints_,
	const std::vector<double>& destinationPoints_,
	const std::vector<double>& affinities_,
	const std::vector<double>& sourceIntrinsics_,
	const std::vector<double>& destinationIntrinsics_,
	std::vector<bool>& inlierMask_,
	std::vector<double>& essentialMatrix_,
	int sourceImageHeight_,
	int sourceImageWidth_,
	int destinationImageHeight_,
	int destinationImageWidth_,
	double spatialCoherenceWeight_,
	double inlierOutlierThreshold_,
	double ransacConfidence_,
	int maximumIterations_)
{
	typedef gcransac::utils::DefaultAffinityBasedEssentialMatrixEstimator Estimator;
	const size_t kCellNumberInNeighborhoodGraph = 8;

	const size_t pointNumber = sourcePoints_.size() / 2; // The number of points in the scene
	cv::Mat points(pointNumber, 8, CV_64F);
	int iterations = 0;

	for (int i = 0; i < pointNumber; ++i)
	{
		points.at<double>(i, 0) = sourcePoints_[2 * i];
		points.at<double>(i, 1) = sourcePoints_[2 * i + 1];
		points.at<double>(i, 2) = destinationPoints_[2 * i];
		points.at<double>(i, 3) = destinationPoints_[2 * i + 1];
		points.at<double>(i, 4) = affinities_[4 * i];
		points.at<double>(i, 5) = affinities_[4 * i + 1];
		points.at<double>(i, 6) = affinities_[4 * i + 2];
		points.at<double>(i, 7) = affinities_[4 * i + 3];
	

	neighborhood::GridNeighborhoodGraph neighborhoodGraph(&points,
		sourceImageWidth_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		sourceImageHeight_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		destinationImageWidth_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		destinationImageHeight_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		kCellNumberInNeighborhoodGraph);

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhoodGraph.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	};

	Eigen::Matrix3d K1, K2;
	K1 << sourceIntrinsics_[0], sourceIntrinsics_[1], sourceIntrinsics_[2],
		sourceIntrinsics_[3], sourceIntrinsics_[4], sourceIntrinsics_[5],
		sourceIntrinsics_[6], sourceIntrinsics_[7], sourceIntrinsics_[8];
	K2 << destinationIntrinsics_[0], destinationIntrinsics_[1], destinationIntrinsics_[2],
		destinationIntrinsics_[3], destinationIntrinsics_[4], destinationIntrinsics_[5],
		destinationIntrinsics_[6], destinationIntrinsics_[7], destinationIntrinsics_[8];

	// Normalize the correspondences by the intrinsic camera matrices
	cv::Mat normalizedMatches(points.size(), points.type());
	Eigen::Vector3d sourcePoint, destinationPoint;
	Eigen::Matrix3d affinity;
	const Eigen::Matrix3d kInstrinsicsSourceInverse = K1.inverse(),
		kInstrinsicsDestinationInverse = K2.inverse();
	for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
	{
		sourcePoint <<
			points.at<double>(pointIdx, 0),
			points.at<double>(pointIdx, 1),
			1;

		destinationPoint <<
			points.at<double>(pointIdx, 2),
			points.at<double>(pointIdx, 3),
			1;

		affinity <<
			points.at<double>(pointIdx, 4), points.at<double>(pointIdx, 5), 0,
			points.at<double>(pointIdx, 6), points.at<double>(pointIdx, 7), 0,
			0, 0, 1;

		sourcePoint = kInstrinsicsSourceInverse * sourcePoint;
		destinationPoint = kInstrinsicsDestinationInverse * destinationPoint;
		affinity = kInstrinsicsDestinationInverse * affinity * K1;

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
	const double kAverageDiagonal = (K1(0, 0) + K1(1, 1) +
		K2(0, 0) + K2(1, 1)) / 4.0;
	const double kNormalizedInlierOutlierThreshold =
		inlierOutlierThreshold_ / kAverageDiagonal;

	Estimator estimator(K1, K2, 0.1);
	EssentialMatrix model;

	if (pointNumber < estimator.sampleSize()) // If there are no points, return
	{
		fprintf(stderr, "There are not enough points for estimation a homography (%d < %d)\n", pointNumber, estimator.sampleSize());
		return 0;
	}

	// Initialize the sampler used for selecting minimal samples
	gcransac::sampler::ProsacSampler mainSampler(&points,
		Estimator::sampleSize());
	gcransac::sampler::UniformSampler localOptimizationSampler(&points);

	// Deciding if affine correspondences are used based on the template parameter
	constexpr size_t kUsingAffineCorrespondences = 1;

	// Defining the combined (SPRT + uncertainty propagation) preemption type based on 
	// whether affine or point correspondences are used
	typedef gcransac::preemption::CombinedPreemptiveVerfication<kUsingAffineCorrespondences,
		Estimator, // The solver used for fitting a model to a non-minimal sample
		gcransac::preemption::EssentialMatrixUncertaintyBasedPreemption<kUsingAffineCorrespondences, Estimator>> // The type used for uncertainty propagation
		CombinedPreemptiveVerification;

	// Initializing the preemptive verification object
	CombinedPreemptiveVerification preemptiveVerification;
	preemptiveVerification.initialize(
		points,
		estimator);

	gcransac::GCRANSAC<Estimator,
		gcransac::neighborhood::GridNeighborhoodGraph,
		gcransac::MSACScoringFunction<Estimator>,
		CombinedPreemptiveVerification> gcransac;
	gcransac.settings.threshold = kNormalizedInlierOutlierThreshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatialCoherenceWeight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = ransacConfidence_; // The required confidence in the results
	gcransac.settings.max_iteration_number = maximumIterations_; // The maximum number of iterations

	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();
	gcransac.run(normalizedMatches,
		estimator,
		&mainSampler,
		&localOptimizationSampler,
		&neighborhoodGraph,
		model,
		preemptiveVerification);
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsedSeconds = end - start;
	std::time_t endTime = std::chrono::system_clock::to_time_t(end);

	const size_t &iterationNumber =
		gcransac.getRansacStatistics().iteration_number; // Number of iterations required

	printf("Iterations: %d\n", iterationNumber);
	printf("Elapsed time: %f secs\n", elapsedSeconds.count());

	const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));

	essentialMatrix_.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			essentialMatrix_[i * 3 + j] = model.descriptor(i, j);
		}
	}

	inlierMask_.resize(pointNumber);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < pointNumber; ++pt_idx) {
		inlierMask_[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inlierMask_[statistics.inliers[pt_idx]] = 1;
	}

	return num_inliers;
}

int findHomography_(
	const std::vector<double>& sourcePoints_,
	const std::vector<double>& destinationPoints_,
	const std::vector<double>& affinities_,
	std::vector<bool>& inlierMask_,
	std::vector<double>& homography_,
	int sourceImageHeight_, 
	int sourceImageWidth_, 
	int destinationImageHeight_, 
	int destinationImageWidth_,
	double spatialCoherenceWeight_,
	double inlierOutlierThreshold_,
	double ransacConfidence_,
	int maximumIterations_)
{
	typedef gcransac::utils::DefaultAffinityBasedHomographyEstimator Estimator;
	const size_t kCellNumberInNeighborhoodGraph = 8;

	const size_t pointNumber = sourcePoints_.size() / 2; // The number of points in the scene
	cv::Mat points(pointNumber, 8, CV_64F);
	int iterations = 0;
	for (int i = 0; i < pointNumber; ++i)
	{
		points.at<double>(i, 0) = sourcePoints_[2 * i];
		points.at<double>(i, 1) = sourcePoints_[2 * i + 1];
		points.at<double>(i, 2) = destinationPoints_[2 * i];
		points.at<double>(i, 3) = destinationPoints_[2 * i + 1];
		points.at<double>(i, 4) = affinities_[4 * i];
		points.at<double>(i, 5) = affinities_[4 * i + 1];
		points.at<double>(i, 6) = affinities_[4 * i + 2];
		points.at<double>(i, 7) = affinities_[4 * i + 3];
	}

	neighborhood::GridNeighborhoodGraph neighborhoodGraph(&points,
		sourceImageWidth_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		sourceImageHeight_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		destinationImageWidth_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		destinationImageHeight_ / static_cast<double>(kCellNumberInNeighborhoodGraph),
		kCellNumberInNeighborhoodGraph);

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhoodGraph.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	}

	Estimator estimator;
	Homography model;

	// The number of points in the datasets

	if (pointNumber < estimator.sampleSize()) // If there are no points, return
	{
		fprintf(stderr, "There are not enough points for estimation a homography (%d < %d)\n", pointNumber, estimator.sampleSize());
		return 0;
	}

	// Initialize the sampler used for selecting minimal samples
	gcransac::sampler::ProsacSampler mainSampler(&points, 
		Estimator::sampleSize());
	gcransac::sampler::UniformSampler localOptimizationSampler(&points);

	// Deciding if affine correspondences are used based on the template parameter
	constexpr size_t kUsingAffineCorrespondences = 1;

	// Defining the combined (SPRT + uncertainty propagation) preemption type based on 
	// whether affine or point correspondences are used
	typedef gcransac::preemption::CombinedPreemptiveVerfication<kUsingAffineCorrespondences,
		Estimator, // The solver used for fitting a model to a non-minimal sample
		gcransac::preemption::HomographyUncertaintyBasedPreemption<kUsingAffineCorrespondences, Estimator>> // The type used for uncertainty propagation
		CombinedPreemptiveVerification;

	// Initializing the preemptive verification object
	CombinedPreemptiveVerification preemptiveVerification;
	preemptiveVerification.initialize(
		points,
		estimator);

	gcransac::GCRANSAC<Estimator,
		gcransac::neighborhood::GridNeighborhoodGraph,
		gcransac::MSACScoringFunction<Estimator>,
		CombinedPreemptiveVerification> gcransac;
	gcransac.settings.threshold = inlierOutlierThreshold_; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatialCoherenceWeight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = ransacConfidence_; // The required confidence in the results
	gcransac.settings.max_iteration_number = maximumIterations_; // The maximum number of iterations

	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();
	gcransac.run(points,
		estimator,
		&mainSampler,
		&localOptimizationSampler,
		&neighborhoodGraph,
		model,
		preemptiveVerification);
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsedSeconds = end - start;
	std::time_t endTime = std::chrono::system_clock::to_time_t(end);

	const size_t &iterationNumber =
		gcransac.getRansacStatistics().iteration_number; // Number of iterations required

	printf("Iterations: %d\n", iterationNumber);
	printf("Elapsed time: %f secs\n", elapsedSeconds.count());

	const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));

	homography_.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			homography_[i * 3 + j] = model.descriptor(i, j);
		}
	}

	inlierMask_.resize(pointNumber);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < pointNumber; ++pt_idx) {
		inlierMask_[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inlierMask_[statistics.inliers[pt_idx]] = 1;
	}

	return num_inliers;
}
