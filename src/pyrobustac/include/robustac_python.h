#include <vector>
#include <string>

void extractACs_(
	const std::string &imageSourcePath_,
	const std::string &imageDestinationPath_,
	std::vector<double> &srcPts_,
	std::vector<double> &dstPts_,
	std::vector<double> &affines_);

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
	int max_iters);

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
	int max_iters);

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
	int maximumIterations_);