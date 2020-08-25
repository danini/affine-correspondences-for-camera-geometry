#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "robustac_python.h"

namespace py = pybind11;

py::tuple extractACs(
	std::string image1_,
	std::string image2_)
{
	std::vector<double> x1y1;
	std::vector<double> x2y2;
	std::vector<double> ACs;

	extractACs_(
		image1_,
		image2_,
		x1y1,
		x2y2,
		ACs);

	const size_t pointNumber = x1y1.size() / 2;

	printf("Point number in the binding function = %d\n", pointNumber);

	py::array_t<double> x1y1_ = py::array_t<double>({ pointNumber, 2 });
	py::buffer_info buf1 = x1y1_.request();
	double *ptr1 = (double *)buf1.ptr;
	for (size_t i = 0; i < 2 * pointNumber; i++)
		ptr1[i] = x1y1[i];

	py::array_t<double> x2y2_ = py::array_t<double>({ pointNumber, 2 });
	py::buffer_info buf2 = x2y2_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 2 * pointNumber; i++)
		ptr2[i] = x2y2[i];

	py::array_t<double> ACs_ = py::array_t<double>({ pointNumber, 4 });
	py::buffer_info buf3 = ACs_.request();
	double *ptr3 = (double *)buf3.ptr;
	for (size_t i = 0; i < 4 * pointNumber; i++)
		ptr3[i] = ACs[i];

	return py::make_tuple(x1y1_, x2y2_, ACs_);
}

py::tuple findHomography(
	py::array_t<double> x1y1_,
	py::array_t<double> x2y2_,
	py::array_t<double> ACs_,
	int h1, int w1, int h2, int w2,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	int max_iters)
{
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=2");
	}
	if (NUM_TENTS < 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=2");
	}

	py::buffer_info buf1a = x2y2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 2) {
		throw std::invalid_argument("x2y2 should be an array with dims [n,2], n>=2");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and x2y2 should be the same size");
	}

	py::buffer_info buf1AC = ACs_.request();
	size_t NUM_TENTSAC = buf1AC.shape[0];
	size_t DIMAC = buf1AC.shape[1];

	if (DIMAC != 4) {
		throw std::invalid_argument("ACs should be an array with dims [n,4], n>=2");
	}
	if (NUM_TENTSAC != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and ACs should have the same number of rows");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

	double *ptr1a = (double *)buf1a.ptr;
	std::vector<double> x2y2;
	x2y2.assign(ptr1a, ptr1a + buf1a.size);

	double *ptr1AC = (double *)buf1AC.ptr;
	std::vector<double> ACs;
	ACs.assign(ptr1AC, ptr1AC + buf1AC.size);

	std::vector<double> H(9);
	std::vector<bool> inliers(NUM_TENTS);

	int num_inl = findHomography_(x1y1,
		x2y2,
		ACs,
		inliers,
		H,
		h1, w1, h2, w2,
		spatial_coherence_weight,
		threshold,
		conf,
		max_iters);

	py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
	py::buffer_info buf3 = inliers_.request();
	bool *ptr3 = (bool *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = inliers[i];

	if (num_inl == 0) {
		return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
	}

	py::array_t<double> H_ = py::array_t<double>({ 3,3 });
	py::buffer_info buf2 = H_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 9; i++)
		ptr2[i] = H[i];

	return py::make_tuple(H_, inliers_);
}

py::tuple findFundamentalMat(
	py::array_t<double> x1y1_,
	py::array_t<double> x2y2_,
	py::array_t<double> ACs_,
	int h1, int w1, int h2, int w2,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	int max_iters)
{
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=2");
	}
	if (NUM_TENTS < 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=2");
	}

	py::buffer_info buf1a = x2y2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 2) {
		throw std::invalid_argument("x2y2 should be an array with dims [n,2], n>=2");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and x2y2 should be the same size");
	}

	py::buffer_info buf1AC = ACs_.request();
	size_t NUM_TENTSAC = buf1AC.shape[0];
	size_t DIMAC = buf1AC.shape[1];

	if (DIMAC != 4) {
		throw std::invalid_argument("ACs should be an array with dims [n,4], n>=2");
	}
	if (NUM_TENTSAC != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and ACs should have the same number of rows");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

	double *ptr1a = (double *)buf1a.ptr;
	std::vector<double> x2y2;
	x2y2.assign(ptr1a, ptr1a + buf1a.size);

	double *ptr1AC = (double *)buf1AC.ptr;
	std::vector<double> ACs;
	ACs.assign(ptr1AC, ptr1AC + buf1AC.size);

	std::vector<double> F(9);
	std::vector<bool> inliers(NUM_TENTS);

	int num_inl = findFundamentalMat_(x1y1,
		x2y2,
		ACs,
		inliers,
		F,
		h1, w1, h2, w2,
		spatial_coherence_weight,
		threshold,
		conf,
		max_iters);

	py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
	py::buffer_info buf3 = inliers_.request();
	bool *ptr3 = (bool *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = inliers[i];

	if (num_inl == 0) {
		return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
	}

	py::array_t<double> F_ = py::array_t<double>({ 3,3 });
	py::buffer_info buf2 = F_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 9; i++)
		ptr2[i] = F[i];

	return py::make_tuple(F_, inliers_);
}

py::tuple findEssentialMat(
	py::array_t<double> x1y1_,
	py::array_t<double> x2y2_,
	py::array_t<double> ACs_,
	py::array_t<double> K1_,
	py::array_t<double> K2_,
	int h1, int w1, int h2, int w2,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	int max_iters)
{
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=2");
	}
	if (NUM_TENTS < 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=2");
	}

	py::buffer_info buf1a = x2y2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 2) {
		throw std::invalid_argument("x2y2 should be an array with dims [n,2], n>=2");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and x2y2 should be the same size");
	}

	py::buffer_info buf1AC = ACs_.request();
	size_t NUM_TENTSAC = buf1AC.shape[0];
	size_t DIMAC = buf1AC.shape[1];

	if (DIMAC != 4) {
		throw std::invalid_argument("ACs should be an array with dims [n,4], n>=2");
	}
	if (NUM_TENTSAC != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and ACs should have the same number of rows");
	}

	py::buffer_info buf1K1 = K1_.request();
	size_t DIMK11 = buf1K1.shape[0];
	size_t DIMK12 = buf1K1.shape[1];

	if (DIMK11 != 3 || DIMK12 != 3) {
		throw std::invalid_argument("K1 should be an array with dims [3,3]");
	}

	py::buffer_info buf1K2 = K2_.request();
	size_t DIMK21 = buf1K2.shape[0];
	size_t DIMK22 = buf1K2.shape[1];

	if (DIMK21 != 3 || DIMK22 != 3) {
		throw std::invalid_argument("K2 should be an array with dims [3,3]");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

	double *ptr1a = (double *)buf1a.ptr;
	std::vector<double> x2y2;
	x2y2.assign(ptr1a, ptr1a + buf1a.size);

	double *ptr1AC = (double *)buf1AC.ptr;
	std::vector<double> ACs;
	ACs.assign(ptr1AC, ptr1AC + buf1AC.size);

	double *ptr1K1 = (double *)buf1K1.ptr;
	std::vector<double> K1;
	K1.assign(ptr1K1, ptr1K1 + buf1K1.size);

	double *ptr1K2 = (double *)buf1K2.ptr;
	std::vector<double> K2;
	K2.assign(ptr1K2, ptr1K2 + buf1K2.size);

	std::vector<double> E(9);
	std::vector<bool> inliers(NUM_TENTS);

	int num_inl = findEssentialMat_(x1y1,
		x2y2,
		ACs,
		K1,
		K2,
		inliers,
		E,
		h1, w1, h2, w2,
		spatial_coherence_weight,
		threshold,
		conf,
		max_iters);

	py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
	py::buffer_info buf3 = inliers_.request();
	bool *ptr3 = (bool *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = inliers[i];

	if (num_inl == 0) {
		return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
	}

	py::array_t<double> F_ = py::array_t<double>({ 3,3 });
	py::buffer_info buf2 = F_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 9; i++)
		ptr2[i] = E[i];

	return py::make_tuple(F_, inliers_);
}

PYBIND11_PLUGIN(pyrobustac) {
                                                                             
    py::module m("pyrobustac", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pygcransac
        .. autosummary::
           :toctree: _generate
           
           extractACs,
           findHomography,
		   findFundamentalMat,
		   findEssentialMat,

    )doc");

	m.def("extractACs", &extractACs, R"doc(some doc)doc",
		py::arg("sourceImagePath"),
		py::arg("destinationImagePath"));
	
	m.def("findHomography", &findHomography, R"doc(some doc)doc",
		py::arg("x1y1"),
		py::arg("x2y2"),
		py::arg("ACs"),
		py::arg("h1"),
		py::arg("w1"),
		py::arg("h2"),
		py::arg("w2"),
		py::arg("threshold") = 1.0,
		py::arg("conf") = 0.99,
		py::arg("spatial_coherence_weight") = 0.975,
		py::arg("max_iters") = 10000);
	
	m.def("findFundamentalMat", &findFundamentalMat, R"doc(some doc)doc",
		py::arg("x1y1"),
		py::arg("x2y2"),
		py::arg("ACs"),
		py::arg("h1"),
		py::arg("w1"),
		py::arg("h2"),
		py::arg("w2"),
		py::arg("threshold") = 1.0,
		py::arg("conf") = 0.99,
		py::arg("spatial_coherence_weight") = 0.975,
		py::arg("max_iters") = 10000);

	m.def("findEssentialMat", &findEssentialMat, R"doc(some doc)doc",
		py::arg("x1y1"),
		py::arg("x2y2"),
		py::arg("ACs"),
		py::arg("K1"),
		py::arg("K2"),
		py::arg("h1"),
		py::arg("w1"),
		py::arg("h2"),
		py::arg("w2"),
		py::arg("threshold") = 1.0,
		py::arg("conf") = 0.99,
		py::arg("spatial_coherence_weight") = 0.975,
		py::arg("max_iters") = 10000);

  return m.ptr();
}
