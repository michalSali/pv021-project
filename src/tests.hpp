#ifndef PV021_FASHIONMNIST_PROJECT_TESTS_HPP
#define PV021_FASHIONMNIST_PROJECT_TESTS_HPP

#include "matrix.hpp"
#include "network.hpp"

class Tester {
	
public:

	Tester() = default;

	void test_exportResults();
	void test_CSVReader();
	void test_displayAccuracy();
	void testAddSubtractMatrix();
	void testMatrixBundle();
	float getAccuracy(const std::vector<int> &testLabels, const std::vector<int> &predictedLabels);
	void gridSearch(activations hiddenLayerActivation = activations::_leakyReLu,
					const std::vector<float> &learningRatesVector = {0.01},
					const std::vector<int> &epochsVector = {20},
					const std::vector<int> &batchSizeVector = {512},
					const std::vector<float> &leakyReLuVector = {0.01});

	void printValues(const std::vector<vector> &v);
	void printValuesWithLabels(const std::vector<vector> &values, const std::vector<int> &labels);
	void printMatrix(const std::vector<vector> &m);
	void printMatrix(const matrix &m);
	std::vector<int> createLabels(const std::vector<vector> &v);
	std::vector<int> createLabels(int count);
	std::vector<int> createRandomLabels(int count);
	std::vector<vector> createValues(int count, int dimension);
	std::vector<vector> createRandomValues(int count, int dimension);

	void testMLPBasic();
	void testMLPBasicHelper(size_t inputDimension, const std::vector<size_t> &layerDimensions);
	void testInitializeWeights(bool printInfo = false);
	void testInitializeWeightsHelper(int rows, int cols, activations activationFunction, bool printInfo);

	void predictTest();
	
	void feedForwardImpl(vector &input, std::vector<Layer> &_layers);
	void setWeightsToValue(matrix &m, valueType value);
	void setBiasToValue(vector &v, valueType value);
	
	void testFeedForward();
	void testFeedForwardHelper(MLP &mlp, vector &input);
	void testFeedForwardHelper_2(MLP &mlp, vector &input);
	void testFeedForwardHelper_3(MLP &mlp, vector &input);
};

#endif //PV021_FASHIONMNIST_PROJECT_TESTS_HPP
