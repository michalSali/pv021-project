#include "IOUtils.hpp"
#include "tests.hpp"
#include <cassert>
#include <iostream>
#include <random>
#include <stdlib.h> 
#include <cmath>
#include <string>


float Tester::getAccuracy(const std::vector<int> &testLabels, const std::vector<int> &predictedLabels) {
    if (testLabels.size() != predictedLabels.size()) {
        return 0;
    }

    float correctHits = 0;
    for (size_t i = 0; i < predictedLabels.size(); i++) {
        if (predictedLabels[i] == testLabels[i]) {
            correctHits++;
        }
    }

    return correctHits / testLabels.size();
}


void Tester::gridSearch(activations hiddenLayerActivation,
                const std::vector<float> &learningRatesVector,
                const std::vector<int> &epochsVector,
                const std::vector<int> &batchSizeVector,
                const std::vector<float> &leakyReLuVector) {

    CSVReader reader;
    auto trainValues = reader.readCSVValues("../data/fashion_mnist_train_vectors.csv");
    auto trainLabels = reader.readCSVLabels("../data/fashion_mnist_train_labels.csv");
    auto testValues = reader.readCSVValues("../data/fashion_mnist_test_vectors.csv");
    auto testLabels = reader.readCSVLabels("../data/fashion_mnist_test_labels.csv");

    float bestLearningRate = 0.01;
    int bestEpochs = 20;
    int bestBatchSize = 512;
    float bestLeakyReLU = 0.01;
    float bestAccuracy = 0.0;

    for (float learningRate : learningRatesVector) {
        for (int epochs : epochsVector) {
            for (int batchSize : batchSizeVector) {
                for (float leakyReLUAlpha : leakyReLuVector) {

                    MLP network(784);
                    //network.addLayer(512, hiddenLayerActivation);
                    //network.addLayer(256, hiddenLayerActivation);
                    network.addLayer(128, hiddenLayerActivation);
                    //network.addLayer(64, hiddenLayerActivation);
                    network.addLayer(32, hiddenLayerActivation);
                    //network.addLayer(16, hiddenLayerActivation);
                    network.addLayer(10, activations::_softmax);

                    network.setLeakyReLUAlpha(leakyReLUAlpha);

                    network.train(trainValues, trainLabels, learningRate, epochs, batchSize);

                    auto predictedLabels = network.predict(testValues);

                    float accuracy = getAccuracy(testLabels, predictedLabels);

                    if (accuracy > bestAccuracy) {
                        bestLearningRate = learningRate;
                        bestEpochs = epochs;
                        bestBatchSize = batchSize;
                        bestLeakyReLU = leakyReLUAlpha;
                        bestAccuracy = accuracy;
                    }

                    std::cout << "learning rate: " << learningRate << " \tepochs: " << epochs;
                    std::cout << " batch size: " << batchSize << " leaky reLU: " << leakyReLUAlpha << " \taccuracy: " << accuracy << "%" << std::endl;
                }
            }
        }
    }

    std::cout << "Best setting:" << std::endl;
    std::cout << "learning rate: " << bestLearningRate << " \tepochs: " << bestEpochs;
    std::cout << " batch size: " << bestBatchSize << " leaky reLU: " << bestLeakyReLU << " \taccuracy: " << bestAccuracy << "%" << std::endl;
}


void Tester::printValues(const std::vector<vector> &v) {
    for (size_t i = 0; i < v.size(); ++i) {
        for (size_t j = 0; j < v[0].size(); ++j) {
            std::cout << v[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


void Tester::printValuesWithLabels(const std::vector<vector> &values, const std::vector<int> &labels) {
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = 0; j < values[0].size(); ++j) {
            std::cout << values[i][j] << " ";
        }
        std::cout << " | " << labels[i] << std::endl;        
    }
    std::cout << std::endl;
}


void Tester::printMatrix(const std::vector<vector> &m) {
    for (const auto &row : m) {
        for (auto n : row.getValues()) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
    }
}

void Tester::printMatrix(const matrix &m) {
    for (const auto &row : m.getValues()) {
        for (auto n : row.getValues()) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
    }
}


void Tester::test_exportResults() {
	CSVReader reader;
    std::vector<int> v1 {1, 2, 3};
    std::string s1 = "./test_results.csv";
    reader.exportResults(s1, v1);
}


/*
void Tester::test_CSVReader() {
    std::string s1 = "./simple.csv";
    auto values = CSVReader().readCSVValues(s1);
    printMatrix(values);
}
*/


void Tester::test_displayAccuracy() {
    std::string expectedValuesPath = "./expectedValues.csv";
    std::string actualValuesPath = "./actualValues.csv";
    //displayAccuracy(expectedValuesPath, actualValuesPath);
    //displayAccuracy(expectedValuesPath, actualValuesPath);
}


std::vector<vector> Tester::createRandomValues(int count, int dimension) {
	std::vector<vector> values;
	vector singleExample(dimension);
	
	for (int i = 0; i < count; ++i) {		
		for (int j = 0; j < dimension; ++j) {
			singleExample[j] = std::rand() % (100); //i*dimension + j;
		}
		values.emplace_back(singleExample);
	}
	return values;
}


std::vector<vector> Tester::createValues(int count, int dimension) {
	std::vector<vector> values;
	vector singleExample(dimension);
	
	for (int i = 0; i < count; ++i) {		
		for (int j = 0; j < dimension; ++j) {
			singleExample[j] = i*dimension + j;
		}
		values.emplace_back(singleExample);
	}
	return values;
}


std::vector<int> Tester::createRandomLabels(int count) {
	std::vector<int> labels(count);
	for (int i = 0; i < count; ++i) {
		labels[i] = std::rand() % (100);
	}
	return labels;
}


std::vector<int> Tester::createLabels(int count) {
	std::vector<int> labels(count);
	for (int i = 0; i < count; ++i) {
		labels[i] = i;
	}
	return labels;
}


// this function sets the example's (vector's) label equal to the first element of the example (vector)
std::vector<int> Tester::createLabels(const std::vector<vector> &v) {
	std::vector<int> labels(v.size());
	for (size_t i = 0; i < v.size(); ++i) {
		labels[i] = v[i][0];
	}
	return labels;
}


// ------------------------[ basic MLP methods ]-----------------------

void Tester::testMLPBasicHelper(size_t inputDimension, const std::vector<size_t> &layerDimensions) {
	MLP mlp(inputDimension);
	
	size_t layerCount = layerDimensions.size();
	// reLU for all layers except for last
	for (size_t i = 0; i < layerCount - 1; ++i) {		
		mlp.addLayer(layerDimensions[i], activations::_reLU);
	}
	
	// add output layer with different activation function
	mlp.addLayer(layerDimensions[layerCount - 1], activations::_softmax);	
	auto layers = mlp.getLayers();
	
	assert(layers.size() == layerCount);	
	size_t rows = inputDimension;	
	
	for (size_t i = 0; i < layerCount; ++i) {
		size_t cols = layerDimensions[i];		
		assert(layerDimensions[i] == layers[i].size());
		assert((size_t)layers[i].getWeights().rows() == rows);
		assert((size_t)layers[i].getWeights().cols() == cols);
		assert(layers[i].getWeights()[0].size() == cols);		
		rows = cols;		
	}
}


void Tester::testMLPBasic() {
	testMLPBasicHelper(4, {2, 2});
	testMLPBasicHelper(20, {10, 5, 5, 3});
	testMLPBasicHelper(2, {2, 1});
}


// ----------------------[ randomizeWeights tests ]----------------------------------

void Tester::testInitializeWeightsHelper(int rows, int cols, activations activationFunction, bool printInfo) {
	
	auto m = initializeWeights(rows, cols, activationFunction);
	assert(m.rows() == rows);
	assert(m.cols() == cols);
	
	initialization init = getInitializationByActivation(activationFunction);
	float multiplier = 3.0;
	float upperBound;	
	
	std::string initFunction = "";
	
	switch (init) {
		case initialization::glorot:
			upperBound = std::sqrt(multiplier * 2.0 / (rows + cols));
			initFunction = "Glorot";           
            break;
        case initialization::he:
            upperBound = std::sqrt(multiplier * 2.0 / rows);
            initFunction = "He"; 
            break;  
        case initialization::lecun:
			upperBound = std::sqrt(multiplier / rows);
			initFunction = "Lecun"; 
			break;
		default:
			upperBound = std::sqrt(multiplier * 2.0 / (rows + cols));
			initFunction = "undefined"; 
			break;
    }
		
	float lowerBound = -upperBound;
	
	if (printInfo) {
		std::cout << "---------- init: " << initFunction << " ; rows: " << rows << " ; cols: " << cols << " ----------" << std::endl;		
		std::cout << "Lowerbound: " << lowerBound << std::endl;
		std::cout << "Upperbound: " << upperBound << std::endl;
		std::cout << "-------------------------------------------" << std::endl;
	}
	
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (printInfo) { std::cout << "weight: " << m[i][j] << std::endl; }
			assert((m[i][j] >= lowerBound) && (m[i][j] <= upperBound));
		}
	}
}

void Tester::testInitializeWeights(bool printInfo) {
	testInitializeWeightsHelper(10, 5, activations::_reLU, printInfo);
	testInitializeWeightsHelper(3, 3, activations::_softmax, printInfo);
	testInitializeWeightsHelper(8, 2, activations::_sigmoid, printInfo);	
}


// ----------------------[ prediction tests ]----------------------------------

void Tester::predictTest() {    //works only with modified version od predict()
    CSVReader reader;
    MLP test(100);
    std::vector<vector> vals = {
            vector( {1, 2, 3}),
            vector({44, 5, 6}),
            vector( {7, 8, 9}),
            vector( {10, 114, 12}),
            vector({13, 14, 15})
    };
    assert (test.predict(vals) == std	::vector<int>({2, 0, 2, 1, 2}));
    reader.exportResults("./predicted.csv", test.predict(vals));
}


void Tester::feedForwardImpl(vector &input, std::vector<Layer> &_layers) {
	auto layerBelowValues = input;
	for (Layer& layer : _layers) {			
		vector innerPotential = (layerBelowValues * layer._weights) + layer._bias;
		layer._values = layer.useActivationFunction(innerPotential);
		//layer._valuesDerivatives = layer.useDerivedActivationFunction(innerPotential);			
		layerBelowValues = layer._values;
	}		
}

void Tester::setWeightsToValue(matrix &m, valueType value) {
	for (int i = 0; i < m.rows(); ++i) {
		for (int j = 0; j < m.cols(); ++j) {
			m[i][j] = value;
		}
	}
}

void Tester::setBiasToValue(vector &v, valueType value) {
	for (size_t i = 0; i < v.size(); ++i) {
		v[i] = value;
	}
}



void Tester::testFeedForwardHelper(MLP &mlp, vector &input) {
	
	std::cout << std::endl << "NEW MLP TEST" << std::endl;
	
	std::cout << "input: " << std::endl;
	printVector(input);
	std::cout << std::endl;
	
	for (auto& layer : mlp._layers) {
		setWeightsToValue(layer._weights, 1);
		setBiasToValue(layer._bias, 1);
	}
	
	feedForwardImpl(input, mlp._layers);
	
	int l = 1;
	for (auto& layer : mlp._layers) {
		std::cout << "weights in hidden layer " << l << ": " << std::endl;
		printMatrix(layer._weights);
		std::cout << std::endl;
		std::cout << "values in hidden layer " << l << ": " << std::endl;
		printVector(layer._values);
		std::cout << std::endl;
		++l;
	}	
	
	std::cout << "input: " << std::endl;
	printVector(input);
	std::cout << std::endl;
}

void Tester::testFeedForwardHelper_2(MLP &mlp, vector &input) {

	std::cout << std::endl << "NEW MLP TEST" << std::endl;

	for (auto& layer : mlp._layers) {
		setWeightsToValue(layer._weights, 0.5);
		setBiasToValue(layer._bias, 0.5);
	}

	feedForwardImpl(input, mlp._layers);
	
	int l = 1;
	for (auto& layer : mlp._layers) {
		std::cout << "weights in hidden layer " << l << ": " << std::endl;
		printMatrix(layer._weights);
		std::cout << std::endl;
		std::cout << "values in hidden layer " << l << ": " << std::endl;
		printVector(layer._values);
		std::cout << std::endl;		
		++l;
	}	
}

void Tester::testFeedForwardHelper_3(MLP &mlp, vector &input) {
	
	std::cout << std::endl << "NEW MLP TEST" << std::endl;

	feedForwardImpl(input, mlp._layers);	
	
	int l = 1;
	for (auto& layer : mlp._layers) {
		std::cout << "weights in hidden layer " << l << ": " << std::endl;
		printMatrix(layer._weights);
		std::cout << std::endl;
		std::cout << "bias in hidden layer " << l << ": " << std::endl;
		printVector(layer._bias);
		std::cout << std::endl;
		std::cout << "values in hidden layer " << l << ": " << std::endl;
		printVector(layer._values);
		std::cout << std::endl;		
		++l;
	}	
}

void Tester::testFeedForward() {
	MLP mlp(4);
	mlp.addLayer(2, activations::_reLU); // 9, 9
	mlp.addLayer(1, activations::_reLU); // 19
	std::vector<float> v {2,2,2,2};
	vector vec(v);
	testFeedForwardHelper(mlp, vec);
	
	MLP mlp_2(4);
	mlp_2.addLayer(4, activations::_reLU); // 5, 5, 5, 5
	mlp_2.addLayer(2, activations::_reLU); // 21, 21
	std::vector<float> v_2(4, 1);
	vector vec_2(v_2);
	testFeedForwardHelper(mlp_2, vec_2);
	
	MLP mlp_3(2);
	mlp_3.addLayer(2, activations::_reLU);
	mlp_3.addLayer(1, activations::_reLU);
	std::vector<float> v_3(2, 1);
	vector vec_3(v_3);
	testFeedForwardHelper_2(mlp_3, vec_3);
	
	MLP mlp_4(2);
	mlp_4.addLayer(2, activations::_reLU);
	mlp_4.addLayer(1, activations::_reLU);
	std::vector<float> v_4(2, 1);
	vector vec_4(v_4);
	testFeedForwardHelper_3(mlp_4, vec_4);
}

