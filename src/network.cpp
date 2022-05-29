#include "network.hpp"
#include "matrix.hpp"
#include <vector>
#include <random>  // uniform_real_distribution
#include <algorithm>  // shuffle
#include <cmath>  // sqrt, pow


// ----------------------------[ top-level MLP methods + related methods ]-----------------------------------

/* 
 * The whole traning data is split into batches of given size, and training examples in a batch are chosen by shuffling
 * the data (indices) and then choosing the minibatches sequentially. We only use batches of given size, therefore we omit
 * a few examples that do not fill a complete batch. Omitting few examples is negligible, as it's only a very small
 * fraction of the whole dataset (at most [(batchSize - 1) / n_examples]), and the indices/data is shuffled each time anyway.
 * 
 * For the training itself we use Adam optimizer.
 */
void MLP::train(const std::vector<vector> &inputValues, const std::vector<int> &inputLabels, valueType learningRate=0.01, int epochs = 10, int batchSize = 64) {

	_learningRate = learningRate;

	std::vector<vector> batchValues(batchSize);
	std::vector<int> batchLabels(batchSize);	

	int n_examples = inputValues.size();
	int batchCount = n_examples / batchSize;
	
	// vector of indices that is shuffled and used to choose training examples
	std::vector<int> indexes(n_examples);
	for (int i = 0; i < n_examples; ++i) {
		indexes[i] = i;
	}

    for (int i = 0; i < epochs; ++i) {
        shuffleIndexes(indexes);        
        
		for (int j = 0; j < batchCount; ++j) {
			// fill the batchValues and batchLabels with coresponding data
			for (int k = 0; k < batchSize; ++k) {
				int index = indexes[j*batchSize + k];
				batchValues[k] = inputValues[index];
				batchLabels[k] = inputLabels[index];
			}

			_inputValues = batchValues;
			_inputLabels = batchLabels;

            // number of steps: 1 epoch = "batchCount" minibatches, each minibatch is one step; we add +1 so that we start from step = 1
			updateWeights(i * batchCount + j + 1);
		}
	}
}

/*
 * Makes prediction according to given test values.
 * Feedforwards the data through the network, selects category from output layer 
 * with the highest probability and returns vector with predicted labels.
 */
std::vector<int> MLP::predict(const std::vector<vector> &testValues) {

    std::vector<int> predictions;
    vector currentMLPOutput;

    for (const auto& input : testValues) {

        feedForward(input);
        currentMLPOutput = getMLPOutput();
		
        valueType maxValue = -1;
        int currentPrediction = 0;

        for (size_t i = 0; i < currentMLPOutput.size(); ++i) {
			if (currentMLPOutput[i] >= maxValue) {
                maxValue = currentMLPOutput[i];
                currentPrediction = i;
			}
		}
		predictions.emplace_back(currentPrediction);
    }

    return predictions;
}


/*
 * Instead of shuffling the data itself (inputValues and inputLabels), we can perform the random choice
 * of training examples for minibatch by using a shuffled vector of indexes. 
 * 
 * Note: The seed for the generator has been determined by std::random_device, however it has been fixed
 *       to avoid extremely bad RNG - throughout the last stages of testing, the accuracy with the current parameters,
 *       architecture, etc. has never sunk below 88.5% AFAIK.
 */
void shuffleIndexes(std::vector<int> &indexes) {

	std::random_device rd;
	// std::mt19937 generator(rd());
	std::mt19937 generator(0);
	std::shuffle(indexes.begin(), indexes.end(), generator);
}


/*
 * get values from output layer
 */
vector MLP::getMLPOutput() {
    return _layers.back().getValues();
}


/*
 * Adds a layer to the network.
 * We need to get number of neurons from the layer below to create matrix of weights of correct dimensions.
 */
void MLP::addLayer(int dimension, activations activationFunction) {

    int oldDimension;
	if (_layers.empty()) {
		oldDimension = _inputDimension;
	} else {
		oldDimension = _layers.back().dimension();
	}

	_layers.emplace_back(oldDimension, dimension, activationFunction);
}


// ----------------------------[ core network-training methods ]----------------------------------

/* First we compute the backpropagated value in the output layer, and then propagate this value
 *    to the layers below. These backpropagated values (_deltas) are then used to compute the gradient
 *    of the error function w.r.t. to weights/biases.
 * In this case, the chosen error/loss function is Cross-Entropy and the activation function in
 *    the output layer is softmax. */
void MLP::backPropagate(size_t inputLabel) {

	// compute for the output layer
    for (size_t i = 0; i < _layers.back().size(); ++i) {
		
		valueType inputLabelHotEncoded;
		if (inputLabel == i) {
			inputLabelHotEncoded = 1.0;
		} else {
			inputLabelHotEncoded = 0.0;
		}

		_layers.back()._deltas[i] = _layers.back()._values[i] - inputLabelHotEncoded;		
	}
				
	// compute for each layer from top to bottom (omitting output layer), for each neuron in the layer above (dense)
	for (int l = (int)_layers.size() - 2; l >= 0; --l) {
		for (size_t i = 0; i < _layers[l].size(); ++i) {  
			valueType deltasSum = 0;
			for (size_t j = 0; j < _layers[l+1].size(); ++j) {											
				deltasSum += _layers[l+1]._deltas[j] * _layers[l+1]._weights[i][j];
			}
			_layers[l]._deltas[i] = deltasSum * _layers[l]._valuesDerivatives[i];
		}
	}
}


/* For each training example in the batch (_inputValues), we perform feedforward and backpropagation, and calculate the gradient
 *    of the error function w.r.t. to each weight and bias. We sum the gradients for all examples in the batch.
 * We then update the weights and biases using the sum of gradients. Currently we use the Adam optimizer, but it's also possible
 *    to update the weights/biases in a simple way using 'updateWeightsNormal' / 'updateBiasNormal'. */
void MLP::updateWeights(int step) {

    // ---------------[ get the sum of gradients for all examples in the batch ]--------------
	for (size_t k = 0; k < _inputValues.size(); ++k) {
		
		auto inputValue = _inputValues[k];
		auto inputLabel = _inputLabels[k];
		feedForward(inputValue);
		backPropagate(inputLabel);

		for (size_t l = 0; l < _layers.size(); ++l) {

			auto layerBelowValues = (l == 0) ? inputValue : _layers[l - 1]._values;
			
            for (int i = 0; i < _layers[l]._weights.rows(); ++i) {				
				for (int j = 0; j < _layers[l]._weights.cols(); ++j) {							
                    _layers[l]._gradients[i][j] += _layers[l]._deltas[j] * layerBelowValues[i];					
				}
			}
			
			for (size_t j = 0; j < _layers[l].size(); ++j) {
				_layers[l]._biasGradients[j] += _layers[l]._deltas[j]; // y_i is always 1
			}
		}
	}

	// -------------------[ update weights and bias using Adam optimizer ]--------------------
	for (auto &layer : _layers) {
		
		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < layer._weights.rows(); ++i) {
			for (int j = 0; j < layer._weights.cols(); ++j) {
				updateWeightAdam(i, j, step, layer);
				layer._gradients[i][j] = 0;
			}
		}

		for (size_t i = 0; i < layer._bias.size(); ++i) {
			updateBiasAdam(i, step, layer);
			layer._biasGradients[i] = 0;
		}
	}
}


/* Input to this function is a single training example (e.g. a vector of 784 'pixels' in case of fashionMNIST). 
 * Gradually for each layer we compute the output values of all neurons (_values) by applying the activation function
 * to the inner potential of each neuron. We also compute derived values by applying the derivate of
 * the activation function (_valuesDerivatives), which are used in backpropagation. */
void MLP::feedForward(const vector &input) {

	vector innerPotential;
	for (size_t i = 0; i < _layers.size(); ++i) {

		if (i == 0) {
			innerPotential = (input * _layers[i]._weights) + _layers[i]._bias;
		} else {
			innerPotential = (_layers[i-1]._values * _layers[i]._weights) + _layers[i]._bias;
		}
		
		_layers[i]._values = _layers[i].useActivationFunction(innerPotential);
		_layers[i]._valuesDerivatives = _layers[i].useDerivedActivationFunction(innerPotential);		
	}	
}


// ------------------------------[ weights/bias initialization and related functions ]------------------------------------

/*
 * Gets initialization according to the activation function
 * (more activation functions can be using the same initialization).
 */
initialization getInitializationByActivation(activations activationFunction) {

	switch (activationFunction) {
		case activations::_reLU:
        case activations::_leakyReLu:
			return initialization::he;

		case activations::_softmax:
        case activations::_sigmoid:
        default:
			return initialization::glorot;
	}
}


/*
 * Initialize weights in matrix of n x m dimension according to its activation function.
 * 
 * Note: The seed for the generator has been determined by std::random_device, however it has been fixed
 * to avoid extremely bad RNG - throughout the last stages of testing, the accuracy with the current parameters,
 * architecture, etc. has never sunk below 88.5%.
 */
matrix initializeWeights(int n, int m, activations activationFunction, bool uniformDistribution) {

	matrix weights(n, m);

	initialization init = getInitializationByActivation(activationFunction);
	valueType multiplier = (uniformDistribution) ? 3.0 : 1.0;
	valueType upperBound;

	// set bound according to initialization
	switch (init) {
		case initialization::glorot:
			upperBound = std::sqrt(multiplier * 2.0 / (n + m));
            break;
        case initialization::he:
            upperBound = std::sqrt(multiplier * 2.0 / n);
            break;
        case initialization::lecun:
			upperBound = std::sqrt(multiplier / n);
			break;
    }
	valueType lowerBound = -upperBound;

	std::random_device rd;
	// std::mt19937 generator(rd());
	std::mt19937 generator(0);
	std::uniform_real_distribution<valueType> distribution(lowerBound, upperBound);

	//set weights
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			weights[i][j] = distribution(generator);
		}
	}

	return weights;
}


vector initializeBias(int dimension) {

    vector bias(dimension); // initialize vector to 0
    return bias;
}


// -------------------------------------[ methods for updating weights / optimizers ]---------------------------------------

/*
 * Updates weights with gradient, provides only very basic correction.
 * Not used in actual network.
 */
void MLP::updateWeightNormal(int i, int j, int step, Layer& layer) const {

	valueType gradient = layer._gradients[i][j];
	layer._weights[i][j] -= (_learningRate / step) * gradient;	
}


/*
 * Updates weights with gradient, provides only very basic correction.
 * Not used in actual network.
 */
void MLP::updateBiasNormal(int i, int step, Layer& layer) const {

	valueType gradient = layer._biasGradients[i];
	layer._bias[i] -= (_learningRate / step) * gradient;	
}


/*
 * Updates the weights of the model using Adam optimizer.
 */
void MLP::updateWeightAdam(int i, int j, int step, Layer& layer) const {

	valueType beta_1 = 0.9;
	valueType beta_2 = 0.999;
	valueType epsilon = 1e-8;  // epsilon (or delta) is a smoothing term to avoid division by 0

	valueType beta_1_t = std::pow(beta_1, step);
	valueType beta_2_t = std::pow(beta_2, step);

	valueType gradient = layer._gradients[i][j];

	layer._adamFirstMoment[i][j] = (beta_1) * layer._adamFirstMoment[i][j] + (1 - beta_1) * gradient;
	layer._adamSecondMoment[i][j] = (beta_2) * layer._adamSecondMoment[i][j] + (1 - beta_2) * (gradient * gradient);

	auto biasCorrectedPastGradient = layer._adamFirstMoment[i][j] / (1 - beta_1_t);
	auto biasCorrectedPastSquaredGradient = layer._adamSecondMoment[i][j] / (1 - beta_2_t);

    layer._weights[i][j] -= _learningRate / (std::sqrt(biasCorrectedPastSquaredGradient) + epsilon) * biasCorrectedPastGradient;
}
	

/*
 * Updates bias of model using Adam optimizer.
 */
void MLP::updateBiasAdam(int i, int step, Layer& layer) const {

	valueType beta_1 = 0.9;
	valueType beta_2 = 0.999;
	valueType epsilon = 1e-8;
	valueType beta_1_t = std::pow(beta_1, step);
	valueType beta_2_t = std::pow(beta_2, step);
	
	valueType gradient = layer._biasGradients[i];
	
	layer._adamBiasFirstMoment[i] = (beta_1) * layer._adamBiasFirstMoment[i] + (1 - beta_1) * gradient;
    layer._adamBiasSecondMoment[i] = (beta_2) * layer._adamBiasSecondMoment[i] + (1 - beta_2) * (gradient * gradient);

    auto biasCorrectedPastGradient = layer._adamBiasFirstMoment[i] / (1 - beta_1_t);
    auto biasCorrectedPastSquaredGradient = layer._adamBiasSecondMoment[i] / (1 - beta_2_t);

    layer._bias[i] -= _learningRate / (std::sqrt(biasCorrectedPastSquaredGradient) + epsilon) * biasCorrectedPastGradient;
}


// --------------------------------------[ other functions ]-----------------------------------------

/*
 * apply activation function to given vector
 */
vector Layer::useActivationFunction(const vector &vec) {

    switch (_activationFunction) {
		case activations::_reLU:
			return reLu(vec);
	    case activations::_leakyReLu:
            return leakyReLu(vec, _leakyReLUAlpha);
		case activations::_softmax:		    
			return softmax(vec);
		case activations::_sigmoid:
			return sigmoid(vec);
		default:
			return vector(vec.size());
	}
}


/*
 * apply derived activation function to given vector
 */
vector Layer::useDerivedActivationFunction(const vector &vec) {

    switch (_activationFunction) {
        case activations::_reLU:
            return reLuDerivative(vec);
        case activations::_leakyReLu:
            return leakyReLuDerivative(vec, _leakyReLUAlpha);
        case activations::_softmax:
            return softmaxDerivative(_values);
        case activations::_sigmoid:
            return sigmoidDerivative_fromInnerPotential(vec);
        default:
            return vector(vec.size());
    }
}

/*
 * set alpha constant for leaky ReLu
 */
void MLP::setLeakyReLUAlpha(valueType alpha) {

    for (auto &layer : _layers) {
        layer._leakyReLUAlpha = alpha;
    }
}


