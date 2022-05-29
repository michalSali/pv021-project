#ifndef PV021_FASHIONMNIST_PROJECT_NETWORK_HPP
#define PV021_FASHIONMNIST_PROJECT_NETWORK_HPP

#include "matrix.hpp"
#include <vector>

enum activations {_reLU, _leakyReLu, _softmax, _sigmoid};
enum initialization {glorot, he, lecun};

initialization getInitializationByActivation(activations activationFunction);
matrix initializeWeights(int n, int m, activations activationFunction, bool uniformDistribution = true);
vector initializeBias(int dimension);


/* _values: output values of neurons computed by applying activation function to the neuron's inner potential
 * _valuesDerivatives: precomputed values of derivatives while applying feed forward, 
 *  				   computed by applying the derivative of the activation function to the neuron's inner potential
 * _weights: matrix of incoming weights from the layer below; initialized accordingly to activation function
 * _bias: bias of neurons, initialized to 0
 * _gradients: used when summing gradients of the error/loss function w.r.t. weights for all examples from batch
 * _biasGradients: used when summing gradients of the error/loss function w.r.t. bias for all examples from batch
 * _deltas: used for computing the derivative of error function w.r.t. y_j (output value of the neuron)
 *          when performing backpropagation, and for computing gradient of the error function w.r.t. weight/bias
 * _adamFirstMoment: when updating weights using Adam optimizer, we have to keep values of previous
 * 					   gradients/first moments for each weight
 * _adamSecondMoment: when updating weights using Adam optimizer, we have to keep values of previous
 * 							  squared gradients/second moments for each weight
 * _adamBiasFirstMoment: same as _adamFirstMoment, except for bias
 * _adamBiasSecondMoment: same as _adamSecondMoment, except for bias
 * _dimension: number of neurons in a layer
 * _activationFunction: activation function of the layer
 * _leakyReLUAlpha: value of the parameter for the leakyReLU activation fuction
 */
class Layer {

	friend class MLP; 
	friend class Tester;
	
	vector _values;
	vector _valuesDerivatives;
	
	matrix _weights;
	vector _bias;
	matrix _gradients;
	vector _biasGradients;
	
	vector _deltas;
	
	matrix _adamFirstMoment;
	matrix _adamSecondMoment;
	vector _adamBiasFirstMoment;
	vector _adamBiasSecondMoment;
	
	int _dimension;
	activations _activationFunction;

	valueType _leakyReLUAlpha;
	
	vector useActivationFunction(const vector &vec);
	vector useDerivedActivationFunction(const vector &vec);

public:
	
	Layer(int oldDimension, int dimension, activations activationFunction)
		: _values(dimension),
          _valuesDerivatives(dimension),
          _weights(initializeWeights(oldDimension, dimension, activationFunction)),
          _bias(initializeBias(dimension)),
          _gradients(oldDimension, dimension),
          _biasGradients(dimension),
          _deltas(dimension),
          _adamFirstMoment(oldDimension, dimension),
          _adamSecondMoment(oldDimension, dimension),
          _adamBiasFirstMoment(dimension),
          _adamBiasSecondMoment(dimension),
          _dimension(dimension),
          _activationFunction(activationFunction),
          _leakyReLUAlpha(0.01) {}

	const matrix& getWeights() const { return _weights; }
	const vector& getBias() const { return _bias; }
	const vector& getValues() const { return _values; }			
	int dimension() const { return _dimension; }
	size_t size() const { return _dimension; }
};


/* _inputValues: contains training examples (size depends on the chosen batchSize)
 * _inputLabels: contains labels corresponding to the training examples
 * _layers: contains layers of the network (input layer is not added to the vector of layers, 
 * 			the values from input layer are extracted directly from _inputValues)
 * _inputDimension: dimension of each example (dimension of 'vector' in _inputValues, e.g. 784 in case of fashionMNIST)
 * _learningRate: value of parameter used when updating the weights
 */
class MLP {

	friend class Tester;

	std::vector<vector> _inputValues;
	std::vector<int> _inputLabels;
	std::vector<Layer> _layers;
	int _inputDimension;
	valueType _learningRate;
	
	void feedForward(const vector &input);
	void backPropagate(size_t label);
	void updateWeights(int step);		
	
	void updateWeightNormal(int i, int j, int step, Layer& layer) const;
	void updateBiasNormal(int i, int step, Layer& layer) const;
	
	void updateWeightAdam(int i, int j, int step, Layer& layer) const;
	void updateBiasAdam(int i, int step, Layer& layer) const;
	
	vector getMLPOutput();

	void setLeakyReLUAlpha(valueType alpha);

public:
	
	/* sets dimension of the input (number of neurons in the input layer) */
	explicit MLP (int inputDimension)
	    : _inputDimension(inputDimension),
	      _learningRate(0.01) {}
	
	void addLayer(int dimension, activations activationFunction);
	const std::vector<Layer>& getLayers() { return _layers; }
	
	void train(const std::vector<vector> &inputValues, const std::vector<int> &inputLabels, valueType learningRate, int epochs, int batchSize);	
	std::vector<int> predict(const std::vector<vector> &testValues);			
};

void shuffleIndexes(std::vector<int> &indexes);

#endif
