/* implementation of methods for matrix and vector */
#include <cmath>
#include <iostream>
#include "matrix.hpp"


// ----------------------------[ vector ]-------------------------------

// a and b are expected to have the same dimension
vector plusMinusVectors( const vector &a, const vector &b, int sign ) {
	int dimension = a.dimension();
	std::vector<valueType> newValues(dimension);

	for (int i = 0; i < dimension; ++i) {
		newValues[i] = a[i] + (b[i] * sign);
	}

	return vector(newValues);	
}


vector operator+( const vector &a, const vector &b ) {
	return plusMinusVectors(a, b, 1);
}


vector operator-( const vector &a, const vector &b ) {
	return plusMinusVectors(a, b, -1);
}


vector operator*( const vector &a, valueType scalar ) {	
	int dimension = a.dimension();
	std::vector<valueType> newValues(dimension);

	for (int i = 0; i < dimension; ++i) {
		newValues[i] = a[i] * scalar;
	}
	
	return vector(newValues);	
}


vector operator*( valueType scalar, const vector &a ) {
	return a * scalar;
}


// a and b are expected to have the same dimension
valueType operator*( const vector &a, const vector &b ) {
	int dimension = a.dimension();
	valueType dotProduct = 0;

	for (int i = 0; i < dimension; ++i) {
		dotProduct += a[i] * b[i];
	}
		
	return dotProduct;
}


// ---------------------------- MATRIX -----------------------------------------------------


// a and b are expected to have the same dimensions
matrix operator+( const matrix &a, const matrix &b ) {
	
	int rows = a.rows();
	int cols = a.cols();
	matrix newMatrix(rows, cols);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0 ; j < cols; ++j) {
			newMatrix[i][j] = a[i][j] + b[i][j];
		}
	}

	return newMatrix;
}


vector operator*( const matrix &m, const vector &v ) {
	vector result(m.rows());
	for (int i = 0; i < m.rows(); ++i) {
		result[i] = m.row(i) * v;
	}
	
	return result;
}


vector operator*( const vector &v, const matrix &m ) {
	vector result(m.cols());
	int i;
#pragma omp parallel for num_threads(16) default(shared) private(i)
    for (i = 0; i < m.cols(); ++i) {
		result[i] = v * m.col(i);
	}
	
	return result;
}


vector matrix::row( int n ) const {
	return _values[n];
}


vector matrix::col( int n ) const {
	vector newColumn(_rows);

	for (int i = 0; i < _rows; ++i) {
		newColumn[i] = _values[i][n]; 
	}
	return newColumn;
}


// ================[ activation functions and their derivatives ]================


// --------------------------[ sigmoid ]-----------------------------

valueType sigmoid(valueType x) {
    return (valueType)1.0 / ((valueType)1.0 + std::exp(-x));
}


vector sigmoid(const vector &inputVector) {
    
    int dimension = inputVector.dimension();
    std::vector<valueType> result(dimension);
	
	for (int i = 0; i < dimension; ++i) {
		result[i] = sigmoid(inputVector[i]);
	}
 
    return vector(result);
}


vector sigmoidDerivative_fromInnerPotential(const vector &inputVector) {
	
	int dimension = inputVector.dimension();
    std::vector<valueType> result(dimension);
	
	for (int i = 0; i < dimension; ++i) {
		valueType y = sigmoid(inputVector[i]);
		result[i] = y * (1 - y);
	}
 
    return vector(result);
}


vector sigmoidDerivative_fromValues(const vector &inputVector) {
	
	int dimension = inputVector.dimension();
    std::vector<valueType> result(dimension);
	
	for (int i = 0; i < dimension; ++i) {
		valueType y = inputVector[i];
		result[i] = y * (1 - y);
	}
 
    return vector(result);
}


// ----------------------------[ reLU & leakyReLU ]---------------------------------

valueType reLu(valueType x) {
	if (x < 0) {
		return 0;
	}	
	return x;    
}

valueType leakyReLu(valueType x, float alpha) {
    if (x < 0) {
        return x * alpha;
    }
    return x;
}


vector reLu(const vector &inputVector) {
	
	int dimension = inputVector.dimension();
    std::vector<valueType> result(dimension);
	
	for (int i = 0; i < dimension; ++i) {
		result[i] = reLu(inputVector[i]) ;
	}
 
    return vector(result);
}


vector leakyReLu(const vector &inputVector, float alpha) {

	int dimension = inputVector.dimension();
    std::vector<valueType> result(dimension);

	for (int i = 0; i < dimension; ++i) {
		result[i] = leakyReLu(inputVector[i], alpha) ;
	}

    return vector(result);
}


vector reLuDerivative(const vector &inputVector) {
	
	int dimension = inputVector.dimension();
    std::vector<valueType> result(dimension);
	
	for (int i = 0; i < dimension; ++i) {
		result[i] = (inputVector[i] <= 0) ? 0 : 1;
	}
 
    return vector(result);	
}


vector leakyReLuDerivative(const vector &inputVector, float alpha) {

	int dimension = inputVector.dimension();
    std::vector<valueType> result(dimension);

	for (int i = 0; i < dimension; ++i) {
		result[i] = (inputVector[i] <= 0) ? alpha : 1;
	}

    return vector(result);
}



// ----------------------------[ softmax ]-------------------------------

/* Implementation of a numerically stable softmax */
vector softmax(const vector &inputVector) {

    valueType maxValue = -INFINITY;
    for (size_t i = 0; i < inputVector.size(); i++) {
        if (inputVector[i] > maxValue) {
            maxValue = inputVector[i];
        }
    }

    if (maxValue < 0) {
        maxValue *= -1.0;
    }

    valueType sum = 0.0;
    for (size_t i = 0; i < inputVector.size(); i++) {
        sum += expf(inputVector[i] - maxValue);
    }

    vector outputVector(inputVector.size());

    valueType offset = maxValue + logf(sum);
    for (size_t i = 0; i < inputVector.size(); ++i) {
        outputVector[i] = expf(inputVector[i] - offset);
    }

    return outputVector;
}


vector softmaxDerivative(const vector &inputVector) {

    std::vector<valueType> numberVector;
    numberVector.resize(inputVector.size());

    auto resultIt = numberVector.begin();
    for (auto it = inputVector.getValues().begin(); it != inputVector.getValues().end(); it++, resultIt++) {
        *resultIt = *it * (1 - *it);
    }

    return vector(numberVector);
}


// -----------------------[ other functions ]---------------------------

void printVector(const vector &vec) {
    for (int i = 0; i < vec.dimension(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}


void printMatrix(const matrix &m) {
	for (int i = 0; i < m.rows(); ++i) {
		printVector(m[i]);
	}
}
