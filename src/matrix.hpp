#ifndef PV021_FASHIONMNIST_PROJECT_MATRIX_HPP
#define PV021_FASHIONMNIST_PROJECT_MATRIX_HPP

/* the linear algebra classes (vector and matrix), and activation functions used in the network */

#include <vector>
#include <cstdint> // INT32_MAX

// used across whole implementation, should be float or double
using valueType = float;


// ---------------------------[ VECTOR ]--------------------------------

class vector {

	int _dimension;
	std::vector<valueType> _values;
	
public:
	
	vector() = default;

	explicit vector( int dimension ) 
		: _dimension(dimension), _values(dimension, 0) {}
		
	explicit vector( const std::vector<valueType> &vec )
		: _dimension(vec.size()), _values(vec) {}

	friend vector operator+( const vector &a, const vector &b );
	friend vector operator-( const vector &a, const vector &b );
	friend vector operator*( const vector &a, valueType scalar );
	friend vector operator*( valueType scalar, const vector &a );
	friend valueType operator*( const vector &a, const vector &b );

	valueType operator[]( int i ) const { return _values[i]; }
	valueType& operator[]( int i ) { return _values[i]; }
	int dimension() const { return _dimension; }
	
	size_t size() const { return _dimension; }
	
	void pop_back() { _values.pop_back(); }
	valueType back() const { return _values.back(); }
	void emplace_back(valueType value) { _values.emplace_back(value); }
	const std::vector<valueType>& getValues() const { return _values; } 
	
};



// ----------------------------[ MATRIX ]-------------------------------

class matrix {
	
	int _rows;
	int _cols;
	std::vector<vector> _values;

public:

	matrix( int rows, int cols ) 
		: _rows(rows), _cols(cols), 
		  _values(rows, vector(cols)) {}
		  
	explicit matrix( const std::vector<vector> &vec )
		: _rows(vec.size()), _cols(vec[0].dimension()),
		  _values(vec) {}

	friend matrix operator+( const matrix &a, const matrix &b );
	friend vector operator*( const matrix &m, const vector &v );
	friend vector operator*( const vector &v, const matrix &m );
	
	const vector& operator[]( int i ) const { return _values[i]; }
	vector& operator[]( int i ) { return _values[i]; }
	
	vector row( int n ) const;
	vector col( int n ) const;

	int cols() const { return _cols; }
	int rows() const { return _rows; }	
	size_t size() const { return _rows; }
	
	const std::vector<vector>& getValues() const { return _values; }
	
};

vector plusMinusVectors( const vector &a, const vector &b, int sign );


// ------------------------[ MATH FUNCTIONS ]---------------------------

valueType sigmoid(valueType x);
valueType reLu(valueType x);
valueType leakyReLu(valueType x, float alpha);

vector sigmoid(const vector &inputVector);
vector reLu(const vector &inputVector);
vector leakyReLu(const vector &inputVector, float alpha);
vector softmax(const vector &inputVector);
vector sigmoidDerivative(const vector &inputVector);
vector reLuDerivative(const vector &inputVector);
vector leakyReLuDerivative(const vector &inputVector, float alpha);
vector softmaxDerivative(const vector &inputVector);

vector sigmoidDerivative_fromInnerPotential(const vector &inputVector);
vector sigmoidDerivative_fromValues(const vector &inputVector);

void printVector(const vector &v);
void printMatrix(const matrix &m);


#endif
