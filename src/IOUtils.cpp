#include "IOUtils.hpp"
#include <fstream>  // ifstream, ofstream
#include <string>   // stoi
#include <iostream>
#include <cmath>


/* Generic function for transforming a single row/line into
 * a vector of given type. */
template <typename T>
std::vector<T> readRow(const std::string &line, char sep) {
	
	std::vector<T> row;

    if (line.empty()) {
		T error = -1;
        return {error};
    }

    size_t index = 0;
    char c;
    std::string currentString;
    T currentValue;

    while (index < line.length()) {
        c = line[index];
        if (c == sep) {
            currentValue = std::stoi(currentString);
            row.emplace_back(currentValue);
            currentString = "";
        } else {
            currentString += c;
        }

        ++index;
    }

    if (!currentString.empty()) {
        currentValue = std::stoi(currentString);
        row.emplace_back(currentValue);
    }
    
    return row;
}

/* Generic function for retrieving maximum value
 * of a given type from a vector. */
template <typename T>
T getMaxValue(const std::vector<T> &v) {
	
	T max = -INFINITY;
	
	for (size_t i = 0; i < v.size(); ++i) {
		if (v[i] > max) {
			max = v[i];
		}
	}
	
	return max;
}


/* Generic function for retrieving minimum value
 * of a given type from a vector. */
template <typename T>
T getMinValue(const std::vector<T> &v) {
	
	T min = INFINITY;
	
	for (size_t i = 0; i < v.size(); ++i) {
		if (v[i] < min) {
			min = v[i];
		}
	}
	
	return min;
}


// -----------------------------------[ labels ]----------------------------------------

/* Can be easily adjusted to parse different datasets */
int CSVReader::readRowLabels(const std::string &line) const {
	
	std::vector<int> row = readRow<int>(line, _sep);	
    return row.front();
}


std::vector<int> CSVReader::readCSVLabels(const std::string &filepath) {

    std::vector<int> values;
    std::ifstream is(filepath);
    std::string line;

    while (std::getline(is, line)) {
        values.emplace_back(readRowLabels(line));
    }
    is.close();
    return values;
}



// ------------------------------[ values ]---------------------------------------

vector CSVReader::readRowValues(const std::string &line) const {
	
	std::vector<valueType> row = readRow<valueType>(line, _sep);	
	return vector(row);
}	


std::vector<vector> CSVReader::readCSVValues(const std::string &filepath, normalization norm) {
	
	std::vector<vector> values;
	std::ifstream is(filepath);
	std::string line;
		
	while (std::getline(is, line)) {
		values.emplace_back(readRowValues(line));
	}
	is.close();
	
	normalizeValues(norm, values);
		
	return values;
}


// ---------------------------------[ normalization, scaling ]---------------------------------------

void CSVReader::normalizeValues(normalization norm, std::vector<vector> &values) const {
	
	switch (norm) {
		case normalization::_basicNormalize:
			basicNormalize(values);
			break;
		case normalization::_halfNormalize:
			halfNormalize(values);
			break;
		case normalization::_standardNormalize:
			standardNormalize(values);
			break;
		case normalization::_smallNormalize:
			smallNormalize(values);
			break;
		case normalization::_minMaxNormalize:
			minMaxNormalize(values);
			break;
		default:
			// do nothing
			break;
	}
}


/* Implementation of a StandardScaler */
void CSVReader::standardNormalize(std::vector<vector> &values) const {

	for (size_t col = 0; col < values[0].size(); ++col) {
		valueType sum = 0.0;
		valueType mean = 0.0;
		valueType variance = 0.0;
		valueType stddev = 0.0;
		
		for (size_t row = 0; row < values.size(); ++row) {
			sum += values[row][col];
		}
        
		mean = sum / values.size();
		for (size_t row = 0; row < values.size(); ++row) {
			variance += std::pow(values[row][col] - mean, 2);
		}
		variance = variance / values.size();
		stddev = std::sqrt(variance);

		for (size_t row = 0; row < values.size(); ++row) {
			values[row][col] = (values[row][col] - mean) / stddev;
		}
	}
	
}


/* Normalizing by squeezing pixel values into range [-0.5 ; 0.5] */
void CSVReader::halfNormalize(std::vector<vector> &values) const {	
	for (size_t i = 0; i < values.size(); ++i) {
		for (size_t j = 0; j < values[0].size(); ++j) {
			values[i][j] = (values[i][j] - 127.5) / 127.5;
		}	
	}
}


/* Normalizing by squeezing pixel values into range [0 ; 1] */
void CSVReader::basicNormalize(std::vector<vector> &values) const {    
    for (size_t i = 0; i < values.size(); ++i) {
		for (size_t j = 0; j < values[0].size(); ++j) {
			values[i][j] = values[i][j] / 255;
		}	
	}
}


/* Implementation of a MinMaxScaler with range [-0.5 ; 0.5].
 * Note: Can be implemented by calling minMaxNormalize and then iterating
 * over all values again and subtracting 0.5, which is slightly 
 * less effective, however. */
void CSVReader::smallNormalize(std::vector<vector> &values) const {
	
	for (size_t col = 0; col < values[0].size(); ++col) {
		
		std::vector<valueType> column(values.size());
		
		for (size_t row = 0; row < values.size(); ++row) {
			column[row] = values[row][col];
		}
				
		auto maxVal = getMaxValue<valueType>(column);
		auto minVal = getMinValue<valueType>(column);

		for (size_t row = 0; row < values.size(); ++row) {
			values[row][col] = ((values[row][col] - minVal) / (maxVal - minVal)) - 0.5;
		}   
	}	   
}


/* Implementation of a MinMaxScaler with range [0 ; 1]  */
void CSVReader::minMaxNormalize(std::vector<vector> &values) const {
	
    for (size_t col = 0; col < values[0].size(); ++col) {
		
		std::vector<valueType> column(values.size());
		
		for (size_t row = 0; row < values.size(); ++row) {
			column[row] = values[row][col];
		}
				
		auto maxVal = getMaxValue<valueType>(column);
		auto minVal = getMinValue<valueType>(column);

		for (size_t row = 0; row < values.size(); ++row) {
			values[row][col] = (values[row][col] - minVal) / (maxVal - minVal);
		}   
	}	   
}


// -----------------------------[ other functions / methods ]-----------------------------------

/* Used to export predictions/labels */
void CSVReader::exportResults(const std::string &filepath, const std::vector<int> &results) {
	std::ofstream os(filepath);
	for (const auto &value : results) {
		os << value << std::endl;
	}
	os.close();
}


/* Can be used 'internally' to display model accuracy by comparing expected labels
 * and actual labels, similarly to python_evaluator. */
void displayAccuracy(const std::string &expectedValuesPath, const std::string &actualValuesPath) {
	
	CSVReader reader;	
	auto expectedLabels = reader.readCSVLabels(expectedValuesPath);
	auto actualLabels = reader.readCSVLabels(actualValuesPath);
	
	if (expectedLabels.size() != actualLabels.size()) {
		std::cout << "Files specified by filepaths have different size." << std::endl;
		return;
	}
	
	size_t size = expectedLabels.size();
	int correct = 0;
	
	for (size_t i = 0; i < size; ++i) {
		if (expectedLabels[i] == actualLabels[i]) {
			++correct;
		}
	}
	
	double acc = (double)correct / size * 100;
	printf("Accuracy of the model is: %d / %ld = %.2f %%\n", correct, size, acc);	
}
