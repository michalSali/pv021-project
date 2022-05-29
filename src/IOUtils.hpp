#ifndef PV021_FASHIONMNIST_PROJECT_IOUTILS_HPP
#define PV021_FASHIONMNIST_PROJECT_IOUTILS_HPP

#include "matrix.hpp"
#include <vector>
#include <string>

enum class normalization {
	_basicNormalize,
	_halfNormalize,
	_smallNormalize,
	_minMaxNormalize,
	_standardNormalize
};

class CSVReader {
		
	char _sep;
	vector readRowValues(const std::string &line) const;
    int readRowLabels(const std::string &line) const;

	void normalizeValues(normalization norm, std::vector<vector> &values) const;
		
    void basicNormalize(std::vector<vector> &values) const;
    void halfNormalize(std::vector<vector> &values) const;
    void smallNormalize(std::vector<vector> &values) const;
    void minMaxNormalize(std::vector<vector> &values) const;
    void standardNormalize(std::vector<vector> &values) const;

public:

	CSVReader(char sep = ',') : _sep(sep) {}	
	
	std::vector<vector> readCSVValues(const std::string &filepath, normalization norm = normalization::_standardNormalize);
    std::vector<int> readCSVLabels(const std::string &filepath);
    void exportResults(const std::string &filepath, const std::vector<int> &results);
};

template <typename T>
std::vector<T> readRow(const std::string &line, char sep);

template <typename T = valueType>
T getMinValue(const std::vector<T> &v);

template <typename T = valueType>
T getMaxValue(const std::vector<T> &v);

void displayAccuracy(const std::string &expectedValuesPath, const std::string &actualValuesPath);

#endif
