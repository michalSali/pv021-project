#include "IOUtils.hpp"
#include "network.hpp"
#include "tests.hpp"

int main() {

    CSVReader reader;
    auto trainValues = reader.readCSVValues("../data/fashion_mnist_train_vectors.csv");
    //auto trainValues = reader.readCSVValues("./data/fashion_mnist_train_vectors.csv");  // for submission
    auto trainLabels = reader.readCSVLabels("../data/fashion_mnist_train_labels.csv");
    //auto trainLabels = reader.readCSVLabels("./data/fashion_mnist_train_labels.csv");  // for submission
    

    MLP network(784);
    network.addLayer(128, activations::_leakyReLu);
    network.addLayer(32, activations::_leakyReLu);
    network.addLayer(10, activations::_softmax);


    network.train(trainValues, trainLabels, 0.001, 15, 512);


    auto testValues = reader.readCSVValues("../data/fashion_mnist_test_vectors.csv");
    //auto testValues = reader.readCSVValues("./data/fashion_mnist_test_vectors.csv");  // for submission

    auto predictedTestLabels = network.predict(testValues);
    auto predictedTrainLabels = network.predict(trainValues);
    
	reader.exportResults("../actualPredictions", predictedTestLabels);
	reader.exportResults("../trainPredictions", predictedTrainLabels);
    //reader.exportResults("./actualPredictions", predictedTestLabels);     // for submission
    //reader.exportResults("./trainPredictions", predictedTrainLabels);    // for submission

    /*
    Tester tester;
    for (int i = 0; i < 20; ++i) {
        tester.gridSearch(activations::_leakyReLu, {0.01}, {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}, {512}, {0.01});
    }
    */

    return 0;
}

