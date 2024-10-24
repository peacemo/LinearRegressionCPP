#include "../header/utils.h"
#include "../header/linearRegression.h"
#include <iostream>

int main() {
    std::vector<std::pair<double, double>> samples = Utils::generateSamples(1000, 2, 1, true);  // generate samples
    
    std::vector<double> X;  // extract samples into X, Y
    std::vector<double> Y;
    for (const auto& sample : samples) {
        X.push_back(sample.first);
        Y.push_back(sample.second);
    }

    double W = 0.1;
    double b = 0.1;
    double lr = 0.001;
    int epoch = 100;
    int batchSize = 32;

    LinearRegression model(X, Y);
    model.train(lr, epoch, batchSize);

    return 0;
}
