# pragma once

#include <vector>

class LinearRegression {
private:
    std::vector<double> X;
    std::vector<double> Y;
    double W;
    double b;
public:
    LinearRegression(std::vector<double> X, std::vector<double> Y);  // constructor
    void train(double lr, int epoch, int batchSize);
    double forward(double X);
    double squareLoss(std::vector<double> Y, std::vector<double> Y_hat);
    void gradientDescent(std::vector<double> X, std::vector<double> Y, std::vector<double> Y_hat, double lr);
};

