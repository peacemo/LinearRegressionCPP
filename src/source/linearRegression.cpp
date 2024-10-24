#include "../header/linearRegression.h"
#include <random>
#include <algorithm>
#include <iostream>

LinearRegression::LinearRegression(std::vector<double> X, std::vector<double> Y) {
    std::normal_distribution<> d(0, 1);
    std::default_random_engine gen;  // random number generator for initial params. 

    this->X = X;
    this->Y = Y;
    this->W = d(gen);
    this->b = d(gen);
}

/**
 * Forward pass to calculate the predicted value.
 * 
 * @param X The input value.
 * @return The predicted value.
 */
double LinearRegression::forward(double X) {
    return this->W * X + this->b;
}

/**
 * Calculate the square loss between the actual and predicted values.
 * 
 * @param Y_bs The actual values.
 * @param Y_hat The predicted values.
 * @return The average square loss.
 */
double LinearRegression::squareLoss(std::vector<double> Y_bs, std::vector<double> Y_hat) {
    double sum_loss = 0;
    for (int i = 0; i < Y_hat.size(); ++i) {
        sum_loss += 0.5 * (Y_hat[i] - Y_bs[i]) * (Y_hat[i] - Y_bs[i]);
    }
    return sum_loss / Y_hat.size();
}

/**
 * Gradient descent to update the model parameters.
 * 
 * @param X_bs The input values.
 * @param Y_bs The actual values.
 * @param Y_hat The predicted values.
 * @param lr The learning rate.
 */
void LinearRegression::gradientDescent(std::vector<double> X_bs, std::vector<double> Y_bs, std::vector<double> Y_hat, double lr) {
    double sum_w_grad = 0;
    double sum_b_grad = 0;

    for (int i = 0; i < Y_hat.size(); ++i) {
        sum_w_grad += X_bs[i] * (Y_hat[i] - Y_bs[i]);
        sum_b_grad += (Y_hat[i] - Y_bs[i]);
    }

    double grad_w = sum_w_grad / Y_hat.size();
    double grad_b = sum_b_grad / Y_hat.size();

    this->W = this->W - lr * grad_w;
    this->b = this->b - lr * grad_b;
}

/**
 * Train the model using gradient descent.
 * 
 * @param lr The learning rate.
 * @param epoch The number of epochs.
 * @param batchSize The batch size.
 */
void LinearRegression::train(double lr, int epoch, int batchSize) {
    for (int e = 0; e < epoch; ++e) {
        std::vector<double> Y_hat;
        std::vector<double> Y_bs;
        std::vector<double> X_bs;
        double loss = 0;
        for (int bs = 0; bs < this->X.size(); bs += batchSize) {
            for (int i = bs; i < bs + batchSize && i < this->X.size(); ++i) {
                X_bs.push_back(this->X[i]);
                Y_hat.push_back(this->forward(this->X[i]));
                Y_bs.push_back(this->Y[i]);
            }
            loss += this->squareLoss(Y_bs, Y_hat);
            this->gradientDescent(X_bs, Y_bs, Y_hat, lr);
            Y_hat.clear();
            Y_bs.clear();
            X_bs.clear();
        }
        std::cout << "epoch " << e << ", loss: " << loss / batchSize << ", W: " << this->W << ", b: " << this->b << std::endl;
    }
}
