#include "../header/utils.h"
#include <random>
#include <algorithm>

/**
 * Generates a specified number of samples with a linear relationship and optional noise.
 * 
 * This function generates 'n' number of samples, each consisting of a pair of 'x' and 'y' values.
 * The 'y' values are calculated based on a linear equation 'y = w * x + b', where 'w' and 'b' are the
 * slope and intercept of the line, respectively. Additionally, the function can optionally add
 * Gaussian noise to the 'y' values. The noise is also normally distributed with a mean of 0 and
 * a standard deviation of 0.01. If 'shuffle' is set to true, the generated samples are shuffled
 * randomly.
 * 
 * @param n The number of samples to generate.
 * @param w The slope of the linear equation.
 * @param b The intercept of the linear equation.
 * @param shuffle Optional. If true, the generated samples are shuffled. Default is true.
 * @return A vector of pairs, each pair containing an 'x' and a 'y' value.
 */
std::vector<std::pair<double, double>> Utils::generateSamples(int n, double w, double b, bool shuffle) {
    std::vector<std::pair<double, double>> samples;

    std::normal_distribution<> d(0, 1);
    std::default_random_engine gen; // Define a random number generator
    for (int i = 0; i < n; ++i) {
        double x = d(gen);
        double y = w * x + b;
        samples.push_back(std::make_pair(x, y));
    }

    // Generate 'n' Gaussian noise data in the range [0, 0.01] and add it to 'y'
    std::normal_distribution<> d2(0, 0.01);
    std::default_random_engine gen2; // Define a random number generator
    for (auto& sample : samples) {
        sample.second += d2(gen2);
    }

    if (shuffle) {
        std::random_device rd; // Get a random seed
        std::mt19937 g(rd());   // Initialize a random number generator
        std::shuffle(samples.begin(), samples.end(), g); // Shuffle the order of the samples
    }

    return samples;
}
