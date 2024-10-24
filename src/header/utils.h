#pragma once

#include <cmath>
#include <vector>
#include <iostream>

class Utils {
public:
    static std::vector<std::pair<double, double>> generateSamples(int n, double w, double b, bool shuffle=true);
};
