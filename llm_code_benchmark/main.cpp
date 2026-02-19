
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>

int main() {
    const auto start = std::chrono::high_resolution_clock::now();
    
    constexpr std::int64_t iterations = 200000000;
    constexpr double param1 = 4.0;
    constexpr double param2 = 1.0;
    
    double acc0 = 0.0;
    double acc1 = 0.0;
    double acc2 = 0.0;
    double acc3 = 0.0;
    
    std::int64_t i = 1;
    
    for (; i + 3 <= iterations; i += 4) {
        double j0 = i * param1 - param2;
        acc0 -= 1.0 / j0;
        j0 = i * param1 + param2;
        acc0 += 1.0 / j0;
        
        double j1 = (i + 1) * param1 - param2;
        acc1 -= 1.0 / j1;
        j1 = (i + 1) * param1 + param2;
        acc1 += 1.0 / j1;
        
        double j2 = (i + 2) * param1 - param2;
        acc2 -= 1.0 / j2;
        j2 = (i + 2) * param1 + param2;
        acc2 += 1.0 / j2;
        
        double j3 = (i + 3) * param1 - param2;
        acc3 -= 1.0 / j3;
        j3 = (i + 3) * param1 + param2;
        acc3 += 1.0 / j3;
    }
    
    double result = 1.0 + acc0 + acc1 + acc2 + acc3;
    
    for (; i <= iterations; ++i) {
        double j = i * param1 - param2;
        result -= 1.0 / j;
        j = i * param1 + param2;
        result += 1.0 / j;
    }
    
    result *= 4.0;
    
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff = end - start;
    
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << "\n";
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << diff.count() << " seconds\n";
    
    return 0;
}
