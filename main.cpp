#include <algorithm>
#include <execution>
#include <iostream>
#include <random>
#include <vector>
#include <thread>
#include <numeric>
#include <string>

using hrclock = std::chrono::high_resolution_clock;
using ms = std::chrono::duration<double, std::milli>;

static double elapsed_ms(const hrclock::time_point &a, const hrclock::time_point &b) {
    return std::chrono::duration_cast<ms>(b - a).count();
}

struct PredCheap {
    bool operator()(int x) const noexcept { return (x & 1) == 0; }
};

struct PredExpensive {
    bool operator()(int n) const {
        if (n < 2) return false;
        if (n % 2 == 0) return n == 2;
        for (int d = 3; (long long)d * d <= n; d += 2)
            if (n % d == 0) return false;
        return true;
    }
};

template <typename It, typename Pred>
size_t parallel_count_if_custom(It first, It last, Pred pred, size_t K) {
    auto n = std::distance(first, last);
    if (n <= 0) return 0;
    if (K <= 1) return std::count_if(first, last, pred);

    std::vector<std::thread> threads;
    std::vector<size_t> results(K, 0);
    threads.reserve(K);

    auto chunk_size = n / K;
    auto rem = n % K;
    It chunk_start = first;

    for (size_t i = 0; i < K; ++i) {
        auto this_chunk = chunk_size + (i < rem ? 1 : 0);
        It chunk_end = std::next(chunk_start, this_chunk);
        threads.emplace_back([chunk_start, chunk_end, &pred, &results, i]() {
            results[i] = std::count_if(chunk_start, chunk_end, pred);
        });
        chunk_start = chunk_end;
    }
    for (auto &t : threads) t.join();
    return std::accumulate(results.begin(), results.end(), size_t(0));
}

template <typename F>
std::pair<size_t, double> time_once(F f) {
    auto t0 = hrclock::now();
    size_t cnt = f();
    auto t1 = hrclock::now();
    return {cnt, elapsed_ms(t0, t1)};
}

template <typename F>
std::pair<size_t, double> time_repeat(F f, int reps = 5) {
    std::vector<double> times;
    times.reserve(reps);
    size_t last_count = 0;
    for (int i = 0; i < reps; ++i) {
        auto r = time_once(f);
        last_count = r.first;
        times.push_back(r.second);
    }
    std::sort(times.begin(), times.end());
    return {last_count, times[times.size() / 2]};
}

int main() {
    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;

    std::cout << "count_if experiments\n";
    std::cout << "hardware threads: " << hw_threads << "\n\n";

    std::vector<size_t> sizes = {100000, 500000, 1000000, 5000000, 10000000};

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, 1000000);

    for (auto N : sizes) {
        std::cout << "================ N = " << N << " ================\n";

        std::vector<int> data(N);
        for (auto &x : data) x = dist(rng);

        std::vector<std::pair<std::string, bool>> preds = {
                {"Cheap predicate (even)", true},
                {"Expensive predicate (prime)", false}
        };

        for (auto &pp : preds) {
            std::cout << "\n" << pp.first << "\n";

            if (pp.second) {
                PredCheap pred;

                auto r0 = time_repeat([&](){ return std::count_if(data.begin(), data.end(), pred); });
                std::cout << "[std::count_if] time " << r0.second << " ms\n";

                try {
                    auto r1 = time_repeat([&](){ return std::count_if(std::execution::par, data.begin(), data.end(), pred); });
                    std::cout << "[par] time " << r1.second << " ms\n";
                } catch (...) {
                    std::cout << "[par unsupported]\n";
                }

                std::cout << "\nK, time(ms), count\n";
                size_t bestK = 1; double bestTime = 1e100;
                size_t maxK = std::min(N, static_cast<size_t>(hw_threads * 2));


                for (size_t K = 1; K <= maxK; K++) {
                    auto r = time_repeat([&](){ return parallel_count_if_custom(data.begin(), data.end(), pred, K); });
                    std::cout << K << ", " << r.second << ", " << r.first << "\n";
                    if (r.second < bestTime) { bestTime = r.second; bestK = K; }
                }
                std::cout << "Best K = " << bestK << " (time=" << bestTime << ")\n";
            }

            else {
                PredExpensive pred;

                auto r0 = time_repeat([&](){ return std::count_if(data.begin(), data.end(), pred); });
                std::cout << "[std::count_if] time " << r0.second << " ms\n";

                try {
                    auto r1 = time_repeat([&](){ return std::count_if(std::execution::par, data.begin(), data.end(), pred); });
                    std::cout << "[par] time " << r1.second << " ms\n";
                } catch (...) {
                    std::cout << "[par unsupported]\n";
                }

                std::cout << "\nK, time(ms), count\n";
                size_t bestK = 1; double bestTime = 1e100;
                size_t maxK = std::min(N, static_cast<size_t>(hw_threads * 2));


                for (size_t K = 1; K <= maxK; K++) {
                    auto r = time_repeat([&](){ return parallel_count_if_custom(data.begin(), data.end(), pred, K); });
                    std::cout << K << ", " << r.second << ", " << r.first << "\n";
                    if (r.second < bestTime) { bestTime = r.second; bestK = K; }
                }
                std::cout << "Best K = " << bestK << " (time=" << bestTime << ")\n";
            }
        }
    }

    std::cout << "\nDone. Run in -O0 and -O3.\n";
    return 0;
}
