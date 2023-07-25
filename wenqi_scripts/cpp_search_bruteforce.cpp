// Compile: g++ -o2 -fopenmp cpp_search_bruteforce.cpp -o a.out

// Adjusted from examples/searchKnnCloserFirst_test.cpp, i.e., the official C++ search demo

#include "../hnswlib/hnswlib.h"

#include <assert.h>

#include <chrono>
#include <vector>
#include <iostream>
#include <unordered_set>


void test() {

    int D = 128;
    size_t n_db_vec = 32768;
    size_t nq = 100;
    size_t topK = 100;
   
    std::vector<float> data(n_db_vec * D);
    std::vector<float> query(nq * D);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (size_t i = 0; i < n_db_vec * D; ++i) {
        data[i] = distrib(rng);
    }
    for (size_t i = 0; i < nq * D; ++i) {
        query[i] = distrib(rng);
    }
      

    hnswlib::L2Space space(D);

    // for brute-force check
    hnswlib::AlgorithmInterface<float>* alg_brute  = new hnswlib::BruteforceSearch<float>(&space, n_db_vec);

    std::cout << "Adding data..." << std::endl;
    for (size_t i = 0; i < n_db_vec; ++i) {
        alg_brute->addPoint(data.data() + D * i, i);
    }

    std::cout << "Checking results..." << std::endl;
    std::vector<size_t> gt_ID(nq * topK);
    std::vector<size_t> hnsw_ID(nq * topK);

	int omp_threads = 8;
	omp_set_num_threads(omp_threads);

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * D;
        // searchKNN return type: std::priority_queue<std::pair<dist_t, labeltype >>
        // auto gd = alg_brute->searchKnn(p, topK);
        auto gd = alg_brute->searchKnnParallel(p, topK);
        assert(gd.size() == topK);
        int cnt = 0;
        while (!gd.empty()) {
            gt_ID[j * topK + cnt] = gd.top().second;
            gd.pop();
            cnt++;
        }
    }
	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
	// calculate QPS
	double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
			end - start).count());
	float QPS = nq / (durationUs / 1000.0 / 1000.0);
	std::cout << "duration: " << durationUs / 1000.0 / 1000.0 << " sec" << std::endl;
	std::cout << "QPS: " << QPS << std::endl;

    delete alg_brute;
}

int main() {
    
    test();

    return 0;
}
