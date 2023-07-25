// Compile: g++ -o2 -fopenmp cpp_search_bruteforce.cpp -o a.out

// Adjusted from examples/searchKnnCloserFirst_test.cpp, i.e., the official C++ search demo

#include "../hnswlib/hnswlib.h"

#include <assert.h>

#include <chrono>
#include <vector>
#include <iostream>
#include <unordered_set>

static std::chrono::system_clock::time_point start;
static std::chrono::system_clock::time_point end;
static double durationUs;

std::vector<std::pair<float, size_t >>
priority_queue_to_vector(std::priority_queue<std::pair<float, size_t >>& queue) {

	std::vector<std::pair<float, size_t >> results;
	while (queue.size() > 0) {
		results.push_back(queue.top());
		queue.pop();
	}
	std::sort(results.begin(), results.end());
	return results;
}

std::vector<std::vector<std::pair<float, size_t >>>
batch_priority_queue_to_vector(std::vector<std::priority_queue<std::pair<float, size_t >>>& queue) {

	std::vector<std::vector<std::pair<float, size_t >>> results;
	for (std::priority_queue<std::pair<float, size_t >> e : queue) {
		results.push_back(priority_queue_to_vector(e));
	}
	return results;
}

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
	// std::vector<std::priority_queue<std::pair<dist_t, labeltype >>>

    // std::vector<size_t> gt_ID(nq * topK);
    // std::vector<size_t> hnsw_ID(nq * topK);

	int omp_threads = 8;
	omp_set_num_threads(omp_threads);

	//// Single query, Serial ///// 
	std::vector<std::priority_queue<std::pair<float, size_t >>> results_single_query_serial_queue(nq);
	start = std::chrono::system_clock::now();
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * D;
        results_single_query_serial_queue.at(j) = alg_brute->searchKnn(p, topK);
    }
	end = std::chrono::system_clock::now();
	durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
			end - start).count());

	float QPS_intra_serial = nq / (durationUs / 1000.0 / 1000.0);
	std::cout << "duration (intra serial): " << durationUs / 1000.0 / 1000.0 << " sec" << std::endl;
	std::cout << "QPS_intra_serial: " << QPS_intra_serial << std::endl;
	std::vector<std::vector<std::pair<float, size_t >>> results_single_query_serial = batch_priority_queue_to_vector(results_single_query_serial_queue);

	//// Single query, Parallel ///// 
	std::vector<std::priority_queue<std::pair<float, size_t >>> results_single_query_parallel_queue(nq);
	start = std::chrono::system_clock::now();
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * D;
        results_single_query_parallel_queue.at(j) = alg_brute->searchKnnParallel(p, topK);
    }
	end = std::chrono::system_clock::now();
	durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
			end - start).count());

	float QPS_intra_parallel = nq / (durationUs / 1000.0 / 1000.0);
	std::cout << "duration (intra parallel): " << durationUs / 1000.0 / 1000.0 << " sec" << std::endl;
	std::cout << "QPS_intra_parallel: " << QPS_intra_parallel << std::endl;
	std::vector<std::vector<std::pair<float, size_t >>> results_single_query_parallel = batch_priority_queue_to_vector(results_single_query_parallel_queue);

	//// Batch queries, Serial ///// 
	start = std::chrono::system_clock::now();
	auto results_batch_serial_queue = 
		alg_brute->searchKnnBatch(query.data(), topK, nq);
	end = std::chrono::system_clock::now();
	durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
			end - start).count());
	
	float QPS_batch_serial = nq / (durationUs / 1000.0 / 1000.0);
	std::cout << "duration (batch serial): " << durationUs / 1000.0 / 1000.0 << " sec" << std::endl;
	std::cout << "QPS_batch_serial: " << QPS_batch_serial << std::endl;
	std::vector<std::vector<std::pair<float, size_t >>> results_batch_serial = batch_priority_queue_to_vector(results_batch_serial_queue);


	//// Batch queries, Parallel ///// 
	start = std::chrono::system_clock::now();
	auto results_batch_parallel_queue = 
		alg_brute->searchKnnBatchParallel(query.data(), topK, nq);
	end = std::chrono::system_clock::now();
	durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
			end - start).count());

	float QPS_batch_parallel = nq / (durationUs / 1000.0 / 1000.0);
	std::cout << "duration (batch parallel): " << durationUs / 1000.0 / 1000.0 << " sec" << std::endl;
	std::cout << "QPS_batch_parallel: " << QPS_batch_parallel << std::endl;
	std::vector<std::vector<std::pair<float, size_t >>> results_batch_parallel = batch_priority_queue_to_vector(results_batch_parallel_queue);


	// Results verification
	std::cout << "Verifying results..." << std::endl;
	for (int b = 0; b < nq; b++) {
		for (int k = 0; k < topK; k++) {
			assert (results_single_query_serial.at(b).at(k) == results_single_query_parallel.at(b).at(k));
			assert (results_single_query_serial.at(b).at(k) == results_batch_serial.at(b).at(k));
			assert (results_single_query_serial.at(b).at(k) == results_batch_parallel.at(b).at(k));
		}
	}
	std::cout << "Results Correct!" << std::endl;

    delete alg_brute;
}

int main() {
    
    test();

    return 0;
}
