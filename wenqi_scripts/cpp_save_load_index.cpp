// Adjusted from examples/searchKnnCloserFirst_test.cpp, i.e., the official C++ search demo

#include "../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>
#include <unordered_set>
#include <string>
#include <fstream>

// https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exists-using-standard-c-c11-14-17-c
inline bool file_exists (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void test() {


    int D = 128;
    size_t n_db_vec = 10000;
    size_t nq = 10;
    size_t topK = 32;
   
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

    // interface:  HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100)
    size_t ef_construction = 128;

    std::string index_dir = "/tmp/hnsw_index.bin";
    hnswlib::AlgorithmInterface<float>* alg_hnsw;
    if (file_exists(index_dir)) {
        std::cout << "Index exists, loading index..." << std::endl;
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_dir);
    }
    else {
        std::cout << "Index does not exist, creating new index..." << std::endl;
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n_db_vec, ef_construction);
        std::cout << "Adding data..." << std::endl;
        for (size_t i = 0; i < n_db_vec; ++i) {
            alg_hnsw->addPoint(data.data() + D * i, i);
        }
        alg_hnsw->saveIndex(index_dir);
    }

    std::cout << "Searching for results..." << std::endl;
    std::vector<size_t> hnsw_ID(nq * topK);

    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * D;
        // searchKNN return type: std::priority_queue<std::pair<dist_t, labeltype >>
        auto res = alg_hnsw->searchKnn(p, topK);
        assert(res.size() == topK);
        int cnt = 0;
        while (!res.empty()) {
            hnsw_ID[j * topK + cnt] = res.top().second;
            res.pop();
            cnt++;
        }
    }

    
    for (size_t i = 0; i < nq; i++) {

        std::cout << "query ID: " << i << std::endl;

        int start_addr = i * topK;
        for (int k = 0; k < topK; k++) {
            std::cout << "hnsw ID: " << hnsw_ID[start_addr + k] << std::endl;
        }
    }
    
    delete alg_hnsw;
}

int main() {
    
    test();

    return 0;
}
