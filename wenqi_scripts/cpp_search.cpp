// Adjusted from examples/searchKnnCloserFirst_test.cpp, i.e., the official C++ search demo

#include "../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>
#include <unordered_set>


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

    // for brute-force check
    hnswlib::AlgorithmInterface<float>* alg_brute  = new hnswlib::BruteforceSearch<float>(&space, n_db_vec);

    // interface:  HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100)
    size_t ef_construction = 128;
    hnswlib::AlgorithmInterface<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n_db_vec, ef_construction);

    std::cout << "Adding data..." << std::endl;
    for (size_t i = 0; i < n_db_vec; ++i) {
        alg_brute->addPoint(data.data() + D * i, i);
        alg_hnsw->addPoint(data.data() + D * i, i);
    }

    std::cout << "Checking results..." << std::endl;
    std::vector<size_t> gt_ID(nq * topK);
    std::vector<size_t> hnsw_ID(nq * topK);

    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * D;
        // searchKNN return type: std::priority_queue<std::pair<dist_t, labeltype >>
        auto gd = alg_brute->searchKnn(p, topK);
        assert(gd.size() == topK);
        int cnt = 0;
        while (!gd.empty()) {
            gt_ID[j * topK + cnt] = gd.top().second;
            gd.pop();
            cnt++;
        }
    }
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
        int start_addr = i * topK;


        // https://en.cppreference.com/w/cpp/container/unordered_set
        std::unordered_set<size_t> gt_set;

        for (int k = 0; k < topK; k++) {
            if (gt_ID[start_addr + k] != hnsw_ID[start_addr + k]) {
                std::cout << "gt : " << gt_ID[start_addr + k] << " hnsw: " << hnsw_ID[start_addr + k] << std::endl;
            }
        }
        int match_cnt = 0;
        for (int k = 0; k < topK; k++) {
            gt_set.insert(gt_ID[start_addr + k]);
        }
        for (int k = 0; k < topK; k++) {
            // count actually means contain here...
            // https://stackoverflow.com/questions/42532550/why-does-stdset-not-have-a-contains-member-function
            if (gt_set.count(hnsw_ID[start_addr + k])) { 
                match_cnt++;
            }
        }
        std::cout << "query ID: " << i << " match cnt: " << match_cnt << std::endl;

        // // set intersection: https://en.cppreference.com/w/cpp/algorithm/set_intersection
        // std::vector<size_t> v_intersection;
        // std::set_intersection(
        //     gt_ID.begin() + start_addr, gt_ID.begin() + start_addr + (topK - 1),
        //     hnsw_ID.begin() + start_addr, hnsw_ID.begin() + start_addr + (topK - 1),
        //     v_intersection.begin());
        // size_t match_cnt = v_intersection.size();
        // std::cout << "query ID: " << i << " match cnt: " << match_cnt;
    }
    
    delete alg_brute;
    delete alg_hnsw;
}

int main() {
    
    test();

    return 0;
}
