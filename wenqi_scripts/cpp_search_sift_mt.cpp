// Adjusted from examples/searchKnnCloserFirst_test.cpp, i.e., the official C++ search demo

#include "../hnswlib/hnswlib.h"
#include "../hnswlib/hnswalg.h"

#include <assert.h>

#include <vector>
#include <iostream>
#include <unordered_set>
#include <string>
#include <fstream>
#include <chrono>

#include <sys/stat.h>

// https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exists-using-standard-c-c11-14-17-c
inline bool file_exists (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

void test() {


    int d = 128;
	int query_num = 10000;
    size_t topK = 32;
	size_t ef = 64; 

	std::string index_dir = "/mnt/scratch/wenqi/hnswlib-eval/indexes/SIFT1M_index_M_32.bin";
   
    const char* fname_query_vectors =  "/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_query.bvecs";
    const char* fname_gt_vec_ID = "/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/idx_1M.ivecs";
    const char* fname_gt_dist = "/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/dis_1M.fvecs";
	
	FILE* f_query_vectors = fopen(fname_query_vectors, "rb");
	FILE* f_gt_vec_ID = fopen(fname_gt_vec_ID, "rb");
	FILE* f_gt_dist = fopen(fname_gt_dist, "rb");

    size_t raw_query_vectors_size = GetFileSize(fname_query_vectors);
    size_t raw_gt_vec_ID_size = GetFileSize(fname_gt_vec_ID);
    size_t raw_gt_dist_size = GetFileSize(fname_gt_dist);

    std::vector<float> raw_query_vectors(raw_query_vectors_size / sizeof(float));
    std::vector<int> raw_gt_vec_ID(raw_gt_vec_ID_size / sizeof(int));
    std::vector<float> raw_gt_dist(raw_gt_dist_size / sizeof(float));

	fread(raw_query_vectors.data(), 1, raw_query_vectors_size, f_query_vectors);
    fclose(f_query_vectors);
    fread(raw_gt_vec_ID.data(), 1, raw_gt_vec_ID_size, f_gt_vec_ID);
    fclose(f_gt_vec_ID);
    fread(raw_gt_dist.data(), 1, raw_gt_dist_size, f_gt_dist);
    fclose(f_gt_dist);


    int max_topK = 100;

    std::vector<float> query_vectors(query_num * d);
    std::vector<int> gt_vec_ID(query_num * max_topK);
    std::vector<float> gt_dist(query_num * max_topK);

	size_t len_per_query = 4 + d;
    for (int qid = 0; qid < query_num; qid++) {
        for (int i = 0; i < d; i++) {
            query_vectors[qid * d + i] = (float) raw_query_vectors[qid * len_per_query + 4 + i];
        }
    }

    // ground truth = 4-byte ID + 1000 * 4-byte ID + 1000 or 4-byte distances
    size_t len_per_gt = (4 + 1000 * 4) / 4;
    for (int qid = 0; qid < query_num; qid++) {
        for (int i = 0; i < max_topK; i++) {
            gt_vec_ID[qid * max_topK + i] = raw_gt_vec_ID[qid * len_per_gt + 1 + i];
            gt_dist[qid * max_topK + i] = raw_gt_dist[qid * len_per_gt + 1 + i];
        }
    }

    hnswlib::L2Space space(d);

    hnswlib::HierarchicalNSW<float> alg_hnsw(&space, index_dir);
    // if (file_exists(index_dir)) {
    //     std::cout << "Index exists, loading index..." << std::endl;
    //     alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_dir);
    // }

	// alg_hnsw.ef_ = ef;

	// insert profiler
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < query_num; ++j) {
        const void* p = query_vectors.data() + j * d;
        // searchKNN return type: std::priority_queue<std::pair<dist_t, labeltype >>
        auto res = alg_hnsw->searchKnn(p, topK);
        assert(res.size() == topK);
        int cnt = 0;
        // while (!res.empty()) {
        //     hnsw_ID[j * topK + cnt] = res.top().second;
        //     res.pop();
        //     cnt++;
        // }
    }
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
	std::cout << "search time: " << duration << " ms" << std::endl;


    
    // for (size_t i = 0; i < query_num; i++) {

    //     std::cout << "query ID: " << i << std::endl;

    //     int start_addr = i * topK;
    //     for (int k = 0; k < topK; k++) {
    //         std::cout << "hnsw ID: " << hnsw_ID[start_addr + k] << std::endl;
    //     }
    // }
    
    delete alg_hnsw;
}

int main() {
    
    test();

    return 0;
}
