#include <unistd.h>

template <typename T>
static void readBinaryPOD(std::istream& in, T& podRef) {
  in.read((char*)&podRef, sizeof(T));
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}

    std::string binaryFile = argv[1];
//////////////////////////////   TEMPLATE START  //////////////////////////////
// read binary file
#ifdef DEBUG
    char const *location = "/home/yuzhuyu/anns/sift_1M_320.bin";
    // char const *location = "/home/yuzhuyu/anns/sift_1M.bin";
#else
    char const *location = "/home/yuzhuyu/anns/sift_1M.bin";
#endif
    char const *gt_location = "/home/yuzhuyu/anns/sift/sift_groundtruth.ivecs";
    char const *query_location = "/home/yuzhuyu/anns/sift/sift_query.fvecs";

    std::cout << "Load index: " << location << std::endl;
    std::ifstream input(location, std::ios::binary);
	// parameters
    uint64_t offsetLevel0_;
    uint64_t max_elements_;
    uint64_t cur_element_count;
    uint64_t size_data_per_element_;
    uint64_t label_offset_;
    uint64_t offsetData_;
    uint32_t maxlevel_;
    uint32_t enterpoint_node_;
    uint64_t maxM_;
    uint64_t maxM0_;
    uint64_t M_;
    double mult_;
    uint64_t ef_construction_;

	int dim = 128;

	if (!input.is_open())
        throw std::runtime_error("Cannot open index file");
    input.seekg(0, input.end); // set the position to be the end of the file
    std::streampos total_filesize = input.tellg();
    std::cout << "Index file size: " << total_filesize << std::endl;
    input.seekg(0, input.beg); // reset the position to be the beginning of the file

	readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);
    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);

	std::cout << "offsetLevel0_: " << offsetLevel0_ << std::endl;
    std::cout << "max_elements_: " << max_elements_ << std::endl;
    std::cout << "cur_element_count: " << cur_element_count << std::endl;
    std::cout << "size_data_per_element_: " << size_data_per_element_ << std::endl;
    std::cout << "label_offset_: " << label_offset_ << std::endl;
    std::cout << "offsetData_: " << offsetData_ << std::endl;
    std::cout << "maxlevel_: " << maxlevel_ << std::endl;
    std::cout << "enterpoint_node_: " << enterpoint_node_ << std::endl;
    std::cout << "maxM_: " << maxM_ << std::endl;
    std::cout << "maxM0_: " << maxM0_ << std::endl;
    std::cout << "M_: " << M_ << std::endl;
    std::cout << "mult_: " << mult_ << std::endl;
    std::cout << "ef_construction_: " << ef_construction_ << std::endl;

    char *data_level0_memory_ = (char*)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

	uint64_t size_links_per_element_ = maxM_ * sizeof(uint32_t) + sizeof(uint32_t);
    char **linkLists_ = (char**)malloc(sizeof(void *) * max_elements_);
    if (linkLists_ == nullptr)
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
    std::vector<int> element_levels_ = std::vector<int>(max_elements_);

	uint32_t links_count = 0;
    for (uint64_t i = 0; i < cur_element_count; i++) {
        uint32_t linkListSize;
        readBinaryPOD(input, linkListSize);
        std::cout << "element "<< i << " linkListSize: " << linkListSize << std::endl;
        if (linkListSize == 0) {
            element_levels_[i] = 0;
            linkLists_[i] = nullptr;
			// links_count += 1; // store the size 0
        } else {
            element_levels_[i] = linkListSize / size_links_per_element_;
            // std::cout << "elements_levels[" << i << "] " << element_levels_[i] << std::endl;
            linkLists_[i] = (char *)malloc(linkListSize);
            if (linkLists_[i] == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
            input.read(linkLists_[i], linkListSize);
			links_count += linkListSize / 4; // store the size and corresponding ids
            // std::cout << links_count << std::endl;
        }
    }
    input.close();

    return 0;
}

