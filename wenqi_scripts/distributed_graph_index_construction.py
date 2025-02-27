"""
This script constructs the indexes for distributed graph search, 
    and save it as pkl file
"""

import hnswlib
import numpy as np 
import struct
import heapq
import time
import pickle
import os

from pathlib import Path
from sklearn.cluster import KMeans


def save_obj(obj, dirc, name):
    # note use "dir/" in dirc
    with open(os.path.join(dirc, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, protocol=4) # for py37,pickle.HIGHEST_PROTOCOL=4

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)
    
def convertBytes(bytestring, dtype='int'):
    """
    convert bytes to a single element
    dtype = {int, long, float, double}
    struct: https://docs.python.org/3/library/struct.html
    """ 
    # int from bytes is much faster than struct.unpack
    if dtype =='int' or dtype == 'long': 
        return int.from_bytes(bytestring, byteorder='little', signed=False)
    elif dtype == 'float': 
        return struct.unpack('f', bytestring)[0]
    elif dtype == 'double': 
        return struct.unpack('d', bytestring)[0]
    else:
        raise ValueError 

# Wenqi: the fastest way to load a bytestring list is to use *** np.frombuffer ***
def convertBytesList(bytestring, dtype='int'):
    """
    Given a byte string, return the value list
    """
    result_list = []
    if dtype == 'int' or dtype == 'float':
        dsize = 4
    elif dtype == 'long' or dtype == 'double':
        dsize = 8
    else:
        raise ValueError 
        
    start_pointer = 0
    for i in range(len(bytestring) // dsize):
        result_list.append(convertBytes(
            bytestring[start_pointer: start_pointer + dsize], dtype=dtype))
        start_pointer += dsize
    return result_list

def calculateDist(query_data, db_vec):
    """
    HNSWLib returns L2 square distance, so do we
        both inputs are 1-d np array
    """
    # return l2 distance between two points
    return np.sum((query_data - db_vec) ** 2)


def merge_two_distance_list(list_A, list_B, k):
    """
    merge two lists by selecting the k pairs of the smallest distance
    input:
        both list has format [(dist, ID), (dist, ID), ...]
    return:
        a result list, with ascending distance (the first contains the largest distance)
    """
    
    results_heap = []
    for i in range(len(list_A)):
        dist, server_ID, vec_ID = list_A[i]
        heapq.heappush(results_heap, (-dist, server_ID, vec_ID))
    for i in range(len(list_B)):
        dist, server_ID, vec_ID = list_B[i]
        heapq.heappush(results_heap, (-dist, server_ID, vec_ID))

    while len(results_heap) > k:
        heapq.heappop(results_heap)

    results = []
    while len(results_heap) > 0:
        dist, server_ID, vec_ID = results_heap[0]
        results.append((-dist, server_ID, vec_ID))
        heapq.heappop(results_heap)
    results.reverse()
            
    return results

        
class HNSW_index():
    
    """
    Returned result list always in the format of (dist, server_ID, vec_ID),
        in ascending distance order (the first result is the nearest neighbor)
    """
    
    def __init__(self, local_server_ID=0, dim=128):
        
        self.dim = dim
        self.local_server_ID = local_server_ID
        
        # Meta Info
        self.offsetLevel0_ = None
        self.max_elements_ = None
        self.cur_element_count = None
        self.size_data_per_element_ = None
        self.label_offset_ = None
        self.offsetData_ = None
        self.maxlevel_ = None
        self.enterpoint_node_ = None
        self.maxM_ = None
        self.maxM0_ = None
        self.M_ = None
        self.mult_ = None # the probability that a node is one a higher level
        self.ef_construction_ = None
        
        # Graph partition info
        self.centroid_vectors = None # centroids for all sub-graphs
        
        # ground layer, all with length of cur_element_count
        self.links_count_l0 = None # a list of link_count
        self.links_l0 = None # a list of links per vector
        self.data_l0 = None # a list of vectors
        self.label_l0 = None # a list of vector IDs
        self.vec_ID_to_local_ID = dict() # mapping from vector ID -> local storage ID (on ground layer)
        
        # upper layers, all with length of cur_element_count
        self.element_levels_ = None # the level per vector
        self.links = None # the upper layer link info (link count + links)
        
        # remote nodes, order according to local ID (not label ID)
        #  remote_links: an 2-D array (cur_element_count, k), 
        #    each element is a tuple: (server_ID, vector_ID)
        self.remote_links_count = None
        self.remote_links = None
        
    def load_meta_info(self, index_bin):
        """
        index_bin = hnswlib index binary 
        
        HNSW save index order:
            https://github.com/WenqiJiang/hnswlib-eval/blob/master/hnswlib/hnswalg.h#L588-L616
        """
        self.offsetLevel0_ = int.from_bytes(index_bin[0:8], byteorder='little', signed=False)
        self.max_elements_ = int.from_bytes(index_bin[8:16], byteorder='little', signed=False)
        self.cur_element_count = int.from_bytes(index_bin[16:24], byteorder='little', signed=False)
        self.size_data_per_element_ = int.from_bytes(index_bin[24:32], byteorder='little', signed=False)
        self.label_offset_ = int.from_bytes(index_bin[32:40], byteorder='little', signed=False)
        self.offsetData_ = int.from_bytes(index_bin[40:48], byteorder='little', signed=False)
        self.maxlevel_ = int.from_bytes(index_bin[48:52], byteorder='little', signed=False)
        self.enterpoint_node_ = int.from_bytes(index_bin[52:56], byteorder='little', signed=False)
        self.maxM_ = int.from_bytes(index_bin[56:64], byteorder='little', signed=False)
        self.maxM0_ = int.from_bytes(index_bin[64:72], byteorder='little', signed=False)
        self.M_ = int.from_bytes(index_bin[72:80], byteorder='little', signed=False)
        self.mult_ = struct.unpack('d', index_bin[80:88])[0] # the probability that a node is one a higher level
        self.ef_construction_ = int.from_bytes(index_bin[88:96], byteorder='little', signed=False)
        

        print("offsetLevel0_", self.offsetLevel0_)
        print("max_elements_", self.max_elements_)
        print("cur_element_count", self.cur_element_count)
        print("size_data_per_element_", self.size_data_per_element_)
        print("label_offset_", self.label_offset_)
        print("offsetData_", self.offsetData_)
        print("maxlevel_", self.maxlevel_)
        print("enterpoint_node_", self.enterpoint_node_)
        print("maxM_", self.maxM_)
        print("maxM0_", self.maxM0_)
        print("M_", self.M_)
        print("mult_", self.mult_)
        print("ef_construction_", self.ef_construction_)
        
    def set_centroid_vectors(self, centroid_vectors):
        """
        Given n subgraphs, there will be n centroid vectors,
            the centroid vector of the current sub-graph is 
            self.centroid_vectors[self.local_server_ID]
        Input:
            cluster centroids: (n, dim)
        """
        assert centroid_vectors.shape[1] == self.dim
        self.centroid_vectors = centroid_vectors
        
        
    def load_ground_layer(self, index_bin):
        """
        Get the ground layer vector ID, vectors, and links:
            links_count_l0: vec_num
            links_l0: maxM0_ * vec_num 
            data_l0: (dim, vec_num)
            label_l0: vec_num
        """
        
        # Layer 0 data 
        start_byte_pointer = 96
        delta = self.cur_element_count * self.size_data_per_element_
        data_level0 = index_bin[start_byte_pointer: start_byte_pointer + delta]
        
        size = len(data_level0)
        self.links_count_l0 = []
        self.links_l0 = np.zeros((self.cur_element_count, self.maxM0_), dtype=int)
        self.data_l0 = np.zeros((self.cur_element_count, self.dim))
        self.label_l0 = []
        self.vec_ID_to_local_ID = dict()

        data_l0_list = []
        
        assert len(data_level0) == self.size_data_per_element_ * self.cur_element_count
        
        size_link_count = 4
        size_links = self.maxM0_ * 4
        size_vectors = self.dim * 4
        size_label = 8
        
        assert self.size_data_per_element_ == \
            size_link_count + size_links + size_vectors + size_label
            
        for i in range(self.cur_element_count):
            # per ground layer node: (link_count (int), links (int array of len=maxM0_), 
            #    vector (float array of len=dim, vector ID (long)))
            
            addr_link_count = i * self.size_data_per_element_ 
            addr_links = addr_link_count + size_link_count
            addr_vectors = addr_links + size_links
            addr_label = addr_vectors + size_vectors
            
            tmp_bytes = data_level0[addr_link_count: addr_link_count + size_link_count]
            self.links_count_l0.append(convertBytes(tmp_bytes, dtype='int'))
        
            tmp_bytes = data_level0[addr_links: addr_links + size_links]
            self.links_l0[i] = np.frombuffer(tmp_bytes, dtype=np.int32)
            
            tmp_bytes = data_level0[addr_vectors: addr_vectors + size_vectors]
            self.data_l0[i] = np.frombuffer(tmp_bytes, dtype=np.float32)
            
            tmp_bytes = data_level0[addr_label: addr_label + size_label]
            vec_ID = convertBytes(tmp_bytes, dtype='long')
            self.label_l0.append(vec_ID)
            self.vec_ID_to_local_ID[vec_ID] = i


    def load_upper_layers(self, index_bin):
        """
        Get the upper layer info:
            element_levels_: the levels of each vector
            links: list of upper links
        """
        
        # meta + ground data
        start_byte_pointer = 96 + self.max_elements_ * self.size_data_per_element_
        
        # Upper layers
        links_count = 0
        size_links_per_element_ = self.maxM_ * 4 + 4
        self.element_levels_ = []
        self.links = []

        for i in range(self.cur_element_count):
            tmp_bytes = index_bin[start_byte_pointer:start_byte_pointer+4]
            linkListSize = convertBytes(tmp_bytes, dtype='int')
            start_byte_pointer += 4
            
            # if an element is only on ground layer, it has no links on upper layers at all
            if linkListSize == 0:
                self.element_levels_.append(0)
                self.links.append([])
            else:
                level = int(linkListSize / size_links_per_element_)
                self.element_levels_.append(level)
                tmp_bytes = index_bin[start_byte_pointer:start_byte_pointer+linkListSize]
                links_tmp = list(np.frombuffer(tmp_bytes, dtype=np.int32))
                start_byte_pointer += linkListSize
                links_count += linkListSize / 4;
                self.links.append(links_tmp)

        assert start_byte_pointer == len(index_bin) # 6606296

    def insertRemote(self, remote_hnswlib_indexes, remote_server_IDs, ef=128):
        """
        Input: 
            remote_hnswlib_indexes: a list of remote_hnswlib_index
                remote_hnswlib_index: index loaded by remote memory (hnswlib object)
            remote_server_IDs: a list of remote index IDs respective to remote_hnswlib_indexes
                e.g., this is server 1, and there are four servers in totol,
                    then the remote index IDs should be [0, 2, 3]
        """
        self.remote_links_count = []
        self.remote_links = []
        
        remote_I_list = []
        remote_D_list = []
        remote_server_ID_list = []
        
        k = self.maxM0_
        
        assert len(remote_hnswlib_indexes) == len(remote_server_IDs)
        
        # query all servers
        for i in range(len(remote_server_IDs)):
            remote_hnswlib_index = remote_hnswlib_indexes[i]
            remote_server_ID = remote_server_IDs[i]
        
            query = self.data_l0
            remote_hnswlib_index.set_ef(ef)
            I, D = remote_hnswlib_index.knn_query(query, k=k)
            remote_I_list.append(I)
            remote_D_list.append(D)
            remote_server_ID_list.append(
                np.ones((I.shape[0], I.shape[1]), dtype=np.int32) * int(remote_server_ID))
    
        # merge results per server
        remote_I = np.concatenate(remote_I_list, axis=1)
        remote_D = np.concatenate(remote_D_list, axis=1)
        remote_server_ID = np.concatenate(remote_server_ID_list, axis=1)
        
        D_server_ID_I_list = [[] for i in range(remote_I.shape[0])]
        server_ID_I_list= [[] for i in range(remote_I.shape[0])]
        for i in range(remote_I.shape[0]):
            for j in range(remote_I.shape[1]):
                D_server_ID_I_list[i].append((remote_D[i][j], remote_server_ID[i][j], remote_I[i][j]))
            D_server_ID_I_list[i].sort()
            D_server_ID_I_list[i] = D_server_ID_I_list[i][:k]
            server_ID_I_list[i] = [(s, i) for d, s, i in D_server_ID_I_list[i]]
                
        
        self.remote_links_count = [k for i in range(self.cur_element_count)]
        #  remote_links: an 2-D array (cur_element_count x k), 
        #    each element is a tuple: (server_ID, vector_ID)
        self.remote_links = server_ID_I_list
        
    def searchKnnGroundLayer(self, q_data, k, ef, ep_local_id, existing_results=None):
        """
        The ground layer searching process.
        Input:
            query vector
            topk
            ef
            ep_local_id: entry point of the ground layer (local ID, not real vec ID)
            existing_results: a list of search results on previous servers
        Output:
            topK results: list of (dist, serverID, vecID) in ascending distance order
            search_path_local_ID_gnd: ground layer search path (local node ID) 
            search_path_vec_ID_gnd: ground layer search path (vector ID)
        """
        
        num_elements = self.cur_element_count
        links_count_l0 = self.links_count_l0
        links_l0 = self.links_l0
        data_l0 = self.data_l0
        label_l0 = self.label_l0
        dim = self.dim
        
        search_path_local_ID_gnd = set()
        search_path_vec_ID_gnd = set()
        
        # dynamic list (result candidates): (-dist, server_ID, vec_ID)
        top_candidates = [] 
        # candidate list: (dist, local_ID)
        candidate_set = []
        # local visisted vectors
        visited_array = set() 
        
        
        ep_dist = calculateDist(q_data, data_l0[ep_local_id])
        distUpperBound = ep_dist 
        # By default heap queue is a min heap: https://docs.python.org/3/library/heapq.html
        # candidate_set = candidate list, min heap
        # top_candidates = dynamic list (potential results), max heap
        # compare min(candidate_set) vs max(top_candidates)
        if not existing_results:
            vec_ID = label_l0[ep_local_id]
            heapq.heappush(top_candidates, (-ep_dist, self.local_server_ID, vec_ID))
            heapq.heappush(candidate_set, (ep_dist, ep_local_id))
            visited_array.add(ep_local_id) 
        else:
            for dist, server_ID, vec_ID in existing_results:
                heapq.heappush(top_candidates, (-dist, server_ID, vec_ID))
                if server_ID == self.local_server_ID: 
                    local_ID = self.vec_ID_to_local_ID[vec_ID]
                    heapq.heappush(candidate_set, (dist, local_ID))
                    visited_array.add(local_ID)
                
                
        while len(candidate_set)!=0:
            current_node_dist, current_node_id = candidate_set[0]
            if ((current_node_dist > distUpperBound)):
                break
            heapq.heappop(candidate_set)
            search_path_local_ID_gnd.add(current_node_id)
            size = links_count_l0[current_node_id]
            
            for i in range(size):
                candidate_id = links_l0[current_node_id][i]
                if (candidate_id not in visited_array):
                    visited_array.add(candidate_id)
                    currVec = data_l0[candidate_id]
                    dist = calculateDist(q_data, currVec)
                    
                    if (len(top_candidates) < ef or distUpperBound > dist):
                        heapq.heappush(candidate_set, (dist, candidate_id))
                        vec_ID = label_l0[candidate_id]
                        heapq.heappush(top_candidates, (-dist, self.local_server_ID, vec_ID))
                    if (len(top_candidates) > ef):
                        heapq.heappop(top_candidates)
                    if (len(top_candidates)!=0):
                        distUpperBound = -top_candidates[0][0] 
                     

        while len(top_candidates) > k:
            heapq.heappop(top_candidates)

        result = []
        while len(top_candidates) > 0:
            minus_dist, server_ID, vec_ID = top_candidates[0]
            result.append([-minus_dist, server_ID, vec_ID])
            heapq.heappop(top_candidates)
        result.reverse()
            
        for local_ID in search_path_local_ID_gnd:
            search_path_vec_ID_gnd.add(label_l0[local_ID])

        return result, search_path_local_ID_gnd, search_path_vec_ID_gnd
        
    def searchKnn(self, q_data, k, ef):
        """
        The HNSW way to search knn, from top layer all the way down to the bottom.
        result a list of (distance, vec_ID) in ascending distance
        """
        
        ep_node = self.enterpoint_node_
        max_level = self.maxlevel_
        links = self.links
        data_l0 = self.data_l0
        label_l0 = self.label_l0
        dim = self.dim
        
        currObj = ep_node
        currVec = data_l0[currObj]
        curdist = calculateDist(q_data, currVec)
        
        search_path_local_ID_upper = set()
        search_path_vec_ID_upper = set()
        
        # search upper layers
        for level in reversed(range(1, max_level+1)):
            
            changed = True
            while changed:
            
                search_path_local_ID_upper.add(currObj)
                changed = False
                if (len(links[currObj])==0):
                    break
                else:
                    start_index = (level-1) * 17
                    size = links[currObj][start_index]
                    neighbors = links[currObj][(start_index+1):(start_index+17)]
                    
                    for i in range(size):
                        cand = neighbors[i]
                        currVec = data_l0[cand]
                        dist = calculateDist(q_data, currVec)
                        if (dist < curdist):
                            curdist = dist
                            currObj = cand
                            changed = True
                            
        for local_ID in search_path_local_ID_upper:
            search_path_vec_ID_upper.add(label_l0[local_ID])
            
        # search in ground layer
        result, search_path_local_ID_gnd, search_path_vec_ID_gnd = \
            self.searchKnnGroundLayer(q_data, k, ef, ep_local_id=currObj)

#         search_path_local_ID = set.union(search_path_local_ID_upper, search_path_local_ID_gnd)
#         search_path_vec_ID = set.union(search_path_vec_ID_upper, search_path_vec_ID_gnd)
        
        return result, search_path_local_ID_upper, search_path_vec_ID_upper, search_path_local_ID_gnd, search_path_vec_ID_gnd
        
    def searchKnnPlusRemoteCache(self, q_data, k, ef, all_vectors, ep_vec_id=-1, existing_results=None):
        """
        Seach local vectors + cached remote vectors
        Input: 
            all vectors = the entire dataset with N_TOTAL d-dimensional vectors, used to do remote search
            if the search is performed on the first server, start the search from scratch, else:
                ep_vec_id (entry point vector ID): if specified, start searching from ground layer using this vector ID
                existing_results: a list of search results on previous servers
        Output:
            a list of local results, in asending distance
            a list of remote results (only with the vectors one hop away from local), in asending distance
            a list of merged results, in asending distance
            whether one should search remote (True/False)
            the remote server ID
            the entry poin vector ID on the remote server
            the graph search path length on the local graph (gnd layer + (if searched) upper layer)
        """
        path_len = None # number of nodes visited in the local graph search
        if ep_vec_id == -1: # search local
            local_results,  search_path_local_ID_upper, search_path_vec_ID_upper, search_path_local_ID_gnd, search_path_vec_ID_gnd = \
                self.searchKnn(q_data, k, ef)
            path_len = len(search_path_local_ID_upper) + len(search_path_local_ID_gnd)
        else:
            ep_local_id = self.vec_ID_to_local_ID[ep_vec_id]
            local_results, search_path_local_ID_gnd, search_path_vec_ID_gnd = \
                self.searchKnnGroundLayer(q_data, k, ef, ep_local_id, existing_results=existing_results)
            path_len = len(search_path_local_ID_gnd)
        # get the list of remote vectors that should be visited
        remote_server_ID_vec_ID_list = []
        
        # only the candidates on ground layer are considered
        for local_ID in search_path_local_ID_gnd:
            link_count = self.remote_links_count[local_ID]
            for i in range(link_count):
                remote_server_ID_vec_ID_list.append(self.remote_links[local_ID][i])
    
        visited_array = set() # visited remote vector IDs
        remote_results_heap = []
        for remote_server_ID, vec_ID in remote_server_ID_vec_ID_list:
            if vec_ID not in visited_array:
                visited_array.add(vec_ID)
                dist = np.sum((q_data - all_vectors[vec_ID]) ** 2)
                heapq.heappush(remote_results_heap, (-dist, remote_server_ID, vec_ID))
            
        while len(remote_results_heap) > k:
            heapq.heappop(remote_results_heap)

        remote_results = []
        while len(remote_results_heap) > 0:
            dist, remote_server_ID, vec_ID = remote_results_heap[0]
            remote_results.append((-dist, remote_server_ID, vec_ID))
            heapq.heappop(remote_results_heap)
        remote_results.reverse()
            
        # Merge local + remote
        results = merge_two_distance_list(local_results, remote_results, k)
            
        if remote_results[0][0] < local_results[0][0]:
            search_remote = True
            remote_server_ID = remote_results[0][1]
        else:
            search_remote = False
            remote_server_ID = -1
            
        remote_ep_vec_ID = remote_results[0][2]
        
        return results, local_results, remote_results, search_remote, remote_server_ID, remote_ep_vec_ID, path_len
        
        
#     def searchKnnPlusRemote(self, q_data, k, ef, all_vectors, remote_hnswlib_indexes, remote_server_IDs, debug=False):
#         """
#         Search local vectors, hop to remote index ***(currently only support 1 index)*** when needed
#             *** Thus, this is only a testing functino, in reality, there should be a global search 
#                 function allowing multiple hops between servers ***
#         Input: 
#             all vectors = the entire dataset with N_TOTAL d-dimensional vectors, used to do remote search
#             remote_hnswlib_indexes: a list of remote_hnswlib_index
#                 remote_hnswlib_index: index loaded by remote memory (hnswlib object)
#             remote_server_IDs: a list of remote index IDs respective to remote_hnswlib_indexes
#                 e.g., this is server 1, and there are four servers in totol,
#                     then the remote index IDs should be [0, 2, 3]
#         Output:
#             a list of local results, in distance ascending order
#             a list of remote results (only with the vectors one hop away from local)
#             a list of merged results
#             whether one should search remote (True/False)
#         """
#         local_plus_cach_results, local_results, remote_results, search_remote, remote_server_ID = \
#             self.searchKnnPlusRemoteCache(q_data, k, ef, all_vectors, debug=debug)
        
#         if search_remote:
#             for i, ids in enumerate(remote_server_IDs):
#                 if ids == remote_server_ID:
#                     remote_hnswlib_index = remote_hnswlib_indexes[i]
            
#             remote_hnswlib_index.set_ef(ef)
#             remote_I, remote_D = remote_hnswlib_index.knn_query(q_data, k=k) # I, D are 2-d array

#             # merge results
#             remote_results = [(remote_D[0][i], remote_server_ID, remote_I[0][i]) for i in range(remote_I.shape[1])]

#             results = merge_two_distance_list(local_results, remote_results, k)
#         else:
#             results = local_plus_cach_results
            
#         return results, search_remote

if __name__ == '__main__':
    
    N_SUBGRAPH = 32
    dim = 128
    
    """ Load hnswlib index """
    all_server_IDs = np.arange(N_SUBGRAPH)
    all_indexes = [hnswlib.Index(space='l2', dim=dim) for i in all_server_IDs]
    parent_dir = '../indexes_subgraph_kmeans/SIFT1M_32_subgraphs'
    all_index_paths=[os.path.join(parent_dir, 'subgraph_{}.bin'.format(i)) for i in all_server_IDs]
    for i in all_server_IDs:
        print("\nLoading hnswlib index from {}\n".format(all_index_paths[i]))
        all_indexes[i].load_index(all_index_paths[i])

    kmeans = load_obj(parent_dir, 'kmeans')
    
    """ Load index as python object, construct distributed index """
    for i in all_server_IDs:
        print("Constructing distributed index from {}\n".format(all_index_paths[i]))
        index = Path(all_index_paths[i]).read_bytes()
        print('size: {} bytes'.format(len(index)))
        
        hnsw_index = HNSW_index(local_server_ID=i, dim=dim)
        hnsw_index.load_meta_info(index)
        
        hnsw_index.set_centroid_vectors(kmeans.cluster_centers_)
        
        ef = hnsw_index.ef_construction_
        
        print('load ground layer...')
        t0 = time.time()
        hnsw_index.load_ground_layer(index)
        t1 = time.time()
        print("time consumption: {:.2f} sec".format(t1 - t0))

        print('load upper layer...')
        t0 = time.time()
        hnsw_index.load_upper_layers(index)
        t1 = time.time()
        print("time consumption: {:.2f} sec".format(t1 - t0))
        
        remote_hnswlib_indexes = []
        remote_server_IDs = []
        for j in all_server_IDs:
            if i == j: 
                continue
            else:
                remote_hnswlib_indexes.append(all_indexes[j])
                remote_server_IDs.append(j)
        print('build remote connections...')
        t0 = time.time()
        hnsw_index.insertRemote(remote_hnswlib_indexes, remote_server_IDs, ef=ef)
        t1 = time.time()
        print("time consumption: {:.2f} sec".format(t1 - t0))
        
        print('saving index object as pkl...')
        save_obj(hnsw_index, parent_dir, 'subgraph_{}_with_remote_edges'.format(i))
        print('finished server {}'.format(i))
        