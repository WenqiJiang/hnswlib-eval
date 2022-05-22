import sys
import os
import time 

import pickle
import hnswlib
import numpy as np


from sklearn.cluster import KMeans

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    # Wenqi: Format of ground truth (for 10000 query vectors):
    #   1000(topK), [1000 ids]
    #   1000(topK), [1000 ids]
    #        ...     ...
    #   1000(topK), [1000 ids]
    # 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def choose_train_size(ncentroids):

    # some training vectors for PQ and the PCA
    n_train = 256 * 1000
    n_train = max(n_train, 100 * ncentroids)
    return n_train

def save_obj(obj, dirc, name):
    # note use "dir/" in dirc
    with open(os.path.join(dirc, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, protocol=4) # for py37,pickle.HIGHEST_PROTOCOL=4

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    N_SUBGRAPH = 4
    dbname = 'SIFT1M'
    
    folder_name = dbname + '_{}_subgraphs'.format(N_SUBGRAPH)
    if not os.path.exists('../indexes_subgraph_kmeans'):
        os.mkdir('../indexes_subgraph_kmeans')
    if not os.path.exists(os.path.join('../indexes_subgraph_kmeans', folder_name)):
        os.mkdir(os.path.join('../indexes_subgraph_kmeans', folder_name))
        
    index_path_list= ['../indexes_subgraph_kmeans/{}/subgraph_{}.bin'.format(folder_name, i) for i in range(N_SUBGRAPH)]
    vec_ID_path_list= ['../indexes_subgraph_kmeans/{}/subgraph_{}_vec_IDs.pkl'.format(folder_name, i) for i in range(N_SUBGRAPH)]
    kmeans_path = '../indexes_subgraph_kmeans/{}/kmeans.pkl'.format(folder_name)

    if dbname.startswith('SIFT'):
        # SIFT1M to SIFT1000M
        dbsize = int(dbname[4:-1])
        xb = mmap_bvecs('/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_base.bvecs')
        xq = mmap_bvecs('/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_query.bvecs')
        gt = ivecs_read('/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/idx_%dM.ivecs' % dbsize)

        N_VEC = int(dbsize * 1000 * 1000)

        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]

        # Wenqi: load xq to main memory and reshape
        xq = xq.astype('float32').copy()
        xq = np.array(xq, dtype=np.float32)
        gt = np.array(gt, dtype=np.int32)

        print("Vector shapes:")
        print("Base vector xb: ", xb.shape)
        print("Query vector xq: ", xq.shape)
        print("Ground truth gt: ", gt.shape)
    else:
        print('unknown dataset', dbname, file=sys.stderr)
        sys.exit(1)

    dim = xb.shape[1] # should be 128
    nq = xq.shape[0]

    if not os.path.exists(kmeans_path):
        # K-means
        print('learning k-means clusters...')
        kmeans = KMeans(n_clusters=N_SUBGRAPH)
        train_size = choose_train_size(N_SUBGRAPH)
        print("k-means on {} vectors".format(train_size))
        xt = xb[:train_size]
        kmeans.fit(xt)
        centroid_vectors = kmeans.cluster_centers_
        print('finish learning k-means clusters...')
        with open(kmeans_path, 'wb') as f:
            pickle.dump(kmeans, f, protocol=4) # for py37,pickle.HIGHEST_PROTOCOL=4
    else:
        print('loadingn k-means from {}'.format(kmeans_path))
        with open(kmeans_path, 'rb') as f:
            kmeans = pickle.load(f)
    
    graph_ID_exists = True
    for i in range(N_SUBGRAPH): 
        if not os.path.exists(vec_ID_path_list[i]):
            graph_ID_exists = False
            break
    if not graph_ID_exists:
        # Partition the vectors
        vec_ID_list = [[] for i in range(N_SUBGRAPH)]
        subgraph_IDs = kmeans.predict(xb)
        for vec_id in range(N_VEC):
            partition_ID = subgraph_IDs[vec_id]
            vec_ID_list[partition_ID].append(vec_id)
        for i in range(N_SUBGRAPH):
            with open(vec_ID_path_list[i], 'wb') as f:
                pickle.dump(vec_ID_list[i], f, protocol=4) # for py37,pickle.HIGHEST_PROTOCOL=4
    else: 
        vec_ID_list = [[] for i in range(N_SUBGRAPH)]
        for i in range(N_SUBGRAPH):
            with open(vec_ID_path_list[i], 'rb') as f:
                vec_ID_list[i] = pickle.load(f)

    # train index if not exist
    for partition_ID, index_path in enumerate(index_path_list):

        if not os.path.exists(index_path):
            
            # Generate the list of vectors & IDs
            N_VEC_SUBGRAPH = len(vec_ID_list[partition_ID])
            xb_subgraph = np.zeros((N_VEC_SUBGRAPH, dim))
            vec_ID_subgraph = vec_ID_list[partition_ID]
            print('adding {} vectors to partition {}'.format(N_VEC_SUBGRAPH, partition_ID))
            for i in range(N_VEC_SUBGRAPH):
                xb_subgraph[i] = xb[vec_ID_subgraph[i]]
            
            print("Building index ", index_path)
            # Declaring index
            p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

            # Initing index
            # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
            # during insertion of an element.
            # The capacity can be increased by saving/loading the index, see below.
            #
            # ef_construction - controls index search speed/build speed tradeoff
            #
            # M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
            # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
            p.init_index(max_elements=N_VEC_SUBGRAPH, ef_construction=128, M=16)

            # Controlling the recall by setting ef:
            # higher ef leads to better accuracy, but slower search
            p.set_ef(128)

            # Set number of threads used during batch search/construction
            # By default using all available cores
#             p.set_num_threads(16)

            batch_size = 10000
            batch_num = int(np.ceil(N_VEC_SUBGRAPH / batch_size))
            for i in range(batch_num):
                print("Adding {} th batch of {} elements".format(i, batch_size))
                start_ID = i * batch_size
                end_ID = (i + 1) * batch_size
                
                xbatch = xb_subgraph[start_ID: end_ID]
                id_batch = vec_ID_subgraph[start_ID: end_ID]
                
                p.add_items(xbatch, id_batch)


            # Serializing and deleting the index:
            print("Saving index to '%s'" % index_path)
            p.save_index(index_path)

            print("Searching...")
            start = time.time()
            I, D = p.knn_query(xq, k=100)
            end = time.time()
            t_consume = end - start
            print(' ' * 4, '\t', 'R@1    R@10   R@100')
            for rank in 1, 10, 100:
                n_ok = (I[:, :rank] == gt[:, :1]).sum()
                print("{:.4f}".format(n_ok / float(nq)), end=' ')
            print("\nSearch {} vectors in {} sec\tQPS={}".format(nq, t_consume, nq / t_consume))

            del p

        else:
            print("Index {} exists, skip...".format(index_path))


    # search from all indexes if not exist
    t_total = 0
    n_ok_total = 0
    for partition_ID, index_path in enumerate(index_path_list):

        # load the index and search
        p = hnswlib.Index(space='l2', dim=dim)  # the space can be changed - keeps the data, alters the distance function.

        print("\nLoading index from {}\n".format(index_path))

        # Increase the total capacity (max_elements), so that it will handle the new data
        p.load_index(index_path, max_elements=N_VEC_SUBGRAPH)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        p.set_ef(1)

        p.set_num_threads(1)

        # Query the elements for themselves and measure recall:
        print("Searching...")
        start = time.time()
        I, D = p.knn_query(xq, k=100)
        end = time.time()
        t_consume = end - start
        t_total += t_consume

        print(' ' * 4, '\t', 'R@1    R@10   R@100')
        rank = 1
        # for rank in 1, 10, 100:
        n_ok = (I[:, :rank] == gt[:, :1]).sum()
        n_ok_total += n_ok
        print("{:.4f}".format(n_ok / float(nq)), end=' ')
        print("\nSearch {} vectors in {} sec\tQPS={}".format(nq, t_consume, nq / t_consume))

        del p


    print("\nOverall performance on 10 subgraphs: ")
    print("Search {} vectors in {} sec\tQPS={}".format(nq, t_total, nq / t_total))
    print("R@1 = {}".format(n_ok_total / float(nq)))
