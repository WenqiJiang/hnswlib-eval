import sys
import os
import time 

import hnswlib
import numpy as np


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


if __name__ == '__main__':

    dbname = 'SIFT10M'
    index_path='../indexes/{}_index.bin'.format(dbname)

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


    # train index if not exist
    if not os.path.exists(index_path):
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
        p.init_index(max_elements=N_VEC, ef_construction=128, M=16)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        p.set_ef(128)

        # Set number of threads used during batch search/construction
        # By default using all available cores
        # p.set_num_threads(16)

        batch_size = 1000
        batch_num = int(np.ceil(N_VEC / batch_size))
        for i in range(batch_num):
            print("Adding {} th batch of {} elements".format(i, batch_size))
            xbatch = xb[i * batch_size: (i + 1) * batch_size]
            p.add_items(xbatch)


        # Serializing and deleting the index:
        print("Saving index to '%s'" % index_path)
        p.save_index(index_path)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        p.set_ef(128)

        p.set_num_threads(1)
        # Query the elements for themselves and measure recall:
        start = time.time()
        I, D = p.knn_query(xq, k=100)
        end = time.time()
        t_consume = end - start

        print("Searching...")
        print(' ' * 4, '\t', 'R@1    R@10   R@100')
        for rank in 1, 10, 100:
            n_ok = (I[:, :rank] == gt[:, :1]).sum()
            print("{:.4f}".format(n_ok / float(nq)), end=' ')
        print("Search {} vectors in {} sec\tQPS={}".format(nq, t_consume, nq / t_consume))


    # If index exists, load the index
    else:
        p = hnswlib.Index(space='l2', dim=dim)  # the space can be changed - keeps the data, alters the distance function.

        print("\nLoading index from {}\n".format(index_path))

        # Increase the total capacity (max_elements), so that it will handle the new data
        p.load_index(index_path, max_elements=N_VEC)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        p.set_ef(256)

        p.set_num_threads(1)

        # Query the elements for themselves and measure recall:
        start = time.time()
        I, D = p.knn_query(xq, k=100)
        end = time.time()
        t_consume = end - start

        print("Searching...")
        print(' ' * 4, '\t', 'R@1    R@10   R@100')
        for rank in 1, 10, 100:
            n_ok = (I[:, :rank] == gt[:, :1]).sum()
            print("{:.4f}".format(n_ok / float(nq)), end=' ')
        print("\nSearch {} vectors in {} sec\tQPS={}".format(nq, t_consume, nq / t_consume))
