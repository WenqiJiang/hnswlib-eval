import kahip
import pickle
import os

#build adjacency array representation of the graph

"""
# Wenqi: node 0 -> start from position 0 of the list below
# Wenqi: node 1 -> start from position 2 of the list below
# Wenqi: dummy node -> start from 12 -> out of range of the adjncy list; 
xadj           = [0,2,5,7,9,12]; # Wenqi: node 0 -> start from position 0 of the list below
adjncy         = [1,4,0,2,4,1,3,2,4,0,1,3];
vwgt           = [1,1,1,1,1]
adjcwgt        = [1,1,1,1,1,1,1,1,1,1,1,1]
print("length of xadj: {}\tadjncy: {}\tvwg: {}\tadjcwgt: {}".format(len(xadj), len(adjncy), len(vwgt), len(adjcwgt)))
supress_output = 0
imbalance      = 0.03
nblocks        = 3 
seed           = 0
"""

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

links = load_obj('/mnt/scratch/wenqi/hnswlib-eval/indexes', 'SIFT1M_upper_layer_links')

print(len(links))
for i in range(20):
    print(links[i])

link_count = 0
xadj = []
for i in range(len(links)):
    xadj.append(link_count) # start
    if len(links[i]) > 0:
        adjncy += links[i]
        link_count += len(links[i])

xadj.append(link_count) # last dummy node

vwgt = list(np.ones(len(links))
adjcwgt = list(np.ones(len(adjncy))
               
supress_output = 0
imbalance      = 0.03
nblocks        = 3 
seed           = 0

# set mode 
#const int FAST           = 0;
#const int ECO            = 1;
#const int STRONG         = 2;
#const int FASTSOCIAL     = 3;
#const int ECOSOCIAL      = 4;
#const int STRONGSOCIAL   = 5;
mode = 2 

edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, 
                              adjncy,  nblocks, imbalance, 
                              supress_output, seed, mode)

print("Edge cut", edgecut)
print("Blocks", blocks)
