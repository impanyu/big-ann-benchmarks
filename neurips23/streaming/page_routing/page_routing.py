from neurips23.streaming.base import BaseStreamANN
import data_structures.page_index as page_index
import random

class PageRouting(BaseStreamANN):
    def __init__(self, metric, index_params):
        self.name = "page_routing"
    
    def index_name(self): 
        return self.name
    
    def setup(self, dtype, max_pts, ndims) -> None:
        '''
        Initialize the data structures for your algorithm
        dtype can be 'uint8', 'int8 'or 'float32'
        max_pts is an upper bound on non-deleted points that the index must support
        ndims is the size of the dataset
        '''
        #raise NotImplementedError
        max_neighbors = 50
        index_file = "index.bin"
        meta_data_file = "meta_data.json"

        self.index = page_index(ndims, max_neighbors, index_file, meta_data_file, k=5, L=50, max_visits=1000, nodes_per_page=20, page_buffer_size=100, max_ios_per_hop = 3)

      

    def insert(self, X: np.array, ids: npt.NDArray[np.uint32]) -> None:
        '''
        Implement this for your algorithm
        X is num_vectos * num_dims matrix 
        ids is num_vectors-sized array which indicates ids for each vector
        '''
        for i in range(len(X)):
            x = X[i]
            x_id = ids[i]
            self.index.insert_node(x,x_id)
        #raise NotImplementedError
    
    def delete(self, ids: npt.NDArray[np.uint32]) -> None:
        '''
        Implement this for your algorithm
        delete the vectors labelled with ids.
        '''
        for id in ids:
            self.index.delete_node(id)
        #raise NotImplementedError

    
    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        rand_idx = random.randint(0,len(self.node_ids))
        start_node_id = list(self.index.node_ids.keys())[rand_idx]
        self.res = []
        for i in range(len(X)):
            x = X[i]    
            top_k_node_ids,visited_node_ids = self.search(x, start_node_id, k, self.index.L, self.index.max_visits)
            self.res.append(top_k_node_ids)

        self.res = np.array(self.res)
        #raise NotImplementedError()

    def set_query_arguments(self, query_args):
        pass

    def __str__(self):
        return f'page_routing({self.name})'