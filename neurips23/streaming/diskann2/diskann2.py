from neurips23.streaming.base import BaseStreamingANN
from .data_structures2 import diskann2_index
import random
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

class diskann2(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.name = "diskann2"
        self.insert_threads = index_params.get("insert_threads")
        self.delete_threads = self.insert_threads
    
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

        self.index = diskann2_index(ndims, max_neighbors, index_file, meta_data_file, k=1, L=20, max_visits=200, nodes_per_page=20, node_buffer_size=1000, max_ios_per_hop = 3)

      

    def insert(self, X: np.array, ids: np.ndarray[np.uint32]) -> None:
        '''
        Implement this for your algorithm
        X is num_vectos * num_dims matrix 
        ids is num_vectors-sized array which indicates ids for each vector
        '''

        for i in range(len(X)):
                x = X[i]
                x_id = ids[i]
                self.index.insert_node(x, x_id)
        '''
        with ThreadPoolExecutor(max_workers=self.insert_threads) as executor:
            # Submit tasks to the executor
            for i in range(len(X)):
                x = X[i]
                x_id = ids[i]
                executor.submit(self.index.insert_node, x, x_id)
        '''
        '''
        threads = []

        for i in range(len(X)):
            x = X[i]
            x_id = ids[i]

            thread = threading.Thread(target=self.index.insert_node, args=(x,x_id))
            thread.start()
            threads.append(thread)

            
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        '''
    
    def delete(self, ids: np.ndarray[np.uint32]) -> None:
        '''
        Implement this for your algorithm
        delete the vectors labelled with ids.
        '''
        '''
        with ThreadPoolExecutor(max_workers=self.delete_threads) as executor:
            # Submit tasks to the executor
            for id in ids:

                executor.submit(self.index.delete_node, id)

        '''
        for id in ids:
            self.index.delete_node(id)


        '''
        threads = []
        for id in ids:
            thread = threading.Thread(target=self.index.delete_node, args=(id))
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        '''

    
    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        #rand_idx = random.randint(0,len(self.index.node_ids)-1)
        #start_node_id = list(self.index.node_ids.keys())[rand_idx]

        self.res = []

        futures = {}

        
        #xs = X.tolist()
        '''
        with ThreadPoolExecutor() as executor:
            for i in range(len(X)):
                x = X[i]
                future = executor.submit(self.index.search, x, 0, k, self.index.L, self.index.max_visits)
                futures[i] = future
            #results = list(executor.map(self.index.search, xs, [start_node_id]*len(xs), [k]*len(xs), [self.index.L]*len(xs), [self.index.max_visits]*len(xs)))
        
        for i in range(len(X)):
            result = futures[i].result()
            self.res.append(result[0])
        
        '''
        
        for x in X:
            self.res.append(self.index.search(x, 0, k, self.index.L, self.index.max_visits)[0])
        
    def set_query_arguments(self, query_args):
        pass

    def __str__(self):
        return f'{self.name}({self.name})'