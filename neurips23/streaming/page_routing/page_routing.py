from neurips23.streaming.base import BaseStreamANN

class PageRouting(BaseStreamANN):
    def __init__(self, metric, index_params):
        self.name = "page_routing"
        


    
    def setup(self, dtype, max_pts, ndims) -> None:
        '''
        Initialize the data structures for your algorithm
        dtype can be 'uint8', 'int8 'or 'float32'
        max_pts is an upper bound on non-deleted points that the index must support
        ndims is the size of the dataset
        '''
        #raise NotImplementedError
        




    def insert(self, X: np.array, ids: npt.NDArray[np.uint32]) -> None:
        '''
        Implement this for your algorithm
        X is num_vectos * num_dims matrix 
        ids is num_vectors-sized array which indicates ids for each vector
        '''
        for x in X:
            self.index.insert_node(x)
        raise NotImplementedError
    
    def delete(self, ids: npt.NDArray[np.uint32]) -> None:
        '''
        Implement this for your algorithm
        delete the vectors labelled with ids.
        '''
        raise NotImplementedError

    
    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        raise NotImplementedError()