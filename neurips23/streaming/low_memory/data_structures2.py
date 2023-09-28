import numpy as np
import heapq
import json
import networkx as nx
import random
import os
import threading
from readerwriterlock import rwlock
import time
from pyclustering.cluster.kmedoids import kmedoids
from sklearn_extra.cluster import KMedoids


    
class Node:
    def __init__(self, vector, node_id,index, max_neighbors=50, alpha=1.2, max_cluster_number=8):
        self.vector = vector #list of floating point numbers
        self.node_id = node_id
        self.max_neighbors = max_neighbors

        self.neighbor_ids = []
        self.alpha = alpha
        self.index = index
        self.max_cluster_number = max_cluster_number
        self.clusters = []

    def form_clusters(self):
        start_time = time.time()
        self.clusters = []
        vectors = []
        vector_ids = []
        for neighbor_id in self.neighbor_ids:
            neighbor = self.index.get_node(neighbor_id)
            vectors.append(neighbor.get_vector())
            vector_ids.append(neighbor_id)

        if len(vectors) == 0:
            return
        if len(vectors) < self.max_cluster_number:
            for i in range(len(vectors)):
                medoid = vectors[i]
                self.clusters.append({"medoid": medoid, "cluster_member_ids": [vector_ids[i]],"cluster_radius": [0]})
            return

        vectors = np.array(vectors)
        #initial_medoids = random.sample(range(0, len(vectors)), self.max_cluster_number)
        #kmedoids_instance = kmedoids(vectors, initial_medoids)
        #kmedoids_instance.process()
        #clusters = kmedoids_instance.get_clusters()
        #medoids = kmedoids_instance.get_medoids()


        # Create KMedoids instance and fit
        kmedoids = KMedoids(n_clusters = self.max_cluster_number, random_state=0).fit(vectors)      
        
        medoids = kmedoids.cluster_centers_

        clusters = []

        for k in range(self.max_cluster_number):
            cluster = [i for i, x in enumerate(kmedoids.labels_) if x == k]
            clusters.append(cluster)

        
        
        print(len(clusters))
        if len(clusters) != self.max_cluster_number:
            print("cluster number not equal to max cluster number")

        

        for i in range(self.max_cluster_number):
            cluster_member_ids = np.array(vector_ids)[clusters[i]]
            medoid = medoids[i]
            cluster_radius = np.linalg.norm(vectors[clusters[i]] - medoid,axis=1)
            self.clusters.append({"medoid": medoid, "cluster_member_ids": list(cluster_member_ids),"cluster_radius": list(cluster_radius)})
 
        
        end_time = time.time()
        #print("form clusters time: ", end_time - start_time)


    def remove_deleted_neighbors(self):
        for i in range(len(self.neighbor_ids)-1,-1,-1):
            neighbor_id = self.neighbor_ids[i]
            #neighbor = self.index.get_node(neighbor_id)
            if neighbor_id not in self.index.node_ids:
                self.neighbor_ids.pop(i)

    def add_neighbor(self, new_neighbor_id):
        self.neighbor_ids.append(new_neighbor_id)

        if len(self.neighbor_ids) > self.max_neighbors:
            self.remove_deleted_neighbors()
            self.prune_neighbors()
        
        self.form_clusters()

    def add_neighbors(self, new_neighbor_ids):
        self.neighbor_ids = self.neighbor_ids + new_neighbor_ids
        if len(self.neighbor_ids) > self.max_neighbors:
            self.remove_deleted_neighbors()
            self.prune_neighbors()
        self.form_clusters()

    def find_nearest_neighbors(self):
        start_time = time.time()
        priority_queue = []
        heapq.heapify(priority_queue)
        for neighbor_id in self.neighbor_ids:
            neighbor = self.index.get_node(neighbor_id)
  
            distance = self.get_distance(neighbor.get_vector())
            heapq.heappush(priority_queue, (distance, neighbor_id))
        if len(priority_queue) == 0:
            return None, None
        distance, nearest_neighbor_id = heapq.heappop(priority_queue)
        end_time = time.time()
        #print("find nearest neighbors time: ", end_time - start_time)
        return distance,nearest_neighbor_id
    
    #
    def prune_neighbors(self):
        start_time = time.time()
        neighbor_ids = []
        while len(self.neighbor_ids) > 0:
            distance, nearest_neighbor_id = self.find_nearest_neighbors()
            if nearest_neighbor_id is None:
                break


            nearest_neighbor = self.index.get_node(nearest_neighbor_id)
            neighbor_ids.append(nearest_neighbor_id)
            if len(neighbor_ids) >= self.max_neighbors:
                break

            for i in range(len(self.neighbor_ids)-1,-1,-1):
                neighbor_id = self.neighbor_ids[i]
                neighbor = self.index.get_node(neighbor_id)
                if neighbor is None:
                    #self.neighbor_ids.remove(neighbor_id)
                    continue

                distance_1 = nearest_neighbor.get_distance(neighbor.get_vector())
                distance_2 = self.get_distance(neighbor.get_vector())

                if self.alpha * distance_1 < distance_2:
                    self.neighbor_ids.pop(i)
     
  
        self.neighbor_ids = neighbor_ids
        end_time = time.time()
        #print("prune time: ", end_time - start_time)

    def remove_neighbor(self, neighbor_id):
        if neighbor_id in self.neighbor_ids:
            self.neighbor_ids.remove(neighbor_id)
        
        self.form_clusters()
        
    def get_neighbor_ids(self):
    
        return self.neighbor_ids
    
    def get_vector(self):
        return self.vector
    
    def set_vector(self, new_vector):
        self.vector = new_vector
    
    def get_id(self):
        return self.node_id
    
    def get_clusters(self):
        return self.clusters

    def get_distance(self, other_vector):
        #print(len(self.vector))
        #print(len(other_vector))
        
        return np.sum(np.square(np.array(self.vector) - np.array(other_vector)))
        #np.linalg.norm(np.array(self.vector) - np.array(other_vector))
    
    def get_neighbor_distance(self, neighbor_id, vector):
        for cluster in self.clusters:
            if neighbor_id in cluster["cluster_member_ids"]:
                radius = cluster["cluster_radius"][cluster["cluster_member_ids"].index(neighbor_id)]
                d = np.linalg.norm(np.array(vector) - np.array(cluster["medoid"]))
                return abs(d-radius),d+radius

        return None
    



class low_memory_index:
    def __init__(self, dim, max_neighbors, index_file, meta_data_file, k=5, L=50, max_visits=200, nodes_per_page=20, node_buffer_size=100, max_ios_per_hop = 3,max_cluster_number = 8):
        self.k = k
        self.L = L
        self.max_visits = max_visits
        self.dim = dim
        self.max_neighbors = max_neighbors

        self.index_file = index_file
        self.meta_data_file = meta_data_file


        self.node_ids = {}

        self.node_w_buffer = []
        self.node_r_buffer = []


        self.marker = rwlock.RWLockFair()
        # Create a lock
        self.lock = threading.Lock()

  
        self.node_buffer_size = node_buffer_size
        self.node_buffer = {}

        self.pq_size = self.dim
        self.max_cluster_number = max_cluster_number

        self.node_size = 1+ self.dim + self.max_neighbors*2 + self.max_cluster_number*(2+self.pq_size) 

        try:
            '''
            if os.path.exists(self.meta_data_file):
                with open(self.meta_data_file, 'r') as f:
                    
                    self.meta_data = json.load(f)
                    self.node_ids = self.meta_data['node_ids']
                    self.available_page_ids = self.meta_data['available_page_ids']
                    self.available_node_ids = self.meta_data['available_node_ids']
            '''
            #else:


            self.available_node_ids = {"first_available_node_id":0, "deleted_node_ids":[]}

            self.meta_data = {'node_ids': self.node_ids, 'available_node_ids':self.available_node_ids}
            with open(self.meta_data_file, 'w') as f:
                json.dump(self.meta_data, f)

            with open(self.index_file, 'wb') as f:
                f.write(np.array([]).tobytes())

        except Exception as e:
            print(f"An error occurred: {e}")



    def add_to_node_r_buffer(self,node):
        #with self.page_buffer_lock:
        for i in range(len(self.node_r_buffer)):
            if node.get_id() == self.node_r_buffer[i]:
                self.node_r_buffer.pop(i)
                break

        self.node_r_buffer.append(node.get_id())
        self.node_buffer[node.get_id()] = node

        # remove the first node from the buffer if the buffer is full
        if len(self.node_r_buffer) > self.node_buffer_size:
            popped_node_id = self.node_r_buffer.pop(0)
            if popped_node_id not in self.node_w_buffer:
                del self.node_buffer[popped_node_id]

    def add_to_node_w_buffer(self,node):
        #with self.page_buffer_lock:
        for i in range(len(self.node_w_buffer)):
            if node.get_id() == self.node_w_buffer[i]:
                return

        self.node_w_buffer.append(node.get_id())
        self.node_buffer[node.get_id()] = node

        if len(self.node_w_buffer) >= self.node_buffer_size:
            self.dump_changed_nodes()

    def remove_from_node_w_buffer(self,node):
        #with self.page_buffer_lock:
        

        for i in range(len(self.node_w_buffer)):
            if node.get_id() == self.node_w_buffer[i]:
                self.node_w_buffer.pop(i)
  
                break

        if node.get_id() in self.node_buffer and node.get_id() not in self.node_r_buffer:
            del self.node_buffer[node.get_id()]
    
    def remove_from_node_r_buffer(self,node):
        #with self.page_buffer_lock:
        for i in range(len(self.node_r_buffer)):
            if node.get_id() == self.node_r_buffer[i]:
                self.node_r_buffer.pop(i)
                break

        if node.get_id() in self.node_buffer and node.get_id() not in self.node_w_buffer:
            del self.node_buffer[node.get_id()]



    def get_aviailable_node_id(self):
        #with  self.available_node_ids_lock:
        if len(self.available_node_ids["deleted_node_ids"]) == 0:
            self.available_node_ids["first_available_node_id"] = self.available_node_ids["first_available_node_id"]+1
            return self.available_node_ids["first_available_node_id"]-1
        else:
            return self.available_node_ids["deleted_node_ids"].pop(0)

            

    def dump_changed_node(self, node):  
        node_id = node.get_id()
        #self.index_file_rw_lock.acquire_write()
  
        with open(self.index_file, 'rb+') as f:
            f.seek(node_id *self.node_size*4)
            clusters = node.get_clusters()

            node_data = np.append([node.get_id()], node.get_vector())
            

            for cluster_id in range(len(node.clusters)):
                cluster = clusters[cluster_id]
                node_data =np.append(node_data,cluster_id)
                node_data =np.append(node_data,len(cluster["cluster_member_ids"]))
                #print(f'cluster_size {len(cluster["cluster_member_ids"])}')
                node_data =np.append(node_data,cluster["medoid"])
                node_data =np.append(node_data,cluster["cluster_member_ids"])
                node_data = np.append(node_data,cluster["cluster_radius"])
                

           
            #padding within node with -1s
            if len(node_data) < self.node_size:
                node_data =np.append(node_data,np.full(self.node_size-len(node_data),-1))
            #print(len(node_data))
            
                        
            f.write(node_data.astype(np.float32).tobytes())


        #self.index_file_rw_lock.release_write()

    def dump_changed_nodes(self):
        for node_id in self.node_w_buffer:
            node = self.node_buffer[node_id]
            self.dump_changed_node(node)
            self.node_w_buffer.remove(node_id)

            if node_id not in self.node_r_buffer:
                del self.node_buffer[node_id]
            
            self.dump_changed_node(node)
           


    def dump_meta_data(self):
        with open(self.meta_data_file, 'w') as f:
            json.dump(self.meta_data, f)



    def get_node_from_file(self, node_id):
        #self.index_file_rw_lock.acquire_read()
        with open(self.index_file, 'rb') as f:
            # index_file is a binary file, so we need to seek to the correct position
            print(node_id )
            f.seek(node_id *self.node_size*4)
            # read the node from the file
            try:
                node_data = np.fromfile(f, dtype=np.float32, count=int(self.node_size))
            except Exception as e:
                return None
        #self.index_file_rw_lock.release_read()
        
        vector = node_data[1:self.dim+1]
        node = Node(vector, node_id, self,self.max_neighbors)

        shift = self.dim+1

        

       
       

        for i in range(self.max_cluster_number):
            
            cluster_id = int(node_data[shift])
            if cluster_id == -1:
                break
            cluster_size = int(node_data[shift+1])
            cluster_medoid = node_data[shift+2:shift+2+self.pq_size]
            cluster_member_ids = node_data[shift+2+self.pq_size:shift+2+self.pq_size+cluster_size]
            cluster_member_ids = [int(cluster_member_id) for cluster_member_id in cluster_member_ids]

            node.neighbor_ids = node.neighbor_ids + cluster_member_ids

            cluster_radius = node_data[shift+2+self.pq_size+cluster_size:shift+2+self.pq_size+cluster_size*2]

            node.clusters.append({"medoid": cluster_medoid, "cluster_member_ids": cluster_member_ids,"cluster_radius": cluster_radius})
           
            shift = shift + 2 + self.pq_size + cluster_size*2

        self.add_to_node_r_buffer(node)

        return node
    
    def get_node(self, node_id):
        #with self.page_buffer_lock:
        if node_id in self.node_buffer:
            return self.node_buffer[node_id]
        else:
            node = self.get_node_from_file(node_id)
            if not node is None:
                self.add_to_node_r_buffer(node)
            return node


    def insert_node(self, vector, new_node_id = None):
        #self.rw_lock.acquire_write()
        #w_lock = self.marker.gen_rlock()
        #w_lock.acquire()
        print(f"insert {new_node_id}")
        start_time = time.time()


        if new_node_id is None:
            new_node_id = self.get_aviailable_node_id()

        else:
            
            if new_node_id in self.node_ids:
                new_node = self.get_node(new_node_id)
                new_node.set_vector(vector)
                return


        new_node = Node(vector, new_node_id, self, self.max_neighbors)

        with self.lock:
            self.add_to_node_w_buffer(new_node)
            self.add_to_node_r_buffer(new_node)
            self.node_ids[new_node_id] = new_node_id

    
 
        top_k_node_ids,visited_node_ids = self.search(vector, 0, self.k, self.L, self.max_visits)



        with self.lock:
            new_node.add_neighbors(list(visited_node_ids))
   
            #print(f"find best page time: {end_time_3-start_time_3}")
            

            #print(f"add to buffer time: {end_time_1-start_time_1}")


        
            # add the new node to the neighbor list of the neighbors
            for neighbor_id in new_node.get_neighbor_ids():
                
    
                if neighbor_id in self.node_ids:
                    neighbor = self.get_node(neighbor_id)

                    neighbor.add_neighbor(new_node_id)
                
                    self.add_to_node_w_buffer(neighbor)

                
        #print(f"add neighbors time: {end_time_2-start_time_2}")

                    #self.changed_pages[neighbor_page_id] = self.get_page(neighbor_page_id)
        #print("after add neighbors")
        #self.rw_lock.release_write()
        #w_lock.release()
        end_time = time.time()
        
        print(f"insert {new_node_id} time: {end_time-start_time}")



   
    #in some case delete_node may not delete the link pointing to the deleted node, so deleted node may still be in the neighbor list of other nodes
    def delete_node(self, node_id):
        #w_lock = self.marker.gen_wlock()
        #w_lock.acquire()

        #print(f"delete {node_id}")
    
        #print("deleting node")
        
        if node_id not in self.node_ids:
            return 
        
        deleted_node = self.get_node(node_id)
        #page.get_lock().release_write()

        
        #self.add_to_page_buffer(page)

        #self.changed_pages[page_id] = page
        #with self.available_node_ids_lock:
        with self.lock:
            self.available_node_ids["deleted_node_ids"].append(node_id)
            self.remove_from_node_r_buffer(deleted_node)
            self.remove_from_node_w_buffer(deleted_node)
            del self.node_ids[node_id]

    
        # iterate through all the neighbors of the node and remove the node from their neighbor list
        # also add the neighbors of the node to the neighbor list of the neighbors to perserve the links: when a->delete_node and delete_node -> b, we add a->b
        #page.get_lock().acquire_read()
            for neighbor_id in deleted_node.get_neighbor_ids():
                
                if neighbor_id not in self.node_ids:
                    continue

                neighbor = self.get_node(neighbor_id)
                
                if node_id in neighbor.get_neighbor_ids():
                    neighbor.remove_neighbor(node_id)
                    other_neighbor_ids =[other_neighbor_id for other_neighbor_id in deleted_node.get_neighbor_ids() if other_neighbor_id != neighbor_id]
                    neighbor.add_neighbors(other_neighbor_ids)

                    self.add_to_node_w_buffer(neighbor)

        

        #w_lock.release()

    def search(self, query_vector, start_node_id, k, L, max_visits):
        start_time = time.time()
        # This priority queue will keep track of nodes to visit
        # Format is (distance, node)
        if len(self.node_ids) == 0:
            
            return [],[]
        


        rand_idx = random.randint(0,len(self.node_ids)-1)
        start_node_id = list(self.node_ids.keys())[rand_idx]

        start_node = self.get_node(start_node_id)
        dis = start_node.get_distance(query_vector)
        to_visit = [ start_node_id]

        to_visit_distances = {start_node_id:(dis,dis)}

        
        #heapq.heapify(to_visit)
        visited = set() # Keep track of visited nodes
        #queried_nodes = set()

    
        #num_visits = 0
    

        while len(visited) < max_visits:
  

            to_visit.sort(key=lambda x: (to_visit_distances[x][0]+to_visit_distances[x][1])/2)

            to_visit = to_visit[:L]

            for i in range(len(to_visit)):
                current_node_id = to_visit[i]
                if current_node_id not in visited:
                    break

            if i == len(to_visit)-1 and current_node_id in visited:
                break
            #print(current_node_id)
            

            # Mark this node as visited
            visited.add(current_node_id)
            

            current_node = self.get_node(current_node_id)

            # Add neighbors to the to_visit queue
            neighbor_ids = current_node.get_neighbor_ids()

            for neighbor_id in neighbor_ids:
                if neighbor_id in visited:
                    continue
                if neighbor_id not in self.node_ids:
                    continue
                neighbor_distance_a, neighbor_distance_b = current_node.get_neighbor_distance(neighbor_id,query_vector)

                if neighbor_id not in to_visit_distances:
                    to_visit_distances[neighbor_id] = (neighbor_distance_a,neighbor_distance_b)
                    to_visit.append(neighbor_id)

                else:
                    to_visit_distances[neighbor_id] = (max(neighbor_distance_a,to_visit_distances[neighbor_id][0]),min(neighbor_distance_b,to_visit_distances[neighbor_id][1]))
                

            



        top_k_node_ids = [to_visit[i] for i in range(min(k,len(to_visit)))]
        if len(top_k_node_ids) < k:
            top_k_node_ids.extend([0] * (k - len(top_k_node_ids)))

        end_time = time.time()
        #print("search time: ", end_time - start_time)
        #print(len(visited))
        return top_k_node_ids,list(visited)




if __name__ == '__main__':

    index = low_memory_index(100, 50, 'index.bin', 'node_ids.json')
    #load data from hdf5 file




