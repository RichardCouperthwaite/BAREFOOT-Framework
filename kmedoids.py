# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:01:34 2020

This code is taken from the following Github Repository

https://github.com/SachinKalsi/kmedoids

"""

import numpy as np
from scipy.sparse import csr_matrix
import random

class KMedoids:
    def __init__(self, n_cluster=2, max_iter=10, tol=0.1, start_prob=0.8, end_prob=0.99):
        '''Kmedoids constructor called'''
        if start_prob < 0 or start_prob >= 1 or end_prob < 0 or end_prob >= 1 or start_prob > end_prob:
            raise ValueError('Invalid input')
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tol = tol
        self.start_prob = start_prob
        self.end_prob = end_prob
        
        self.medoids = []
        self.clusters = {}
        self.tol_reached = float('inf')
        self.current_distance = 0
        
        self.__data = None
        self.__is_csr = None
        self.__rows = 0
        self.__columns = 0
        self.cluster_distances = {}
        
        
    def fit(self, data):
        self.__data = data
        self.__set_data_type()     
        self.__start_algo()
        return self
    
    def __start_algo(self):
        self.__initialize_medoids()
        self.clusters, self.cluster_distances = self.__calculate_clusters(self.medoids)
        self.__update_clusters()
 
    def __update_clusters(self):
        for i in range(self.max_iter):
            cluster_dist_with_new_medoids = self.__swap_and_recalculate_clusters()
            if self.__is_new_cluster_dist_small(cluster_dist_with_new_medoids) == True:
                self.clusters, self.cluster_distances = self.__calculate_clusters(self.medoids)
            else:
                break

    def __is_new_cluster_dist_small(self, cluster_dist_with_new_medoids):
        existance_dist = self.calculate_distance_of_clusters()
        new_dist = self.calculate_distance_of_clusters(cluster_dist_with_new_medoids)
        
        if existance_dist > new_dist and (existance_dist - new_dist) > self.tol:
            self.medoids = cluster_dist_with_new_medoids.keys()
            return True
        return False
    
    def calculate_distance_of_clusters(self, cluster_dist=None):
        if cluster_dist == None:
            cluster_dist = self.cluster_distances
        dist = 0
        for medoid in cluster_dist.keys():
            dist += cluster_dist[medoid]
        return dist
        
    def __swap_and_recalculate_clusters(self):
        # http://www.math.le.ac.uk/people/ag153/homepage/KmeansKmedoids/Kmeans_Kmedoids.html
        cluster_dist = {}
        for medoid in self.medoids:
            is_shortest_medoid_found = False
            for data_index in self.clusters[medoid]:
                if data_index != medoid:
                    cluster_list = list(self.clusters[medoid])
                    cluster_list[self.clusters[medoid].index(data_index)] = medoid
                    new_distance = self.calculate_inter_cluster_distance(data_index, cluster_list)
                    if new_distance < self.cluster_distances[medoid]:
                        cluster_dist[data_index] = new_distance
                        is_shortest_medoid_found = True
                        break
            if is_shortest_medoid_found == False:
                cluster_dist[medoid] = self.cluster_distances[medoid]
        return cluster_dist
    
    def calculate_inter_cluster_distance(self, medoid, cluster_list):
        distance = 0
        for data_index in cluster_list:
            distance += self.__get_distance(medoid, data_index)
        return distance/len(cluster_list)
        
    def __calculate_clusters(self, medoids):
        clusters = {}
        cluster_distances = {}
        for medoid in medoids:
            clusters[medoid] = []
            cluster_distances[medoid] = 0
            
        for row in range(self.__rows):
            nearest_medoid, nearest_distance = self.__get_shortest_distance_to_mediod(row, medoids)
            cluster_distances[nearest_medoid] += nearest_distance
            clusters[nearest_medoid].append(row)
        
        for medoid in medoids:
            cluster_distances[medoid] /= len(clusters[medoid])
        return clusters, cluster_distances
        
        
    def __get_shortest_distance_to_mediod(self, row_index, medoids):
        min_distance = float('inf')
        current_medoid = None
        
        for medoid in medoids:
            current_distance = self.__get_distance(medoid, row_index)
            if current_distance < min_distance:
                min_distance = current_distance
                current_medoid = medoid
        return current_medoid, min_distance

    def __initialize_medoids(self):
        '''Kmeans++ initialisation'''
        self.medoids.append(random.randint(0,self.__rows-1))
        while len(self.medoids) != self.n_cluster:
            self.medoids.append(self.__find_distant_medoid())
    
    def __find_distant_medoid(self):
        distances = []
        indices = []
        for row in range(self.__rows):
            indices.append(row)
            distances.append(self.__get_shortest_distance_to_mediod(row,self.medoids)[1])
        distances_index = np.argsort(distances)
        choosen_dist = self.__select_distant_medoid(distances_index)
        return indices[choosen_dist]
    
    def __select_distant_medoid(self, distances_index):
        start_index = round(self.start_prob*len(distances_index))
        end_index = round(self.end_prob*(len(distances_index)-1)) 
        return distances_index[random.randint(start_index, end_index)]

                           
    def __get_distance(self, x1, x2):
        a = self.__data[x1].toarray() if self.__is_csr == True else np.array(self.__data[x1])
        b = self.__data[x2].toarray() if self.__is_csr == True else np.array(self.__data[x2])
        return np.linalg.norm(a-b)
    
    def __set_data_type(self):
        '''to check whether the given input is of type "list" or "csr" '''
        if isinstance(self.__data,csr_matrix):
            self.__is_csr = True
            self.__rows = self.__data.shape[0]
            self.__columns = self.__data.shape[1]
        elif isinstance(self.__data,list):
            self.__is_csr = False
            self.__rows = len(self.__data)
            self.__columns = len(self.__data[0])
        else:
            raise ValueError('Invalid input')
            
            

def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C