import os
from math import *
import random
import numpy as np 
import data 
import time
import matplotlib.pyplot as plt

class TabuSearch(object):
    """The tabu search algorithm
        Args:
            base_name [str]: [the base name of the problem, a280 for example]
            n_iters [int]: [the max iteration time]
            n_candidates [int]: [the number of candidates]
            forbidden_length [int]: [the length of forbidden list]
    """
    
    def __init__(self, base_name='burma14', n_iters=500, n_candidates=20, forbidden_length=5):
        #load data
        self.N, self.graph, self.ground_truth = data.read_tsp_data(base_name)
        
        
        #super parameters
        self.n_iters = n_iters
        self.n_candidates = n_candidates
        self.forbidden_length = forbidden_length
        
        #init basic
        self.current_solution = np.arange(self.N)
        np.random.seed(1453)
        random.seed(1453)
        np.random.shuffle(self.current_solution) #random shuffle to get current best
        self.current_best = self.get_total_distance(self.current_solution)
        self.global_solution = self.current_solution
        self.forbidden_list = []
        for i in range(self.forbidden_length):
            self.forbidden_list.append(None)
        
        #recording
        self.result_list = []
        self.result_list.append(self.current_best)
        self.ground_truth_list = [self.ground_truth] * (1 + self.n_iters)
    
    def solve(self):
        """The main algorithm of tabu search
        """
        start = time.time()
        for iter_ in range(1, self.n_iters + 1):
            #get the random candidates and sort them by cost
            candidates = []
            candidate_shuffles = self.sample_shuffles()
            for i in range(self.n_candidates):
                p = candidate_shuffles[i][0]
                q = candidate_shuffles[i][1]
                pth = self.current_solution[p]
                qth = self.current_solution[q]
                smaller = min(pth, qth)
                bigger = max(pth, qth)
                new_travel_order, smaller, bigger = self.shuffle_travel_order(self.current_solution, p, q)
                new_distance = self.get_total_distance(new_travel_order)
                new_candidate = {'travel_order': new_travel_order, 'distance': new_distance, 'smaller': smaller, 'bigger': bigger}
                candidates.append(new_candidate)
            candidates.sort(key=lambda x: x['distance'])
            
            #get the best candidate that is not forbidden or aspirated
            for candidate in candidates:
                travel_order = candidate['travel_order']
                distance = candidate['distance']
                swap = [candidate['smaller'], candidate['bigger']]
                forbidden_place = self.search_in_forbidden_list(swap)
                if distance < self.current_best:
                    self.current_best = distance 
                    self.global_solution = travel_order
                    
                #not in forbidden list
                if forbidden_place == -1:
                    self.current_solution = travel_order
                    self.forbidden_list.pop()
                    self.forbidden_list.insert(0, swap)
                    break
                    
                #aspirated
                elif distance < self.current_best:
                    self.current_solution = travel_order
                    self.current_best = distance 
                    del(self.forbidden_list[forbidden_place])
                    self.forbidden_list.insert(0, swap)
                    break
            
            self.result_list.append(self.current_best)
            print('iter {}/{}:'.format(iter_, self.n_iters))
            print('current best result: {:.4f}'.format(self.current_best))
            print('ground truth result: {:.4f}'.format(self.ground_truth))
            print('rate: {:.4f}'.format(self.current_best / self.ground_truth))
        end = time.time()
        print('Total time cost: {:.2f}s'.format(end - start))

            
    def get_total_distance(self, travel_order):
        """Get the total distance of a travel order

        Args:
            travel_order [numpy int array], [N]: [the travelling order]
        
        Returns:
            total_distance [double]: [the total distance]
        """
        total_distance = 0.0 
        for i in range(self.N):
            this_ = travel_order[i]
            next_ = travel_order[(i + 1) % self.N]
            distance = self.graph[this_][next_]
            total_distance += distance 
        return total_distance
    
    def shuffle_travel_order(self, travel_order, p, q):
        """Shuffle the travel order to get an adjacent travel order

        Args:
            travel_order [numpy int array], [N]: [the original travelling order]
            p [int]: [the id of the travel order to be shuffled, 0 <= p < q < N]
            q [int]: [the id of the travel order to be shuffled, 0 <= p < q < N]
        
        Returns:
            new_travel_order [numpy int array], [N]: [the new travelling order]
            smaller [int]: [the smaller id to be shuffled]
            bigger [int]: [the bigger id to be shuffled]
        """
        #get new travel order
        new_travel_order = travel_order.copy()
        pth = travel_order[p]
        qth = travel_order[q]
        new_travel_order[q] = pth 
        new_travel_order[p] = qth
        smaller = min(pth, qth)
        bigger = max(pth, qth)
        return new_travel_order, smaller, bigger
    
    def sample_shuffles(self):
        """Sample the random shuffles
        
        Returns:
            random_shuffles [numpy int array][n_candidates * 2]: [the K random shuffles to form the candidates]
        """
        #get random integers
        random_shuffles = np.zeros((self.n_candidates, 2), dtype=np.int32)
        max_number = int(self.N * (self.N - 1) / 2)
        random_list = random.sample(range(0, max_number), self.n_candidates)
        
        #switch them to shuffles, get p and q for each integers
        floor = []
        floor.append(0)
        for i in range(self.N - 1):
            floor.append(floor[i] + self.N - i - 1)
        for i in range(len(random_list)):
            m = random_list[i]
            for j in range(self.N - 1):
                if m >= floor[j] and m < floor[j + 1]:
                    p = j
                    q = m - floor[j] + 1 + p
                    break 
            random_shuffles[i][0] = p 
            random_shuffles[i][1] = q 
        return random_shuffles
                
    def search_in_forbidden_list(self, swap):
        """Search one travel order in the forbidden list
        Args:
            swap [list of two integers]: [swapping [p, q], 0 <= p < q < N]
        Returns:
            place [int]: [the place of the travel order in the forbidden list. if not in forbidden list, return -1]
        """    
        place = -1
        for i in range(self.forbidden_length):
            the_forbidden = self.forbidden_list[i]
            if the_forbidden is None:
                continue 
            if the_forbidden[0] == swap[0] and the_forbidden[1] == swap[1]:
                place = i 
                break 
        return place 
    
    def plot_result(self):
        """Plot the results
        """
        x = range(0, self.n_iters + 1)
        plt.plot(x, self.result_list)
        plt.plot(x, self.ground_truth_list)

        plt.xlabel('Iterations')
        plt.ylabel('TSP min distances')
        plt.title("The TSP results of tabu search")
        plt.legend(['Current Best Results', 'Ground Truth Results'])    
        plt.show()
        
if __name__ == '__main__':
    a = TabuSearch(base_name='burma14', n_iters=500, n_candidates=20, forbidden_length=5)
    a.solve()
    a.plot_result()

        
        



