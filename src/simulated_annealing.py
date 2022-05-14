import os
from math import *
import random
import numpy as np 
import data 
import time
import matplotlib
import matplotlib.pyplot as plt

class SimulatedAnnealing(object):
    """The simulated annealing algorithm

    Args:
        base_name [str]: [the base name of the problem, a280 for example]
        n_iters [int]: [the max iteration time of each T]
        decay_rate [double]: [the decay rate of T]
        start_T [double]: [the starting temperature]
        end_T [double]: [the ending temperature]
    """
    def __init__(self, base_name='burma14', n_iters=50, decay_rate=0.99, start_T=100, end_T=0.1):
        #load data
        self.N, self.graph, self.ground_truth = data.read_tsp_data(base_name)
        
        #super parameters
        self.base_name = base_name
        self.n_iters = n_iters
        self.decay_rate = decay_rate
        self.start_T = start_T
        self.end_T = end_T

        #init basic
        np.random.seed(1453)
        random.seed(1453)
        self.current_solution = np.arange(self.N)
        np.random.shuffle(self.current_solution) #random shuffle to get current best
        self.current_solution = self.current_solution.tolist()
        self.current_best = self.get_total_distance(self.current_solution)
        self.best_solution = self.current_solution
        self.T = self.start_T
        self.result_list = []
        self.ground_truth_list = []
        
    def solve(self):
        """The main function of simulated annealing
        """
        start = time.time()
        while self.T >= self.end_T:
            self.T = self.T * self.decay_rate
            for iter_ in range(self.n_iters):
                new_travel_order = self.sample_adjacent(self.current_solution)
                new_distance = self.get_total_distance(new_travel_order)
                if new_distance < self.current_best:
                    self.current_solution = new_travel_order
                    self.best_solution = new_travel_order
                    self.current_best = new_distance
                else: 
                    accept_probability = exp(-(new_distance - self.current_best) / self.T)
                    p = random.random()
                    if p <= accept_probability:
                        self.current_solution = new_travel_order
                        
                self.result_list.append(self.current_best)
                self.ground_truth_list.append(self.ground_truth)
                print('T: {:.4f}'.format(self.T))
                print('iter {}/{}:'.format(iter_ + 1, self.n_iters))
                print('current best result: {:.4f}'.format(self.current_best))
                print('ground truth result: {:.4f}'.format(self.ground_truth))
                print('rate: {:.4f}'.format(self.current_best / self.ground_truth))
        end = time.time()
        self.total_time = end - start
        print('Total time cost: {:.2f}s'.format(self.total_time))
                
        
    def get_total_distance(self, travel_order):
        """Get the total distance of a travel order

        Args:
            travel_order [int array], [N]: [the travelling order]
        
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
    
    def sample_adjacent(self, travel_order):
        """Swap the travel order to get an adjacent travel order

        Args:
            travel_order [int array], [N]: [the original travelling order]
        
        Returns:
            new_travel_order [int array], [N]: [the new travelling order]
        """
        #sample p and q, 0 <= p < q < N
        p = random.randint(0, self.N - 2)
        q = random.randint(p + 1, self.N - 1)
        
        #get new travel order
        new_travel_order = travel_order.copy()
        pth = travel_order[p]
        qth = travel_order[q]
        new_travel_order[q] = pth 
        new_travel_order[p] = qth
        return new_travel_order
    
    def plot_result(self, save_path):
        """Plot the results
        Args:
            save_path [str]: [the full saving path]
        """
        matplotlib.use('agg')
        x = range(0, len(self.ground_truth_list))
        plt.plot(x, self.result_list)
        plt.plot(x, self.ground_truth_list)
        plt.xlabel('Iterations')
        plt.ylabel('TSP min distances')
        plt.title("The TSP results of simulated annealing")
        plt.legend(['Current Best Results', 'Ground Truth Results'])   
        plt.savefig(save_path) 
        plt.close()
        
    def save_result(self):
        """Save the results
        """
        base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, 'results'))
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        save_path = os.path.join(base_path, 'simulated_annealing')
        if not os.path.exists(save_path):
            os.mkdir(save_path)        
        txt_path = os.path.join(save_path, self.base_name + '.txt')
        png_path = os.path.join(save_path, self.base_name + '.png')

        f = open(txt_path, 'w')
        f.write('Algorithm: Simulated Annealing\n')
        f.write('data: ' + self.base_name + '\n')
        f.write('N: '+ str(self.N) + '\n')
        f.write('iteration times of each T: '+ str(self.n_iters) + '\n')
        f.write('decay rate: ' + str(self.decay_rate) + '\n')
        f.write('start T: ' + str(self.start_T) + '\n')
        f.write('end T: ' + str(self.end_T) + '\n')
        f.write('ground truth: {:.2f}'.format(self.ground_truth) + '\n')
        f.write('my result: {:.2f}'.format(self.current_best) + '\n')
        f.write('rate: {:.2f}'.format(self.current_best / self.ground_truth) + '\n')
        f.write('time cost: {:.2f}s'.format(self.total_time) + '\n')
        f.write('best route: \n')
        text = ''
        for i in range(len(self.best_solution)):
            text += str(self.best_solution[i])
            text += ' '
        f.write(text)
        f.close()
        self.plot_result(png_path)
        
        
if __name__ == '__main__':  
    a = SimulatedAnnealing()     
    a.solve()
    a.save_result()
        