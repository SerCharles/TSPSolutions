import os
from math import *
import random
import numpy as np 
import data 
import time
import matplotlib
import matplotlib.pyplot as plt

class GeneticAlgorithm(object):
    """The Genetic Algorithm
    
    Args:
        base_name [str]: [the base name of the problem, a280 for example]
        n_iters [int]: [the max iteration time]
        group_number [int]: [the number of the group at the starting of each iteration]
        cross_probability [double]: [the probability of genetic cross]
        mutation_probability [double]: [the probability of mutation]
        fitness_mode [str]: [the mode of fitness function, order/distance]
    """
    def __init__(self, base_name='burma14', n_iters=200, group_number=50, cross_probability=1, mutation_probability=0.05, fitness_mode='distance'):
        #load data
        self.N, self.graph, self.ground_truth = data.read_tsp_data(base_name)
        
        #super parameters
        self.base_name = base_name
        self.n_iters = n_iters
        self.group_number = group_number
        self.cross_probability = cross_probability
        self.mutation_probability = mutation_probability
        self.fitness_mode = fitness_mode
        
        #init
        np.random.seed(1453)
        random.seed(1453)
        self.group = []
        for i in range(self.group_number):
            new_travel_order = np.arange(self.N)
            np.random.shuffle(new_travel_order)
            new_travel_order = new_travel_order.tolist()
            new_distance = self.get_total_distance(new_travel_order)
            new_item = {"travel_order": new_travel_order, "distance": new_distance}
            self.group.append(new_item)
        self.group.sort(key=lambda x: x['distance'])
        self.best_solution = self.group[0]["travel_order"]
        self.current_best = self.group[0]["distance"]
        
        #recording
        self.result_list = []
        self.result_list.append(self.current_best)
        self.ground_truth_list = [self.ground_truth] * (1 + self.n_iters)
        
    def solve(self):
        """The main algorithm of genetic algorithm
        """
        start = time.time()
        for iter_ in range(self.n_iters):
            self.cross()
            self.mutate()
            self.group.sort(key=lambda x: x['distance'])
            the_best = self.group[0]["distance"]
            if the_best < self.current_best:
                self.best_solution = self.group[0]["travel_order"]
                self.current_best = self.group[0]["distance"]
            self.fitness = self.get_fitness()
            self.select()
            
            self.result_list.append(self.current_best)
            print('iter {}/{}:'.format(iter_ + 1, self.n_iters))
            print('current best result: {:.4f}'.format(self.current_best))
            print('ground truth result: {:.4f}'.format(self.ground_truth))
            print('rate: {:.4f}'.format(self.current_best / self.ground_truth))
        end = time.time()
        self.total_time = end - start
        print('Total time cost: {:.2f}s'.format(self.total_time))

    def cross(self):
        """Random cross the travel orders in the group
        """
        cross_list = np.arange(len(self.group))
        np.random.shuffle(cross_list)
        new_items = []
        for i in range(len(cross_list) // 2):
            p = random.random()
            if p <= self.cross_probability:
                new_travel_order_1, new_travel_order_2 = self.cross_two_travel_orders(self.group[2 * i]["travel_order"], self.group[2 * i + 1]["travel_order"])
                new_distance_1 = self.get_total_distance(new_travel_order_1)
                new_distance_2 = self.get_total_distance(new_travel_order_2)
                new_item_1 = {"travel_order": new_travel_order_1, "distance": new_distance_1}
                new_item_2 = {"travel_order": new_travel_order_2, "distance": new_distance_2}
                new_items.append(new_item_1)
                new_items.append(new_item_2)
        self.group.extend(new_items)
        
    def mutate(self):
        """Random mutate the travel orders in the group
        """
        for i in range(len(self.group)):
            p = random.random()
            if p <= self.mutation_probability:
                new_travel_order = self.mutate_one_travel_order(self.group[i]["travel_order"])
                new_distance = self.get_total_distance(new_travel_order)
                self.group[i]["travel_order"] = new_travel_order
                self.group[i]["distance"] = new_distance
                

                
    def select(self):
        """Random select the item in the groups by fitness function
        """
        #get fitness
        fitness_total = 0.0
        for i in range(len(self.group)):
            fitness_total += self.fitness[i]
        
        #accumulated probability
        accumulated_probability = [0.0]
        for i in range(len(self.group)):
            the_probability = self.fitness[i] / fitness_total
            accumulated_probability.append(accumulated_probability[i] + the_probability)
        
        #roulette wheel selection
        new_group = []
        for i in range(self.group_number):
            p = random.random()
            j = 0
            while True:
                if p >= accumulated_probability[j] and p <= accumulated_probability[j + 1]:
                    break 
                j += 1 
            new_group.append(self.group[j])
        self.group = new_group
            
    def get_fitness(self, mode="order"):
        """Get the fitness of current group

        Args:
            mode [str]: [the mode of fitness function, order/distance]
        
        Returns:
            fitness [double array], [M]: [the list of fitness]
        """  
        fitness = []
        for i in range(len(self.group)):
            if mode == 'order': 
                the_fitness = len(self.group) - i
            else: 
                the_fitness = 1 / (self.group[i]["distance"] ** 2)
            fitness.append(the_fitness)
        return fitness
            
        
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
    
    def cross_two_travel_orders(self, order_1, order_2):
        """Random cross two travel orders

        Args:
            order_1 [int array], [N]: [the travel order to be crossed]
            order_2 [int array], [N]: [the travel order to be crossed]
        
        Returns:
            new_order_1 [int array], [N]: [the result travel order]
            new_order_2 [int array], [N]: [the result travel order]
        """
        #get p and q
        p = random.randint(0, self.N - 1)
        q = random.randint(0, self.N - 1)
        start = min(p, q)
        end = max(p, q)
        
        #copy and swap
        new_order_1 = order_1.copy()
        new_order_2 = order_2.copy()
        swap_1 = order_1[start:end + 1]
        swap_2 = order_2[start:end + 1]
        new_order_1[start:end + 1] = swap_2
        new_order_2[start:end + 1] = swap_1 
        
        #solve duplication:front
        for i in range(start):
            while new_order_1[i] in new_order_1[start:end + 1]:
                duplicate_index = swap_2.index(new_order_1[i])
                new_order_1[i] = swap_1[duplicate_index]
                
            while new_order_2[i] in new_order_2[start:end + 1]:
                duplicate_index = swap_1.index(new_order_2[i])
                new_order_2[i] = swap_2[duplicate_index]
                
        #back 
        for i in range(end + 1, self.N):
            while new_order_1[i] in new_order_1[start:end + 1]:
                duplicate_index = swap_2.index(new_order_1[i])
                new_order_1[i] = swap_1[duplicate_index]
                
            while new_order_2[i] in new_order_2[start:end + 1]:
                duplicate_index = swap_1.index(new_order_2[i])
                new_order_2[i] = swap_2[duplicate_index]
        return new_order_1, new_order_2
        
    def mutate_one_travel_order(self, travel_order):
        """Mutate one travel order randomly

        Args:
            travel_order [int array], [N]: [the travel order to be mutated]
            
        Returns:
            new_travel_order [int array], [N]: [the mutated travel order]
        """
        p = random.randint(0, self.N - 1)
        q = random.randint(0, self.N - 1)
        start = min(p, q)
        end = max(p, q)
        new_travel_order = travel_order.copy()
        for i in range(start, end + 1):
            new_travel_order[i] = travel_order[end + start - i]
        return new_travel_order

    
    def plot_result(self, save_path):
        """Plot the results
        Args:
            save_path [str]: [the full saving path]
        """
        matplotlib.use('agg')
        x = range(0, self.n_iters + 1)
        plt.plot(x, self.result_list)
        plt.plot(x, self.ground_truth_list)
        plt.xlabel('Iterations')
        plt.ylabel('TSP min distances')
        plt.title("The TSP results of genetic algorithm")
        plt.legend(['Current Best Results', 'Ground Truth Results'])   
        plt.savefig(save_path) 
        plt.close()
        
    def save_result(self):
        """Save the results
        """
        base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, 'results'))
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        save_path = os.path.join(base_path, 'genetic_algorithm')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        '''
        txt_path = os.path.join(save_path, self.base_name + '.txt')
        png_path = os.path.join(save_path, self.base_name + '.png')
        '''
        filename = self.base_name + '_group' + str(self.group_number) + \
            '_cross0' + str(int(self.cross_probability * 10)) + '_mute0' + str(int(self.mutation_probability * 100)) + '_fitness' + self.fitness_mode
        txt_path = os.path.join(save_path, filename + '.txt')
        png_path = os.path.join(save_path, filename + '.png')
        
        f = open(txt_path, 'w')
        f.write('Algorithm: Genetic Algorithm\n')
        f.write('data: ' + self.base_name + '\n')
        f.write('N: '+ str(self.N) + '\n')
        f.write('group number: ' + str(self.group_number) + '\n')
        f.write('cross probability: ' + str(self.cross_probability) + '\n')
        f.write('mutation probability: ' + str(self.mutation_probability) + '\n')
        f.write('fitness mode: ' + self.fitness_mode + '\n')
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
    a = GeneticAlgorithm()     
    a.solve()
    a.save_result()
