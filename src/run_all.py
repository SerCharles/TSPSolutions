import tabu_search
import genetic_algorithm
import simulated_annealing

def run_all():
    '''
    #tabu search
    a = tabu_search.TabuSearch(base_name='burma14', n_iters=500, n_candidates=20, forbidden_length=5)
    a.solve()
    a.save_result()
    a = tabu_search.TabuSearch(base_name='att48', n_iters=10000, n_candidates=100, forbidden_length=10)
    a.solve()
    a.save_result()
    a = tabu_search.TabuSearch(base_name='eil101', n_iters=10000, n_candidates=500, forbidden_length=5)
    a.solve()
    a.save_result()
    a = tabu_search.TabuSearch(base_name='a280', n_iters=10000, n_candidates=2000, forbidden_length=5)
    a.solve()
    a.save_result()
    
    #genetic algorithm
    a = genetic_algorithm.GeneticAlgorithm(base_name='burma14', n_iters=200, group_number=50, cross_probability=1, mutation_probability=0.05, fitness_mode='distance')
    a.solve()
    a.save_result()
    a = genetic_algorithm.GeneticAlgorithm(base_name='att48', n_iters=10000, group_number=100, cross_probability=1, mutation_probability=0.2, fitness_mode='distance')
    a.solve()
    a.save_result()
    a = genetic_algorithm.GeneticAlgorithm(base_name='eil101', n_iters=10000, group_number=200, cross_probability=0.5, mutation_probability=0.05, fitness_mode='distance')
    a.solve()
    a.save_result()
    a = genetic_algorithm.GeneticAlgorithm(base_name='a280', n_iters=10000, group_number=500, cross_probability=1, mutation_probability=0.05, fitness_mode='distance')
    a.solve()
    a.save_result()
    '''
    #simulated annealing
    a = simulated_annealing.SimulatedAnnealing(base_name='burma14', n_iters=50, decay_rate=0.99, start_T=100, end_T=1e-1)
    a.solve()
    a.save_result()
    a = simulated_annealing.SimulatedAnnealing(base_name='att48', n_iters=200, decay_rate=0.99, start_T=500, end_T=1e-1)
    a.solve()
    a.save_result()
    a = simulated_annealing.SimulatedAnnealing(base_name='eil101', n_iters=500, decay_rate=0.98, start_T=5, end_T=1e-1)
    a.solve()
    a.save_result()
    a = simulated_annealing.SimulatedAnnealing(base_name='a280', n_iters=200, decay_rate=0.99, start_T=5, end_T=1e-1)
    a.solve()
    a.save_result()

if __name__ == '__main__':
    run_all()