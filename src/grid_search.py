import tabu_search
import genetic_algorithm

def tabu_search_grid_search():
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
    
def genetic_algorithm_grid_search():
    a = genetic_algorithm.GeneticAlgorithm(base_name='burma14', n_iters=2000, group_number=50, cross_probability=1, mutation_probability=0.05, fitness_mode='distance')
    a.solve()
    a.save_result()
    

    group_number = [1000, 500, 200, 100]
    #group_number = [10, 20, 50]
    cross_probability = [1]
    mutation_probability = [0.2]
    mode = ['distance', 'order']
    base_name = ['eil101']
    #base_name = ['eil101', 'a280']
    
    for bn in base_name:
        for gn in group_number:
            for cp in cross_probability:
                for mp in mutation_probability:
                    for md in mode:
                        a = genetic_algorithm.GeneticAlgorithm(base_name = bn, n_iters = 10000, group_number=gn, cross_probability=cp, mutation_probability=mp, fitness_mode=md)
                        a.solve()
                        a.save_result()
                    
if __name__ == '__main__':
    genetic_algorithm_grid_search()
    #tabu_search_grid_search()
