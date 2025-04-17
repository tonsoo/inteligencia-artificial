import numpy as np
from typing import Dict, Tuple, List
from scipy.spatial import distance

class GeneticAlgorithm:
    def __init__(self, problem_type: str, data: Dict):
        self.problem_type = problem_type
        self.data = data
        self._setup_problem()
    
    def _setup_problem(self):
        """Configura funções específicas para cada problema."""
        if self.problem_type == "tsp":
            self._setup_tsp()
        elif self.problem_type == "knapsack":
            self._setup_knapsack()
        elif self.problem_type == "function":
            self._setup_function()
    
    def _setup_tsp(self):
        coords = self.data["tsp"][['X', 'Y']].values
        self.dist_matrix = distance.cdist(coords, coords, 'euclidean')
        self.fitness_func = self._tsp_fitness
        self.init_pop_func = self._init_tsp_pop
        self.mutate_func = self._tsp_mutate
    
    def _setup_knapsack(self):
        self.items = self.data["knapsack_items"]
        self.capacity = self.data["knapsack_capacity"]
        self.fitness_func = self._knapsack_fitness
        self.init_pop_func = self._init_knapsack_pop
        self.mutate_func = self._knapsack_mutate
    
    def _setup_function(self):
        self.coeffs = self.data["function_coeffs"].set_index('Variável')['Coeficiente'].to_dict()
        self.fitness_func = self._function_fitness
        self.init_pop_func = self._init_function_pop
        self.mutate_func = self._function_mutate

    # ========== TSP METHODS ==========
    def _tsp_fitness(self, individual: np.ndarray) -> float:
        """Calcula a distância total do percurso (quanto menor, melhor)."""
        total_distance = 0
        for i in range(len(individual)):
            city1 = individual[i]
            city2 = individual[(i+1)%len(individual)]
            total_distance += self.dist_matrix[city1, city2]
        return -total_distance  # Negativo porque queremos minimizar
    
    def _init_tsp_pop(self, pop_size: int) -> np.ndarray:
        """Inicializa população com permutações aleatórias de cidades."""
        num_cities = len(self.data["tsp"])
        return np.array([np.random.permutation(num_cities) for _ in range(pop_size)])
    
    def _tsp_mutate(self, population: np.ndarray, rate: float) -> np.ndarray:
        """Mutação por swap em permutações."""
        for i in range(len(population)):
            if np.random.rand() < rate:
                idx1, idx2 = np.random.choice(len(population[i]), 2, replace=False)
                population[i][idx1], population[i][idx2] = population[i][idx2], population[i][idx1]
        return population

    # ========== KNAPSACK METHODS ==========
    def _knapsack_fitness(self, individual: np.ndarray) -> float:
        """Calcula o valor total dos itens selecionados."""
        total_weight = np.sum(self.items['Peso'] * individual)
        if total_weight > self.capacity:
            return 0  # Solução inválida
        return np.sum(self.items['Valor'] * individual)
    
    def _init_knapsack_pop(self, pop_size: int) -> np.ndarray:
        """Inicializa população com cromossomos binários."""
        num_items = len(self.items)
        return np.random.randint(0, 2, (pop_size, num_items))
    
    def _knapsack_mutate(self, population: np.ndarray, rate: float) -> np.ndarray:
        """Mutação por flip de bits."""
        mask = np.random.rand(*population.shape) < rate
        return np.where(mask, 1 - population, population)

    # ========== FUNCTION OPTIMIZATION METHODS ==========
    def _function_fitness(self, individual: np.ndarray) -> float:
        """Avalia a função linear."""
        return self.coeffs['x1'] * individual[0] + self.coeffs['x2'] * individual[1]
    
    def _init_function_pop(self, pop_size: int) -> np.ndarray:
        """Inicializa população com valores reais."""
        return np.random.uniform(-10, 10, (pop_size, 2))
    
    def _function_mutate(self, population: np.ndarray, rate: float) -> np.ndarray:
        """Mutação por perturbação gaussiana."""
        mask = np.random.rand(*population.shape) < rate
        noise = np.random.normal(0, 1, population.shape)
        return np.where(mask, population + noise, population)

    # ========== CORE GA METHODS ==========
    def run(self, pop_size: int = 50, generations: int = 100, 
            crossover_type: str = "one_point", mutation_rate: float = 0.05,
            elitism: bool = True) -> Tuple[np.ndarray, List[float]]:
        """Executa o algoritmo genético."""
        population = self.init_pop_func(pop_size)
        best_fitness_history = []
        
        for gen in range(generations):
            # Avaliação
            fitness = np.array([self.fitness_func(ind) for ind in population])
            best_fitness_history.append(np.max(fitness))
            
            # Seleção
            parents = self._tournament_selection(population, fitness)
            
            # Crossover
            offspring = self._crossover(parents, crossover_type)
            
            # Mutação
            offspring = self.mutate_func(offspring, mutation_rate)
            
            # Substituição (com elitismo)
            if elitism:
                best_idx = np.argmax(fitness)
                population = np.vstack([offspring, population[best_idx]])
            else:
                population = offspring
        
        # Retorna a melhor solução e histórico
        final_fitness = np.array([self.fitness_func(ind) for ind in population])
        return population[np.argmax(final_fitness)], best_fitness_history
    
    def _tournament_selection(self, population: np.ndarray, 
                            fitness: np.ndarray, k: int = 3) -> np.ndarray:
        """Seleção por torneio."""
        selected = []
        for _ in range(len(population)):
            contenders = np.random.choice(len(population), k, replace=False)
            winner = contenders[np.argmax(fitness[contenders])]
            selected.append(population[winner])
        return np.array(selected)
    
    def _crossover(self, parents: np.ndarray, 
                 crossover_type: str) -> np.ndarray:
        """Aplica crossover conforme o tipo especificado."""
        offspring = []
        for i in range(0, len(parents), 2):
            if i+1 >= len(parents):
                offspring.append(parents[i])
                break
                
            parent1, parent2 = parents[i], parents[i+1]
            
            if crossover_type == "one_point":
                point = np.random.randint(1, len(parent1))
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
            elif crossover_type == "two_point":
                points = sorted(np.random.choice(len(parent1), 2, replace=False))
                child1 = np.concatenate([
                    parent1[:points[0]], 
                    parent2[points[0]:points[1]], 
                    parent1[points[1]:]
                ])
                child2 = np.concatenate([
                    parent2[:points[0]], 
                    parent1[points[0]:points[1]], 
                    parent2[points[1]:]
                ])
            elif crossover_type == "uniform":
                mask = np.random.randint(0, 2, len(parent1))
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
            else:
                raise ValueError(f"Tipo de crossover inválido: {crossover_type}")
            
            offspring.extend([child1, child2])
        return np.array(offspring)