import numpy as np

def tournament_selection(population: np.ndarray, fitness: np.ndarray, 
                       tournament_size: int = 3) -> np.ndarray:
    """Seleção por torneio."""
    selected = []
    for _ in range(len(population)):
        contenders = np.random.choice(len(population), tournament_size, replace=False)
        winner = contenders[np.argmax(fitness[contenders])]
        selected.append(population[winner])
    return np.array(selected)

def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple:
    """Crossover de um ponto."""
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

# ... (outros operadores)