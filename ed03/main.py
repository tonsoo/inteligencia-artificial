from problem_parsers import *
from genetic_algorithm import GeneticAlgorithm
import os

def main():
    print("Algoritmo Genético - Solucionador de Problemas")
    print("1. Caixeiro Viajante (TSP)\n2. Problema da Mochila\n3. Otimização de Função")
    
    choice = input("Escolha o problema (1-3): ")
    file_path = input("Escolha o arquivo CSV: ")
    
    try:
        if choice == "1":
            data = load_tsp_data(file_path)
            ga = GeneticAlgorithm("tsp", data)
            
        elif choice == "2":
            data = load_knapsack_data(file_path)  # Agora lê de um único arquivo
            ga = GeneticAlgorithm("knapsack", data)
            
        elif choice == "3":
            data = load_function_data(file_path)
            ga = GeneticAlgorithm("function", data)
            
        else:
            print("Opção inválida!")
            return
        
        # Configura e executa o AG
        best_solution = ga.run(pop_size=50, generations=100)
        print(f"Melhor solução encontrada:\n{best_solution}")
        
    except Exception as e:
        print(f"Erro: {str(e)}")

if __name__ == "__main__":
    main()