import pandas as pd
from typing import Dict
import os

def load_tsp_data(file_path: str) -> Dict:
    """Carrega dados do problema TSP."""
    df = pd.read_csv(file_path)
    if not {'Cidade', 'X', 'Y'}.issubset(df.columns):
        raise ValueError("Arquivo TSP deve conter colunas: Cidade, X, Y")
    return {"tsp": df}

def load_knapsack_data(file_path: str) -> Dict:
    """Carrega dados do problema da mochila de um único arquivo CSV.
    Formato esperado:
        Item,Peso,Valor
        1,71,385
        2,23,496
        ...
        Capacidade da Mochila,50,
    """
    df = pd.read_csv(file_path)
    
    # Extrai a capacidade (última linha)
    capacity_row = df.iloc[-1]
    if not capacity_row['Item'].startswith('Capacidade'):
        raise ValueError("Última linha deve conter 'Capacidade da Mochila'")
    
    capacity = int(capacity_row['Peso'])  # Usa a coluna 'Peso' para o valor
    items = df[:-1]  # Todas as linhas exceto a última
    
    if not {'Item', 'Peso', 'Valor'}.issubset(items.columns):
        raise ValueError("Arquivo deve conter colunas: Item, Peso, Valor")
    
    return {
        "knapsack_items": items,
        "knapsack_capacity": capacity
    }

def load_function_data(file_path: str) -> Dict:
    """Carrega coeficientes para otimização de função."""
    df = pd.read_csv(file_path)
    if not {'Variável', 'Coeficiente'}.issubset(df.columns):
        raise ValueError("Arquivo deve conter colunas: Variável, Coeficiente")
    return {"function_coeffs": df}