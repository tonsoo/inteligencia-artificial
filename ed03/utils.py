import matplotlib.pyplot as plt

def plot_convergence(history: list, title: str = ""):
    """Plota curva de convergência."""
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title(title or "Convergência do AG")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()

def validate_file(path: str, required_cols: list) -> bool:
    """Valida se arquivo CSV contém colunas necessárias."""
    try:
        df = pd.read_csv(path)
        return set(required_cols).issubset(df.columns)
    except:
        return False