from data.data import Data, load_config
import pandas as pd
from pathlib import Path

def main():
    # Cargar la configuración
    config = load_config()

    # Definir la ruta base del proyecto
    base_path = Path(__file__).parent.parent.parent.parent

    # Crear paths relativos utilizando pathlib y la configuración
    path_raw = base_path / config['paths']['raw']
    path_clean = base_path / config['paths']['clean']

    data = Data(config)

    df_clean = data.load_clean()

    df_test = pd.read_csv(path_clean, index_col='Unnamed: 0').fillna(0)
    
    print((df_clean == df_test).all(axis=1).all())

if __name__ == "__main__":
    main()
