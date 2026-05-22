"""Pipeline principal del proyecto mundialista de Machine Learning."""

from src.data_loader import load_world_cup_data
from src import clustering_model
from src import evaluation
from src import feature_engineering
from src import preprocessing
from src import supervised_model
from src import visualization


def main() -> None:
    """Ejecuta el pipeline completo del proyecto cuando todas las fases esten implementadas."""
    print("1. Cargando datos...")
    data = load_world_cup_data()

    print("2. Limpiando datos...")
    clean_data = preprocessing.clean_world_cup_data(data)

    print("3. Construyendo features...")
    supervised_df = feature_engineering.build_supervised_dataset(clean_data)

    print("4. Entrenando modelo supervisado...")
    supervised_model.train_supervised_model(supervised_df)

    print("5. Entrenando modelos de clustering...")
    clustering_model.train_clustering_models(clean_data)

    print("6. Generando metricas y visualizaciones...")
    evaluation.generate_all_metrics()
    visualization.generate_all_figures()

    print("Pipeline completo. Resultados en outputs/")


if __name__ == "__main__":
    main()
