from app import EstatePricePredictionApp
from data_import import HousingDataLoader
from model_training import HousingModelTrainer


def main():
    data_loader = HousingDataLoader(dataset_name="yasserh/housing-prices-dataset")
    model_trainer = HousingModelTrainer(target_column="price")
    df = data_loader.load_data()

    app = EstatePricePredictionApp(df, model_trainer)
    app.run()


if __name__ == "__main__":
    main()
