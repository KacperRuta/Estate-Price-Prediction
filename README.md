# Estate Price Prediction

Desktop application for exploring housing data and predicting property prices with regression models.

The project uses a Kaggle housing dataset with features such as area, number of bedrooms, bathrooms, stories, parking spaces, road access, air conditioning, basement, guest room, heating, preferred area, and furnishing status.

## Features

- Downloads the housing dataset from Kaggle.
- Uses local cache in `data_cache`, so the dataset is not downloaded again if `Housing.csv` already exists.
- Shows exploratory data analysis charts in the GUI:
  - missing values,
  - price distribution,
  - price boxplot,
  - numeric feature correlation heatmap.
- Trains and compares three regression models:
  - `LinearRegression`,
  - `RandomForestRegressor`,
  - `GradientBoostingRegressor`.
- Uses cross-validation and RMSE to compare model quality.
- Shows feature importance from the Random Forest model.
- Provides a price calculator where the user enters property details and receives predictions from all three models.

## Project Structure

- `main.py` - application entry point.
- `app.py` - desktop GUI built with `tkinter`.
- `data_import.py` - dataset download and cache handling.
- `eda.py` - EDA chart generation with `matplotlib` and `seaborn`.
- `model_training.py` - preprocessing, model training, cross-validation, feature importance, and prediction logic.
- `requirements.txt` - Python packages required to run the project.
- `data_cache/` - local cached dataset directory.

## Machine Learning Flow

The model pipeline separates features from the target column `price`.

Numeric features are filled with the median value when missing. Categorical features are filled with `unknown` and encoded with OneHotEncoding. This preprocessing is inside a scikit-learn `Pipeline`, so cross-validation is evaluated correctly without data leakage.

The application trains three models on the same processed data and compares them using RMSE. Lower RMSE means better average prediction error.

## GUI

The application starts with three options:

- `Calculate price` - enter property data and calculate predicted prices.
- `Look at the data charts` - view EDA charts.
- `Model information` - view model RMSE results and top feature importances.

The GUI keeps one main window and changes its content depending on the selected option.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

On the first run, the app downloads the dataset from Kaggle. Later runs use the cached file from `data_cache/Housing.csv`.

## Notes

- `tkinter` is used for the GUI and is part of the Python standard library on most desktop Python installations.
- The dataset price currency is not defined by the application. Predictions are shown in the same currency units as the source dataset.
- `area` is measured in square feet, following the source dataset.
