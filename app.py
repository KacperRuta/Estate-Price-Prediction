from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from eda import HousingEDA
from model_training import HousingModelTrainer


class EstatePricePredictionApp:
    """Desktop application for charts and price predictions."""

    def __init__(self, dataframe: pd.DataFrame, model_trainer: HousingModelTrainer) -> None:
        self._df = dataframe
        self._model_trainer = model_trainer
        self._trained_models = None
        self._model_evaluation_results = None
        self._feature_importance_results = None
        self._root = tk.Tk()
        self._root.title("Estate Price Prediction")
        self._maximize_root_window()
        self._content = ttk.Frame(self._root, padding=24)
        self._content.pack(fill=tk.BOTH, expand=True)

        self._numeric_fields = ["area", "bedrooms", "bathrooms", "stories", "parking"]
        self._integer_fields = {"bedrooms", "bathrooms", "stories", "parking"}
        self._field_labels = {
            "area": "Area",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "stories": "Stories",
            "parking": "Parking spaces",
            "mainroad": "Main road access",
            "guestroom": "Guest room",
            "basement": "Basement",
            "hotwaterheating": "Hot water heating",
            "airconditioning": "Air conditioning",
            "prefarea": "Preferred area",
            "furnishingstatus": "Furnishing status",
        }
        self._field_hints = {
            "area": "Property area in square feet, as used in the source dataset.",
            "bedrooms": "Number of bedrooms.",
            "bathrooms": "Number of bathrooms.",
            "stories": "Number of building levels.",
            "parking": "Number of available parking spaces.",
            "mainroad": "Select yes if the property has direct main road access.",
            "guestroom": "Select yes if the property includes a guest room.",
            "basement": "Select yes if the property includes a basement.",
            "hotwaterheating": "Select yes if hot water heating is available.",
            "airconditioning": "Select yes if air conditioning is available.",
            "prefarea": "Select yes if the property is in a preferred area.",
            "furnishingstatus": "Choose the current furnishing level.",
        }
        self._binary_fields = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
        ]
        self._input_variables: dict[str, tk.StringVar] = {}
        self._current_figure = None

    def run(self) -> None:
        """Start the application."""
        self._show_home_view()
        self._root.mainloop()

    def _show_home_view(self) -> None:
        self._clear_content()

        container = ttk.Frame(self._content)
        container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        title = ttk.Label(container, text="Estate Price Prediction", font=("Segoe UI", 24, "bold"))
        title.pack(pady=(0, 28))

        calculate_button = ttk.Button(
            container,
            text="Calculate price",
            command=self._show_calculate_view,
            width=28,
        )
        calculate_button.pack(pady=8, ipady=8)

        charts_button = ttk.Button(
            container,
            text="Look at the data charts",
            command=self._show_charts_view,
            width=28,
        )
        charts_button.pack(pady=8, ipady=8)

        model_info_button = ttk.Button(
            container,
            text="Model information",
            command=self._show_model_info_view,
            width=28,
        )
        model_info_button.pack(pady=8, ipady=8)

    def _show_charts_view(self) -> None:
        self._clear_content()
        self._add_navigation("Data charts")

        figure = HousingEDA(self._df).create_figure()
        self._current_figure = figure
        canvas = FigureCanvasTkAgg(figure, master=self._content)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_model_info_view(self) -> None:
        self._clear_content()
        self._add_navigation("Model information")
        self._ensure_model_information()

        container = ttk.Frame(self._content)
        container.pack(fill=tk.BOTH, expand=True, pady=(18, 0))

        metrics_frame = ttk.LabelFrame(container, text="Cross-validation RMSE", padding=16)
        metrics_frame.pack(fill=tk.X, pady=(0, 18))

        for result in self._model_evaluation_results:
            label = ttk.Label(
                metrics_frame,
                text=(
                    f"{result.model_name}: "
                    f"mean RMSE={result.mean_rmse:,.2f}, "
                    f"std RMSE={result.std_rmse:,.2f}"
                ),
                font=("Segoe UI", 12),
            )
            label.pack(anchor=tk.W, pady=4)

        importance_frame = ttk.LabelFrame(container, text="Top feature importances", padding=16)
        importance_frame.pack(fill=tk.X)

        for result in self._feature_importance_results:
            label = ttk.Label(
                importance_frame,
                text=(
                    f"{self._format_feature_name(result.feature_name)}: "
                    f"{result.importance:.4f}"
                ),
                font=("Segoe UI", 12),
            )
            label.pack(anchor=tk.W, pady=4)

    def _show_calculate_view(self) -> None:
        self._clear_content()
        self._input_variables.clear()
        self._add_navigation("Calculate price")

        description = ttk.Label(
            self._content,
            text=(
                "Fill in the apartment details below. Area is measured in square feet, "
                "numeric fields should be whole property counts, and dropdowns describe "
                "whether a feature is available."
            ),
            wraplength=900,
            font=("Segoe UI", 11),
        )
        description.pack(anchor=tk.W, pady=(16, 8))

        form = ttk.Frame(self._content)
        form.pack(fill=tk.X, pady=(8, 0))

        for row_index, field_name in enumerate(self._numeric_fields):
            self._add_numeric_input(form, field_name, row_index)

        start_row = len(self._numeric_fields)
        for index, field_name in enumerate(self._binary_fields, start=start_row):
            self._add_dropdown_input(form, field_name, ["yes", "no", "unknown"], index)

        self._add_dropdown_input(
            form,
            "furnishingstatus",
            ["furnished", "semi-furnished", "unfurnished", "unknown"],
            start_row + len(self._binary_fields),
        )

        calculate_button = ttk.Button(
            self._content,
            text="Calculate",
            command=self._calculate_price,
        )
        calculate_button.pack(pady=20, ipady=6)

        self._results_frame = ttk.Frame(self._content)
        self._results_frame.pack(fill=tk.X)

    def _add_navigation(self, title: str) -> None:
        header = ttk.Frame(self._content)
        header.pack(fill=tk.X)

        back_button = ttk.Button(header, text="Back", command=self._show_home_view)
        back_button.pack(side=tk.LEFT)

        title_label = ttk.Label(header, text=title, font=("Segoe UI", 18, "bold"))
        title_label.pack(side=tk.LEFT, padx=16)

    def _add_numeric_input(self, parent: ttk.Frame, field_name: str, row_index: int) -> None:
        label = ttk.Label(parent, text=self._field_labels[field_name])
        label.grid(row=row_index, column=0, sticky=tk.W, padx=(0, 16), pady=6)

        variable = tk.StringVar(value=self._get_default_numeric_value(field_name))
        entry = ttk.Entry(parent, textvariable=variable, width=24)
        entry.grid(row=row_index, column=1, sticky=tk.W, pady=6)

        hint = ttk.Label(parent, text=self._field_hints[field_name], foreground="#555555")
        hint.grid(row=row_index, column=2, sticky=tk.W, padx=(16, 0), pady=6)
        self._input_variables[field_name] = variable

    def _add_dropdown_input(
        self,
        parent: ttk.Frame,
        field_name: str,
        values: list[str],
        row_index: int,
    ) -> None:
        label = ttk.Label(parent, text=self._field_labels[field_name])
        label.grid(row=row_index, column=0, sticky=tk.W, padx=(0, 16), pady=6)

        variable = tk.StringVar(value=self._get_default_category_value(field_name, values))
        dropdown = ttk.Combobox(parent, textvariable=variable, values=values, width=22, state="readonly")
        dropdown.grid(row=row_index, column=1, sticky=tk.W, pady=6)

        hint = ttk.Label(parent, text=self._field_hints[field_name], foreground="#555555")
        hint.grid(row=row_index, column=2, sticky=tk.W, padx=(16, 0), pady=6)
        self._input_variables[field_name] = variable

    def _calculate_price(self) -> None:
        try:
            input_data = self._collect_input_data()
        except ValueError as error:
            messagebox.showerror("Invalid input", str(error))
            return

        predictions = self._model_trainer.predict_prices(self._get_trained_models(), input_data)
        self._show_prediction_results(predictions)

    def _collect_input_data(self) -> dict[str, float | str]:
        input_data: dict[str, float | str] = {}

        for field_name in self._numeric_fields:
            raw_value = self._input_variables[field_name].get().strip()
            if not raw_value:
                raise ValueError(f"{self._field_labels[field_name]} is required.")

            try:
                value = float(raw_value)
            except ValueError as error:
                raise ValueError(f"{self._field_labels[field_name]} must be a number.") from error

            if value < 0:
                raise ValueError(f"{self._field_labels[field_name]} cannot be negative.")

            if field_name in self._integer_fields:
                if not value.is_integer():
                    raise ValueError(f"{self._field_labels[field_name]} must be a whole number.")

                input_data[field_name] = int(value)
            else:
                input_data[field_name] = value

        for field_name in [*self._binary_fields, "furnishingstatus"]:
            input_data[field_name] = self._input_variables[field_name].get()

        return input_data

    def _show_prediction_results(self, predictions) -> None:
        for child in self._results_frame.winfo_children():
            child.destroy()

        for prediction in predictions:
            label = ttk.Label(
                self._results_frame,
                text=(
                    f"{prediction.model_name}: "
                    f"{prediction.predicted_price:,.2f} dataset currency units"
                ),
                font=("Segoe UI", 13),
            )
            label.pack(anchor=tk.W, pady=4)

    def _ensure_model_information(self) -> None:
        if self._model_evaluation_results is None:
            self._model_evaluation_results = self._model_trainer.evaluate_models(self._df)

        if self._feature_importance_results is None:
            self._feature_importance_results = self._model_trainer.get_feature_importance(
                self._df,
                model_name="RandomForest",
                top_n=10,
            )

    def _get_trained_models(self):
        if self._trained_models is None:
            self._trained_models = self._model_trainer.train_models(self._df)

        return self._trained_models

    def _format_feature_name(self, feature_name: str) -> str:
        cleaned_name = feature_name.replace("numeric__", "").replace("categorical__", "")
        base_name = cleaned_name.split("_", maxsplit=1)[0]
        label = self._field_labels.get(base_name)

        if label is None:
            return cleaned_name

        suffix = cleaned_name.removeprefix(base_name).lstrip("_")
        return f"{label}: {suffix}" if suffix else label

    def _get_default_numeric_value(self, field_name: str) -> str:
        if field_name not in self._df.columns:
            return ""

        return str(round(float(self._df[field_name].median()), 2))

    def _get_default_category_value(self, field_name: str, values: list[str]) -> str:
        if field_name not in self._df.columns:
            return values[0]

        mode = self._df[field_name].mode(dropna=True)
        if mode.empty:
            return values[0]

        mode_value = str(mode.iloc[0])
        return mode_value if mode_value in values else values[0]

    def _clear_content(self) -> None:
        if self._current_figure is not None:
            plt.close(self._current_figure)
            self._current_figure = None

        for child in self._content.winfo_children():
            child.destroy()

    def _maximize_root_window(self) -> None:
        try:
            self._root.state("zoomed")
        except tk.TclError:
            width = self._root.winfo_screenwidth()
            height = self._root.winfo_screenheight()
            self._root.geometry(f"{width}x{height}+0+0")
