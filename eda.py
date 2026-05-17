from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class HousingEDA:
    """Run a compact exploratory data analysis workflow."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._df = dataframe
        sns.set_theme(style="whitegrid")

    def show_all(self) -> None:
        """Display all EDA plots in a single window."""
        fig = self.create_figure()
        self._maximize_window(fig)
        plt.show()

    def create_figure(self) -> plt.Figure:
        """Create all EDA plots in a single figure."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle("Housing Dataset EDA", fontsize=16)

        self._plot_missing_values(axes[0, 0])
        self._plot_price_distribution(axes[0, 1], axes[1, 0])
        self._plot_correlation_heatmap(axes[1, 1])

        plt.tight_layout()
        return fig

    def _maximize_window(self, fig: plt.Figure) -> None:
        manager = fig.canvas.manager

        try:
            manager.window.state("zoomed")
            return
        except AttributeError:
            pass

        try:
            manager.window.showMaximized()
            return
        except AttributeError:
            pass

        try:
            manager.full_screen_toggle()
        except AttributeError:
            pass

    def _plot_missing_values(self, ax: plt.Axes) -> None:
        missing_ratio = (self._df.isna().mean() * 100).sort_values(ascending=False)
        missing_ratio = missing_ratio[missing_ratio > 0]

        if missing_ratio.empty:
            ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            ax.set_title("Missing Values")
            return

        sns.barplot(x=missing_ratio.index, y=missing_ratio.values, ax=ax, palette="Blues_d")
        ax.set_title("Missing Values by Column")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Missing Values (%)")
        ax.tick_params(axis="x", rotation=45)

    def _plot_price_distribution(self, histogram_ax: plt.Axes, boxplot_ax: plt.Axes) -> None:
        price_column = self._find_price_column()
        if price_column is None:
            histogram_ax.text(0.5, 0.5, "Price column not found", ha="center", va="center", fontsize=12)
            histogram_ax.set_axis_off()
            boxplot_ax.set_axis_off()
            return

        sns.histplot(self._df[price_column].dropna(), kde=True, ax=histogram_ax, color="#4C72B0")
        histogram_ax.set_title(f"{price_column} Distribution")
        histogram_ax.set_xlabel(price_column)

        sns.boxplot(x=self._df[price_column].dropna(), ax=boxplot_ax, color="#55A868")
        boxplot_ax.set_title(f"{price_column} Boxplot")
        boxplot_ax.set_xlabel(price_column)

    def _plot_correlation_heatmap(self, ax: plt.Axes) -> None:
        numeric_df = self._df.select_dtypes(include="number")
        if numeric_df.empty:
            ax.text(0.5, 0.5, "No numeric columns detected", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            return

        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Heatmap")

    def _find_price_column(self) -> str | None:
        candidates = ("price", "cost", "value", "target")

        for column in self._df.columns:
            normalized_column = column.strip().lower()
            if any(candidate in normalized_column for candidate in candidates):
                return column

        numeric_columns = self._df.select_dtypes(include="number").columns
        return numeric_columns[0] if len(numeric_columns) > 0 else None
