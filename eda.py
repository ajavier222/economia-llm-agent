"""
eda.py
------

This module collects utility functions for performing an exploratory
data analysis (EDA) on economic time series data.  The functions
provided here are designed to be lightweight and easily composable
within a Streamlit application.  A typical workflow involves
downloading data (for example using ``yfinance``), computing basic
statistics, identifying missing or out‑of‑range values, and
constructing simple visualisations.

Although any CSV dataset with at least 300 rows and six columns can
be analysed, this module includes helpers to download daily price
information for a given ticker symbol via the ``yfinance`` library,
which serves as a convenient example of publicly available economic
data.  Users may upload their own CSV files using the Streamlit
interface, in which case these functions may still be applied to
compute statistics and figures.

The plotting functions use ``matplotlib`` directly rather than more
interactive libraries in order to save the resulting figures as PNG
files.  When embedded in a Streamlit app the same information is
presented using ``st.line_chart`` and similar helpers.

"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import yfinance as yf  # type: ignore
except Exception:
    # yfinance may not be installed in certain environments.  In that case
    # ``load_stock_data`` will attempt to import it lazily and raise a
    # descriptive error if unavailable.
    yf = None

import matplotlib.pyplot as plt
import seaborn as sns


def generate_synthetic_economic_data(
    num_days: int = 365,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic economic time series dataset.

    This helper creates a DataFrame with daily observations of
    hypothetical economic indicators such as GDP growth, inflation rate,
    unemployment rate, interest rate, consumer sentiment and a stock
    market index.  The generated data exhibits seasonal trends and
    random noise, providing a realistic dataset for demonstration and
    educational purposes.  The function requires no external data
    sources.

    Parameters
    ----------
    num_days : int, default 365
        Number of daily observations to generate.
    start_date : str, default "2023-01-01"
        ISO‑format date string for the first observation.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by date with columns:
        ``GDP_Growth``, ``Inflation_Rate``, ``Unemployment_Rate``,
        ``Interest_Rate``, ``Consumer_Sentiment``, ``Stock_Index``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=num_days, freq="D")
    # Create seasonal patterns using sine waves and random noise
    t = np.arange(num_days)
    gdp_growth = 2 + 0.5 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 0.2, num_days)
    inflation_rate = 3 + 0.3 * np.sin(2 * np.pi * (t + 30) / 365) + rng.normal(0, 0.1, num_days)
    unemployment_rate = 5 - 0.2 * np.sin(2 * np.pi * (t + 60) / 365) + rng.normal(0, 0.15, num_days)
    interest_rate = 4 + 0.1 * np.sin(2 * np.pi * (t + 90) / 365) + rng.normal(0, 0.05, num_days)
    consumer_sentiment = 100 + 5 * np.sin(2 * np.pi * (t + 120) / 365) + rng.normal(0, 2, num_days)
    # Simulate a stock index with drift and noise
    stock_index = np.cumsum(0.1 + 0.02 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 1, num_days)) + 3000
    df = pd.DataFrame(
        {
            "GDP_Growth": gdp_growth,
            "Inflation_Rate": inflation_rate,
            "Unemployment_Rate": unemployment_rate,
            "Interest_Rate": interest_rate,
            "Consumer_Sentiment": consumer_sentiment,
            "Stock_Index": stock_index,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def load_stock_data(symbol: str = "GOOG", period: str = "2y") -> pd.DataFrame:
    """Download daily price data for a given ticker using yfinance.

    Parameters
    ----------
    symbol : str, default "GOOG"
        Ticker symbol of the security to download.  Examples include
        ``"AAPL"`` for Apple, ``"MSFT"`` for Microsoft, or index
        tickers such as ``"^GSPC"`` for the S&P 500.  The dataset
        returned will contain at least six columns: ``Open``,
        ``High``, ``Low``, ``Close``, ``Adj Close``, and ``Volume``.
    period : str, default "2y"
        Length of historical data to download.  Accepts periods such as
        ``"1y"``, ``"2y"``, ``"5y"`` or ``"max"``.  A two‑year period
        ensures roughly 500 trading days (>300 samples), satisfying
        the evaluation requirement.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by date with columns containing daily price
        data.  If the download fails, an empty DataFrame is returned.
    """
    # Import yfinance lazily in case it is unavailable at module import time
    global yf
    if yf is None:
        try:
            import yfinance as yfi  # type: ignore
            yf = yfi
        except Exception:
            # yfinance is not installed; return an empty DataFrame
            return pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, progress=False)
        df = df.dropna(how="all")
        df.index.name = "Date"
        return df
    except Exception:
        # If download fails, return an empty DataFrame
        return pd.DataFrame()


def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic descriptive statistics for numeric columns.

    The returned DataFrame includes mean, standard deviation, minimum,
    median, and maximum for each numeric column.  Non‑numeric
    columns are ignored.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by statistic names with columns
        corresponding to numeric columns in ``df``.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    # Compute statistics into a dictionary keyed by statistic name.  Each
    # value is a Series indexed by column names.
    stats_dict = {
        "mean": numeric_df.mean(),
        "std": numeric_df.std(),
        "min": numeric_df.min(),
        "median": numeric_df.median(),
        "max": numeric_df.max(),
    }
    stats = pd.DataFrame(stats_dict)
    # Transpose so that statistics form rows (mean, std, etc.) and
    # columns correspond to variables.  This orientation aligns with
    # typical presentation where each row is a statistic.
    return stats.T.round(4)


def identify_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the count of missing values for each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.Series
        Series whose index corresponds to column names and values to
        the count of missing entries in that column.
    """
    return df.isna().sum()


def create_correlation_heatmap(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Create a correlation heatmap for numeric columns.

    This function computes the Pearson correlation between numeric
    features of the DataFrame and plots a heatmap using seaborn.  The
    resulting figure is saved to ``output_path``, or to a temporary
    file if no path is provided.  The returned Path points to the
    image file on disk.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric columns.
    output_path : Path, optional
        Destination file path.  If ``None``, a file will be created in
        the current working directory with a generated name.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    if output_path is None:
        output_path = Path("correlation_heatmap.png")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def create_time_series_plot(
    df: pd.DataFrame,
    column: str = "Close",
    output_path: Optional[Path] = None,
) -> Path:
    """Create a time series line plot for a specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame indexed by date.
    column : str, default "Close"
        Name of the column to plot.  The column must be numeric.
    output_path : Path, optional
        Destination file path.  If ``None``, a file will be created in
        the current working directory with a generated name.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if output_path is None:
        output_path = Path(f"time_series_{column}.png")
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df[column], label=column)
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.title(f"Time series of {column}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Convert a DataFrame to a Markdown table string.

    Streamlit can display DataFrames directly, but generating a
    Markdown representation is useful for summarising statistics in
    reports or for passing structured information to the language
    model.  Only the first ``max_rows`` rows are included.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to convert.
    max_rows : int, default 10
        Maximum number of rows to include in the output.

    Returns
    -------
    str
        A Markdown formatted table.
    """
    if df.empty:
        return "No data available"
    limited_df = df.head(max_rows)
    # Use the built‑in pandas method to convert to markdown.  We set
    # ``tablefmt='pipe'`` for GitHub‑friendly formatting.
    try:
        import tabulate
    except ImportError:
        # Fallback: convert to CSV and wrap in code block
        return "\n".join(
            [", ".join(map(str, limited_df.columns))]
            + [", ".join(map(str, row)) for row in limited_df.values]
        )
    return limited_df.to_markdown(tablefmt="pipe", index=True)
