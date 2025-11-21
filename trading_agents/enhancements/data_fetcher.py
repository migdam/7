"""
Market Data Fetcher
Fetch real market data from various sources (yfinance, Alpha Vantage, etc.)
"""

import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta


class DataFetcher:
    """Base class for data fetchers"""

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch OHLCV data"""
        raise NotImplementedError


class YahooFinanceFetcher(DataFetcher):
    """Fetch data from Yahoo Finance via yfinance"""

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError(
                "yfinance not installed. Install with: pip install yfinance"
            )

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'SPY')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', '5m', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        ticker = self.yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)

        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]

        # Keep only OHLCV
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        data = data[required_cols]

        return data

    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> dict:
        """
        Fetch data for multiple symbols

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.fetch(symbol, start_date, end_date, interval)
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
                result[symbol] = None

        return result

    def get_info(self, symbol: str) -> dict:
        """
        Get stock information

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with stock info
        """
        ticker = self.yf.Ticker(symbol)
        return ticker.info


class AlphaVantageFetcher(DataFetcher):
    """Fetch data from Alpha Vantage API"""

    def __init__(self, api_key: str):
        """
        Initialize Alpha Vantage fetcher

        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        try:
            from alpha_vantage.timeseries import TimeSeries
            self.ts = TimeSeries(key=api_key, output_format='pandas')
        except ImportError:
            raise ImportError(
                "alpha_vantage not installed. Install with: pip install alpha-vantage"
            )

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with OHLCV data
        """
        # Alpha Vantage returns daily data
        data, meta_data = self.ts.get_daily(symbol=symbol, outputsize='full')

        # Rename columns
        data.columns = ['open', 'high', 'low', 'close', 'volume']

        # Filter by date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]

        return data


class CSVDataFetcher(DataFetcher):
    """Fetch data from CSV files"""

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d',
        file_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data from CSV file

        Args:
            symbol: Symbol name (used for file path if not provided)
            start_date: Start date
            end_date: End date
            interval: Data interval (ignored for CSV)
            file_path: Path to CSV file (optional)

        Returns:
            DataFrame with OHLCV data
        """
        if file_path is None:
            file_path = f"data/{symbol}.csv"

        data = pd.read_csv(file_path, parse_dates=True, index_col=0)

        # Ensure column names are lowercase
        data.columns = [col.lower() for col in data.columns]

        # Filter by date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]

        return data


def get_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d',
    source: str = 'yahoo',
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to fetch market data

    Args:
        symbol: Ticker symbol
        start_date: Start date (defaults to 1 year ago)
        end_date: End date (defaults to today)
        interval: Data interval
        source: Data source ('yahoo', 'alphavantage', 'csv')
        api_key: API key (required for alphavantage)

    Returns:
        DataFrame with OHLCV data
    """
    # Set default dates
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Select fetcher
    if source == 'yahoo':
        fetcher = YahooFinanceFetcher()
    elif source == 'alphavantage':
        if api_key is None:
            raise ValueError("API key required for Alpha Vantage")
        fetcher = AlphaVantageFetcher(api_key)
    elif source == 'csv':
        fetcher = CSVDataFetcher()
    else:
        raise ValueError(f"Unknown source: {source}")

    return fetcher.fetch(symbol, start_date, end_date, interval)


# Example usage
if __name__ == "__main__":
    # Fetch Apple stock data
    data = get_data('AAPL', start_date='2024-01-01', end_date='2024-12-31')
    print(data.head())
    print(f"\nFetched {len(data)} days of data")
