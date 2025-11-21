"""
Monte Carlo Simulation for Backtests
Simulate multiple market scenarios and assess strategy robustness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
from concurrent.futures import ProcessPoolExecutor


class MonteCarloSimulator:
    """Monte Carlo simulation for trading strategies"""

    def __init__(
        self,
        backtest_function: Callable,
        base_data: pd.DataFrame,
        n_simulations: int = 1000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo simulator

        Args:
            backtest_function: Function that runs backtest on data
            base_data: Original market data
            n_simulations: Number of simulations to run
            random_seed: Random seed for reproducibility
        """
        self.backtest_function = backtest_function
        self.base_data = base_data
        self.n_simulations = n_simulations
        self.random_seed = random_seed

        if random_seed:
            np.random.seed(random_seed)

        self.results = []

    def bootstrap_resample(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Bootstrap resample returns

        Args:
            data: Original data

        Returns:
            Resampled data
        """
        returns = data['close'].pct_change().dropna()

        # Resample returns
        resampled_returns = np.random.choice(returns, size=len(returns), replace=True)

        # Reconstruct price series
        initial_price = data['close'].iloc[0]
        resampled_prices = initial_price * (1 + resampled_returns).cumprod()

        # Create new dataframe
        new_data = data.copy()
        new_data['close'] = [initial_price] + list(resampled_prices)

        # Adjust OHLC
        for i in range(len(new_data)):
            close = new_data.iloc[i]['close']
            new_data.iloc[i, new_data.columns.get_loc('open')] = close * np.random.uniform(0.99, 1.01)
            new_data.iloc[i, new_data.columns.get_loc('high')] = close * np.random.uniform(1.0, 1.02)
            new_data.iloc[i, new_data.columns.get_loc('low')] = close * np.random.uniform(0.98, 1.0)

        return new_data

    def geometric_brownian_motion(
        self,
        data: pd.DataFrame,
        drift: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate price path using Geometric Brownian Motion

        Args:
            data: Base data for structure
            drift: Annual drift (defaults to historical)
            volatility: Annual volatility (defaults to historical)

        Returns:
            Simulated data
        """
        returns = data['close'].pct_change().dropna()

        if drift is None:
            drift = returns.mean() * 252

        if volatility is None:
            volatility = returns.std() * np.sqrt(252)

        # Generate returns
        dt = 1 / 252  # Daily
        n_steps = len(data)

        random_shocks = np.random.normal(0, 1, n_steps)
        price_changes = drift * dt + volatility * np.sqrt(dt) * random_shocks

        # Create price series
        initial_price = data['close'].iloc[0]
        prices = initial_price * np.exp(np.cumsum(price_changes))

        # Create new dataframe
        new_data = data.copy()
        new_data['close'] = prices

        # Adjust OHLC
        for i in range(len(new_data)):
            close = new_data.iloc[i]['close']
            new_data.iloc[i, new_data.columns.get_loc('open')] = close * np.random.uniform(0.99, 1.01)
            new_data.iloc[i, new_data.columns.get_loc('high')] = close * np.random.uniform(1.0, 1.02)
            new_data.iloc[i, new_data.columns.get_loc('low')] = close * np.random.uniform(0.98, 1.0)

        return new_data

    def run_simulation(
        self,
        method: str = 'bootstrap',
        parallel: bool = False,
        n_jobs: int = 4,
        **method_kwargs
    ) -> Dict:
        """
        Run Monte Carlo simulation

        Args:
            method: Simulation method ('bootstrap' or 'gbm')
            parallel: Use parallel processing
            n_jobs: Number of parallel jobs
            **method_kwargs: Method-specific parameters

        Returns:
            Simulation results
        """
        print(f"Running {self.n_simulations} Monte Carlo simulations...")

        if parallel:
            self.results = self._run_parallel(method, n_jobs, **method_kwargs)
        else:
            self.results = self._run_sequential(method, **method_kwargs)

        return self.analyze_results()

    def _run_sequential(self, method: str, **method_kwargs) -> List[Dict]:
        """Run simulations sequentially"""
        results = []

        for i in range(self.n_simulations):
            # Generate simulated data
            if method == 'bootstrap':
                sim_data = self.bootstrap_resample(self.base_data)
            elif method == 'gbm':
                sim_data = self.geometric_brownian_motion(self.base_data, **method_kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Run backtest
            try:
                metrics = self.backtest_function(sim_data)
                results.append(metrics)
            except Exception as e:
                print(f"Simulation {i} failed: {e}")

            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{self.n_simulations}")

        return results

    def _run_parallel(self, method: str, n_jobs: int, **method_kwargs) -> List[Dict]:
        """Run simulations in parallel"""
        def single_sim(i):
            if method == 'bootstrap':
                sim_data = self.bootstrap_resample(self.base_data)
            elif method == 'gbm':
                sim_data = self.geometric_brownian_motion(self.base_data, **method_kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")

            return self.backtest_function(sim_data)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(single_sim, range(self.n_simulations)))

        return results

    def analyze_results(self) -> Dict:
        """Analyze simulation results"""
        if not self.results:
            return {}

        # Extract metrics
        metrics_df = pd.DataFrame(self.results)

        analysis = {}

        for column in metrics_df.columns:
            values = metrics_df[column].dropna()

            if len(values) == 0:
                continue

            analysis[column] = {
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'percentile_5': values.quantile(0.05),
                'percentile_95': values.quantile(0.95),
                'probability_positive': (values > 0).sum() / len(values) if column.endswith('_pct') or column.endswith('_ratio') else None
            }

        return analysis

    def get_confidence_intervals(self, metric: str, confidence: float = 0.95) -> Dict:
        """
        Get confidence intervals for a specific metric

        Args:
            metric: Metric name
            confidence: Confidence level (0.95 = 95%)

        Returns:
            Confidence interval dictionary
        """
        if not self.results:
            return {}

        values = [r.get(metric, np.nan) for r in self.results]
        values = [v for v in values if not np.isnan(v)]

        if not values:
            return {}

        alpha = 1 - confidence
        lower_percentile = alpha / 2
        upper_percentile = 1 - (alpha / 2)

        return {
            'mean': np.mean(values),
            'lower_bound': np.percentile(values, lower_percentile * 100),
            'upper_bound': np.percentile(values, upper_percentile * 100),
            'confidence': confidence
        }

    def print_summary(self):
        """Print Monte Carlo simulation summary"""
        analysis = self.analyze_results()

        print("\n" + "=" * 70)
        print(f"MONTE CARLO SIMULATION RESULTS ({self.n_simulations} simulations)")
        print("=" * 70)

        key_metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate']

        for metric in key_metrics:
            if metric not in analysis:
                continue

            data = analysis[metric]
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print(f"  Mean:           {data['mean']:>10.2f}")
            print(f"  Median:         {data['median']:>10.2f}")
            print(f"  Std Dev:        {data['std']:>10.2f}")
            print(f"  5th Percentile: {data['percentile_5']:>10.2f}")
            print(f"  95th Percentile: {data['percentile_95']:>10.2f}")
            print(f"  Min:            {data['min']:>10.2f}")
            print(f"  Max:            {data['max']:>10.2f}")

            if data.get('probability_positive') is not None:
                print(f"  Prob > 0:       {data['probability_positive']:>10.1%}")

        print("\n" + "=" * 70)


# Example usage
if __name__ == "__main__":
    # Mock backtest function
    def mock_backtest(data):
        returns = data['close'].pct_change().dropna()
        total_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        return {
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': -abs(total_return) * 0.3,
            'win_rate': 0.55 + np.random.normal(0, 0.05)
        }

    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    prices = 100 * (1 + np.random.normal(0.001, 0.02, 252)).cumprod()

    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 252)
    }, index=dates)

    # Run Monte Carlo simulation
    mc = MonteCarloSimulator(
        backtest_function=mock_backtest,
        base_data=data,
        n_simulations=100,
        random_seed=42
    )

    results = mc.run_simulation(method='bootstrap')
    mc.print_summary()

    # Get confidence intervals
    ci = mc.get_confidence_intervals('total_return_pct', confidence=0.95)
    print(f"\n95% Confidence Interval for Total Return:")
    print(f"  {ci['lower_bound']:.2f}% to {ci['upper_bound']:.2f}%")
