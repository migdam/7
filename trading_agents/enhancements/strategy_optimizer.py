"""
Strategy Parameter Optimizer
Grid search and optimization for strategy parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Tuple
from itertools import product
import concurrent.futures


class StrategyOptimizer:
    """Optimize strategy parameters using grid search"""

    def __init__(
        self,
        backtest_function: Callable,
        parameter_grid: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio',
        maximize: bool = True
    ):
        """
        Initialize strategy optimizer

        Args:
            backtest_function: Function that runs backtest and returns metrics
            parameter_grid: Dictionary of parameter names and values to test
            metric: Metric to optimize
            maximize: True to maximize metric, False to minimize
        """
        self.backtest_function = backtest_function
        self.parameter_grid = parameter_grid
        self.metric = metric
        self.maximize = maximize

        self.results = []
        self.best_params = None
        self.best_score = None

    def _run_single_backtest(self, params: Dict) -> Dict:
        """Run backtest with specific parameters"""
        try:
            metrics = self.backtest_function(**params)
            return {
                'params': params,
                'metrics': metrics,
                'score': metrics.get(self.metric, 0)
            }
        except Exception as e:
            return {
                'params': params,
                'metrics': {},
                'score': -np.inf if self.maximize else np.inf,
                'error': str(e)
            }

    def grid_search(self, n_jobs: int = 1, verbose: bool = True) -> Dict:
        """
        Perform grid search optimization

        Args:
            n_jobs: Number of parallel jobs
            verbose: Print progress

        Returns:
            Dictionary with best parameters and results
        """
        # Generate all parameter combinations
        param_names = list(self.parameter_grid.keys())
        param_values = list(self.parameter_grid.values())
        combinations = list(product(*param_values))

        if verbose:
            print(f"Testing {len(combinations)} parameter combinations...")

        # Run backtests
        if n_jobs == 1:
            # Sequential execution
            for i, combo in enumerate(combinations):
                params = dict(zip(param_names, combo))
                result = self._run_single_backtest(params)
                self.results.append(result)

                if verbose and (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{len(combinations)}")
        else:
            # Parallel execution
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                param_dicts = [dict(zip(param_names, combo)) for combo in combinations]
                futures = [executor.submit(self._run_single_backtest, p) for p in param_dicts]

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    result = future.result()
                    self.results.append(result)

                    if verbose and (i + 1) % 10 == 0:
                        print(f"  Completed {i + 1}/{len(combinations)}")

        # Find best parameters
        if self.maximize:
            best_result = max(self.results, key=lambda x: x['score'])
        else:
            best_result = min(self.results, key=lambda x: x['score'])

        self.best_params = best_result['params']
        self.best_score = best_result['score']

        if verbose:
            print(f"\n✓ Optimization complete!")
            print(f"  Best {self.metric}: {self.best_score:.4f}")
            print(f"  Best parameters: {self.best_params}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_metrics': best_result['metrics'],
            'all_results': self.results
        }

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis"""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for result in self.results:
            row = result['params'].copy()
            row['score'] = result['score']
            row.update(result['metrics'])
            rows.append(row)

        return pd.DataFrame(rows)

    def plot_parameter_impact(self, parameter: str):
        """
        Plot impact of single parameter on performance
        (Requires matplotlib)

        Args:
            parameter: Parameter name to analyze
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        df = self.get_results_dataframe()

        if parameter not in df.columns:
            print(f"Parameter {parameter} not found")
            return

        grouped = df.groupby(parameter)['score'].mean()

        plt.figure(figsize=(10, 6))
        plt.plot(grouped.index, grouped.values, marker='o')
        plt.xlabel(parameter)
        plt.ylabel(self.metric)
        plt.title(f'Impact of {parameter} on {self.metric}')
        plt.grid(True)
        plt.show()


class RandomSearchOptimizer:
    """Random search optimization (faster for large parameter spaces)"""

    def __init__(
        self,
        backtest_function: Callable,
        parameter_distributions: Dict[str, Callable],
        metric: str = 'sharpe_ratio',
        maximize: bool = True
    ):
        """
        Initialize random search optimizer

        Args:
            backtest_function: Backtest function
            parameter_distributions: Dict of parameter sampling functions
            metric: Metric to optimize
            maximize: Maximize or minimize
        """
        self.backtest_function = backtest_function
        self.parameter_distributions = parameter_distributions
        self.metric = metric
        self.maximize = maximize

        self.results = []
        self.best_params = None
        self.best_score = None

    def search(self, n_iter: int = 50, verbose: bool = True) -> Dict:
        """
        Perform random search

        Args:
            n_iter: Number of iterations
            verbose: Print progress

        Returns:
            Best parameters and results
        """
        if verbose:
            print(f"Running {n_iter} random parameter combinations...")

        for i in range(n_iter):
            # Sample parameters
            params = {
                name: sampler()
                for name, sampler in self.parameter_distributions.items()
            }

            # Run backtest
            try:
                metrics = self.backtest_function(**params)
                score = metrics.get(self.metric, 0)
                error = None
            except Exception as e:
                metrics = {}
                score = -np.inf if self.maximize else np.inf
                error = str(e)

            result = {
                'params': params,
                'metrics': metrics,
                'score': score,
                'error': error
            }
            self.results.append(result)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_iter}")

        # Find best
        if self.maximize:
            best_result = max(self.results, key=lambda x: x['score'])
        else:
            best_result = min(self.results, key=lambda x: x['score'])

        self.best_params = best_result['params']
        self.best_score = best_result['score']

        if verbose:
            print(f"\n✓ Random search complete!")
            print(f"  Best {self.metric}: {self.best_score:.4f}")
            print(f"  Best parameters: {self.best_params}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }


# Example usage
if __name__ == "__main__":
    # Mock backtest function
    def mock_backtest(ma_period=20, rsi_threshold=70, position_size=5):
        # Simulate metrics
        sharpe = np.random.normal(1.0, 0.5) - abs(ma_period - 30) * 0.01
        return {
            'sharpe_ratio': sharpe,
            'total_return': sharpe * 10,
            'max_drawdown': -abs(sharpe) * 5
        }

    # Grid search
    param_grid = {
        'ma_period': [10, 20, 30, 50],
        'rsi_threshold': [60, 70, 80],
        'position_size': [3, 5, 7]
    }

    optimizer = StrategyOptimizer(
        backtest_function=mock_backtest,
        parameter_grid=param_grid,
        metric='sharpe_ratio',
        maximize=True
    )

    results = optimizer.grid_search(verbose=True)
    print(f"\nBest parameters found: {results['best_params']}")
