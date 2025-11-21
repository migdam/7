"""
Advanced Risk Metrics Calculator
Calculate sophisticated risk metrics: VaR, CVaR, Beta, Alpha, Tracking Error, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy import stats


class RiskMetrics:
    """Advanced risk metrics calculator"""

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize risk metrics calculator

        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns (optional)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate

    def value_at_risk(self, confidence: float = 0.95, method: str = 'historical') -> Dict:
        """
        Calculate Value at Risk (VaR)

        Args:
            confidence: Confidence level (0.95 = 95%)
            method: 'historical', 'parametric', or 'cornish_fisher'

        Returns:
            VaR metrics dictionary
        """
        if method == 'historical':
            var = np.percentile(self.returns, (1 - confidence) * 100)

        elif method == 'parametric':
            # Assume normal distribution
            mean = self.returns.mean()
            std = self.returns.std()
            z_score = stats.norm.ppf(1 - confidence)
            var = mean + z_score * std

        elif method == 'cornish_fisher':
            # Modified VaR accounting for skewness and kurtosis
            mean = self.returns.mean()
            std = self.returns.std()
            skew = self.returns.skew()
            kurt = self.returns.kurtosis()

            z = stats.norm.ppf(1 - confidence)
            z_cf = (z +
                   (z**2 - 1) * skew / 6 +
                   (z**3 - 3*z) * kurt / 24 -
                   (2*z**3 - 5*z) * skew**2 / 36)

            var = mean + z_cf * std

        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'var': round(var * 100, 2),
            'var_dollar': round(var, 4),
            'confidence': confidence,
            'method': method
        }

    def conditional_var(self, confidence: float = 0.95) -> Dict:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)

        Args:
            confidence: Confidence level

        Returns:
            CVaR metrics
        """
        var = np.percentile(self.returns, (1 - confidence) * 100)
        cvar = self.returns[self.returns <= var].mean()

        return {
            'cvar': round(cvar * 100, 2),
            'cvar_dollar': round(cvar, 4),
            'confidence': confidence,
            'var': round(var * 100, 2)
        }

    def beta(self) -> float:
        """
        Calculate portfolio beta relative to benchmark

        Returns:
            Beta value
        """
        if self.benchmark_returns is None:
            return 1.0

        # Align indices
        aligned = pd.concat([self.returns, self.benchmark_returns], axis=1, join='inner')
        if len(aligned) < 2:
            return 1.0

        covariance = aligned.cov().iloc[0, 1]
        benchmark_variance = aligned.iloc[:, 1].var()

        if benchmark_variance == 0:
            return 1.0

        return covariance / benchmark_variance

    def alpha(self, annualized: bool = True) -> float:
        """
        Calculate Jensen's Alpha

        Alpha = Portfolio Return - (Risk Free + Beta * (Benchmark Return - Risk Free))

        Args:
            annualized: Return annualized alpha

        Returns:
            Alpha value
        """
        if self.benchmark_returns is None:
            return 0.0

        beta = self.beta()
        portfolio_return = self.returns.mean()
        benchmark_return = self.benchmark_returns.mean()
        rf_daily = self.risk_free_rate / 252

        alpha = portfolio_return - (rf_daily + beta * (benchmark_return - rf_daily))

        if annualized:
            alpha *= 252

        return round(alpha * 100, 2)

    def tracking_error(self, annualized: bool = True) -> float:
        """
        Calculate tracking error (standard deviation of excess returns)

        Args:
            annualized: Return annualized tracking error

        Returns:
            Tracking error
        """
        if self.benchmark_returns is None:
            return 0.0

        aligned = pd.concat([self.returns, self.benchmark_returns], axis=1, join='inner')
        if len(aligned) < 2:
            return 0.0

        excess_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        te = excess_returns.std()

        if annualized:
            te *= np.sqrt(252)

        return round(te * 100, 2)

    def information_ratio(self) -> float:
        """
        Calculate Information Ratio (excess return / tracking error)

        Returns:
            Information Ratio
        """
        if self.benchmark_returns is None:
            return 0.0

        aligned = pd.concat([self.returns, self.benchmark_returns], axis=1, join='inner')
        if len(aligned) < 2:
            return 0.0

        excess_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        mean_excess = excess_returns.mean() * 252  # Annualized
        te = self.tracking_error(annualized=True) / 100  # Convert back to decimal

        if te == 0:
            return 0.0

        return round((mean_excess * 100) / (te * 100), 2)

    def downside_deviation(self, target_return: float = 0.0, annualized: bool = True) -> float:
        """
        Calculate downside deviation (semi-deviation)

        Args:
            target_return: Minimum acceptable return
            annualized: Return annualized value

        Returns:
            Downside deviation
        """
        downside_returns = self.returns[self.returns < target_return]
        if len(downside_returns) == 0:
            return 0.0

        dd = np.sqrt(np.mean(downside_returns**2))

        if annualized:
            dd *= np.sqrt(252)

        return round(dd * 100, 2)

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio

        Omega = Probability Weighted Gains / Probability Weighted Losses

        Args:
            threshold: Return threshold

        Returns:
            Omega Ratio
        """
        gains = self.returns[self.returns > threshold] - threshold
        losses = threshold - self.returns[self.returns < threshold]

        if len(losses) == 0 or losses.sum() == 0:
            return float('inf') if len(gains) > 0 else 0.0

        omega = gains.sum() / losses.sum()
        return round(omega, 2)

    def tail_ratio(self) -> float:
        """
        Calculate Tail Ratio (95th percentile / 5th percentile)

        Returns:
            Tail Ratio
        """
        p95 = np.percentile(self.returns, 95)
        p5 = np.percentile(self.returns, 5)

        if p5 == 0:
            return 0.0

        return round(abs(p95 / p5), 2)

    def gain_to_pain_ratio(self) -> float:
        """
        Calculate Gain to Pain Ratio (sum of returns / abs(sum of negative returns))

        Returns:
            Gain to Pain Ratio
        """
        total_return = self.returns.sum()
        pain = abs(self.returns[self.returns < 0].sum())

        if pain == 0:
            return float('inf') if total_return > 0 else 0.0

        return round(total_return / pain, 2)

    def get_all_metrics(self) -> Dict:
        """Get all risk metrics"""
        var_95 = self.value_at_risk(0.95, 'historical')
        var_99 = self.value_at_risk(0.99, 'historical')
        cvar_95 = self.conditional_var(0.95)

        metrics = {
            'var_95': var_95['var'],
            'var_99': var_99['var'],
            'cvar_95': cvar_95['cvar'],
            'downside_deviation': self.downside_deviation(),
            'omega_ratio': self.omega_ratio(),
            'tail_ratio': self.tail_ratio(),
            'gain_to_pain_ratio': self.gain_to_pain_ratio()
        }

        # Add benchmark-relative metrics if available
        if self.benchmark_returns is not None:
            metrics.update({
                'beta': round(self.beta(), 2),
                'alpha': self.alpha(),
                'tracking_error': self.tracking_error(),
                'information_ratio': self.information_ratio()
            })

        return metrics

    def print_report(self):
        """Print risk metrics report"""
        metrics = self.get_all_metrics()

        print("\n" + "=" * 70)
        print("RISK METRICS REPORT")
        print("=" * 70)

        print("\n📉 VALUE AT RISK")
        print(f"  VaR (95%):              {metrics['var_95']:>10.2f}%")
        print(f"  VaR (99%):              {metrics['var_99']:>10.2f}%")
        print(f"  CVaR (95%):             {metrics['cvar_95']:>10.2f}%")

        print("\n⚠️  DOWNSIDE RISK")
        print(f"  Downside Deviation:     {metrics['downside_deviation']:>10.2f}%")
        print(f"  Tail Ratio (95/5):      {metrics['tail_ratio']:>10.2f}")

        print("\n📊 RISK-ADJUSTED METRICS")
        print(f"  Omega Ratio:            {metrics['omega_ratio']:>10.2f}")
        print(f"  Gain-to-Pain Ratio:     {metrics['gain_to_pain_ratio']:>10.2f}")

        if 'beta' in metrics:
            print("\n🎯 BENCHMARK-RELATIVE METRICS")
            print(f"  Beta:                   {metrics['beta']:>10.2f}")
            print(f"  Alpha:                  {metrics['alpha']:>10.2f}%")
            print(f"  Tracking Error:         {metrics['tracking_error']:>10.2f}%")
            print(f"  Information Ratio:      {metrics['information_ratio']:>10.2f}")

        print("\n" + "=" * 70)


# Example usage
if __name__ == "__main__":
    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 252), index=dates)

    # Calculate risk metrics
    risk = RiskMetrics(returns, benchmark_returns)
    risk.print_report()
