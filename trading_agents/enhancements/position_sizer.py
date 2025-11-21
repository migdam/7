"""
Advanced Position Sizing Calculator
Implements multiple position sizing strategies: Kelly Criterion, ATR-based, Fixed Fractional, etc.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


class PositionSizer:
    """Base class for position sizing strategies"""

    def calculate_size(
        self,
        capital: float,
        price: float,
        **kwargs
    ) -> float:
        """
        Calculate position size

        Args:
            capital: Available capital
            price: Current price
            **kwargs: Strategy-specific parameters

        Returns:
            Position size (number of units)
        """
        raise NotImplementedError


class FixedFractionalSizer(PositionSizer):
    """Fixed fractional position sizing"""

    def __init__(self, fraction: float = 0.02):
        """
        Args:
            fraction: Fraction of capital to risk per trade (default 2%)
        """
        self.fraction = fraction

    def calculate_size(
        self,
        capital: float,
        price: float,
        **kwargs
    ) -> float:
        """
        Calculate size as fixed fraction of capital

        Args:
            capital: Available capital
            price: Current price

        Returns:
            Position size
        """
        risk_amount = capital * self.fraction
        size = risk_amount / price
        return size


class KellyCriterionSizer(PositionSizer):
    """Kelly Criterion position sizing"""

    def __init__(self, fraction: float = 1.0):
        """
        Args:
            fraction: Fraction of Kelly to use (1.0 = full Kelly, 0.5 = half Kelly)
        """
        self.fraction = fraction

    def calculate_size(
        self,
        capital: float,
        price: float,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        **kwargs
    ) -> float:
        """
        Calculate size using Kelly Criterion

        Kelly % = W - [(1 - W) / R]
        Where:
            W = Win rate
            R = Avg Win / Avg Loss ratio

        Args:
            capital: Available capital
            price: Current price
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Position size
        """
        if avg_loss == 0:
            avg_loss = 0.01  # Prevent division by zero

        win_loss_ratio = avg_win / avg_loss
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply fraction and clamp to reasonable bounds
        kelly_pct = max(0, min(kelly_pct, 1.0))  # Clamp between 0 and 100%
        kelly_pct *= self.fraction

        risk_amount = capital * kelly_pct
        size = risk_amount / price

        return size


class ATRBasedSizer(PositionSizer):
    """ATR-based position sizing for volatility adjustment"""

    def __init__(self, risk_per_atr: float = 1.0, max_risk_pct: float = 0.02):
        """
        Args:
            risk_per_atr: Risk amount per ATR unit
            max_risk_pct: Maximum % of capital to risk (default 2%)
        """
        self.risk_per_atr = risk_per_atr
        self.max_risk_pct = max_risk_pct

    def calculate_size(
        self,
        capital: float,
        price: float,
        atr: float = None,
        **kwargs
    ) -> float:
        """
        Calculate size based on ATR

        Position size = (Capital * Risk%) / (ATR * Multiplier)

        Args:
            capital: Available capital
            price: Current price
            atr: Average True Range

        Returns:
            Position size
        """
        if atr is None or atr == 0:
            # Fallback to fixed fractional if no ATR
            atr = price * 0.02  # Assume 2% volatility

        max_risk_amount = capital * self.max_risk_pct
        stop_distance = atr * self.risk_per_atr

        size = max_risk_amount / stop_distance

        return size


class VolatilityScaledSizer(PositionSizer):
    """Volatility-scaled position sizing (inverse volatility)"""

    def __init__(
        self,
        target_volatility: float = 0.15,
        base_allocation: float = 0.1
    ):
        """
        Args:
            target_volatility: Target portfolio volatility (annualized)
            base_allocation: Base allocation percentage
        """
        self.target_volatility = target_volatility
        self.base_allocation = base_allocation

    def calculate_size(
        self,
        capital: float,
        price: float,
        volatility: float = None,
        **kwargs
    ) -> float:
        """
        Calculate size inversely proportional to volatility

        Size = (Target Vol / Asset Vol) * Base Allocation * Capital / Price

        Args:
            capital: Available capital
            price: Current price
            volatility: Asset volatility (annualized)

        Returns:
            Position size
        """
        if volatility is None or volatility == 0:
            volatility = 0.20  # Default to 20% annual vol

        vol_scalar = self.target_volatility / volatility
        position_value = capital * self.base_allocation * vol_scalar

        size = position_value / price

        return size


class OptimalFSizer(PositionSizer):
    """Optimal F position sizing (Ralph Vince)"""

    def __init__(self, f_value: float = 0.2):
        """
        Args:
            f_value: Optimal F value (typically 0.1 to 0.3)
        """
        self.f_value = f_value

    def calculate_size(
        self,
        capital: float,
        price: float,
        largest_loss: float = None,
        **kwargs
    ) -> float:
        """
        Calculate size using Optimal F

        Size = (Capital * F) / Largest Historical Loss

        Args:
            capital: Available capital
            price: Current price
            largest_loss: Largest historical loss amount

        Returns:
            Position size
        """
        if largest_loss is None or largest_loss == 0:
            # Fallback to conservative sizing
            largest_loss = capital * 0.05  # Assume 5% max loss

        risk_amount = capital * self.f_value
        size = risk_amount / largest_loss * (largest_loss / price)

        return size


class PositionSizingEngine:
    """
    Unified position sizing engine with multiple strategies
    """

    def __init__(
        self,
        strategy: str = 'fixed_fractional',
        **strategy_params
    ):
        """
        Initialize position sizing engine

        Args:
            strategy: Strategy name
            **strategy_params: Strategy-specific parameters
        """
        self.strategy_name = strategy
        self.strategy = self._create_strategy(strategy, **strategy_params)

    def _create_strategy(self, strategy: str, **params) -> PositionSizer:
        """Create position sizing strategy"""
        strategies = {
            'fixed_fractional': FixedFractionalSizer,
            'kelly': KellyCriterionSizer,
            'atr': ATRBasedSizer,
            'volatility': VolatilityScaledSizer,
            'optimal_f': OptimalFSizer
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        return strategies[strategy](**params)

    def calculate(
        self,
        capital: float,
        price: float,
        **context
    ) -> Dict:
        """
        Calculate position size with metadata

        Args:
            capital: Available capital
            price: Current price
            **context: Additional context (volatility, win_rate, etc.)

        Returns:
            Dictionary with size and metadata
        """
        size = self.strategy.calculate_size(capital, price, **context)

        # Calculate position value and percentage
        position_value = size * price
        position_pct = (position_value / capital * 100) if capital > 0 else 0

        return {
            'size': round(size, 4),
            'position_value': round(position_value, 2),
            'position_pct': round(position_pct, 2),
            'strategy': self.strategy_name,
            'capital': capital,
            'price': price
        }


def calculate_optimal_position(
    capital: float,
    price: float,
    strategy: str = 'atr',
    **context
) -> float:
    """
    Convenience function for position sizing

    Args:
        capital: Available capital
        price: Current price
        strategy: Sizing strategy
        **context: Strategy context

    Returns:
        Position size
    """
    engine = PositionSizingEngine(strategy=strategy)
    result = engine.calculate(capital, price, **context)
    return result['size']


# Example usage
if __name__ == "__main__":
    capital = 100000
    price = 150.50

    print("Position Sizing Examples")
    print("=" * 60)

    # Fixed fractional (2% risk)
    sizer1 = PositionSizingEngine('fixed_fractional', fraction=0.02)
    result1 = sizer1.calculate(capital, price)
    print(f"\nFixed Fractional (2%):")
    print(f"  Size: {result1['size']:.2f} units")
    print(f"  Value: ${result1['position_value']:,.2f}")
    print(f"  % of Capital: {result1['position_pct']:.2f}%")

    # Kelly Criterion
    sizer2 = PositionSizingEngine('kelly', fraction=0.5)  # Half Kelly
    result2 = sizer2.calculate(
        capital, price,
        win_rate=0.55,
        avg_win=100,
        avg_loss=50
    )
    print(f"\nKelly Criterion (Half Kelly):")
    print(f"  Size: {result2['size']:.2f} units")
    print(f"  Value: ${result2['position_value']:,.2f}")
    print(f"  % of Capital: {result2['position_pct']:.2f}%")

    # ATR-based
    sizer3 = PositionSizingEngine('atr', risk_per_atr=2.0, max_risk_pct=0.02)
    result3 = sizer3.calculate(capital, price, atr=3.5)
    print(f"\nATR-Based:")
    print(f"  Size: {result3['size']:.2f} units")
    print(f"  Value: ${result3['position_value']:,.2f}")
    print(f"  % of Capital: {result3['position_pct']:.2f}%")
