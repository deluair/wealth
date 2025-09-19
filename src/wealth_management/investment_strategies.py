"""
Investment Strategies

Comprehensive investment strategy framework including factor models,
systematic strategies, tactical allocation, and strategy backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats, optimize
from abc import ABC, abstractmethod
import warnings
from datetime import datetime, timedelta

class StrategyType(Enum):
    """Types of investment strategies"""
    BUY_AND_HOLD = "buy_and_hold"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    GROWTH = "growth"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"
    DIVIDEND = "dividend"
    FACTOR_BASED = "factor_based"
    TACTICAL_ALLOCATION = "tactical_allocation"
    RISK_PARITY = "risk_parity"
    VOLATILITY_TARGETING = "volatility_targeting"
    TREND_FOLLOWING = "trend_following"
    PAIRS_TRADING = "pairs_trading"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"

class RebalancingFrequency(Enum):
    """Portfolio rebalancing frequencies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"
    THRESHOLD_BASED = "threshold_based"

class FactorType(Enum):
    """Investment factor types"""
    VALUE = "value"
    MOMENTUM = "momentum"
    SIZE = "size"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"
    PROFITABILITY = "profitability"
    INVESTMENT = "investment"
    DIVIDEND_YIELD = "dividend_yield"
    EARNINGS_YIELD = "earnings_yield"
    BOOK_TO_MARKET = "book_to_market"

class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

@dataclass
class Factor:
    """Investment factor definition"""
    name: str
    factor_type: FactorType
    description: str
    
    # Factor calculation parameters
    lookback_period: int = 252  # Days
    calculation_method: str = "standard"
    
    # Factor values and scores
    factor_values: Dict[str, float] = field(default_factory=dict)
    factor_scores: Dict[str, float] = field(default_factory=dict)
    
    # Factor performance metrics
    factor_returns: Optional[np.ndarray] = None
    information_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Factor loadings
    factor_loadings: Dict[str, float] = field(default_factory=dict)

@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    asset: str
    signal_type: SignalType
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    timestamp: str
    
    # Signal details
    target_weight: float = 0.0
    current_weight: float = 0.0
    expected_return: float = 0.0
    expected_risk: float = 0.0
    
    # Signal rationale
    factors_contributing: List[str] = field(default_factory=list)
    signal_description: str = ""

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    start_date: str
    end_date: str
    
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    var_95: float = 0.0
    beta: float = 1.0
    
    # Activity metrics
    turnover: float = 0.0
    number_of_trades: int = 0
    win_rate: float = 0.0
    
    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    
    # Period returns
    monthly_returns: List[float] = field(default_factory=list)
    yearly_returns: List[float] = field(default_factory=list)
    
    # Drawdown analysis
    drawdown_periods: List[Dict] = field(default_factory=list)

@dataclass
class StrategyConfig:
    """Strategy configuration parameters"""
    strategy_type: StrategyType
    name: str
    description: str
    
    # Rebalancing settings
    rebalancing_frequency: RebalancingFrequency = RebalancingFrequency.MONTHLY
    rebalancing_threshold: float = 0.05  # For threshold-based rebalancing
    
    # Risk management
    max_position_size: float = 0.1  # Maximum weight per position
    max_sector_exposure: float = 0.3  # Maximum sector exposure
    stop_loss: Optional[float] = None  # Stop loss threshold
    take_profit: Optional[float] = None  # Take profit threshold
    
    # Strategy-specific parameters
    lookback_period: int = 252
    momentum_period: int = 126
    mean_reversion_period: int = 20
    volatility_target: float = 0.15
    
    # Factor parameters
    factors: List[Factor] = field(default_factory=list)
    factor_weights: Dict[str, float] = field(default_factory=dict)
    
    # Transaction costs
    transaction_cost: float = 0.001  # 10 bps
    management_fee: float = 0.01  # 1% annual
    
    # Other parameters
    min_weight: float = 0.01
    max_weight: float = 0.2
    cash_buffer: float = 0.02

class InvestmentStrategy(ABC):
    """Abstract base class for investment strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.current_weights = {}
        self.historical_weights = []
        self.performance_history = []
        self.signals_history = []
        
    @abstractmethod
    def generate_signals(self, market_data: Dict, current_date: str) -> List[TradingSignal]:
        """Generate trading signals based on market data"""
        pass
    
    @abstractmethod
    def calculate_target_weights(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Calculate target portfolio weights from signals"""
        pass
    
    def should_rebalance(self, current_date: str, last_rebalance_date: str) -> bool:
        """Determine if portfolio should be rebalanced"""
        if self.config.rebalancing_frequency == RebalancingFrequency.DAILY:
            return True
        elif self.config.rebalancing_frequency == RebalancingFrequency.WEEKLY:
            # Rebalance on Mondays (simplified)
            return datetime.strptime(current_date, "%Y-%m-%d").weekday() == 0
        elif self.config.rebalancing_frequency == RebalancingFrequency.MONTHLY:
            current = datetime.strptime(current_date, "%Y-%m-%d")
            last = datetime.strptime(last_rebalance_date, "%Y-%m-%d")
            return current.month != last.month
        elif self.config.rebalancing_frequency == RebalancingFrequency.QUARTERLY:
            current = datetime.strptime(current_date, "%Y-%m-%d")
            last = datetime.strptime(last_rebalance_date, "%Y-%m-%d")
            return (current.month - 1) // 3 != (last.month - 1) // 3
        elif self.config.rebalancing_frequency == RebalancingFrequency.THRESHOLD_BASED:
            # Check if any position has drifted beyond threshold
            for asset, current_weight in self.current_weights.items():
                target_weight = self.target_weights.get(asset, 0)
                if abs(current_weight - target_weight) > self.config.rebalancing_threshold:
                    return True
            return False
        
        return False
    
    def apply_risk_constraints(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply risk management constraints to target weights"""
        constrained_weights = target_weights.copy()
        
        # Apply maximum position size constraint
        for asset in constrained_weights:
            if constrained_weights[asset] > self.config.max_position_size:
                constrained_weights[asset] = self.config.max_position_size
        
        # Apply minimum weight constraint
        for asset in list(constrained_weights.keys()):
            if constrained_weights[asset] < self.config.min_weight:
                del constrained_weights[asset]
        
        # Normalize weights to sum to 1 (minus cash buffer)
        total_weight = sum(constrained_weights.values())
        target_invested = 1.0 - self.config.cash_buffer
        
        if total_weight > 0:
            scaling_factor = target_invested / total_weight
            for asset in constrained_weights:
                constrained_weights[asset] *= scaling_factor
        
        return constrained_weights

class MomentumStrategy(InvestmentStrategy):
    """Momentum-based investment strategy"""
    
    def generate_signals(self, market_data: Dict, current_date: str) -> List[TradingSignal]:
        """Generate momentum signals"""
        signals = []
        
        for asset, price_data in market_data.items():
            if len(price_data) < self.config.momentum_period:
                continue
            
            # Calculate momentum score
            recent_prices = price_data[-self.config.momentum_period:]
            momentum_return = (recent_prices[-1] / recent_prices[0]) - 1
            
            # Calculate volatility for risk adjustment
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            
            # Risk-adjusted momentum
            risk_adjusted_momentum = momentum_return / volatility if volatility > 0 else 0
            
            # Generate signal
            if risk_adjusted_momentum > 0.1:  # Positive momentum threshold
                signal_type = SignalType.BUY
                strength = min(1.0, risk_adjusted_momentum / 0.5)
            elif risk_adjusted_momentum < -0.1:  # Negative momentum threshold
                signal_type = SignalType.SELL
                strength = min(1.0, abs(risk_adjusted_momentum) / 0.5)
            else:
                signal_type = SignalType.HOLD
                strength = 0.5
            
            signal = TradingSignal(
                asset=asset,
                signal_type=signal_type,
                strength=strength,
                confidence=min(1.0, abs(risk_adjusted_momentum) / 0.3),
                timestamp=current_date,
                expected_return=momentum_return,
                expected_risk=volatility,
                signal_description=f"Momentum: {momentum_return:.3f}, Risk-adj: {risk_adjusted_momentum:.3f}"
            )
            
            signals.append(signal)
        
        return signals
    
    def calculate_target_weights(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Calculate target weights based on momentum signals"""
        target_weights = {}
        
        # Filter for buy signals
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        
        if not buy_signals:
            return target_weights
        
        # Calculate weights based on signal strength
        total_strength = sum(s.strength for s in buy_signals)
        
        for signal in buy_signals:
            if total_strength > 0:
                weight = (signal.strength / total_strength) * (1.0 - self.config.cash_buffer)
                target_weights[signal.asset] = weight
        
        return self.apply_risk_constraints(target_weights)

class MeanReversionStrategy(InvestmentStrategy):
    """Mean reversion investment strategy"""
    
    def generate_signals(self, market_data: Dict, current_date: str) -> List[TradingSignal]:
        """Generate mean reversion signals"""
        signals = []
        
        for asset, price_data in market_data.items():
            if len(price_data) < self.config.mean_reversion_period:
                continue
            
            # Calculate mean reversion metrics
            recent_prices = price_data[-self.config.mean_reversion_period:]
            mean_price = np.mean(recent_prices)
            current_price = recent_prices[-1]
            std_price = np.std(recent_prices)
            
            # Z-score for mean reversion
            z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
            
            # Generate signal based on z-score
            if z_score < -1.5:  # Oversold
                signal_type = SignalType.BUY
                strength = min(1.0, abs(z_score) / 3.0)
            elif z_score > 1.5:  # Overbought
                signal_type = SignalType.SELL
                strength = min(1.0, abs(z_score) / 3.0)
            else:
                signal_type = SignalType.HOLD
                strength = 0.5
            
            signal = TradingSignal(
                asset=asset,
                signal_type=signal_type,
                strength=strength,
                confidence=min(1.0, abs(z_score) / 2.0),
                timestamp=current_date,
                signal_description=f"Z-score: {z_score:.3f}, Mean: {mean_price:.2f}"
            )
            
            signals.append(signal)
        
        return signals
    
    def calculate_target_weights(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Calculate target weights for mean reversion"""
        target_weights = {}
        
        # Filter for buy signals (oversold assets)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        
        if not buy_signals:
            return target_weights
        
        # Equal weight allocation to oversold assets
        weight_per_asset = (1.0 - self.config.cash_buffer) / len(buy_signals)
        
        for signal in buy_signals:
            target_weights[signal.asset] = weight_per_asset * signal.strength
        
        return self.apply_risk_constraints(target_weights)

class FactorStrategy(InvestmentStrategy):
    """Multi-factor investment strategy"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.factor_models = {}
        self.factor_scores = {}
    
    def calculate_factor_scores(self, market_data: Dict, fundamental_data: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate factor scores for all assets"""
        factor_scores = {}
        
        for asset in market_data.keys():
            asset_scores = {}
            
            # Value factor (P/E, P/B ratios)
            if 'value' in [f.factor_type.value for f in self.config.factors]:
                pe_ratio = fundamental_data.get(asset, {}).get('pe_ratio', 15)
                pb_ratio = fundamental_data.get(asset, {}).get('pb_ratio', 1.5)
                # Lower ratios = higher value score
                value_score = 1.0 / (1.0 + pe_ratio/15 + pb_ratio/1.5)
                asset_scores['value'] = value_score
            
            # Momentum factor
            if 'momentum' in [f.factor_type.value for f in self.config.factors]:
                prices = market_data[asset]
                if len(prices) >= 252:
                    momentum_return = (prices[-1] / prices[-252]) - 1
                    momentum_score = max(0, momentum_return)  # Positive momentum only
                    asset_scores['momentum'] = momentum_score
            
            # Quality factor (ROE, Debt/Equity)
            if 'quality' in [f.factor_type.value for f in self.config.factors]:
                roe = fundamental_data.get(asset, {}).get('roe', 0.1)
                debt_equity = fundamental_data.get(asset, {}).get('debt_equity', 0.5)
                quality_score = roe / (1.0 + debt_equity)
                asset_scores['quality'] = quality_score
            
            # Low volatility factor
            if 'low_volatility' in [f.factor_type.value for f in self.config.factors]:
                prices = market_data[asset]
                if len(prices) >= 60:
                    returns = np.diff(prices[-60:]) / prices[-61:-1]
                    volatility = np.std(returns) * np.sqrt(252)
                    # Lower volatility = higher score
                    low_vol_score = 1.0 / (1.0 + volatility)
                    asset_scores['low_volatility'] = low_vol_score
            
            factor_scores[asset] = asset_scores
        
        return factor_scores
    
    def generate_signals(self, market_data: Dict, current_date: str, 
                        fundamental_data: Optional[Dict] = None) -> List[TradingSignal]:
        """Generate factor-based signals"""
        if fundamental_data is None:
            fundamental_data = {}
        
        # Calculate factor scores
        factor_scores = self.calculate_factor_scores(market_data, fundamental_data)
        
        signals = []
        
        for asset, scores in factor_scores.items():
            # Combine factor scores using configured weights
            combined_score = 0.0
            total_weight = 0.0
            
            for factor in self.config.factors:
                factor_name = factor.factor_type.value
                if factor_name in scores:
                    weight = self.config.factor_weights.get(factor_name, 1.0)
                    combined_score += scores[factor_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                combined_score /= total_weight
            
            # Generate signal based on combined score
            if combined_score > 0.7:
                signal_type = SignalType.BUY
                strength = min(1.0, combined_score)
            elif combined_score < 0.3:
                signal_type = SignalType.SELL
                strength = min(1.0, 1.0 - combined_score)
            else:
                signal_type = SignalType.HOLD
                strength = 0.5
            
            signal = TradingSignal(
                asset=asset,
                signal_type=signal_type,
                strength=strength,
                confidence=combined_score,
                timestamp=current_date,
                factors_contributing=list(scores.keys()),
                signal_description=f"Combined factor score: {combined_score:.3f}"
            )
            
            signals.append(signal)
        
        return signals
    
    def calculate_target_weights(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Calculate target weights based on factor signals"""
        target_weights = {}
        
        # Filter for buy signals
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        
        if not buy_signals:
            return target_weights
        
        # Weight by confidence (factor score)
        total_confidence = sum(s.confidence for s in buy_signals)
        
        for signal in buy_signals:
            if total_confidence > 0:
                weight = (signal.confidence / total_confidence) * (1.0 - self.config.cash_buffer)
                target_weights[signal.asset] = weight
        
        return self.apply_risk_constraints(target_weights)

class RiskParityStrategy(InvestmentStrategy):
    """Risk parity investment strategy"""
    
    def generate_signals(self, market_data: Dict, current_date: str) -> List[TradingSignal]:
        """Generate risk parity signals (all assets get equal risk allocation)"""
        signals = []
        
        for asset in market_data.keys():
            # Risk parity gives equal signals to all assets
            signal = TradingSignal(
                asset=asset,
                signal_type=SignalType.BUY,
                strength=1.0,
                confidence=1.0,
                timestamp=current_date,
                signal_description="Risk parity allocation"
            )
            signals.append(signal)
        
        return signals
    
    def calculate_target_weights(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Calculate risk parity weights"""
        # This is a simplified risk parity - equal volatility contribution
        # In practice, would need covariance matrix for true risk parity
        
        target_weights = {}
        n_assets = len(signals)
        
        if n_assets == 0:
            return target_weights
        
        # Equal weight as simplified risk parity
        equal_weight = (1.0 - self.config.cash_buffer) / n_assets
        
        for signal in signals:
            target_weights[signal.asset] = equal_weight
        
        return self.apply_risk_constraints(target_weights)

class VolatilityTargetingStrategy(InvestmentStrategy):
    """Volatility targeting strategy"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.target_volatility = config.volatility_target
        self.current_volatility = 0.15  # Initial estimate
    
    def calculate_portfolio_volatility(self, market_data: Dict, weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility"""
        if not weights:
            return 0.0
        
        # Simplified volatility calculation
        asset_vols = []
        asset_weights = []
        
        for asset, weight in weights.items():
            if asset in market_data and len(market_data[asset]) >= 30:
                prices = market_data[asset][-30:]  # Last 30 days
                returns = np.diff(prices) / prices[:-1]
                vol = np.std(returns) * np.sqrt(252)
                asset_vols.append(vol)
                asset_weights.append(weight)
        
        if not asset_vols:
            return 0.15  # Default volatility
        
        # Weighted average volatility (simplified)
        portfolio_vol = np.average(asset_vols, weights=asset_weights)
        return portfolio_vol
    
    def generate_signals(self, market_data: Dict, current_date: str) -> List[TradingSignal]:
        """Generate volatility-adjusted signals"""
        signals = []
        
        # Calculate current portfolio volatility
        self.current_volatility = self.calculate_portfolio_volatility(market_data, self.current_weights)
        
        # Volatility scaling factor
        vol_scaling = self.target_volatility / self.current_volatility if self.current_volatility > 0 else 1.0
        vol_scaling = max(0.5, min(2.0, vol_scaling))  # Limit scaling
        
        for asset in market_data.keys():
            # Adjust signal strength based on volatility targeting
            if vol_scaling > 1.1:  # Need to increase risk
                signal_type = SignalType.BUY
                strength = min(1.0, vol_scaling - 1.0)
            elif vol_scaling < 0.9:  # Need to decrease risk
                signal_type = SignalType.SELL
                strength = min(1.0, 1.0 - vol_scaling)
            else:
                signal_type = SignalType.HOLD
                strength = 0.5
            
            signal = TradingSignal(
                asset=asset,
                signal_type=signal_type,
                strength=strength,
                confidence=0.8,
                timestamp=current_date,
                signal_description=f"Vol targeting: current={self.current_volatility:.3f}, target={self.target_volatility:.3f}"
            )
            
            signals.append(signal)
        
        return signals
    
    def calculate_target_weights(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Calculate volatility-adjusted target weights"""
        target_weights = {}
        
        # Start with current weights and adjust
        for signal in signals:
            current_weight = self.current_weights.get(signal.asset, 0.0)
            
            if signal.signal_type == SignalType.BUY:
                # Increase weight
                new_weight = current_weight * (1.0 + signal.strength * 0.1)
            elif signal.signal_type == SignalType.SELL:
                # Decrease weight
                new_weight = current_weight * (1.0 - signal.strength * 0.1)
            else:
                new_weight = current_weight
            
            if new_weight > self.config.min_weight:
                target_weights[signal.asset] = new_weight
        
        # Normalize weights
        total_weight = sum(target_weights.values())
        if total_weight > 0:
            scaling_factor = (1.0 - self.config.cash_buffer) / total_weight
            for asset in target_weights:
                target_weights[asset] *= scaling_factor
        
        return self.apply_risk_constraints(target_weights)

class StrategyBacktester:
    """Backtesting engine for investment strategies"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.transaction_costs = {}
        self.benchmark_data = None
    
    def set_benchmark(self, benchmark_returns: np.ndarray) -> None:
        """Set benchmark for performance comparison"""
        self.benchmark_data = benchmark_returns
    
    def backtest_strategy(self, strategy: InvestmentStrategy,
                         market_data: Dict[str, np.ndarray],
                         start_date: str, end_date: str,
                         fundamental_data: Optional[Dict] = None) -> StrategyPerformance:
        """
        Backtest an investment strategy
        
        Args:
            strategy: Investment strategy to backtest
            market_data: Historical price data
            start_date: Backtest start date
            end_date: Backtest end date
            fundamental_data: Optional fundamental data
            
        Returns:
            StrategyPerformance object with results
        """
        # Initialize tracking variables
        portfolio_value = self.initial_capital
        portfolio_values = [portfolio_value]
        dates = []
        weights_history = []
        trades = []
        
        # Convert dates to datetime for easier handling
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Simulate daily trading (simplified)
        current_date = start_dt
        last_rebalance_date = start_dt
        
        while current_date <= end_dt:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Get market data up to current date
            current_market_data = self._get_market_data_up_to_date(market_data, current_date)
            
            # Check if we should rebalance
            if strategy.should_rebalance(date_str, last_rebalance_date.strftime("%Y-%m-%d")):
                # Generate signals
                if hasattr(strategy, 'generate_signals'):
                    if fundamental_data and 'fundamental_data' in strategy.generate_signals.__code__.co_varnames:
                        signals = strategy.generate_signals(current_market_data, date_str, fundamental_data)
                    else:
                        signals = strategy.generate_signals(current_market_data, date_str)
                
                    # Calculate target weights
                    target_weights = strategy.calculate_target_weights(signals)
                    
                    # Execute trades (simplified)
                    trades_executed = self._execute_trades(strategy.current_weights, target_weights, portfolio_value)
                    trades.extend(trades_executed)
                    
                    # Update strategy weights
                    strategy.current_weights = target_weights.copy()
                    last_rebalance_date = current_date
            
            # Update portfolio value based on market movements
            if strategy.current_weights:
                daily_return = self._calculate_daily_return(current_market_data, strategy.current_weights, current_date)
                portfolio_value *= (1 + daily_return)
            
            # Record data
            portfolio_values.append(portfolio_value)
            dates.append(current_date)
            weights_history.append(strategy.current_weights.copy())
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Calculate performance metrics
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        performance = StrategyPerformance(
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate metrics
        performance.total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        performance.annualized_return = (1 + performance.total_return) ** (252 / len(portfolio_returns)) - 1
        performance.volatility = np.std(portfolio_returns) * np.sqrt(252)
        performance.sharpe_ratio = performance.annualized_return / performance.volatility if performance.volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        performance.max_drawdown = np.min(drawdowns)
        
        # Trading metrics
        performance.number_of_trades = len(trades)
        performance.turnover = self._calculate_turnover(weights_history)
        
        # Benchmark comparison
        if self.benchmark_data is not None and len(self.benchmark_data) >= len(portfolio_returns):
            benchmark_returns = self.benchmark_data[:len(portfolio_returns)]
            performance.benchmark_return = np.prod(1 + benchmark_returns) - 1
            performance.alpha = performance.total_return - performance.benchmark_return
            
            # Beta calculation
            if len(portfolio_returns) == len(benchmark_returns):
                covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                performance.beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                
                # Tracking error and information ratio
                excess_returns = portfolio_returns - benchmark_returns
                performance.tracking_error = np.std(excess_returns) * np.sqrt(252)
                performance.information_ratio = np.mean(excess_returns) * 252 / performance.tracking_error if performance.tracking_error > 0 else 0
        
        return performance
    
    def _get_market_data_up_to_date(self, market_data: Dict[str, np.ndarray], 
                                   current_date: datetime) -> Dict[str, np.ndarray]:
        """Get market data up to current date (simplified)"""
        # In a real implementation, this would filter data by date
        # For now, return all data (assuming it's already filtered)
        return market_data
    
    def _calculate_daily_return(self, market_data: Dict[str, np.ndarray],
                               weights: Dict[str, float], current_date: datetime) -> float:
        """Calculate daily portfolio return (simplified)"""
        if not weights:
            return 0.0
        
        # Simplified: assume 0.1% daily return for demonstration
        # In practice, would calculate based on actual price changes
        total_return = 0.0
        for asset, weight in weights.items():
            # Simulate random daily return
            daily_return = np.random.normal(0.0005, 0.02)  # 0.05% mean, 2% daily vol
            total_return += weight * daily_return
        
        return total_return
    
    def _execute_trades(self, current_weights: Dict[str, float],
                       target_weights: Dict[str, float],
                       portfolio_value: float) -> List[Dict]:
        """Execute trades to reach target weights"""
        trades = []
        
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        
        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)
            
            if abs(current_weight - target_weight) > 0.001:  # 0.1% threshold
                trade_amount = (target_weight - current_weight) * portfolio_value
                
                trade = {
                    'asset': asset,
                    'amount': trade_amount,
                    'from_weight': current_weight,
                    'to_weight': target_weight,
                    'transaction_cost': abs(trade_amount) * 0.001  # 10 bps
                }
                trades.append(trade)
        
        return trades
    
    def _calculate_turnover(self, weights_history: List[Dict[str, float]]) -> float:
        """Calculate portfolio turnover"""
        if len(weights_history) < 2:
            return 0.0
        
        total_turnover = 0.0
        
        for i in range(1, len(weights_history)):
            current_weights = weights_history[i]
            previous_weights = weights_history[i-1]
            
            all_assets = set(current_weights.keys()) | set(previous_weights.keys())
            
            period_turnover = 0.0
            for asset in all_assets:
                current_weight = current_weights.get(asset, 0.0)
                previous_weight = previous_weights.get(asset, 0.0)
                period_turnover += abs(current_weight - previous_weight)
            
            total_turnover += period_turnover / 2  # Divide by 2 for one-way turnover
        
        return total_turnover / (len(weights_history) - 1)  # Average turnover

class StrategyManager:
    """Manager for multiple investment strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.backtester = StrategyBacktester()
        self.performance_history = {}
    
    def add_strategy(self, strategy: InvestmentStrategy) -> None:
        """Add a strategy to the manager"""
        self.strategies[strategy.name] = strategy
    
    def create_momentum_strategy(self, name: str = "Momentum Strategy") -> MomentumStrategy:
        """Create a momentum strategy with default parameters"""
        config = StrategyConfig(
            strategy_type=StrategyType.MOMENTUM,
            name=name,
            description="12-month momentum strategy with risk adjustment",
            momentum_period=252,
            rebalancing_frequency=RebalancingFrequency.MONTHLY,
            max_position_size=0.15,
            transaction_cost=0.001
        )
        
        strategy = MomentumStrategy(config)
        self.add_strategy(strategy)
        return strategy
    
    def create_mean_reversion_strategy(self, name: str = "Mean Reversion Strategy") -> MeanReversionStrategy:
        """Create a mean reversion strategy with default parameters"""
        config = StrategyConfig(
            strategy_type=StrategyType.MEAN_REVERSION,
            name=name,
            description="Short-term mean reversion strategy",
            mean_reversion_period=20,
            rebalancing_frequency=RebalancingFrequency.WEEKLY,
            max_position_size=0.1,
            transaction_cost=0.002
        )
        
        strategy = MeanReversionStrategy(config)
        self.add_strategy(strategy)
        return strategy
    
    def create_factor_strategy(self, name: str = "Multi-Factor Strategy",
                              factors: Optional[List[FactorType]] = None) -> FactorStrategy:
        """Create a multi-factor strategy"""
        if factors is None:
            factors = [FactorType.VALUE, FactorType.MOMENTUM, FactorType.QUALITY, FactorType.LOW_VOLATILITY]
        
        factor_objects = []
        factor_weights = {}
        
        for factor_type in factors:
            factor = Factor(
                name=factor_type.value,
                factor_type=factor_type,
                description=f"{factor_type.value.title()} factor"
            )
            factor_objects.append(factor)
            factor_weights[factor_type.value] = 1.0 / len(factors)  # Equal weight
        
        config = StrategyConfig(
            strategy_type=StrategyType.FACTOR_BASED,
            name=name,
            description="Multi-factor investment strategy",
            factors=factor_objects,
            factor_weights=factor_weights,
            rebalancing_frequency=RebalancingFrequency.MONTHLY,
            max_position_size=0.12,
            transaction_cost=0.001
        )
        
        strategy = FactorStrategy(config)
        self.add_strategy(strategy)
        return strategy
    
    def create_risk_parity_strategy(self, name: str = "Risk Parity Strategy") -> RiskParityStrategy:
        """Create a risk parity strategy"""
        config = StrategyConfig(
            strategy_type=StrategyType.RISK_PARITY,
            name=name,
            description="Equal risk contribution strategy",
            rebalancing_frequency=RebalancingFrequency.QUARTERLY,
            max_position_size=0.25,
            transaction_cost=0.0005
        )
        
        strategy = RiskParityStrategy(config)
        self.add_strategy(strategy)
        return strategy
    
    def create_volatility_targeting_strategy(self, name: str = "Vol Target Strategy",
                                           target_vol: float = 0.15) -> VolatilityTargetingStrategy:
        """Create a volatility targeting strategy"""
        config = StrategyConfig(
            strategy_type=StrategyType.VOLATILITY_TARGETING,
            name=name,
            description=f"Volatility targeting strategy ({target_vol:.1%} target)",
            volatility_target=target_vol,
            rebalancing_frequency=RebalancingFrequency.WEEKLY,
            max_position_size=0.2,
            transaction_cost=0.001
        )
        
        strategy = VolatilityTargetingStrategy(config)
        self.add_strategy(strategy)
        return strategy
    
    def backtest_all_strategies(self, market_data: Dict[str, np.ndarray],
                               start_date: str, end_date: str,
                               fundamental_data: Optional[Dict] = None) -> Dict[str, StrategyPerformance]:
        """Backtest all strategies"""
        results = {}
        
        for name, strategy in self.strategies.items():
            try:
                performance = self.backtester.backtest_strategy(
                    strategy, market_data, start_date, end_date, fundamental_data
                )
                results[name] = performance
                self.performance_history[name] = performance
            except Exception as e:
                print(f"Error backtesting {name}: {str(e)}")
                continue
        
        return results
    
    def compare_strategies(self) -> pd.DataFrame:
        """Compare performance of all strategies"""
        if not self.performance_history:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, performance in self.performance_history.items():
            comparison_data.append({
                'Strategy': name,
                'Total Return': performance.total_return,
                'Annualized Return': performance.annualized_return,
                'Volatility': performance.volatility,
                'Sharpe Ratio': performance.sharpe_ratio,
                'Max Drawdown': performance.max_drawdown,
                'Alpha': performance.alpha,
                'Beta': performance.beta,
                'Information Ratio': performance.information_ratio,
                'Turnover': performance.turnover,
                'Number of Trades': performance.number_of_trades
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_strategy_recommendations(self, risk_profile: str = "moderate") -> List[str]:
        """Get strategy recommendations based on risk profile"""
        recommendations = []
        
        if risk_profile.lower() == "conservative":
            recommendations.extend([
                "Consider Risk Parity Strategy for stable risk-adjusted returns",
                "Low Volatility Factor Strategy may suit conservative investors",
                "Volatility Targeting with low target (10-12%) recommended"
            ])
        elif risk_profile.lower() == "moderate":
            recommendations.extend([
                "Multi-Factor Strategy provides good diversification",
                "Combination of Momentum and Mean Reversion strategies",
                "Volatility Targeting at 15% provides balanced risk exposure"
            ])
        elif risk_profile.lower() == "aggressive":
            recommendations.extend([
                "Pure Momentum Strategy for higher return potential",
                "Factor Strategy with Growth and Momentum emphasis",
                "Higher volatility targeting (20%+) acceptable"
            ])
        
        return recommendations