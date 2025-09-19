"""
Performance Analyzer

Comprehensive performance analysis system for portfolios and investment strategies.
Includes return analysis, risk metrics, attribution analysis, and benchmarking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
import warnings
from datetime import datetime, timedelta

class PerformancePeriod(Enum):
    """Performance measurement periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    INCEPTION_TO_DATE = "inception_to_date"

class BenchmarkType(Enum):
    """Types of benchmarks"""
    MARKET_INDEX = "market_index"
    PEER_GROUP = "peer_group"
    CUSTOM_BENCHMARK = "custom_benchmark"
    RISK_FREE_RATE = "risk_free_rate"
    TARGET_RETURN = "target_return"

class AttributionType(Enum):
    """Types of performance attribution"""
    ASSET_ALLOCATION = "asset_allocation"
    SECURITY_SELECTION = "security_selection"
    INTERACTION = "interaction"
    CURRENCY = "currency"
    SECTOR = "sector"
    STYLE = "style"

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    
    # Basic return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    compound_annual_growth_rate: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    downside_volatility: float = 0.0
    tracking_error: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    
    # Drawdown metrics
    maximum_drawdown: float = 0.0
    average_drawdown: float = 0.0
    maximum_drawdown_duration: int = 0
    recovery_time: int = 0
    
    # Relative performance
    alpha: float = 0.0
    beta: float = 1.0
    correlation: float = 0.0
    r_squared: float = 0.0
    
    # Distribution metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # Activity metrics
    hit_rate: float = 0.0  # Percentage of positive periods
    up_capture: float = 1.0  # Upside capture ratio
    down_capture: float = 1.0  # Downside capture ratio
    
    # Consistency metrics
    consistency_ratio: float = 0.0
    pain_index: float = 0.0
    ulcer_index: float = 0.0
    
    # Time-weighted metrics
    time_weighted_return: float = 0.0
    money_weighted_return: float = 0.0

@dataclass
class AttributionAnalysis:
    """Performance attribution analysis"""
    
    # Asset allocation attribution
    asset_allocation_effect: Dict[str, float] = field(default_factory=dict)
    security_selection_effect: Dict[str, float] = field(default_factory=dict)
    interaction_effect: Dict[str, float] = field(default_factory=dict)
    
    # Total attribution
    total_attribution: float = 0.0
    unexplained_return: float = 0.0
    
    # Sector/Style attribution
    sector_attribution: Dict[str, float] = field(default_factory=dict)
    style_attribution: Dict[str, float] = field(default_factory=dict)
    
    # Currency attribution (for international portfolios)
    currency_attribution: Dict[str, float] = field(default_factory=dict)
    
    # Summary statistics
    attribution_summary: Dict[str, float] = field(default_factory=dict)

@dataclass
class BenchmarkComparison:
    """Benchmark comparison analysis"""
    
    benchmark_name: str
    benchmark_type: BenchmarkType
    
    # Relative performance
    excess_return: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    
    # Regression statistics
    alpha: float = 0.0
    beta: float = 1.0
    r_squared: float = 0.0
    correlation: float = 0.0
    
    # Capture ratios
    up_capture_ratio: float = 1.0
    down_capture_ratio: float = 1.0
    
    # Rolling metrics
    rolling_correlation: List[float] = field(default_factory=list)
    rolling_beta: List[float] = field(default_factory=list)
    rolling_alpha: List[float] = field(default_factory=list)
    
    # Outperformance statistics
    outperformance_frequency: float = 0.0
    average_outperformance: float = 0.0
    average_underperformance: float = 0.0

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    
    portfolio_name: str
    report_date: str
    analysis_period: str
    
    # Core metrics
    performance_metrics: PerformanceMetrics
    
    # Benchmark comparisons
    benchmark_comparisons: List[BenchmarkComparison] = field(default_factory=list)
    
    # Attribution analysis
    attribution_analysis: Optional[AttributionAnalysis] = None
    
    # Period returns
    period_returns: Dict[str, List[float]] = field(default_factory=dict)
    
    # Risk analysis
    risk_analysis: Dict[str, float] = field(default_factory=dict)
    
    # Performance summary
    performance_summary: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)

class PerformanceAnalyzer:
    """Comprehensive performance analysis system"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.benchmarks = {}
        self.performance_cache = {}
        
    def add_benchmark(self, name: str, returns: np.ndarray, 
                     benchmark_type: BenchmarkType = BenchmarkType.MARKET_INDEX) -> None:
        """Add a benchmark for comparison"""
        self.benchmarks[name] = {
            'returns': returns,
            'type': benchmark_type
        }
    
    def calculate_performance_metrics(self, returns: np.ndarray, 
                                    benchmark_returns: Optional[np.ndarray] = None,
                                    frequency: int = 252) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns for relative metrics
            frequency: Return frequency (252 for daily, 12 for monthly)
            
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        
        if len(returns) == 0:
            return metrics
        
        # Basic return metrics
        metrics.total_return = np.prod(1 + returns) - 1
        metrics.annualized_return = (1 + metrics.total_return) ** (frequency / len(returns)) - 1
        metrics.compound_annual_growth_rate = metrics.annualized_return
        
        # Risk metrics
        metrics.volatility = np.std(returns) * np.sqrt(frequency)
        
        # Downside volatility (below risk-free rate)
        downside_returns = returns[returns < self.risk_free_rate / frequency]
        if len(downside_returns) > 0:
            metrics.downside_volatility = np.std(downside_returns) * np.sqrt(frequency)
        
        # Risk-adjusted returns
        excess_return = metrics.annualized_return - self.risk_free_rate
        if metrics.volatility > 0:
            metrics.sharpe_ratio = excess_return / metrics.volatility
        
        if metrics.downside_volatility > 0:
            metrics.sortino_ratio = excess_return / metrics.downside_volatility
        
        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(returns)
        metrics.maximum_drawdown = drawdown_metrics['max_drawdown']
        metrics.average_drawdown = drawdown_metrics['avg_drawdown']
        metrics.maximum_drawdown_duration = drawdown_metrics['max_duration']
        metrics.recovery_time = drawdown_metrics['recovery_time']
        
        # Calmar ratio
        if abs(metrics.maximum_drawdown) > 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.maximum_drawdown)
        
        # Distribution metrics
        metrics.skewness = stats.skew(returns)
        metrics.kurtosis = stats.kurtosis(returns)
        
        # Value at Risk
        metrics.var_95 = np.percentile(returns, 5)
        metrics.var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk
        tail_95 = returns[returns <= metrics.var_95]
        tail_99 = returns[returns <= metrics.var_99]
        metrics.cvar_95 = np.mean(tail_95) if len(tail_95) > 0 else metrics.var_95
        metrics.cvar_99 = np.mean(tail_99) if len(tail_99) > 0 else metrics.var_99
        
        # Hit rate
        metrics.hit_rate = np.mean(returns > 0)
        
        # Consistency metrics
        metrics.consistency_ratio = self._calculate_consistency_ratio(returns)
        metrics.pain_index = self._calculate_pain_index(returns)
        metrics.ulcer_index = self._calculate_ulcer_index(returns)
        
        # Benchmark-relative metrics
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns, frequency)
            metrics.alpha = benchmark_metrics['alpha']
            metrics.beta = benchmark_metrics['beta']
            metrics.correlation = benchmark_metrics['correlation']
            metrics.r_squared = benchmark_metrics['r_squared']
            metrics.tracking_error = benchmark_metrics['tracking_error']
            metrics.information_ratio = benchmark_metrics['information_ratio']
            metrics.treynor_ratio = benchmark_metrics['treynor_ratio']
            metrics.up_capture = benchmark_metrics['up_capture']
            metrics.down_capture = benchmark_metrics['down_capture']
        
        return metrics
    
    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics"""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)
        
        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0
        
        # Drawdown duration analysis
        in_drawdown = drawdowns < -0.001  # 0.1% threshold
        drawdown_periods = []
        current_period = 0
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Recovery time (time to recover from max drawdown)
        recovery_time = 0
        if max_dd_idx < len(drawdowns) - 1:
            recovery_level = running_max[max_dd_idx]
            for i in range(max_dd_idx + 1, len(cumulative_returns)):
                if cumulative_returns[i] >= recovery_level:
                    recovery_time = i - max_dd_idx
                    break
            else:
                recovery_time = len(drawdowns) - max_dd_idx  # Still recovering
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_duration': max_duration,
            'recovery_time': recovery_time,
            'drawdowns': drawdowns
        }
    
    def _calculate_benchmark_metrics(self, returns: np.ndarray, 
                                   benchmark_returns: np.ndarray,
                                   frequency: int) -> Dict[str, float]:
        """Calculate benchmark-relative metrics"""
        # Excess returns
        excess_returns = returns - benchmark_returns
        
        # Regression analysis
        slope, intercept, correlation, p_value, std_err = stats.linregress(benchmark_returns, returns)
        
        alpha = intercept * frequency
        beta = slope
        r_squared = correlation ** 2
        
        # Tracking error
        tracking_error = np.std(excess_returns) * np.sqrt(frequency)
        
        # Information ratio
        information_ratio = 0
        if tracking_error > 0:
            information_ratio = (np.mean(excess_returns) * frequency) / tracking_error
        
        # Treynor ratio
        treynor_ratio = 0
        if beta != 0:
            portfolio_return = np.mean(returns) * frequency
            treynor_ratio = (portfolio_return - self.risk_free_rate) / beta
        
        # Capture ratios
        up_periods = benchmark_returns > 0
        down_periods = benchmark_returns < 0
        
        up_capture = 0
        down_capture = 0
        
        if np.any(up_periods):
            portfolio_up = np.mean(returns[up_periods])
            benchmark_up = np.mean(benchmark_returns[up_periods])
            if benchmark_up != 0:
                up_capture = portfolio_up / benchmark_up
        
        if np.any(down_periods):
            portfolio_down = np.mean(returns[down_periods])
            benchmark_down = np.mean(benchmark_returns[down_periods])
            if benchmark_down != 0:
                down_capture = portfolio_down / benchmark_down
        
        return {
            'alpha': alpha,
            'beta': beta,
            'correlation': correlation,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'up_capture': up_capture,
            'down_capture': down_capture
        }
    
    def _calculate_consistency_ratio(self, returns: np.ndarray) -> float:
        """Calculate consistency ratio (percentage of periods beating average)"""
        if len(returns) == 0:
            return 0.0
        
        average_return = np.mean(returns)
        above_average = np.sum(returns > average_return)
        return above_average / len(returns)
    
    def _calculate_pain_index(self, returns: np.ndarray) -> float:
        """Calculate pain index (average drawdown)"""
        drawdown_metrics = self._calculate_drawdown_metrics(returns)
        drawdowns = drawdown_metrics['drawdowns']
        
        # Average of all drawdowns (negative values)
        negative_drawdowns = drawdowns[drawdowns < 0]
        return np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0.0
    
    def _calculate_ulcer_index(self, returns: np.ndarray) -> float:
        """Calculate Ulcer Index (RMS of drawdowns)"""
        drawdown_metrics = self._calculate_drawdown_metrics(returns)
        drawdowns = drawdown_metrics['drawdowns']
        
        # Root mean square of drawdowns
        return np.sqrt(np.mean(drawdowns ** 2))
    
    def calculate_rolling_metrics(self, returns: np.ndarray, 
                                 benchmark_returns: Optional[np.ndarray] = None,
                                 window: int = 252) -> Dict[str, np.ndarray]:
        """Calculate rolling performance metrics"""
        if len(returns) < window:
            return {}
        
        n_periods = len(returns) - window + 1
        rolling_metrics = {
            'returns': np.zeros(n_periods),
            'volatility': np.zeros(n_periods),
            'sharpe': np.zeros(n_periods),
            'max_drawdown': np.zeros(n_periods)
        }
        
        if benchmark_returns is not None:
            rolling_metrics.update({
                'alpha': np.zeros(n_periods),
                'beta': np.zeros(n_periods),
                'correlation': np.zeros(n_periods),
                'tracking_error': np.zeros(n_periods)
            })
        
        for i in range(n_periods):
            window_returns = returns[i:i+window]
            
            # Basic metrics
            rolling_metrics['returns'][i] = (np.prod(1 + window_returns) - 1) * (252 / window)
            rolling_metrics['volatility'][i] = np.std(window_returns) * np.sqrt(252)
            
            excess_return = rolling_metrics['returns'][i] - self.risk_free_rate
            if rolling_metrics['volatility'][i] > 0:
                rolling_metrics['sharpe'][i] = excess_return / rolling_metrics['volatility'][i]
            
            # Drawdown
            dd_metrics = self._calculate_drawdown_metrics(window_returns)
            rolling_metrics['max_drawdown'][i] = dd_metrics['max_drawdown']
            
            # Benchmark metrics
            if benchmark_returns is not None:
                window_benchmark = benchmark_returns[i:i+window]
                benchmark_metrics = self._calculate_benchmark_metrics(
                    window_returns, window_benchmark, 252
                )
                rolling_metrics['alpha'][i] = benchmark_metrics['alpha']
                rolling_metrics['beta'][i] = benchmark_metrics['beta']
                rolling_metrics['correlation'][i] = benchmark_metrics['correlation']
                rolling_metrics['tracking_error'][i] = benchmark_metrics['tracking_error']
        
        return rolling_metrics
    
    def perform_attribution_analysis(self, portfolio_weights: np.ndarray,
                                   portfolio_returns: np.ndarray,
                                   benchmark_weights: np.ndarray,
                                   benchmark_returns: np.ndarray,
                                   asset_returns: np.ndarray) -> AttributionAnalysis:
        """
        Perform Brinson attribution analysis
        
        Args:
            portfolio_weights: Portfolio weights matrix (time x assets)
            portfolio_returns: Portfolio returns
            benchmark_weights: Benchmark weights matrix (time x assets)
            benchmark_returns: Benchmark returns
            asset_returns: Individual asset returns matrix (time x assets)
            
        Returns:
            AttributionAnalysis object
        """
        attribution = AttributionAnalysis()
        
        if len(portfolio_weights) != len(asset_returns) or len(benchmark_weights) != len(asset_returns):
            return attribution
        
        n_periods, n_assets = asset_returns.shape
        
        # Calculate attribution effects for each period
        asset_allocation_effects = np.zeros((n_periods, n_assets))
        security_selection_effects = np.zeros((n_periods, n_assets))
        interaction_effects = np.zeros((n_periods, n_assets))
        
        for t in range(n_periods):
            pw = portfolio_weights[t]  # Portfolio weights
            bw = benchmark_weights[t]  # Benchmark weights
            ar = asset_returns[t]     # Asset returns
            br = np.dot(benchmark_weights[t], asset_returns[t])  # Benchmark return
            
            # Asset allocation effect: (wp - wb) * (rb - rb_total)
            asset_allocation_effects[t] = (pw - bw) * (ar - br)
            
            # Security selection effect: wb * (rp - rb)
            # For simplicity, assume rp = rb (no security selection within asset class)
            security_selection_effects[t] = bw * (ar - ar)  # Zero for now
            
            # Interaction effect: (wp - wb) * (rp - rb)
            interaction_effects[t] = (pw - bw) * (ar - ar)  # Zero for now
        
        # Aggregate effects
        for i in range(n_assets):
            asset_name = f"Asset_{i}"
            attribution.asset_allocation_effect[asset_name] = np.sum(asset_allocation_effects[:, i])
            attribution.security_selection_effect[asset_name] = np.sum(security_selection_effects[:, i])
            attribution.interaction_effect[asset_name] = np.sum(interaction_effects[:, i])
        
        # Total attribution
        attribution.total_attribution = (
            sum(attribution.asset_allocation_effect.values()) +
            sum(attribution.security_selection_effect.values()) +
            sum(attribution.interaction_effect.values())
        )
        
        # Unexplained return
        total_portfolio_return = np.sum(portfolio_returns)
        total_benchmark_return = np.sum(benchmark_returns)
        attribution.unexplained_return = (
            total_portfolio_return - total_benchmark_return - attribution.total_attribution
        )
        
        # Summary
        attribution.attribution_summary = {
            'asset_allocation': sum(attribution.asset_allocation_effect.values()),
            'security_selection': sum(attribution.security_selection_effect.values()),
            'interaction': sum(attribution.interaction_effect.values()),
            'total': attribution.total_attribution,
            'unexplained': attribution.unexplained_return
        }
        
        return attribution
    
    def compare_to_benchmark(self, returns: np.ndarray, 
                           benchmark_name: str) -> Optional[BenchmarkComparison]:
        """Compare portfolio to a specific benchmark"""
        if benchmark_name not in self.benchmarks:
            return None
        
        benchmark_data = self.benchmarks[benchmark_name]
        benchmark_returns = benchmark_data['returns']
        benchmark_type = benchmark_data['type']
        
        if len(returns) != len(benchmark_returns):
            # Align lengths
            min_length = min(len(returns), len(benchmark_returns))
            returns = returns[:min_length]
            benchmark_returns = benchmark_returns[:min_length]
        
        comparison = BenchmarkComparison(
            benchmark_name=benchmark_name,
            benchmark_type=benchmark_type
        )
        
        # Basic comparison metrics
        excess_returns = returns - benchmark_returns
        comparison.excess_return = np.sum(excess_returns)
        comparison.tracking_error = np.std(excess_returns) * np.sqrt(252)
        
        if comparison.tracking_error > 0:
            comparison.information_ratio = (np.mean(excess_returns) * 252) / comparison.tracking_error
        
        # Regression metrics
        benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns, 252)
        comparison.alpha = benchmark_metrics['alpha']
        comparison.beta = benchmark_metrics['beta']
        comparison.r_squared = benchmark_metrics['r_squared']
        comparison.correlation = benchmark_metrics['correlation']
        comparison.up_capture_ratio = benchmark_metrics['up_capture']
        comparison.down_capture_ratio = benchmark_metrics['down_capture']
        
        # Outperformance statistics
        outperformance_periods = excess_returns > 0
        comparison.outperformance_frequency = np.mean(outperformance_periods)
        
        if np.any(outperformance_periods):
            comparison.average_outperformance = np.mean(excess_returns[outperformance_periods])
        
        underperformance_periods = excess_returns < 0
        if np.any(underperformance_periods):
            comparison.average_underperformance = np.mean(excess_returns[underperformance_periods])
        
        # Rolling metrics
        if len(returns) >= 252:
            rolling_metrics = self.calculate_rolling_metrics(returns, benchmark_returns, 252)
            comparison.rolling_correlation = rolling_metrics.get('correlation', []).tolist()
            comparison.rolling_beta = rolling_metrics.get('beta', []).tolist()
            comparison.rolling_alpha = rolling_metrics.get('alpha', []).tolist()
        
        return comparison
    
    def generate_performance_report(self, returns: np.ndarray,
                                  portfolio_name: str = "Portfolio",
                                  benchmark_names: Optional[List[str]] = None) -> PerformanceReport:
        """Generate comprehensive performance report"""
        report = PerformanceReport(
            portfolio_name=portfolio_name,
            report_date=datetime.now().strftime("%Y-%m-%d"),
            analysis_period=f"{len(returns)} periods"
        )
        
        # Calculate core performance metrics
        primary_benchmark = None
        if benchmark_names and benchmark_names[0] in self.benchmarks:
            primary_benchmark = self.benchmarks[benchmark_names[0]]['returns']
        
        report.performance_metrics = self.calculate_performance_metrics(
            returns, primary_benchmark
        )
        
        # Benchmark comparisons
        if benchmark_names:
            for benchmark_name in benchmark_names:
                comparison = self.compare_to_benchmark(returns, benchmark_name)
                if comparison:
                    report.benchmark_comparisons.append(comparison)
        
        # Period returns analysis
        report.period_returns = self._calculate_period_returns(returns)
        
        # Risk analysis
        report.risk_analysis = self._generate_risk_analysis(returns)
        
        # Performance summary
        report.performance_summary = self._generate_performance_summary(
            report.performance_metrics, report.benchmark_comparisons
        )
        
        # Recommendations
        report.recommendations = self._generate_recommendations(
            report.performance_metrics, report.benchmark_comparisons
        )
        
        return report
    
    def _calculate_period_returns(self, returns: np.ndarray) -> Dict[str, List[float]]:
        """Calculate returns for different periods"""
        period_returns = {}
        
        # Convert to pandas for easier period calculations
        dates = pd.date_range(start='2020-01-01', periods=len(returns), freq='D')
        returns_series = pd.Series(returns, index=dates)
        
        # Monthly returns
        monthly_returns = returns_series.resample('M').apply(lambda x: np.prod(1 + x) - 1)
        period_returns['monthly'] = monthly_returns.tolist()
        
        # Quarterly returns
        quarterly_returns = returns_series.resample('Q').apply(lambda x: np.prod(1 + x) - 1)
        period_returns['quarterly'] = quarterly_returns.tolist()
        
        # Yearly returns
        yearly_returns = returns_series.resample('Y').apply(lambda x: np.prod(1 + x) - 1)
        period_returns['yearly'] = yearly_returns.tolist()
        
        return period_returns
    
    def _generate_risk_analysis(self, returns: np.ndarray) -> Dict[str, float]:
        """Generate risk analysis summary"""
        risk_analysis = {}
        
        # Volatility analysis
        risk_analysis['annualized_volatility'] = np.std(returns) * np.sqrt(252)
        risk_analysis['downside_volatility'] = np.std(returns[returns < 0]) * np.sqrt(252) if np.any(returns < 0) else 0
        
        # Tail risk
        risk_analysis['var_95'] = np.percentile(returns, 5)
        risk_analysis['var_99'] = np.percentile(returns, 1)
        risk_analysis['skewness'] = stats.skew(returns)
        risk_analysis['kurtosis'] = stats.kurtosis(returns)
        
        # Drawdown risk
        dd_metrics = self._calculate_drawdown_metrics(returns)
        risk_analysis['max_drawdown'] = dd_metrics['max_drawdown']
        risk_analysis['avg_drawdown'] = dd_metrics['avg_drawdown']
        
        return risk_analysis
    
    def _generate_performance_summary(self, metrics: PerformanceMetrics,
                                    benchmark_comparisons: List[BenchmarkComparison]) -> List[str]:
        """Generate performance summary points"""
        summary = []
        
        # Return summary
        summary.append(f"Total return: {metrics.total_return:.2%}")
        summary.append(f"Annualized return: {metrics.annualized_return:.2%}")
        summary.append(f"Volatility: {metrics.volatility:.2%}")
        summary.append(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        
        # Risk summary
        summary.append(f"Maximum drawdown: {metrics.maximum_drawdown:.2%}")
        summary.append(f"VaR (95%): {metrics.var_95:.2%}")
        
        # Benchmark comparison
        if benchmark_comparisons:
            primary_benchmark = benchmark_comparisons[0]
            summary.append(f"Alpha vs {primary_benchmark.benchmark_name}: {primary_benchmark.alpha:.2%}")
            summary.append(f"Beta vs {primary_benchmark.benchmark_name}: {primary_benchmark.beta:.2f}")
            summary.append(f"Information ratio: {primary_benchmark.information_ratio:.2f}")
        
        return summary
    
    def _generate_recommendations(self, metrics: PerformanceMetrics,
                                benchmark_comparisons: List[BenchmarkComparison]) -> List[str]:
        """Generate performance-based recommendations"""
        recommendations = []
        
        # Risk recommendations
        if metrics.volatility > 0.25:
            recommendations.append("Consider reducing portfolio volatility through diversification")
        
        if metrics.maximum_drawdown < -0.20:
            recommendations.append("Implement risk management strategies to limit drawdowns")
        
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Focus on improving risk-adjusted returns")
        
        # Benchmark comparison recommendations
        if benchmark_comparisons:
            primary_benchmark = benchmark_comparisons[0]
            
            if primary_benchmark.information_ratio < 0:
                recommendations.append("Portfolio is underperforming benchmark - review strategy")
            
            if primary_benchmark.tracking_error > 0.10:
                recommendations.append("High tracking error - consider closer benchmark alignment")
            
            if primary_benchmark.beta > 1.5:
                recommendations.append("High beta indicates elevated market risk")
        
        # Distribution recommendations
        if metrics.skewness < -1:
            recommendations.append("Negative skew suggests tail risk - consider hedging strategies")
        
        if metrics.kurtosis > 5:
            recommendations.append("High kurtosis indicates fat-tail risk")
        
        if not recommendations:
            recommendations.append("Performance metrics appear reasonable for stated objectives")
        
        return recommendations
    
    def calculate_style_analysis(self, returns: np.ndarray,
                               style_benchmarks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Perform style analysis to determine portfolio exposures
        
        Args:
            returns: Portfolio returns
            style_benchmarks: Dictionary of style benchmark returns
            
        Returns:
            Dictionary of style exposures
        """
        if not style_benchmarks:
            return {}
        
        # Prepare data
        benchmark_matrix = np.column_stack(list(style_benchmarks.values()))
        benchmark_names = list(style_benchmarks.keys())
        
        # Ensure same length
        min_length = min(len(returns), len(benchmark_matrix))
        returns = returns[:min_length]
        benchmark_matrix = benchmark_matrix[:min_length]
        
        # Constrained regression (weights sum to 1, non-negative)
        from scipy.optimize import minimize
        
        def objective(weights):
            predicted_returns = np.dot(benchmark_matrix, weights)
            return np.sum((returns - predicted_returns) ** 2)
        
        def constraint_sum_to_one(weights):
            return np.sum(weights) - 1.0
        
        # Initial guess
        n_benchmarks = len(benchmark_names)
        initial_weights = np.ones(n_benchmarks) / n_benchmarks
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': constraint_sum_to_one}]
        bounds = [(0, 1) for _ in range(n_benchmarks)]
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            style_exposures = dict(zip(benchmark_names, result.x))
            return style_exposures
        else:
            # Fallback to equal weights
            return dict(zip(benchmark_names, [1.0/n_benchmarks] * n_benchmarks))
    
    def calculate_performance_persistence(self, returns: np.ndarray,
                                        periods: List[int] = [252, 504, 756]) -> Dict[str, float]:
        """
        Calculate performance persistence across different periods
        
        Args:
            returns: Portfolio returns
            periods: List of period lengths to analyze
            
        Returns:
            Dictionary of persistence metrics
        """
        persistence_metrics = {}
        
        for period in periods:
            if len(returns) < period * 2:
                continue
            
            # Split into consecutive periods
            n_periods = len(returns) // period
            period_returns = []
            
            for i in range(n_periods):
                start_idx = i * period
                end_idx = (i + 1) * period
                period_return = np.prod(1 + returns[start_idx:end_idx]) - 1
                period_returns.append(period_return)
            
            if len(period_returns) >= 2:
                # Calculate correlation between consecutive periods
                first_half = period_returns[:-1]
                second_half = period_returns[1:]
                
                if len(first_half) > 0 and len(second_half) > 0:
                    correlation = np.corrcoef(first_half, second_half)[0, 1]
                    persistence_metrics[f'{period}_day_persistence'] = correlation
        
        return persistence_metrics