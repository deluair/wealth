"""
Risk Manager

Comprehensive risk management system for portfolio and wealth management.
Includes various risk metrics, stress testing, scenario analysis, and
risk monitoring capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import warnings
from datetime import datetime, timedelta

class RiskType(Enum):
    """Types of financial risks"""
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    CONCENTRATION_RISK = "concentration_risk"
    CURRENCY_RISK = "currency_risk"
    INTEREST_RATE_RISK = "interest_rate_risk"
    INFLATION_RISK = "inflation_risk"
    REGULATORY_RISK = "regulatory_risk"
    TAIL_RISK = "tail_risk"

class RiskProfile(Enum):
    """Investor risk profiles"""
    CONSERVATIVE = "conservative"
    MODERATE_CONSERVATIVE = "moderate_conservative"
    MODERATE = "moderate"
    MODERATE_AGGRESSIVE = "moderate_aggressive"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"

class StressTestType(Enum):
    """Types of stress tests"""
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    PARAMETRIC = "parametric"
    SCENARIO_BASED = "scenario_based"
    EXTREME_VALUE = "extreme_value"

class RiskMeasure(Enum):
    """Risk measurement methods"""
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"

@dataclass
class RiskMetrics:
    """Container for various risk metrics"""
    
    # Value at Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    var_99_9: float = 0.0
    
    # Conditional Value at Risk
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # Expected Shortfall
    expected_shortfall_95: float = 0.0
    expected_shortfall_99: float = 0.0
    
    # Volatility measures
    volatility: float = 0.0
    downside_volatility: float = 0.0
    upside_volatility: float = 0.0
    
    # Drawdown measures
    maximum_drawdown: float = 0.0
    average_drawdown: float = 0.0
    drawdown_duration: int = 0
    
    # Relative risk measures
    beta: float = 1.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    
    # Tail risk measures
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0
    
    # Concentration measures
    concentration_index: float = 0.0
    effective_number_assets: float = 0.0
    
    # Liquidity measures
    liquidity_score: float = 1.0
    bid_ask_spread: float = 0.0
    
    # Time-varying measures
    volatility_of_volatility: float = 0.0
    correlation_breakdown: float = 0.0

@dataclass
class RiskScenario:
    """Risk scenario definition"""
    name: str
    description: str
    probability: float
    
    # Market shocks
    equity_shock: float = 0.0
    bond_shock: float = 0.0
    currency_shock: float = 0.0
    commodity_shock: float = 0.0
    
    # Correlation changes
    correlation_increase: float = 0.0
    volatility_increase: float = 0.0
    
    # Liquidity conditions
    liquidity_reduction: float = 0.0
    
    # Duration
    duration_days: int = 1

@dataclass
class StressTest:
    """Stress test configuration and results"""
    name: str
    test_type: StressTestType
    scenarios: List[RiskScenario]
    
    # Results
    portfolio_impact: Dict[str, float] = field(default_factory=dict)
    asset_impacts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    risk_metrics_impact: Dict[str, float] = field(default_factory=dict)
    
    # Summary statistics
    worst_case_loss: float = 0.0
    expected_loss: float = 0.0
    probability_of_loss: float = 0.0
    
    # Recovery analysis
    recovery_time: int = 0
    recovery_probability: float = 0.0

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    assessment_date: str
    portfolio_name: str
    
    # Risk metrics
    risk_metrics: RiskMetrics
    
    # Risk profile
    risk_profile: RiskProfile
    risk_tolerance: float  # 0-1 scale
    risk_capacity: float   # 0-1 scale
    
    # Stress test results
    stress_tests: List[StressTest] = field(default_factory=list)
    
    # Risk decomposition
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    risk_by_asset_class: Dict[str, float] = field(default_factory=dict)
    risk_by_geography: Dict[str, float] = field(default_factory=dict)
    
    # Risk limits and alerts
    risk_limits: Dict[str, float] = field(default_factory=dict)
    limit_breaches: List[str] = field(default_factory=list)
    
    # Recommendations
    risk_recommendations: List[str] = field(default_factory=list)
    
    # Overall risk score
    overall_risk_score: float = 0.0  # 0-100 scale

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99, 0.999]):
        self.confidence_levels = confidence_levels
        self.risk_scenarios = {}
        self.stress_tests = {}
        self.risk_limits = {}
        
        # Historical data for risk calculations
        self.returns_data = None
        self.benchmark_returns = None
        
        # Risk model parameters
        self.risk_model_params = {
            'lookback_window': 252,  # 1 year
            'decay_factor': 0.94,    # EWMA decay
            'monte_carlo_sims': 10000,
            'bootstrap_samples': 1000
        }
    
    def set_returns_data(self, returns: np.ndarray, 
                        benchmark_returns: Optional[np.ndarray] = None) -> None:
        """Set historical returns data for risk calculations"""
        self.returns_data = returns
        self.benchmark_returns = benchmark_returns
    
    def calculate_risk_metrics(self, returns: Optional[np.ndarray] = None,
                              weights: Optional[np.ndarray] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Return series (uses stored data if None)
            weights: Portfolio weights for multi-asset portfolios
            
        Returns:
            RiskMetrics object with calculated metrics
        """
        if returns is None:
            if self.returns_data is None:
                raise ValueError("No returns data available")
            returns = self.returns_data
        
        # If weights provided, calculate portfolio returns
        if weights is not None and returns.ndim > 1:
            returns = np.dot(returns, weights)
        
        metrics = RiskMetrics()
        
        # Basic statistics
        metrics.volatility = np.std(returns) * np.sqrt(252)  # Annualized
        metrics.skewness = stats.skew(returns)
        metrics.kurtosis = stats.kurtosis(returns)
        
        # Value at Risk calculations
        for i, confidence in enumerate(self.confidence_levels):
            var_value = self._calculate_var(returns, confidence)
            cvar_value = self._calculate_cvar(returns, confidence)
            
            if confidence == 0.95:
                metrics.var_95 = var_value
                metrics.cvar_95 = cvar_value
                metrics.expected_shortfall_95 = cvar_value
            elif confidence == 0.99:
                metrics.var_99 = var_value
                metrics.cvar_99 = cvar_value
                metrics.expected_shortfall_99 = cvar_value
            elif confidence == 0.999:
                metrics.var_99_9 = var_value
        
        # Drawdown calculations
        drawdown_metrics = self._calculate_drawdown_metrics(returns)
        metrics.maximum_drawdown = drawdown_metrics['max_drawdown']
        metrics.average_drawdown = drawdown_metrics['avg_drawdown']
        metrics.drawdown_duration = drawdown_metrics['max_duration']
        
        # Volatility decomposition
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0:
            metrics.upside_volatility = np.std(positive_returns) * np.sqrt(252)
        if len(negative_returns) > 0:
            metrics.downside_volatility = np.std(negative_returns) * np.sqrt(252)
        
        # Relative risk measures (if benchmark available)
        if self.benchmark_returns is not None:
            benchmark_returns = self.benchmark_returns
            if len(benchmark_returns) == len(returns):
                metrics.beta = self._calculate_beta(returns, benchmark_returns)
                metrics.tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
                metrics.information_ratio = self._calculate_information_ratio(returns, benchmark_returns)
        
        # Tail risk measures
        metrics.tail_ratio = self._calculate_tail_ratio(returns)
        
        # Time-varying risk measures
        metrics.volatility_of_volatility = self._calculate_vol_of_vol(returns)
        
        return metrics
    
    def _calculate_var(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var_threshold = self._calculate_var(returns, confidence)
        tail_returns = returns[returns <= var_threshold]
        return np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
    
    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown-related metrics"""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        
        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0
        
        # Maximum drawdown duration
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_duration': max_duration,
            'drawdowns': drawdowns
        }
    
    def _calculate_beta(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate beta relative to benchmark"""
        if len(returns) != len(benchmark_returns):
            return 1.0
        
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
    
    def _calculate_tracking_error(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate tracking error"""
        if len(returns) != len(benchmark_returns):
            return 0.0
        
        excess_returns = returns - benchmark_returns
        return np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_information_ratio(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio"""
        if len(returns) != len(benchmark_returns):
            return 0.0
        
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        
        if tracking_error > 0:
            return (np.mean(excess_returns) * 252) / tracking_error
        return 0.0
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        return abs(p95 / p5) if p5 != 0 else 0.0
    
    def _calculate_vol_of_vol(self, returns: np.ndarray, window: int = 30) -> float:
        """Calculate volatility of volatility"""
        if len(returns) < window * 2:
            return 0.0
        
        # Rolling volatility
        rolling_vols = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            rolling_vols.append(np.std(window_returns))
        
        return np.std(rolling_vols) if len(rolling_vols) > 0 else 0.0
    
    def create_stress_scenario(self, name: str, description: str,
                              **shock_parameters) -> RiskScenario:
        """Create a stress test scenario"""
        scenario = RiskScenario(
            name=name,
            description=description,
            probability=shock_parameters.get('probability', 0.05),
            equity_shock=shock_parameters.get('equity_shock', 0.0),
            bond_shock=shock_parameters.get('bond_shock', 0.0),
            currency_shock=shock_parameters.get('currency_shock', 0.0),
            commodity_shock=shock_parameters.get('commodity_shock', 0.0),
            correlation_increase=shock_parameters.get('correlation_increase', 0.0),
            volatility_increase=shock_parameters.get('volatility_increase', 0.0),
            liquidity_reduction=shock_parameters.get('liquidity_reduction', 0.0),
            duration_days=shock_parameters.get('duration_days', 1)
        )
        
        self.risk_scenarios[name] = scenario
        return scenario
    
    def run_stress_test(self, portfolio_weights: np.ndarray,
                       asset_returns: np.ndarray,
                       scenarios: List[RiskScenario],
                       test_name: str = "Custom Stress Test") -> StressTest:
        """
        Run stress test on portfolio
        
        Args:
            portfolio_weights: Portfolio weights
            asset_returns: Historical asset returns matrix
            scenarios: List of stress scenarios
            test_name: Name of the stress test
            
        Returns:
            StressTest object with results
        """
        stress_test = StressTest(
            name=test_name,
            test_type=StressTestType.SCENARIO_BASED,
            scenarios=scenarios
        )
        
        portfolio_impacts = []
        
        for scenario in scenarios:
            # Apply shocks to returns
            shocked_returns = self._apply_scenario_shocks(asset_returns, scenario)
            
            # Calculate portfolio impact
            portfolio_returns = np.dot(shocked_returns, portfolio_weights)
            scenario_impact = np.sum(portfolio_returns)  # Total impact over scenario duration
            
            portfolio_impacts.append(scenario_impact)
            stress_test.portfolio_impact[scenario.name] = scenario_impact
            
            # Calculate individual asset impacts
            asset_impacts = {}
            for i, weight in enumerate(portfolio_weights):
                if weight > 0:
                    asset_impact = np.sum(shocked_returns[:, i]) * weight
                    asset_impacts[f"Asset_{i}"] = asset_impact
            
            stress_test.asset_impacts[scenario.name] = asset_impacts
        
        # Calculate summary statistics
        stress_test.worst_case_loss = min(portfolio_impacts)
        stress_test.expected_loss = np.mean(portfolio_impacts)
        stress_test.probability_of_loss = sum(1 for impact in portfolio_impacts if impact < 0) / len(portfolio_impacts)
        
        self.stress_tests[test_name] = stress_test
        return stress_test
    
    def _apply_scenario_shocks(self, returns: np.ndarray, scenario: RiskScenario) -> np.ndarray:
        """Apply scenario shocks to return matrix"""
        shocked_returns = returns.copy()
        
        # Apply asset class shocks (simplified - assumes first columns are equity, bonds, etc.)
        n_assets = returns.shape[1]
        
        # Equity shock (assume first 60% of assets are equity)
        equity_end = int(n_assets * 0.6)
        if scenario.equity_shock != 0:
            shocked_returns[:scenario.duration_days, :equity_end] *= (1 + scenario.equity_shock)
        
        # Bond shock (assume next 30% are bonds)
        bond_start = equity_end
        bond_end = int(n_assets * 0.9)
        if scenario.bond_shock != 0:
            shocked_returns[:scenario.duration_days, bond_start:bond_end] *= (1 + scenario.bond_shock)
        
        # Increase volatility if specified
        if scenario.volatility_increase > 0:
            volatility_multiplier = 1 + scenario.volatility_increase
            shocked_returns[:scenario.duration_days] *= volatility_multiplier
        
        # Increase correlations (simplified implementation)
        if scenario.correlation_increase > 0:
            # Add common factor to increase correlations
            common_factor = np.random.normal(0, scenario.correlation_increase, scenario.duration_days)
            for i in range(n_assets):
                shocked_returns[:scenario.duration_days, i] += common_factor
        
        return shocked_returns
    
    def monte_carlo_risk_simulation(self, portfolio_weights: np.ndarray,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   time_horizon: int = 252,
                                   n_simulations: int = 10000) -> Dict:
        """
        Monte Carlo simulation for risk assessment
        
        Args:
            portfolio_weights: Portfolio weights
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            time_horizon: Time horizon in days
            n_simulations: Number of simulation paths
            
        Returns:
            Dictionary with simulation results
        """
        # Portfolio expected return and volatility
        portfolio_return = np.dot(portfolio_weights, expected_returns) / 252  # Daily
        portfolio_variance = np.dot(portfolio_weights.T, np.dot(covariance_matrix, portfolio_weights)) / 252
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Monte Carlo simulation
        simulation_results = []
        final_values = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(portfolio_return, portfolio_vol, time_horizon)
            
            # Calculate cumulative path
            cumulative_path = np.cumprod(1 + random_returns)
            simulation_results.append(cumulative_path)
            final_values.append(cumulative_path[-1])
            
            # Calculate maximum drawdown for this path
            running_max = np.maximum.accumulate(cumulative_path)
            drawdowns = (cumulative_path - running_max) / running_max
            max_drawdowns.append(np.min(drawdowns))
        
        simulation_results = np.array(simulation_results)
        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate risk metrics from simulation
        var_95 = np.percentile(final_values - 1, 5)  # 5th percentile of returns
        var_99 = np.percentile(final_values - 1, 1)  # 1st percentile of returns
        
        cvar_95 = np.mean((final_values - 1)[(final_values - 1) <= var_95])
        cvar_99 = np.mean((final_values - 1)[(final_values - 1) <= var_99])
        
        # Probability of loss
        prob_loss = np.mean(final_values < 1.0)
        
        # Expected maximum drawdown
        expected_max_drawdown = np.mean(max_drawdowns)
        
        return {
            'simulation_paths': simulation_results,
            'final_values': final_values,
            'max_drawdowns': max_drawdowns,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'probability_of_loss': prob_loss,
            'expected_max_drawdown': expected_max_drawdown,
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values)
        }
    
    def assess_portfolio_risk(self, portfolio_weights: np.ndarray,
                             asset_returns: np.ndarray,
                             portfolio_name: str = "Portfolio",
                             risk_profile: RiskProfile = RiskProfile.MODERATE) -> RiskAssessment:
        """
        Comprehensive portfolio risk assessment
        
        Args:
            portfolio_weights: Portfolio weights
            asset_returns: Historical asset returns
            portfolio_name: Name of the portfolio
            risk_profile: Investor risk profile
            
        Returns:
            RiskAssessment object
        """
        # Calculate portfolio returns
        portfolio_returns = np.dot(asset_returns, portfolio_weights)
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(portfolio_returns)
        
        # Create risk assessment
        assessment = RiskAssessment(
            assessment_date=datetime.now().strftime("%Y-%m-%d"),
            portfolio_name=portfolio_name,
            risk_metrics=risk_metrics,
            risk_profile=risk_profile,
            risk_tolerance=self._get_risk_tolerance(risk_profile),
            risk_capacity=0.7  # Default, should be determined based on investor circumstances
        )
        
        # Risk decomposition
        assessment.risk_contributions = self._calculate_risk_contributions(
            portfolio_weights, asset_returns
        )
        
        # Set risk limits based on profile
        assessment.risk_limits = self._get_risk_limits(risk_profile)
        
        # Check for limit breaches
        assessment.limit_breaches = self._check_limit_breaches(risk_metrics, assessment.risk_limits)
        
        # Generate recommendations
        assessment.risk_recommendations = self._generate_risk_recommendations(
            risk_metrics, risk_profile, assessment.limit_breaches
        )
        
        # Calculate overall risk score
        assessment.overall_risk_score = self._calculate_overall_risk_score(risk_metrics, risk_profile)
        
        return assessment
    
    def _calculate_risk_contributions(self, weights: np.ndarray, 
                                    returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk contributions of each asset"""
        portfolio_returns = np.dot(returns, weights)
        portfolio_var = np.var(portfolio_returns)
        
        risk_contributions = {}
        
        for i, weight in enumerate(weights):
            if weight > 0:
                # Marginal contribution to risk
                asset_returns = returns[:, i]
                covariance = np.cov(portfolio_returns, asset_returns)[0, 1]
                marginal_contrib = covariance / portfolio_var if portfolio_var > 0 else 0
                
                # Risk contribution
                risk_contrib = weight * marginal_contrib
                risk_contributions[f"Asset_{i}"] = risk_contrib
        
        return risk_contributions
    
    def _get_risk_tolerance(self, risk_profile: RiskProfile) -> float:
        """Get risk tolerance based on risk profile"""
        tolerance_map = {
            RiskProfile.CONSERVATIVE: 0.2,
            RiskProfile.MODERATE_CONSERVATIVE: 0.35,
            RiskProfile.MODERATE: 0.5,
            RiskProfile.MODERATE_AGGRESSIVE: 0.65,
            RiskProfile.AGGRESSIVE: 0.8,
            RiskProfile.SPECULATIVE: 0.95
        }
        return tolerance_map.get(risk_profile, 0.5)
    
    def _get_risk_limits(self, risk_profile: RiskProfile) -> Dict[str, float]:
        """Get risk limits based on risk profile"""
        if risk_profile == RiskProfile.CONSERVATIVE:
            return {
                'max_volatility': 0.08,
                'max_var_95': -0.02,
                'max_drawdown': -0.05,
                'max_concentration': 0.1
            }
        elif risk_profile == RiskProfile.MODERATE_CONSERVATIVE:
            return {
                'max_volatility': 0.12,
                'max_var_95': -0.03,
                'max_drawdown': -0.08,
                'max_concentration': 0.15
            }
        elif risk_profile == RiskProfile.MODERATE:
            return {
                'max_volatility': 0.16,
                'max_var_95': -0.04,
                'max_drawdown': -0.12,
                'max_concentration': 0.2
            }
        elif risk_profile == RiskProfile.MODERATE_AGGRESSIVE:
            return {
                'max_volatility': 0.20,
                'max_var_95': -0.05,
                'max_drawdown': -0.15,
                'max_concentration': 0.25
            }
        elif risk_profile == RiskProfile.AGGRESSIVE:
            return {
                'max_volatility': 0.25,
                'max_var_95': -0.07,
                'max_drawdown': -0.20,
                'max_concentration': 0.3
            }
        else:  # SPECULATIVE
            return {
                'max_volatility': 0.35,
                'max_var_95': -0.10,
                'max_drawdown': -0.30,
                'max_concentration': 0.5
            }
    
    def _check_limit_breaches(self, risk_metrics: RiskMetrics, 
                             risk_limits: Dict[str, float]) -> List[str]:
        """Check for risk limit breaches"""
        breaches = []
        
        if risk_metrics.volatility > risk_limits.get('max_volatility', 1.0):
            breaches.append(f"Volatility breach: {risk_metrics.volatility:.3f} > {risk_limits['max_volatility']:.3f}")
        
        if risk_metrics.var_95 < risk_limits.get('max_var_95', -1.0):
            breaches.append(f"VaR 95% breach: {risk_metrics.var_95:.3f} < {risk_limits['max_var_95']:.3f}")
        
        if risk_metrics.maximum_drawdown < risk_limits.get('max_drawdown', -1.0):
            breaches.append(f"Max drawdown breach: {risk_metrics.maximum_drawdown:.3f} < {risk_limits['max_drawdown']:.3f}")
        
        if risk_metrics.concentration_index > risk_limits.get('max_concentration', 1.0):
            breaches.append(f"Concentration breach: {risk_metrics.concentration_index:.3f} > {risk_limits['max_concentration']:.3f}")
        
        return breaches
    
    def _generate_risk_recommendations(self, risk_metrics: RiskMetrics,
                                     risk_profile: RiskProfile,
                                     limit_breaches: List[str]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if limit_breaches:
            recommendations.append("Portfolio exceeds risk limits - consider rebalancing")
        
        if risk_metrics.volatility > 0.25:
            recommendations.append("High portfolio volatility - consider adding defensive assets")
        
        if risk_metrics.maximum_drawdown < -0.20:
            recommendations.append("Large historical drawdowns - implement stop-loss or hedging strategies")
        
        if risk_metrics.concentration_index > 0.3:
            recommendations.append("High concentration risk - diversify across more assets")
        
        if risk_metrics.skewness < -1.0:
            recommendations.append("Negative skew detected - consider tail risk hedging")
        
        if risk_metrics.kurtosis > 5.0:
            recommendations.append("High kurtosis - fat tail risk present, consider risk management overlays")
        
        if not recommendations:
            recommendations.append("Risk profile appears appropriate for stated objectives")
        
        return recommendations
    
    def _calculate_overall_risk_score(self, risk_metrics: RiskMetrics,
                                    risk_profile: RiskProfile) -> float:
        """Calculate overall risk score (0-100)"""
        # Normalize key risk metrics
        vol_score = min(100, risk_metrics.volatility * 400)  # 25% vol = 100 points
        var_score = min(100, abs(risk_metrics.var_95) * 1000)  # 10% VaR = 100 points
        drawdown_score = min(100, abs(risk_metrics.maximum_drawdown) * 400)  # 25% DD = 100 points
        concentration_score = risk_metrics.concentration_index * 100
        
        # Weight the scores
        overall_score = (
            vol_score * 0.3 +
            var_score * 0.3 +
            drawdown_score * 0.25 +
            concentration_score * 0.15
        )
        
        return min(100, max(0, overall_score))
    
    def create_predefined_scenarios(self) -> List[RiskScenario]:
        """Create predefined stress test scenarios"""
        scenarios = []
        
        # 2008 Financial Crisis
        scenarios.append(self.create_stress_scenario(
            "Financial Crisis 2008",
            "Severe market downturn similar to 2008 financial crisis",
            probability=0.02,
            equity_shock=-0.40,
            bond_shock=0.05,
            correlation_increase=0.3,
            volatility_increase=0.5,
            liquidity_reduction=0.3,
            duration_days=30
        ))
        
        # COVID-19 Market Crash
        scenarios.append(self.create_stress_scenario(
            "Pandemic Market Crash",
            "Rapid market decline due to pandemic-like event",
            probability=0.05,
            equity_shock=-0.35,
            bond_shock=-0.05,
            correlation_increase=0.4,
            volatility_increase=0.8,
            liquidity_reduction=0.2,
            duration_days=14
        ))
        
        # Interest Rate Shock
        scenarios.append(self.create_stress_scenario(
            "Interest Rate Shock",
            "Rapid increase in interest rates",
            probability=0.10,
            equity_shock=-0.15,
            bond_shock=-0.20,
            correlation_increase=0.1,
            volatility_increase=0.2,
            duration_days=7
        ))
        
        # Inflation Surge
        scenarios.append(self.create_stress_scenario(
            "Inflation Surge",
            "Unexpected surge in inflation",
            probability=0.15,
            equity_shock=-0.10,
            bond_shock=-0.15,
            commodity_shock=0.25,
            volatility_increase=0.3,
            duration_days=21
        ))
        
        # Geopolitical Crisis
        scenarios.append(self.create_stress_scenario(
            "Geopolitical Crisis",
            "Major geopolitical event causing market disruption",
            probability=0.08,
            equity_shock=-0.25,
            bond_shock=0.02,
            currency_shock=0.15,
            commodity_shock=0.20,
            volatility_increase=0.4,
            liquidity_reduction=0.15,
            duration_days=10
        ))
        
        return scenarios