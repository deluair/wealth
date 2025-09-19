"""
Portfolio Optimizer

Advanced portfolio optimization using modern portfolio theory, machine learning,
and behavioral finance insights. Supports multiple optimization objectives,
constraints, and risk models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import optimize
from scipy.stats import norm
import cvxpy as cp
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import warnings

class AssetClass(Enum):
    """Asset class categories"""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"
    ALTERNATIVES = "alternatives"
    CASH = "cash"
    CRYPTO = "crypto"
    PRIVATE_EQUITY = "private_equity"
    HEDGE_FUNDS = "hedge_funds"

class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    BLACK_LITTERMAN = "black_litterman"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ESG_OPTIMIZED = "esg_optimized"

class RiskModel(Enum):
    """Risk model types"""
    SAMPLE_COVARIANCE = "sample_covariance"
    SHRINKAGE = "shrinkage"
    FACTOR_MODEL = "factor_model"
    GARCH = "garch"
    EWMA = "ewma"
    ROBUST = "robust"

class RebalancingFrequency(Enum):
    """Portfolio rebalancing frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"
    THRESHOLD_BASED = "threshold_based"

@dataclass
class Asset:
    """Individual asset representation"""
    symbol: str
    name: str
    asset_class: AssetClass
    
    # Historical data
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    prices: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Asset characteristics
    expected_return: float = 0.0
    volatility: float = 0.0
    beta: float = 1.0
    
    # ESG and other factors
    esg_score: float = 0.0
    liquidity_score: float = 1.0
    expense_ratio: float = 0.0
    
    # Constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    
    def calculate_statistics(self) -> None:
        """Calculate basic statistics from returns"""
        if len(self.returns) > 0:
            self.expected_return = np.mean(self.returns)
            self.volatility = np.std(self.returns)

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    
    # Weight constraints
    min_weights: Dict[str, float] = field(default_factory=dict)
    max_weights: Dict[str, float] = field(default_factory=dict)
    
    # Asset class constraints
    asset_class_min: Dict[AssetClass, float] = field(default_factory=dict)
    asset_class_max: Dict[AssetClass, float] = field(default_factory=dict)
    
    # Risk constraints
    max_portfolio_volatility: Optional[float] = None
    max_tracking_error: Optional[float] = None
    max_var: Optional[float] = None  # Value at Risk
    max_cvar: Optional[float] = None  # Conditional Value at Risk
    
    # Concentration constraints
    max_single_asset_weight: float = 0.1
    max_sector_concentration: float = 0.3
    
    # Turnover constraints
    max_turnover: Optional[float] = None
    transaction_costs: Dict[str, float] = field(default_factory=dict)
    
    # ESG constraints
    min_esg_score: Optional[float] = None
    
    # Other constraints
    long_only: bool = True
    integer_shares: bool = False

@dataclass
class Portfolio:
    """Portfolio representation"""
    assets: List[Asset]
    weights: np.ndarray
    
    # Portfolio metadata
    name: str = "Portfolio"
    benchmark: Optional[str] = None
    creation_date: Optional[str] = None
    
    # Performance metrics
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    
    # Other metrics
    diversification_ratio: float = 0.0
    concentration_index: float = 0.0
    
    def __post_init__(self):
        """Validate portfolio after initialization"""
        if len(self.assets) != len(self.weights):
            raise ValueError("Number of assets must match number of weights")
        
        if not np.isclose(np.sum(self.weights), 1.0, atol=1e-6):
            warnings.warn("Portfolio weights do not sum to 1.0")
    
    def get_asset_symbols(self) -> List[str]:
        """Get list of asset symbols"""
        return [asset.symbol for asset in self.assets]
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Get weights as dictionary"""
        return dict(zip(self.get_asset_symbols(), self.weights))

class PortfolioOptimizer:
    """Advanced portfolio optimizer with multiple objectives and constraints"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.assets = []
        self.returns_matrix = None
        self.covariance_matrix = None
        self.expected_returns = None
        
        # Optimization results
        self.optimization_results = {}
        
    def add_asset(self, asset: Asset) -> None:
        """Add asset to the optimizer"""
        self.assets.append(asset)
        self._update_matrices()
    
    def add_assets(self, assets: List[Asset]) -> None:
        """Add multiple assets to the optimizer"""
        self.assets.extend(assets)
        self._update_matrices()
    
    def _update_matrices(self) -> None:
        """Update returns matrix and covariance matrix"""
        if not self.assets:
            return
        
        # Create returns matrix
        returns_list = []
        expected_returns_list = []
        
        for asset in self.assets:
            if len(asset.returns) > 0:
                returns_list.append(asset.returns)
                expected_returns_list.append(asset.expected_return)
            else:
                # Generate synthetic returns if not provided
                synthetic_returns = np.random.normal(0.08, 0.15, 252)  # Daily returns for 1 year
                returns_list.append(synthetic_returns)
                expected_returns_list.append(np.mean(synthetic_returns))
        
        if returns_list:
            # Ensure all return series have the same length
            min_length = min(len(returns) for returns in returns_list)
            returns_list = [returns[-min_length:] for returns in returns_list]
            
            self.returns_matrix = np.column_stack(returns_list)
            self.expected_returns = np.array(expected_returns_list)
            
            # Calculate covariance matrix
            self.covariance_matrix = np.cov(self.returns_matrix.T)
    
    def estimate_covariance_matrix(self, method: RiskModel = RiskModel.SHRINKAGE) -> np.ndarray:
        """Estimate covariance matrix using different methods"""
        if self.returns_matrix is None:
            raise ValueError("No returns data available")
        
        if method == RiskModel.SAMPLE_COVARIANCE:
            return np.cov(self.returns_matrix.T)
        
        elif method == RiskModel.SHRINKAGE:
            lw = LedoitWolf()
            cov_matrix, _ = lw.fit(self.returns_matrix).covariance_, lw.shrinkage_
            return cov_matrix
        
        elif method == RiskModel.EWMA:
            # Exponentially Weighted Moving Average
            lambda_param = 0.94
            weights = np.array([(1 - lambda_param) * lambda_param**i 
                               for i in range(len(self.returns_matrix))])
            weights = weights[::-1] / np.sum(weights)
            
            weighted_returns = self.returns_matrix * weights.reshape(-1, 1)
            return np.cov(weighted_returns.T)
        
        elif method == RiskModel.ROBUST:
            # Robust covariance estimation
            from sklearn.covariance import MinCovDet
            robust_cov = MinCovDet().fit(self.returns_matrix)
            return robust_cov.covariance_
        
        else:
            return self.covariance_matrix
    
    def optimize_portfolio(self, 
                          objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                          constraints: Optional[OptimizationConstraints] = None,
                          risk_model: RiskModel = RiskModel.SHRINKAGE) -> Portfolio:
        """
        Optimize portfolio based on objective and constraints
        
        Args:
            objective: Optimization objective
            constraints: Portfolio constraints
            risk_model: Risk model for covariance estimation
            
        Returns:
            Optimized Portfolio object
        """
        if not self.assets:
            raise ValueError("No assets added to optimizer")
        
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Update covariance matrix with specified method
        self.covariance_matrix = self.estimate_covariance_matrix(risk_model)
        
        # Perform optimization based on objective
        if objective == OptimizationObjective.MAX_SHARPE:
            weights = self._optimize_max_sharpe(constraints)
        elif objective == OptimizationObjective.MIN_VARIANCE:
            weights = self._optimize_min_variance(constraints)
        elif objective == OptimizationObjective.MAX_RETURN:
            weights = self._optimize_max_return(constraints)
        elif objective == OptimizationObjective.RISK_PARITY:
            weights = self._optimize_risk_parity(constraints)
        elif objective == OptimizationObjective.MAX_DIVERSIFICATION:
            weights = self._optimize_max_diversification(constraints)
        elif objective == OptimizationObjective.BLACK_LITTERMAN:
            weights = self._optimize_black_litterman(constraints)
        else:
            raise ValueError(f"Optimization objective {objective} not implemented")
        
        # Create portfolio
        portfolio = Portfolio(
            assets=self.assets.copy(),
            weights=weights,
            name=f"{objective.value}_portfolio"
        )
        
        # Calculate portfolio metrics
        self._calculate_portfolio_metrics(portfolio)
        
        # Store optimization results
        self.optimization_results[objective.value] = {
            'portfolio': portfolio,
            'objective': objective,
            'constraints': constraints,
            'risk_model': risk_model
        }
        
        return portfolio
    
    def _optimize_max_sharpe(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize for maximum Sharpe ratio"""
        n_assets = len(self.assets)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Portfolio return and risk
        portfolio_return = self.expected_returns.T @ weights
        portfolio_risk = cp.quad_form(weights, self.covariance_matrix)
        
        # Objective: maximize Sharpe ratio (minimize negative Sharpe)
        # We use the approximation: max(return - rf) / sqrt(risk)
        objective = cp.Maximize(portfolio_return - self.risk_free_rate)
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,  # Weights sum to 1
            portfolio_risk <= 1,   # Risk constraint (will be adjusted)
        ]
        
        # Add weight constraints
        if constraints.long_only:
            constraints_list.append(weights >= 0)
        
        # Individual weight constraints
        for i, asset in enumerate(self.assets):
            min_w = constraints.min_weights.get(asset.symbol, asset.min_weight)
            max_w = constraints.max_weights.get(asset.symbol, asset.max_weight)
            constraints_list.extend([weights[i] >= min_w, weights[i] <= max_w])
        
        # Maximum single asset weight
        if constraints.max_single_asset_weight < 1.0:
            constraints_list.append(weights <= constraints.max_single_asset_weight)
        
        # Portfolio volatility constraint
        if constraints.max_portfolio_volatility is not None:
            constraints_list.append(
                cp.sqrt(portfolio_risk) <= constraints.max_portfolio_volatility
            )
        
        # Solve optimization
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status == cp.OPTIMAL:
                return weights.value
            else:
                # Fallback to scipy optimization
                return self._scipy_max_sharpe_optimization(constraints)
                
        except Exception as e:
            print(f"CVXPY optimization failed: {e}")
            return self._scipy_max_sharpe_optimization(constraints)
    
    def _scipy_max_sharpe_optimization(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Fallback Sharpe ratio optimization using scipy"""
        n_assets = len(self.assets)
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_std
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Bounds
        bounds = []
        for i, asset in enumerate(self.assets):
            min_w = constraints.min_weights.get(asset.symbol, asset.min_weight)
            max_w = constraints.max_weights.get(asset.symbol, asset.max_weight)
            if constraints.long_only:
                min_w = max(0, min_w)
            bounds.append((min_w, max_w))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            negative_sharpe, x0, method='SLSQP',
            bounds=bounds, constraints=cons
        )
        
        if result.success:
            return result.x
        else:
            print("Optimization failed, returning equal weights")
            return np.array([1/n_assets] * n_assets)
    
    def _optimize_min_variance(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize for minimum variance"""
        n_assets = len(self.assets)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(weights, self.covariance_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints_list = [cp.sum(weights) == 1]
        
        if constraints.long_only:
            constraints_list.append(weights >= 0)
        
        # Individual weight constraints
        for i, asset in enumerate(self.assets):
            min_w = constraints.min_weights.get(asset.symbol, asset.min_weight)
            max_w = constraints.max_weights.get(asset.symbol, asset.max_weight)
            constraints_list.extend([weights[i] >= min_w, weights[i] <= max_w])
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            return weights.value
        else:
            return np.array([1/n_assets] * n_assets)
    
    def _optimize_max_return(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize for maximum return"""
        n_assets = len(self.assets)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Objective: maximize portfolio return
        portfolio_return = self.expected_returns.T @ weights
        objective = cp.Maximize(portfolio_return)
        
        # Constraints
        constraints_list = [cp.sum(weights) == 1]
        
        if constraints.long_only:
            constraints_list.append(weights >= 0)
        
        # Risk constraint
        if constraints.max_portfolio_volatility is not None:
            portfolio_risk = cp.quad_form(weights, self.covariance_matrix)
            constraints_list.append(
                cp.sqrt(portfolio_risk) <= constraints.max_portfolio_volatility
            )
        
        # Individual weight constraints
        for i, asset in enumerate(self.assets):
            min_w = constraints.min_weights.get(asset.symbol, asset.min_weight)
            max_w = constraints.max_weights.get(asset.symbol, asset.max_weight)
            constraints_list.extend([weights[i] >= min_w, weights[i] <= max_w])
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            return weights.value
        else:
            return np.array([1/n_assets] * n_assets)
    
    def _optimize_risk_parity(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize for risk parity (equal risk contribution)"""
        n_assets = len(self.assets)
        
        def risk_parity_objective(weights):
            """Risk parity objective function"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            
            # Risk contributions
            marginal_contrib = np.dot(self.covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target equal risk contributions
            target_contrib = portfolio_vol / n_assets
            
            # Sum of squared deviations from target
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = []
        for i, asset in enumerate(self.assets):
            min_w = constraints.min_weights.get(asset.symbol, max(0.001, asset.min_weight))
            max_w = constraints.max_weights.get(asset.symbol, asset.max_weight)
            bounds.append((min_w, max_w))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            risk_parity_objective, x0, method='SLSQP',
            bounds=bounds, constraints=cons
        )
        
        if result.success:
            return result.x
        else:
            return np.array([1/n_assets] * n_assets)
    
    def _optimize_max_diversification(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize for maximum diversification ratio"""
        n_assets = len(self.assets)
        
        def diversification_ratio(weights):
            """Calculate diversification ratio"""
            # Weighted average volatility
            individual_vols = np.sqrt(np.diag(self.covariance_matrix))
            weighted_avg_vol = np.dot(weights, individual_vols)
            
            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            
            # Diversification ratio (to maximize, we minimize negative)
            return -weighted_avg_vol / portfolio_vol
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = []
        for i, asset in enumerate(self.assets):
            min_w = constraints.min_weights.get(asset.symbol, asset.min_weight)
            max_w = constraints.max_weights.get(asset.symbol, asset.max_weight)
            if constraints.long_only:
                min_w = max(0, min_w)
            bounds.append((min_w, max_w))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            diversification_ratio, x0, method='SLSQP',
            bounds=bounds, constraints=cons
        )
        
        if result.success:
            return result.x
        else:
            return np.array([1/n_assets] * n_assets)
    
    def _optimize_black_litterman(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Black-Litterman optimization"""
        # Simplified Black-Litterman implementation
        # In practice, this would require market cap weights and investor views
        
        n_assets = len(self.assets)
        
        # Market equilibrium returns (simplified)
        risk_aversion = 3.0
        market_weights = np.array([1/n_assets] * n_assets)  # Equal weights as proxy
        pi = risk_aversion * np.dot(self.covariance_matrix, market_weights)
        
        # Without specific views, this reduces to reverse optimization
        # Return market cap weighted portfolio
        return market_weights
    
    def _calculate_portfolio_metrics(self, portfolio: Portfolio) -> None:
        """Calculate portfolio performance metrics"""
        weights = portfolio.weights
        
        # Expected return and volatility
        portfolio.expected_return = np.dot(weights, self.expected_returns)
        portfolio.volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        
        # Sharpe ratio
        if portfolio.volatility > 0:
            portfolio.sharpe_ratio = (portfolio.expected_return - self.risk_free_rate) / portfolio.volatility
        
        # Risk metrics (simplified calculations)
        if self.returns_matrix is not None:
            portfolio_returns = np.dot(self.returns_matrix, weights)
            
            # VaR and CVaR (95% confidence)
            portfolio.var_95 = np.percentile(portfolio_returns, 5)
            portfolio.cvar_95 = np.mean(portfolio_returns[portfolio_returns <= portfolio.var_95])
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            portfolio.max_drawdown = np.min(drawdowns)
        
        # Diversification ratio
        individual_vols = np.sqrt(np.diag(self.covariance_matrix))
        weighted_avg_vol = np.dot(weights, individual_vols)
        if portfolio.volatility > 0:
            portfolio.diversification_ratio = weighted_avg_vol / portfolio.volatility
        
        # Concentration index (Herfindahl index)
        portfolio.concentration_index = np.sum(weights ** 2)
    
    def generate_efficient_frontier(self, 
                                   n_portfolios: int = 100,
                                   constraints: Optional[OptimizationConstraints] = None) -> Dict:
        """
        Generate efficient frontier
        
        Args:
            n_portfolios: Number of portfolios on the frontier
            constraints: Portfolio constraints
            
        Returns:
            Dictionary with returns, volatilities, and weights
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Range of target returns
        min_ret = np.min(self.expected_returns)
        max_ret = np.max(self.expected_returns)
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        frontier_portfolios = []
        frontier_returns = []
        frontier_volatilities = []
        frontier_sharpe_ratios = []
        
        for target_return in target_returns:
            try:
                # Optimize for minimum variance given target return
                weights = self._optimize_target_return(target_return, constraints)
                
                # Calculate metrics
                portfolio_return = np.dot(weights, self.expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                
                frontier_portfolios.append(weights)
                frontier_returns.append(portfolio_return)
                frontier_volatilities.append(portfolio_vol)
                frontier_sharpe_ratios.append(sharpe_ratio)
                
            except Exception as e:
                print(f"Failed to optimize for target return {target_return}: {e}")
                continue
        
        return {
            'returns': np.array(frontier_returns),
            'volatilities': np.array(frontier_volatilities),
            'sharpe_ratios': np.array(frontier_sharpe_ratios),
            'weights': np.array(frontier_portfolios),
            'target_returns': target_returns
        }
    
    def _optimize_target_return(self, target_return: float, 
                               constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize for minimum variance given target return"""
        n_assets = len(self.assets)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(weights, self.covariance_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,  # Weights sum to 1
            self.expected_returns.T @ weights == target_return  # Target return
        ]
        
        if constraints.long_only:
            constraints_list.append(weights >= 0)
        
        # Individual weight constraints
        for i, asset in enumerate(self.assets):
            min_w = constraints.min_weights.get(asset.symbol, asset.min_weight)
            max_w = constraints.max_weights.get(asset.symbol, asset.max_weight)
            constraints_list.extend([weights[i] >= min_w, weights[i] <= max_w])
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            return weights.value
        else:
            raise ValueError(f"Optimization failed for target return {target_return}")
    
    def backtest_portfolio(self, portfolio: Portfolio, 
                          start_date: str, end_date: str,
                          rebalancing_freq: RebalancingFrequency = RebalancingFrequency.MONTHLY) -> Dict:
        """
        Backtest portfolio performance
        
        Args:
            portfolio: Portfolio to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            rebalancing_freq: Rebalancing frequency
            
        Returns:
            Backtest results dictionary
        """
        # This is a simplified backtest implementation
        # In practice, you would use actual historical data
        
        if self.returns_matrix is None:
            raise ValueError("No returns data available for backtesting")
        
        # Simulate portfolio returns
        portfolio_returns = np.dot(self.returns_matrix, portfolio.weights)
        
        # Calculate performance metrics
        total_return = np.prod(1 + portfolio_returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        # Calculate drawdowns
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calculate other metrics
        positive_returns = portfolio_returns[portfolio_returns > 0]
        negative_returns = portfolio_returns[portfolio_returns < 0]
        
        win_rate = len(positive_returns) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'drawdowns': drawdowns
        }
    
    def monte_carlo_simulation(self, portfolio: Portfolio, 
                              n_simulations: int = 1000,
                              time_horizon: int = 252) -> Dict:
        """
        Monte Carlo simulation of portfolio performance
        
        Args:
            portfolio: Portfolio to simulate
            n_simulations: Number of simulation paths
            time_horizon: Time horizon in days
            
        Returns:
            Simulation results
        """
        # Generate random returns based on portfolio characteristics
        portfolio_return = portfolio.expected_return / 252  # Daily return
        portfolio_vol = portfolio.volatility / np.sqrt(252)  # Daily volatility
        
        # Monte Carlo simulation
        simulated_paths = []
        final_values = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(portfolio_return, portfolio_vol, time_horizon)
            
            # Calculate cumulative path
            cumulative_path = np.cumprod(1 + random_returns)
            simulated_paths.append(cumulative_path)
            final_values.append(cumulative_path[-1])
        
        simulated_paths = np.array(simulated_paths)
        final_values = np.array(final_values)
        
        # Calculate statistics
        mean_final_value = np.mean(final_values)
        std_final_value = np.std(final_values)
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = np.percentile(final_values, percentiles)
        
        # Probability of loss
        prob_loss = np.mean(final_values < 1.0)
        
        return {
            'simulated_paths': simulated_paths,
            'final_values': final_values,
            'mean_final_value': mean_final_value,
            'std_final_value': std_final_value,
            'percentiles': dict(zip(percentiles, percentile_values)),
            'probability_of_loss': prob_loss,
            'expected_return': mean_final_value - 1,
            'value_at_risk_5': percentile_values[0] - 1
        }