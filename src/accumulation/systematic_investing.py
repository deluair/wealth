"""
Systematic Investing Models

Models for systematic investment strategies including dollar-cost averaging,
value averaging, and momentum-based investing approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

class InvestmentStrategy(Enum):
    """Types of systematic investment strategies"""
    DOLLAR_COST_AVERAGING = "dollar_cost_averaging"
    VALUE_AVERAGING = "value_averaging"
    MOMENTUM_INVESTING = "momentum_investing"
    CONTRARIAN_INVESTING = "contrarian_investing"
    TARGET_DATE = "target_date"

class RebalancingFrequency(Enum):
    """Frequency of rebalancing"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"

@dataclass
class InvestmentPlan:
    """Parameters for systematic investment plan"""
    name: str
    strategy: InvestmentStrategy
    initial_amount: float
    periodic_amount: float
    frequency: RebalancingFrequency
    time_horizon: int  # in years
    target_allocation: Dict[str, float]  # asset class allocations
    rebalancing_threshold: float = 0.05  # 5% deviation triggers rebalancing
    target_growth_rate: float = 0.07  # for value averaging
    risk_tolerance: float = 0.5  # 0-1 scale

@dataclass
class InvestmentResult:
    """Results from systematic investment simulation"""
    plan_name: str
    final_value: float
    total_invested: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    investment_history: List[Dict]
    allocation_history: List[Dict]
    rebalancing_events: List[Dict]

class SystematicInvestor:
    """
    Systematic investment simulator with multiple strategies
    """
    
    def __init__(self):
        self.plans = {}
        self.results = {}
        self.market_data = None
    
    def add_plan(self, plan: InvestmentPlan) -> None:
        """Add an investment plan"""
        self.plans[plan.name] = plan
    
    def set_market_data(self, market_data: Dict[str, List[float]]) -> None:
        """Set historical market data for simulation"""
        self.market_data = market_data
    
    def simulate_plan(self, plan_name: str) -> InvestmentResult:
        """Simulate a specific investment plan"""
        if plan_name not in self.plans:
            raise ValueError(f"Plan '{plan_name}' not found")
        
        plan = self.plans[plan_name]
        
        if plan.strategy == InvestmentStrategy.DOLLAR_COST_AVERAGING:
            result = self._simulate_dollar_cost_averaging(plan)
        elif plan.strategy == InvestmentStrategy.VALUE_AVERAGING:
            result = self._simulate_value_averaging(plan)
        elif plan.strategy == InvestmentStrategy.MOMENTUM_INVESTING:
            result = self._simulate_momentum_investing(plan)
        elif plan.strategy == InvestmentStrategy.CONTRARIAN_INVESTING:
            result = self._simulate_contrarian_investing(plan)
        else:  # TARGET_DATE
            result = self._simulate_target_date(plan)
        
        self.results[plan_name] = result
        return result
    
    def _simulate_dollar_cost_averaging(self, plan: InvestmentPlan) -> InvestmentResult:
        """Simulate dollar-cost averaging strategy"""
        periods_per_year = self._get_periods_per_year(plan.frequency)
        total_periods = plan.time_horizon * periods_per_year
        
        # Generate market returns if not provided
        if self.market_data is None:
            market_returns = self._generate_synthetic_returns(total_periods, plan.target_allocation)
        else:
            market_returns = self._process_market_data(total_periods, plan.target_allocation)
        
        portfolio_value = {}
        total_invested = plan.initial_amount
        investment_history = []
        allocation_history = []
        rebalancing_events = []
        
        # Initialize portfolio
        for asset_class, allocation in plan.target_allocation.items():
            portfolio_value[asset_class] = plan.initial_amount * allocation
        
        # Record initial state
        investment_history.append({
            'period': 0,
            'investment': plan.initial_amount,
            'total_value': sum(portfolio_value.values()),
            'total_invested': total_invested
        })
        allocation_history.append(portfolio_value.copy())
        
        for period in range(1, total_periods + 1):
            # Apply market returns
            for asset_class in portfolio_value:
                return_rate = market_returns[asset_class][period - 1]
                portfolio_value[asset_class] *= (1 + return_rate)
            
            # Make periodic investment
            total_value = sum(portfolio_value.values())
            for asset_class, allocation in plan.target_allocation.items():
                investment_amount = plan.periodic_amount * allocation
                portfolio_value[asset_class] += investment_amount
            
            total_invested += plan.periodic_amount
            
            # Check for rebalancing
            current_allocations = self._calculate_current_allocations(portfolio_value)
            if self._needs_rebalancing(current_allocations, plan.target_allocation, plan.rebalancing_threshold):
                old_value = portfolio_value.copy()
                portfolio_value = self._rebalance_portfolio(portfolio_value, plan.target_allocation)
                rebalancing_events.append({
                    'period': period,
                    'old_allocation': old_value,
                    'new_allocation': portfolio_value.copy()
                })
            
            # Record state
            investment_history.append({
                'period': period,
                'investment': plan.periodic_amount,
                'total_value': sum(portfolio_value.values()),
                'total_invested': total_invested
            })
            allocation_history.append(portfolio_value.copy())
        
        return self._create_investment_result(plan, investment_history, allocation_history, 
                                            rebalancing_events, total_invested)
    
    def _simulate_value_averaging(self, plan: InvestmentPlan) -> InvestmentResult:
        """Simulate value averaging strategy"""
        periods_per_year = self._get_periods_per_year(plan.frequency)
        total_periods = plan.time_horizon * periods_per_year
        
        if self.market_data is None:
            market_returns = self._generate_synthetic_returns(total_periods, plan.target_allocation)
        else:
            market_returns = self._process_market_data(total_periods, plan.target_allocation)
        
        portfolio_value = {}
        total_invested = plan.initial_amount
        investment_history = []
        allocation_history = []
        rebalancing_events = []
        
        # Initialize portfolio
        for asset_class, allocation in plan.target_allocation.items():
            portfolio_value[asset_class] = plan.initial_amount * allocation
        
        target_growth_per_period = plan.target_growth_rate / periods_per_year
        
        for period in range(1, total_periods + 1):
            # Apply market returns
            for asset_class in portfolio_value:
                return_rate = market_returns[asset_class][period - 1]
                portfolio_value[asset_class] *= (1 + return_rate)
            
            # Calculate target value
            target_value = plan.initial_amount * ((1 + target_growth_per_period) ** period)
            current_value = sum(portfolio_value.values())
            
            # Calculate required investment
            required_investment = target_value - current_value
            
            # Limit investment to reasonable bounds
            min_investment = -plan.periodic_amount * 2  # Allow some selling
            max_investment = plan.periodic_amount * 3   # Limit buying
            actual_investment = max(min_investment, min(max_investment, required_investment))
            
            # Apply investment
            if actual_investment != 0:
                for asset_class, allocation in plan.target_allocation.items():
                    investment_amount = actual_investment * allocation
                    portfolio_value[asset_class] += investment_amount
                
                total_invested += actual_investment
            
            # Record state
            investment_history.append({
                'period': period,
                'investment': actual_investment,
                'total_value': sum(portfolio_value.values()),
                'total_invested': total_invested,
                'target_value': target_value
            })
            allocation_history.append(portfolio_value.copy())
        
        return self._create_investment_result(plan, investment_history, allocation_history,
                                            rebalancing_events, total_invested)
    
    def _simulate_momentum_investing(self, plan: InvestmentPlan) -> InvestmentResult:
        """Simulate momentum-based investing"""
        periods_per_year = self._get_periods_per_year(plan.frequency)
        total_periods = plan.time_horizon * periods_per_year
        
        if self.market_data is None:
            market_returns = self._generate_synthetic_returns(total_periods, plan.target_allocation)
        else:
            market_returns = self._process_market_data(total_periods, plan.target_allocation)
        
        portfolio_value = {}
        total_invested = plan.initial_amount
        investment_history = []
        allocation_history = []
        rebalancing_events = []
        
        # Initialize portfolio
        for asset_class, allocation in plan.target_allocation.items():
            portfolio_value[asset_class] = plan.initial_amount * allocation
        
        lookback_periods = min(12, total_periods // 4)  # Look back 12 periods or 1/4 of total
        
        for period in range(1, total_periods + 1):
            # Apply market returns
            for asset_class in portfolio_value:
                return_rate = market_returns[asset_class][period - 1]
                portfolio_value[asset_class] *= (1 + return_rate)
            
            # Calculate momentum scores
            if period > lookback_periods:
                momentum_scores = {}
                for asset_class in plan.target_allocation:
                    recent_returns = market_returns[asset_class][period - lookback_periods:period]
                    momentum_scores[asset_class] = np.mean(recent_returns)
                
                # Adjust allocation based on momentum
                adjusted_allocation = self._adjust_allocation_for_momentum(
                    plan.target_allocation, momentum_scores, plan.risk_tolerance
                )
            else:
                adjusted_allocation = plan.target_allocation
            
            # Make periodic investment with adjusted allocation
            for asset_class, allocation in adjusted_allocation.items():
                investment_amount = plan.periodic_amount * allocation
                portfolio_value[asset_class] += investment_amount
            
            total_invested += plan.periodic_amount
            
            # Record state
            investment_history.append({
                'period': period,
                'investment': plan.periodic_amount,
                'total_value': sum(portfolio_value.values()),
                'total_invested': total_invested
            })
            allocation_history.append(portfolio_value.copy())
        
        return self._create_investment_result(plan, investment_history, allocation_history,
                                            rebalancing_events, total_invested)
    
    def _simulate_contrarian_investing(self, plan: InvestmentPlan) -> InvestmentResult:
        """Simulate contrarian investing strategy"""
        # Similar to momentum but with opposite logic
        periods_per_year = self._get_periods_per_year(plan.frequency)
        total_periods = plan.time_horizon * periods_per_year
        
        if self.market_data is None:
            market_returns = self._generate_synthetic_returns(total_periods, plan.target_allocation)
        else:
            market_returns = self._process_market_data(total_periods, plan.target_allocation)
        
        portfolio_value = {}
        total_invested = plan.initial_amount
        investment_history = []
        allocation_history = []
        rebalancing_events = []
        
        # Initialize portfolio
        for asset_class, allocation in plan.target_allocation.items():
            portfolio_value[asset_class] = plan.initial_amount * allocation
        
        lookback_periods = min(12, total_periods // 4)
        
        for period in range(1, total_periods + 1):
            # Apply market returns
            for asset_class in portfolio_value:
                return_rate = market_returns[asset_class][period - 1]
                portfolio_value[asset_class] *= (1 + return_rate)
            
            # Calculate contrarian scores (opposite of momentum)
            if period > lookback_periods:
                momentum_scores = {}
                for asset_class in plan.target_allocation:
                    recent_returns = market_returns[asset_class][period - lookback_periods:period]
                    momentum_scores[asset_class] = -np.mean(recent_returns)  # Negative for contrarian
                
                adjusted_allocation = self._adjust_allocation_for_momentum(
                    plan.target_allocation, momentum_scores, plan.risk_tolerance
                )
            else:
                adjusted_allocation = plan.target_allocation
            
            # Make investment
            for asset_class, allocation in adjusted_allocation.items():
                investment_amount = plan.periodic_amount * allocation
                portfolio_value[asset_class] += investment_amount
            
            total_invested += plan.periodic_amount
            
            investment_history.append({
                'period': period,
                'investment': plan.periodic_amount,
                'total_value': sum(portfolio_value.values()),
                'total_invested': total_invested
            })
            allocation_history.append(portfolio_value.copy())
        
        return self._create_investment_result(plan, investment_history, allocation_history,
                                            rebalancing_events, total_invested)
    
    def _simulate_target_date(self, plan: InvestmentPlan) -> InvestmentResult:
        """Simulate target-date strategy with glide path"""
        periods_per_year = self._get_periods_per_year(plan.frequency)
        total_periods = plan.time_horizon * periods_per_year
        
        if self.market_data is None:
            market_returns = self._generate_synthetic_returns(total_periods, plan.target_allocation)
        else:
            market_returns = self._process_market_data(total_periods, plan.target_allocation)
        
        portfolio_value = {}
        total_invested = plan.initial_amount
        investment_history = []
        allocation_history = []
        rebalancing_events = []
        
        # Initialize portfolio
        for asset_class, allocation in plan.target_allocation.items():
            portfolio_value[asset_class] = plan.initial_amount * allocation
        
        for period in range(1, total_periods + 1):
            # Apply market returns
            for asset_class in portfolio_value:
                return_rate = market_returns[asset_class][period - 1]
                portfolio_value[asset_class] *= (1 + return_rate)
            
            # Calculate glide path allocation
            years_remaining = plan.time_horizon - (period / periods_per_year)
            glide_allocation = self._calculate_glide_path_allocation(
                plan.target_allocation, years_remaining, plan.time_horizon
            )
            
            # Make periodic investment
            for asset_class, allocation in glide_allocation.items():
                investment_amount = plan.periodic_amount * allocation
                portfolio_value[asset_class] += investment_amount
            
            total_invested += plan.periodic_amount
            
            # Rebalance to glide path if needed
            current_allocations = self._calculate_current_allocations(portfolio_value)
            if self._needs_rebalancing(current_allocations, glide_allocation, plan.rebalancing_threshold):
                old_value = portfolio_value.copy()
                portfolio_value = self._rebalance_portfolio(portfolio_value, glide_allocation)
                rebalancing_events.append({
                    'period': period,
                    'old_allocation': old_value,
                    'new_allocation': portfolio_value.copy(),
                    'target_allocation': glide_allocation
                })
            
            investment_history.append({
                'period': period,
                'investment': plan.periodic_amount,
                'total_value': sum(portfolio_value.values()),
                'total_invested': total_invested
            })
            allocation_history.append(portfolio_value.copy())
        
        return self._create_investment_result(plan, investment_history, allocation_history,
                                            rebalancing_events, total_invested)
    
    def _get_periods_per_year(self, frequency: RebalancingFrequency) -> int:
        """Get number of periods per year"""
        mapping = {
            RebalancingFrequency.MONTHLY: 12,
            RebalancingFrequency.QUARTERLY: 4,
            RebalancingFrequency.SEMI_ANNUALLY: 2,
            RebalancingFrequency.ANNUALLY: 1
        }
        return mapping[frequency]
    
    def _generate_synthetic_returns(self, periods: int, allocations: Dict[str, float]) -> Dict[str, List[float]]:
        """Generate synthetic market returns"""
        np.random.seed(42)
        
        # Default return and volatility assumptions
        return_assumptions = {
            'stocks': {'return': 0.10, 'volatility': 0.16},
            'bonds': {'return': 0.04, 'volatility': 0.05},
            'real_estate': {'return': 0.08, 'volatility': 0.12},
            'commodities': {'return': 0.06, 'volatility': 0.20},
            'cash': {'return': 0.02, 'volatility': 0.01}
        }
        
        returns = {}
        for asset_class in allocations:
            if asset_class in return_assumptions:
                params = return_assumptions[asset_class]
            else:
                params = {'return': 0.07, 'volatility': 0.15}  # Default
            
            annual_return = params['return']
            annual_vol = params['volatility']
            
            # Convert to period returns
            period_return = annual_return / 12  # Assuming monthly
            period_vol = annual_vol / math.sqrt(12)
            
            returns[asset_class] = np.random.normal(period_return, period_vol, periods).tolist()
        
        return returns
    
    def _process_market_data(self, periods: int, allocations: Dict[str, float]) -> Dict[str, List[float]]:
        """Process provided market data"""
        processed_returns = {}
        for asset_class in allocations:
            if asset_class in self.market_data:
                data = self.market_data[asset_class][:periods]
                processed_returns[asset_class] = data
            else:
                # Generate synthetic data for missing asset classes
                processed_returns[asset_class] = np.random.normal(0.007, 0.04, periods).tolist()
        
        return processed_returns
    
    def _calculate_current_allocations(self, portfolio_value: Dict[str, float]) -> Dict[str, float]:
        """Calculate current portfolio allocations"""
        total_value = sum(portfolio_value.values())
        if total_value == 0:
            return {asset: 0 for asset in portfolio_value}
        
        return {asset: value / total_value for asset, value in portfolio_value.items()}
    
    def _needs_rebalancing(self, current: Dict[str, float], target: Dict[str, float], 
                          threshold: float) -> bool:
        """Check if rebalancing is needed"""
        for asset_class in target:
            deviation = abs(current.get(asset_class, 0) - target[asset_class])
            if deviation > threshold:
                return True
        return False
    
    def _rebalance_portfolio(self, portfolio_value: Dict[str, float], 
                           target_allocation: Dict[str, float]) -> Dict[str, float]:
        """Rebalance portfolio to target allocation"""
        total_value = sum(portfolio_value.values())
        rebalanced = {}
        
        for asset_class, target_pct in target_allocation.items():
            rebalanced[asset_class] = total_value * target_pct
        
        return rebalanced
    
    def _adjust_allocation_for_momentum(self, base_allocation: Dict[str, float],
                                      momentum_scores: Dict[str, float],
                                      risk_tolerance: float) -> Dict[str, float]:
        """Adjust allocation based on momentum scores"""
        # Normalize momentum scores
        max_score = max(momentum_scores.values())
        min_score = min(momentum_scores.values())
        
        if max_score == min_score:
            return base_allocation
        
        normalized_scores = {
            asset: (score - min_score) / (max_score - min_score)
            for asset, score in momentum_scores.items()
        }
        
        # Adjust allocations
        adjusted = {}
        adjustment_factor = risk_tolerance * 0.3  # Max 30% adjustment
        
        for asset_class, base_pct in base_allocation.items():
            momentum_adjustment = (normalized_scores.get(asset_class, 0.5) - 0.5) * adjustment_factor
            adjusted[asset_class] = max(0.05, base_pct + momentum_adjustment)  # Min 5% allocation
        
        # Normalize to sum to 1
        total = sum(adjusted.values())
        adjusted = {asset: pct / total for asset, pct in adjusted.items()}
        
        return adjusted
    
    def _calculate_glide_path_allocation(self, base_allocation: Dict[str, float],
                                       years_remaining: float, total_years: int) -> Dict[str, float]:
        """Calculate target-date glide path allocation"""
        # Simple glide path: reduce equity allocation as target date approaches
        progress = 1 - (years_remaining / total_years)
        
        glide_allocation = {}
        for asset_class, base_pct in base_allocation.items():
            if asset_class in ['stocks', 'equity']:
                # Reduce equity allocation over time
                reduction_factor = progress * 0.4  # Max 40% reduction
                glide_allocation[asset_class] = max(0.2, base_pct * (1 - reduction_factor))
            elif asset_class in ['bonds', 'fixed_income']:
                # Increase bond allocation over time
                increase_factor = progress * 0.5
                glide_allocation[asset_class] = min(0.7, base_pct * (1 + increase_factor))
            else:
                glide_allocation[asset_class] = base_pct
        
        # Normalize
        total = sum(glide_allocation.values())
        glide_allocation = {asset: pct / total for asset, pct in glide_allocation.items()}
        
        return glide_allocation
    
    def _create_investment_result(self, plan: InvestmentPlan, investment_history: List[Dict],
                                allocation_history: List[Dict], rebalancing_events: List[Dict],
                                total_invested: float) -> InvestmentResult:
        """Create investment result object"""
        final_value = investment_history[-1]['total_value']
        total_return = final_value - total_invested
        
        # Calculate annualized return
        if total_invested > 0 and plan.time_horizon > 0:
            annualized_return = (final_value / plan.initial_amount) ** (1 / plan.time_horizon) - 1
        else:
            annualized_return = 0.0
        
        # Calculate volatility
        values = [entry['total_value'] for entry in investment_history]
        returns = [values[i] / values[i-1] - 1 for i in range(1, len(values))]
        volatility = np.std(returns) * math.sqrt(12) if returns else 0.0  # Annualized
        
        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(values)
        
        return InvestmentResult(
            plan_name=plan.name,
            final_value=final_value,
            total_invested=total_invested,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            investment_history=investment_history,
            allocation_history=allocation_history,
            rebalancing_events=rebalancing_events
        )
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not values:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def compare_strategies(self, plan_names: List[str]) -> pd.DataFrame:
        """Compare multiple investment strategies"""
        comparison_data = []
        
        for name in plan_names:
            if name in self.results:
                result = self.results[name]
                comparison_data.append({
                    'Strategy': name,
                    'Final Value': result.final_value,
                    'Total Invested': result.total_invested,
                    'Total Return': result.total_return,
                    'Annualized Return': result.annualized_return,
                    'Volatility': result.volatility,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Max Drawdown': result.max_drawdown
                })
        
        return pd.DataFrame(comparison_data)