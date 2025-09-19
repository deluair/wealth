"""
Wealth Creation Sources

Detailed models for different pathways to wealth creation including
business ventures, investments, employment, inheritance, and real estate.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

@dataclass
class SourceParameters:
    """Base parameters for wealth creation sources"""
    initial_investment: float
    time_horizon: int
    risk_level: float  # 0-1 scale
    market_conditions: float  # 0-1 scale

class WealthSource(ABC):
    """Abstract base class for wealth creation sources"""
    
    def __init__(self, params: SourceParameters):
        self.params = params
        
    @abstractmethod
    def simulate(self) -> Dict:
        """Simulate wealth creation for this source"""
        pass

class BusinessVentureModel(WealthSource):
    """Model for business venture wealth creation"""
    
    def __init__(self, params: SourceParameters, 
                 industry_type: str = "technology",
                 team_size: int = 3,
                 has_mentor: bool = False):
        super().__init__(params)
        self.industry_type = industry_type
        self.team_size = team_size
        self.has_mentor = has_mentor
        
        # Industry-specific parameters
        self.industry_params = {
            "technology": {"growth_rate": 0.25, "failure_rate": 0.7, "scale_potential": 10},
            "retail": {"growth_rate": 0.12, "failure_rate": 0.5, "scale_potential": 3},
            "manufacturing": {"growth_rate": 0.08, "failure_rate": 0.4, "scale_potential": 5},
            "services": {"growth_rate": 0.15, "failure_rate": 0.45, "scale_potential": 4},
            "healthcare": {"growth_rate": 0.18, "failure_rate": 0.3, "scale_potential": 6}
        }
    
    def simulate(self) -> Dict:
        """Simulate business venture outcomes"""
        industry = self.industry_params.get(self.industry_type, 
                                          self.industry_params["services"])
        
        # Calculate success probability
        base_success_prob = 1 - industry["failure_rate"]
        
        # Adjust for team size (optimal around 3-4 people)
        team_factor = 1 - abs(self.team_size - 3.5) * 0.05
        
        # Mentor bonus
        mentor_bonus = 0.15 if self.has_mentor else 0
        
        # Market conditions impact
        market_factor = 0.5 + 0.5 * self.params.market_conditions
        
        success_prob = base_success_prob * team_factor * market_factor + mentor_bonus
        success_prob = min(0.8, max(0.1, success_prob))  # Clamp between 10-80%
        
        # Simulate business lifecycle
        years = self.params.time_horizon
        wealth_trajectory = [self.params.initial_investment]
        
        # Business stages: Startup -> Growth -> Maturity -> Exit/Decline
        stages = self._determine_business_stages(years)
        
        for year in range(1, years + 1):
            stage = self._get_current_stage(year, stages)
            
            if np.random.random() > success_prob and year <= 3:
                # Business failure in early years
                wealth_trajectory.append(0)
                break
            
            # Stage-specific growth
            if stage == "startup":
                growth_rate = np.random.normal(-0.2, 0.4)  # High volatility
            elif stage == "growth":
                growth_rate = np.random.normal(industry["growth_rate"], 0.3)
            elif stage == "maturity":
                growth_rate = np.random.normal(0.05, 0.1)
            else:  # decline
                growth_rate = np.random.normal(-0.05, 0.15)
            
            # Scale events (rare but high impact)
            if np.random.random() < 0.05 and stage == "growth":
                scale_multiplier = np.random.uniform(2, industry["scale_potential"])
                growth_rate *= scale_multiplier
            
            new_wealth = wealth_trajectory[-1] * (1 + growth_rate)
            wealth_trajectory.append(max(0, new_wealth))
        
        # Exit event simulation
        exit_value = self._simulate_exit_event(wealth_trajectory[-1] if wealth_trajectory else 0)
        
        return {
            "source_type": "business_venture",
            "industry": self.industry_type,
            "success_probability": success_prob,
            "wealth_trajectory": wealth_trajectory,
            "final_wealth": wealth_trajectory[-1] if wealth_trajectory else 0,
            "exit_value": exit_value,
            "business_stages": stages,
            "total_return": (wealth_trajectory[-1] / self.params.initial_investment - 1) if wealth_trajectory else -1
        }
    
    def _determine_business_stages(self, years: int) -> Dict:
        """Determine business lifecycle stages"""
        return {
            "startup": (0, min(3, years // 4)),
            "growth": (min(3, years // 4), min(8, years * 3 // 4)),
            "maturity": (min(8, years * 3 // 4), years),
            "decline": (max(years - 2, years * 3 // 4), years)
        }
    
    def _get_current_stage(self, year: int, stages: Dict) -> str:
        """Get current business stage"""
        for stage, (start, end) in stages.items():
            if start <= year < end:
                return stage
        return "maturity"
    
    def _simulate_exit_event(self, current_value: float) -> float:
        """Simulate business exit (IPO, acquisition, etc.)"""
        if current_value < self.params.initial_investment * 2:
            return current_value  # No exit event
        
        # Exit probability increases with business value
        exit_prob = min(0.3, current_value / (self.params.initial_investment * 10) * 0.1)
        
        if np.random.random() < exit_prob:
            # Exit multiplier based on industry and performance
            multiplier = np.random.lognormal(1, 0.5)  # Log-normal distribution
            return current_value * multiplier
        
        return current_value

class InvestmentModel(WealthSource):
    """Model for investment-based wealth creation"""
    
    def __init__(self, params: SourceParameters,
                 asset_allocation: Dict[str, float] = None,
                 rebalancing_frequency: int = 12,  # months
                 dollar_cost_averaging: bool = True):
        super().__init__(params)
        
        # Default asset allocation
        self.asset_allocation = asset_allocation or {
            "stocks": 0.6,
            "bonds": 0.3,
            "real_estate": 0.05,
            "commodities": 0.03,
            "crypto": 0.02
        }
        
        self.rebalancing_frequency = rebalancing_frequency
        self.dollar_cost_averaging = dollar_cost_averaging
        
        # Asset class parameters (annual returns and volatility)
        self.asset_params = {
            "stocks": {"return": 0.10, "volatility": 0.16, "correlation": 1.0},
            "bonds": {"return": 0.04, "volatility": 0.05, "correlation": -0.2},
            "real_estate": {"return": 0.08, "volatility": 0.12, "correlation": 0.3},
            "commodities": {"return": 0.06, "volatility": 0.20, "correlation": 0.1},
            "crypto": {"return": 0.15, "volatility": 0.60, "correlation": 0.05}
        }
    
    def simulate(self) -> Dict:
        """Simulate investment portfolio performance"""
        months = self.params.time_horizon * 12
        monthly_investment = self.params.initial_investment / months if self.dollar_cost_averaging else 0
        
        # Initialize portfolio
        portfolio_value = self.params.initial_investment if not self.dollar_cost_averaging else 0
        portfolio_history = []
        asset_values = {asset: 0 for asset in self.asset_allocation.keys()}
        
        # Market regime simulation (bull, bear, sideways)
        market_regimes = self._simulate_market_regimes(months)
        
        for month in range(months):
            # Add monthly investment if dollar cost averaging
            if self.dollar_cost_averaging:
                portfolio_value += monthly_investment
                for asset, allocation in self.asset_allocation.items():
                    asset_values[asset] += monthly_investment * allocation
            
            # Generate monthly returns for each asset class
            monthly_returns = self._generate_monthly_returns(market_regimes[month])
            
            # Update asset values
            for asset in asset_values:
                monthly_return = monthly_returns[asset]
                asset_values[asset] *= (1 + monthly_return)
            
            # Calculate total portfolio value
            portfolio_value = sum(asset_values.values())
            
            # Rebalancing
            if month % self.rebalancing_frequency == 0 and month > 0:
                asset_values = self._rebalance_portfolio(portfolio_value)
            
            portfolio_history.append({
                "month": month,
                "portfolio_value": portfolio_value,
                "market_regime": market_regimes[month],
                **{f"{asset}_value": value for asset, value in asset_values.items()}
            })
        
        # Calculate performance metrics
        total_invested = (self.params.initial_investment + 
                         monthly_investment * months if self.dollar_cost_averaging 
                         else self.params.initial_investment)
        
        total_return = (portfolio_value / total_invested) - 1
        annualized_return = (portfolio_value / total_invested) ** (1/self.params.time_horizon) - 1
        
        return {
            "source_type": "investment",
            "portfolio_history": portfolio_history,
            "final_value": portfolio_value,
            "total_invested": total_invested,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "asset_allocation": self.asset_allocation,
            "max_drawdown": self._calculate_max_drawdown(portfolio_history),
            "sharpe_ratio": self._calculate_sharpe_ratio(portfolio_history)
        }
    
    def _simulate_market_regimes(self, months: int) -> List[str]:
        """Simulate market regimes (bull, bear, sideways)"""
        regimes = []
        current_regime = "bull"
        regime_duration = 0
        
        # Regime transition probabilities
        transition_probs = {
            "bull": {"bull": 0.85, "bear": 0.10, "sideways": 0.05},
            "bear": {"bull": 0.20, "bear": 0.70, "sideways": 0.10},
            "sideways": {"bull": 0.30, "bear": 0.20, "sideways": 0.50}
        }
        
        for month in range(months):
            regimes.append(current_regime)
            regime_duration += 1
            
            # Force regime change after extended periods
            if regime_duration > 24:  # 2 years
                if current_regime == "bull":
                    current_regime = np.random.choice(["bear", "sideways"], p=[0.6, 0.4])
                else:
                    current_regime = "bull"
                regime_duration = 0
            else:
                # Normal regime transition
                if np.random.random() < 0.1:  # 10% chance of regime change each month
                    probs = transition_probs[current_regime]
                    current_regime = np.random.choice(list(probs.keys()), p=list(probs.values()))
                    regime_duration = 0
        
        return regimes
    
    def _generate_monthly_returns(self, market_regime: str) -> Dict[str, float]:
        """Generate monthly returns for each asset class"""
        regime_multipliers = {
            "bull": {"stocks": 1.2, "bonds": 1.0, "real_estate": 1.1, "commodities": 1.0, "crypto": 1.3},
            "bear": {"stocks": 0.6, "bonds": 1.1, "real_estate": 0.8, "commodities": 0.9, "crypto": 0.5},
            "sideways": {"stocks": 0.9, "bonds": 1.0, "real_estate": 0.95, "commodities": 0.95, "crypto": 0.8}
        }
        
        returns = {}
        for asset, params in self.asset_params.items():
            # Base monthly return
            monthly_return = params["return"] / 12
            monthly_volatility = params["volatility"] / math.sqrt(12)
            
            # Apply regime multiplier
            regime_mult = regime_multipliers[market_regime][asset]
            adjusted_return = monthly_return * regime_mult
            
            # Generate return with volatility
            returns[asset] = np.random.normal(adjusted_return, monthly_volatility)
        
        return returns
    
    def _rebalance_portfolio(self, total_value: float) -> Dict[str, float]:
        """Rebalance portfolio to target allocation"""
        return {asset: total_value * allocation 
                for asset, allocation in self.asset_allocation.items()}
    
    def _calculate_max_drawdown(self, history: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        values = [h["portfolio_value"] for h in history]
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, history: List[Dict]) -> float:
        """Calculate Sharpe ratio"""
        values = [h["portfolio_value"] for h in history]
        returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
        
        if len(returns) == 0:
            return 0
        
        excess_return = np.mean(returns) - 0.02/12  # Risk-free rate 2% annually
        return excess_return / np.std(returns) * math.sqrt(12) if np.std(returns) > 0 else 0

class EmploymentModel(WealthSource):
    """Model for employment-based wealth creation"""
    
    def __init__(self, params: SourceParameters,
                 starting_salary: float = 50000,
                 industry: str = "technology",
                 education_level: str = "bachelor",
                 career_ambition: float = 0.7):
        super().__init__(params)
        self.starting_salary = starting_salary
        self.industry = industry
        self.education_level = education_level
        self.career_ambition = career_ambition  # 0-1 scale
        
        # Industry salary multipliers and growth rates
        self.industry_params = {
            "technology": {"multiplier": 1.3, "growth_rate": 0.08, "volatility": 0.15},
            "finance": {"multiplier": 1.4, "growth_rate": 0.06, "volatility": 0.12},
            "healthcare": {"multiplier": 1.2, "growth_rate": 0.05, "volatility": 0.08},
            "education": {"multiplier": 0.8, "growth_rate": 0.03, "volatility": 0.05},
            "government": {"multiplier": 0.9, "growth_rate": 0.04, "volatility": 0.03},
            "retail": {"multiplier": 0.7, "growth_rate": 0.02, "volatility": 0.10}
        }
        
        # Education level multipliers
        self.education_multipliers = {
            "high_school": 0.7,
            "associate": 0.85,
            "bachelor": 1.0,
            "master": 1.25,
            "phd": 1.4,
            "professional": 1.6
        }
    
    def simulate(self) -> Dict:
        """Simulate career progression and wealth accumulation"""
        industry_params = self.industry_params.get(self.industry, 
                                                 self.industry_params["technology"])
        education_mult = self.education_multipliers.get(self.education_level, 1.0)
        
        # Adjusted starting salary
        current_salary = (self.starting_salary * 
                         industry_params["multiplier"] * 
                         education_mult)
        
        years = self.params.time_horizon
        salary_history = [current_salary]
        wealth_history = [self.params.initial_investment]
        savings_rate_history = []
        
        for year in range(1, years + 1):
            # Career progression events
            promotion_prob = self._calculate_promotion_probability(year)
            job_change_prob = 0.15 * self.career_ambition  # Higher ambition = more job changes
            
            if np.random.random() < promotion_prob:
                # Promotion
                salary_increase = np.random.uniform(0.15, 0.30)
                current_salary *= (1 + salary_increase)
            elif np.random.random() < job_change_prob:
                # Job change
                salary_change = np.random.uniform(-0.05, 0.25)  # Risk/reward of job change
                current_salary *= (1 + salary_change)
            else:
                # Regular raise
                base_growth = industry_params["growth_rate"]
                volatility = industry_params["volatility"]
                salary_growth = np.random.normal(base_growth, volatility * 0.5)
                current_salary *= (1 + salary_growth)
            
            # Economic cycles impact
            if np.random.random() < 0.1:  # Recession
                current_salary *= np.random.uniform(0.9, 1.0)
            
            # Calculate savings rate (increases with income)
            base_savings_rate = 0.1
            income_bonus = min(0.3, (current_salary - self.starting_salary) / self.starting_salary * 0.1)
            savings_rate = base_savings_rate + income_bonus
            
            # Annual savings
            annual_savings = current_salary * savings_rate
            
            # Investment of savings (conservative portfolio)
            investment_return = np.random.normal(0.06, 0.08)
            wealth_growth = wealth_history[-1] * investment_return + annual_savings
            
            new_wealth = wealth_history[-1] + wealth_growth
            
            salary_history.append(current_salary)
            wealth_history.append(new_wealth)
            savings_rate_history.append(savings_rate)
        
        return {
            "source_type": "employment",
            "industry": self.industry,
            "education_level": self.education_level,
            "salary_history": salary_history,
            "wealth_history": wealth_history,
            "savings_rate_history": savings_rate_history,
            "final_salary": salary_history[-1],
            "final_wealth": wealth_history[-1],
            "total_earnings": sum(salary_history),
            "total_savings": sum([salary_history[i] * savings_rate_history[i-1] 
                                for i in range(1, len(salary_history))]),
            "career_growth_rate": (salary_history[-1] / salary_history[0]) ** (1/years) - 1
        }
    
    def _calculate_promotion_probability(self, year: int) -> float:
        """Calculate promotion probability based on career stage"""
        # Higher probability early in career, then decreases
        base_prob = 0.2 * self.career_ambition
        
        if year <= 5:
            return base_prob * 1.5  # Early career boost
        elif year <= 15:
            return base_prob
        else:
            return base_prob * 0.5  # Senior career plateau

class InheritanceModel(WealthSource):
    """Model for inheritance-based wealth transfer"""
    
    def __init__(self, params: SourceParameters,
                 family_wealth_level: str = "middle_class",
                 inheritance_probability: float = 0.3,
                 family_size: int = 2):
        super().__init__(params)
        self.family_wealth_level = family_wealth_level
        self.inheritance_probability = inheritance_probability
        self.family_size = family_size  # Number of heirs
        
        # Wealth level distributions (log-normal parameters)
        self.wealth_distributions = {
            "low_income": {"mu": 10, "sigma": 0.5},      # ~$22k median
            "middle_class": {"mu": 12, "sigma": 0.8},    # ~$163k median
            "upper_middle": {"mu": 13.5, "sigma": 1.0},  # ~$730k median
            "wealthy": {"mu": 15, "sigma": 1.2},         # ~$3.3M median
            "ultra_wealthy": {"mu": 17, "sigma": 1.5}    # ~$24M median
        }
    
    def simulate(self) -> Dict:
        """Simulate inheritance events"""
        # Determine if inheritance occurs
        receives_inheritance = np.random.random() < self.inheritance_probability
        
        if not receives_inheritance:
            return {
                "source_type": "inheritance",
                "receives_inheritance": False,
                "inheritance_amount": 0,
                "inheritance_year": None,
                "family_wealth_level": self.family_wealth_level
            }
        
        # Generate inheritance amount
        wealth_params = self.wealth_distributions[self.family_wealth_level]
        total_estate = np.random.lognormal(wealth_params["mu"], wealth_params["sigma"])
        
        # Account for taxes and family size
        tax_rate = self._calculate_estate_tax_rate(total_estate)
        after_tax_estate = total_estate * (1 - tax_rate)
        inheritance_amount = after_tax_estate / self.family_size
        
        # Determine timing of inheritance
        inheritance_year = np.random.randint(10, min(30, self.params.time_horizon))
        
        # Simulate inheritance growth/decline over time
        years_to_simulate = self.params.time_horizon - inheritance_year
        inheritance_trajectory = self._simulate_inheritance_management(
            inheritance_amount, years_to_simulate)
        
        return {
            "source_type": "inheritance",
            "receives_inheritance": True,
            "inheritance_amount": inheritance_amount,
            "inheritance_year": inheritance_year,
            "total_estate": total_estate,
            "tax_rate": tax_rate,
            "family_size": self.family_size,
            "family_wealth_level": self.family_wealth_level,
            "inheritance_trajectory": inheritance_trajectory,
            "final_inheritance_value": inheritance_trajectory[-1] if inheritance_trajectory else inheritance_amount
        }
    
    def _calculate_estate_tax_rate(self, estate_value: float) -> float:
        """Calculate estate tax rate based on value"""
        if estate_value < 12_000_000:  # Below federal exemption
            return 0.0
        elif estate_value < 50_000_000:
            return 0.40  # Federal estate tax
        else:
            return 0.45  # Higher rate for very large estates
    
    def _simulate_inheritance_management(self, initial_amount: float, years: int) -> List[float]:
        """Simulate how inheritance is managed over time"""
        if years <= 0:
            return [initial_amount]
        
        trajectory = [initial_amount]
        current_value = initial_amount
        
        # Assume conservative investment approach for inherited wealth
        for year in range(years):
            # Conservative portfolio return
            annual_return = np.random.normal(0.05, 0.08)
            
            # Some heirs may spend inheritance faster
            spending_rate = np.random.uniform(0.02, 0.08)  # 2-8% annual spending
            
            current_value = current_value * (1 + annual_return) * (1 - spending_rate)
            trajectory.append(max(0, current_value))
        
        return trajectory

class RealEstateModel(WealthSource):
    """Model for real estate investment wealth creation"""
    
    def __init__(self, params: SourceParameters,
                 property_type: str = "residential",
                 leverage_ratio: float = 0.8,
                 location_tier: str = "tier_2"):
        super().__init__(params)
        self.property_type = property_type
        self.leverage_ratio = leverage_ratio  # Loan-to-value ratio
        self.location_tier = location_tier
        
        # Property type parameters
        self.property_params = {
            "residential": {"appreciation": 0.04, "rental_yield": 0.06, "volatility": 0.08},
            "commercial": {"appreciation": 0.05, "rental_yield": 0.08, "volatility": 0.12},
            "industrial": {"appreciation": 0.03, "rental_yield": 0.07, "volatility": 0.06},
            "retail": {"appreciation": 0.02, "rental_yield": 0.09, "volatility": 0.15}
        }
        
        # Location tier multipliers
        self.location_multipliers = {
            "tier_1": {"appreciation": 1.3, "rental_yield": 0.9, "volatility": 1.2},  # Major cities
            "tier_2": {"appreciation": 1.0, "rental_yield": 1.0, "volatility": 1.0},  # Mid-size cities
            "tier_3": {"appreciation": 0.7, "rental_yield": 1.1, "volatility": 0.8}   # Small cities/rural
        }
    
    def simulate(self) -> Dict:
        """Simulate real estate investment performance"""
        property_params = self.property_params[self.property_type]
        location_mult = self.location_multipliers[self.location_tier]
        
        # Initial property purchase
        down_payment = self.params.initial_investment
        property_value = down_payment / (1 - self.leverage_ratio)
        mortgage_amount = property_value - down_payment
        
        years = self.params.time_horizon
        property_values = [property_value]
        mortgage_balances = [mortgage_amount]
        rental_income = []
        net_cash_flows = []
        equity_values = [down_payment]
        
        # Mortgage parameters
        interest_rate = 0.04  # 4% mortgage rate
        monthly_payment = self._calculate_mortgage_payment(mortgage_amount, interest_rate, years)
        
        for year in range(1, years + 1):
            # Property appreciation
            base_appreciation = property_params["appreciation"] * location_mult["appreciation"]
            volatility = property_params["volatility"] * location_mult["volatility"]
            appreciation_rate = np.random.normal(base_appreciation, volatility)
            
            # Market cycles
            if np.random.random() < 0.1:  # Market downturn
                appreciation_rate *= np.random.uniform(0.5, 0.8)
            elif np.random.random() < 0.05:  # Market boom
                appreciation_rate *= np.random.uniform(1.3, 1.8)
            
            new_property_value = property_values[-1] * (1 + appreciation_rate)
            property_values.append(new_property_value)
            
            # Rental income
            base_yield = property_params["rental_yield"] * location_mult["rental_yield"]
            annual_rent = new_property_value * base_yield
            
            # Vacancy and maintenance costs
            vacancy_rate = np.random.uniform(0.05, 0.15)
            maintenance_rate = 0.02  # 2% of property value
            
            net_rental_income = (annual_rent * (1 - vacancy_rate) - 
                               new_property_value * maintenance_rate)
            rental_income.append(net_rental_income)
            
            # Mortgage payments
            annual_mortgage_payment = monthly_payment * 12
            
            # Principal paydown (simplified)
            principal_payment = mortgage_balances[-1] * 0.03  # Approximate
            new_mortgage_balance = max(0, mortgage_balances[-1] - principal_payment)
            mortgage_balances.append(new_mortgage_balance)
            
            # Net cash flow
            net_cash_flow = net_rental_income - annual_mortgage_payment
            net_cash_flows.append(net_cash_flow)
            
            # Equity calculation
            equity = new_property_value - new_mortgage_balance
            equity_values.append(equity)
        
        # Calculate returns
        total_cash_invested = down_payment + sum([max(0, -cf) for cf in net_cash_flows])
        final_equity = equity_values[-1]
        total_return = (final_equity / total_cash_invested) - 1 if total_cash_invested > 0 else 0
        
        return {
            "source_type": "real_estate",
            "property_type": self.property_type,
            "location_tier": self.location_tier,
            "initial_property_value": property_value,
            "down_payment": down_payment,
            "leverage_ratio": self.leverage_ratio,
            "property_values": property_values,
            "mortgage_balances": mortgage_balances,
            "rental_income": rental_income,
            "net_cash_flows": net_cash_flows,
            "equity_values": equity_values,
            "final_property_value": property_values[-1],
            "final_equity": final_equity,
            "total_return": total_return,
            "cash_on_cash_return": sum(net_cash_flows) / down_payment if down_payment > 0 else 0,
            "total_appreciation": (property_values[-1] / property_values[0]) - 1
        }
    
    def _calculate_mortgage_payment(self, principal: float, annual_rate: float, years: int) -> float:
        """Calculate monthly mortgage payment"""
        monthly_rate = annual_rate / 12
        num_payments = years * 12
        
        if monthly_rate == 0:
            return principal / num_payments
        
        return principal * (monthly_rate * (1 + monthly_rate)**num_payments) / \
               ((1 + monthly_rate)**num_payments - 1)