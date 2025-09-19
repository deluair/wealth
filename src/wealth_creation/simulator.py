"""
Wealth Creation Simulator

Main simulation engine for modeling various pathways to wealth creation
using Monte Carlo methods and stochastic processes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

class WealthSource(Enum):
    BUSINESS = "business"
    INVESTMENT = "investment"
    EMPLOYMENT = "employment"
    INHERITANCE = "inheritance"
    REAL_ESTATE = "real_estate"
    ENTREPRENEURSHIP = "entrepreneurship"
    INNOVATION = "innovation"

@dataclass
class WealthCreationParams:
    """Parameters for wealth creation simulation"""
    initial_capital: float = 10000
    time_horizon: int = 30  # years
    risk_tolerance: float = 0.5  # 0-1 scale
    education_level: int = 3  # 1-5 scale
    network_strength: float = 0.5  # 0-1 scale
    market_conditions: float = 0.5  # 0-1 scale
    innovation_factor: float = 0.3  # 0-1 scale
    ai_adoption_rate: float = 0.2  # 0-1 scale

class WealthCreationSimulator:
    """
    Comprehensive wealth creation simulation engine
    """
    
    def __init__(self, params: Optional[WealthCreationParams] = None):
        self.params = params or WealthCreationParams()
        self.random_seed = 42
        np.random.seed(self.random_seed)
        
    def simulate_business_venture(self, years: int = 10) -> Dict:
        """Simulate wealth creation through business ventures"""
        np.random.seed(self.random_seed)
        
        # Business success probability based on various factors
        success_prob = (
            0.3 * self.params.education_level / 5 +
            0.3 * self.params.network_strength +
            0.2 * self.params.market_conditions +
            0.2 * self.params.innovation_factor
        )
        
        wealth_trajectory = [self.params.initial_capital]
        annual_returns = []
        
        for year in range(years):
            # Market volatility and business cycles
            market_shock = np.random.normal(0, 0.15)
            base_return = np.random.normal(0.12, 0.25)  # High risk, high reward
            
            # Success/failure events
            if np.random.random() < success_prob * 0.1:  # Major success
                multiplier = np.random.uniform(2, 10)
                annual_return = base_return * multiplier
            elif np.random.random() < 0.05:  # Business failure
                annual_return = -0.8  # Lose 80% of investment
            else:
                annual_return = base_return + market_shock
            
            # AI impact on business efficiency
            ai_boost = self.params.ai_adoption_rate * np.random.uniform(0.05, 0.15)
            annual_return += ai_boost
            
            new_wealth = wealth_trajectory[-1] * (1 + annual_return)
            wealth_trajectory.append(max(0, new_wealth))  # Can't go negative
            annual_returns.append(annual_return)
        
        return {
            'source': WealthSource.BUSINESS,
            'wealth_trajectory': wealth_trajectory,
            'annual_returns': annual_returns,
            'final_wealth': wealth_trajectory[-1],
            'total_return': (wealth_trajectory[-1] / wealth_trajectory[0]) - 1,
            'success_probability': success_prob
        }
    
    def simulate_investment_portfolio(self, years: int = 30) -> Dict:
        """Simulate wealth creation through investment portfolio"""
        np.random.seed(self.random_seed + 1)
        
        # Portfolio allocation based on risk tolerance
        stock_allocation = self.params.risk_tolerance
        bond_allocation = 1 - stock_allocation
        
        wealth_trajectory = [self.params.initial_capital]
        annual_returns = []
        
        for year in range(years):
            # Stock returns (higher volatility)
            stock_return = np.random.normal(0.10, 0.16)
            
            # Bond returns (lower volatility)
            bond_return = np.random.normal(0.04, 0.05)
            
            # Portfolio return
            portfolio_return = (stock_allocation * stock_return + 
                              bond_allocation * bond_return)
            
            # Market cycles and economic shocks
            if np.random.random() < 0.1:  # Market crash
                portfolio_return *= np.random.uniform(0.6, 0.8)
            elif np.random.random() < 0.05:  # Bull market
                portfolio_return *= np.random.uniform(1.3, 1.8)
            
            # AI-driven investment optimization
            ai_alpha = self.params.ai_adoption_rate * np.random.uniform(0.01, 0.03)
            portfolio_return += ai_alpha
            
            new_wealth = wealth_trajectory[-1] * (1 + portfolio_return)
            wealth_trajectory.append(new_wealth)
            annual_returns.append(portfolio_return)
        
        return {
            'source': WealthSource.INVESTMENT,
            'wealth_trajectory': wealth_trajectory,
            'annual_returns': annual_returns,
            'final_wealth': wealth_trajectory[-1],
            'total_return': (wealth_trajectory[-1] / wealth_trajectory[0]) - 1,
            'stock_allocation': stock_allocation,
            'bond_allocation': bond_allocation
        }
    
    def simulate_employment_income(self, years: int = 40) -> Dict:
        """Simulate wealth creation through employment and career progression"""
        np.random.seed(self.random_seed + 2)
        
        # Starting salary based on education
        base_salary = 30000 + (self.params.education_level * 15000)
        current_salary = base_salary
        
        wealth_trajectory = [self.params.initial_capital]
        annual_savings = []
        salaries = [current_salary]
        
        for year in range(years):
            # Career progression
            promotion_prob = (0.1 + 0.05 * self.params.network_strength + 
                            0.03 * self.params.education_level / 5)
            
            if np.random.random() < promotion_prob:
                salary_increase = np.random.uniform(0.15, 0.30)
                current_salary *= (1 + salary_increase)
            else:
                # Regular raise
                current_salary *= np.random.uniform(1.02, 1.05)
            
            # AI impact on salary (automation vs augmentation)
            ai_impact = self.params.ai_adoption_rate * np.random.uniform(-0.05, 0.10)
            current_salary *= (1 + ai_impact)
            
            # Savings rate based on income level
            savings_rate = min(0.4, 0.1 + (current_salary - base_salary) / base_salary * 0.1)
            annual_saving = current_salary * savings_rate
            
            # Investment of savings (conservative approach)
            investment_return = np.random.normal(0.06, 0.08)
            wealth_growth = wealth_trajectory[-1] * investment_return + annual_saving
            
            new_wealth = wealth_trajectory[-1] + wealth_growth
            wealth_trajectory.append(new_wealth)
            annual_savings.append(annual_saving)
            salaries.append(current_salary)
        
        return {
            'source': WealthSource.EMPLOYMENT,
            'wealth_trajectory': wealth_trajectory,
            'annual_savings': annual_savings,
            'salary_progression': salaries,
            'final_wealth': wealth_trajectory[-1],
            'final_salary': current_salary,
            'total_savings': sum(annual_savings)
        }
    
    def simulate_inheritance_wealth(self) -> Dict:
        """Simulate wealth creation through inheritance"""
        np.random.seed(self.random_seed + 3)
        
        # Inheritance probability and amount based on family background
        inheritance_prob = 0.15 + 0.1 * self.params.network_strength
        
        if np.random.random() < inheritance_prob:
            # Inheritance amount follows log-normal distribution
            inheritance_amount = np.random.lognormal(11, 1.5)  # Mean ~$150k
            inheritance_year = np.random.randint(10, 25)  # Receive between year 10-25
        else:
            inheritance_amount = 0
            inheritance_year = None
        
        return {
            'source': WealthSource.INHERITANCE,
            'inheritance_amount': inheritance_amount,
            'inheritance_year': inheritance_year,
            'received_inheritance': inheritance_amount > 0
        }
    
    def simulate_real_estate_investment(self, years: int = 25) -> Dict:
        """Simulate wealth creation through real estate investment"""
        np.random.seed(self.random_seed + 4)
        
        # Initial property investment
        property_value = self.params.initial_capital * 5  # Leverage
        down_payment = self.params.initial_capital
        mortgage = property_value - down_payment
        
        wealth_trajectory = [down_payment]  # Equity
        property_values = [property_value]
        rental_income = []
        
        for year in range(years):
            # Property appreciation
            appreciation_rate = np.random.normal(0.04, 0.06)
            property_value *= (1 + appreciation_rate)
            
            # Rental yield
            rental_yield = np.random.normal(0.06, 0.02)
            annual_rent = property_value * rental_yield
            
            # Mortgage payments (simplified)
            mortgage_payment = mortgage * 0.06  # 6% interest
            net_rental_income = annual_rent - mortgage_payment
            
            # Equity calculation
            mortgage *= 0.95  # Principal paydown
            equity = property_value - mortgage
            
            wealth_trajectory.append(equity)
            property_values.append(property_value)
            rental_income.append(net_rental_income)
        
        return {
            'source': WealthSource.REAL_ESTATE,
            'wealth_trajectory': wealth_trajectory,
            'property_values': property_values,
            'rental_income': rental_income,
            'final_wealth': wealth_trajectory[-1],
            'total_appreciation': property_values[-1] - property_values[0]
        }
    
    def run_comprehensive_simulation(self, num_simulations: int = 1000) -> pd.DataFrame:
        """Run comprehensive Monte Carlo simulation across all wealth sources"""
        results = []
        
        for sim in range(num_simulations):
            self.random_seed = sim  # Different seed for each simulation
            
            # Run all simulations
            business_result = self.simulate_business_venture()
            investment_result = self.simulate_investment_portfolio()
            employment_result = self.simulate_employment_income()
            inheritance_result = self.simulate_inheritance_wealth()
            real_estate_result = self.simulate_real_estate_investment()
            
            # Combine results
            total_wealth = (business_result['final_wealth'] + 
                          investment_result['final_wealth'] + 
                          employment_result['final_wealth'] + 
                          inheritance_result['inheritance_amount'] + 
                          real_estate_result['final_wealth'])
            
            results.append({
                'simulation_id': sim,
                'business_wealth': business_result['final_wealth'],
                'investment_wealth': investment_result['final_wealth'],
                'employment_wealth': employment_result['final_wealth'],
                'inheritance_wealth': inheritance_result['inheritance_amount'],
                'real_estate_wealth': real_estate_result['final_wealth'],
                'total_wealth': total_wealth,
                'primary_source': max([
                    ('business', business_result['final_wealth']),
                    ('investment', investment_result['final_wealth']),
                    ('employment', employment_result['final_wealth']),
                    ('inheritance', inheritance_result['inheritance_amount']),
                    ('real_estate', real_estate_result['final_wealth'])
                ], key=lambda x: x[1])[0]
            })
        
        return pd.DataFrame(results)
    
    def analyze_wealth_creation_factors(self) -> Dict:
        """Analyze key factors affecting wealth creation"""
        
        # Sensitivity analysis
        factors = {
            'education_impact': [],
            'network_impact': [],
            'risk_tolerance_impact': [],
            'ai_adoption_impact': []
        }
        
        base_params = WealthCreationParams()
        
        # Test education impact
        for education in range(1, 6):
            params = WealthCreationParams(education_level=education)
            simulator = WealthCreationSimulator(params)
            result = simulator.simulate_business_venture()
            factors['education_impact'].append(result['final_wealth'])
        
        # Test network impact
        for network in np.linspace(0, 1, 5):
            params = WealthCreationParams(network_strength=network)
            simulator = WealthCreationSimulator(params)
            result = simulator.simulate_business_venture()
            factors['network_impact'].append(result['final_wealth'])
        
        # Test risk tolerance impact
        for risk in np.linspace(0, 1, 5):
            params = WealthCreationParams(risk_tolerance=risk)
            simulator = WealthCreationSimulator(params)
            result = simulator.simulate_investment_portfolio()
            factors['risk_tolerance_impact'].append(result['final_wealth'])
        
        # Test AI adoption impact
        for ai_rate in np.linspace(0, 1, 5):
            params = WealthCreationParams(ai_adoption_rate=ai_rate)
            simulator = WealthCreationSimulator(params)
            result = simulator.simulate_business_venture()
            factors['ai_adoption_impact'].append(result['final_wealth'])
        
        return factors