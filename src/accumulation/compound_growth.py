"""
Compound Growth Models

Models for wealth accumulation through compound growth with various
scenarios, tax considerations, and inflation adjustments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

class GrowthType(Enum):
    """Types of growth patterns"""
    CONSTANT = "constant"
    VARIABLE = "variable"
    CYCLICAL = "cyclical"
    STOCHASTIC = "stochastic"

class TaxTreatment(Enum):
    """Tax treatment types"""
    TAXABLE = "taxable"
    TAX_DEFERRED = "tax_deferred"
    TAX_FREE = "tax_free"

@dataclass
class GrowthScenario:
    """Parameters for a compound growth scenario"""
    name: str
    initial_amount: float
    annual_return: float
    volatility: float = 0.0
    time_horizon: int = 30
    annual_contribution: float = 0.0
    contribution_growth: float = 0.0
    tax_rate: float = 0.0
    inflation_rate: float = 0.03
    tax_treatment: TaxTreatment = TaxTreatment.TAXABLE
    growth_type: GrowthType = GrowthType.CONSTANT

@dataclass
class GrowthResult:
    """Results from compound growth simulation"""
    scenario_name: str
    final_value: float
    real_value: float  # inflation-adjusted
    total_contributions: float
    total_growth: float
    annual_values: List[float]
    annual_real_values: List[float]
    annual_contributions: List[float]
    effective_annual_return: float
    years_to_double: float
    years_to_million: Optional[float]

class CompoundGrowthModel:
    """
    Comprehensive compound growth model for wealth accumulation
    """
    
    def __init__(self):
        self.scenarios = {}
        self.results = {}
    
    def add_scenario(self, scenario: GrowthScenario) -> None:
        """Add a growth scenario"""
        self.scenarios[scenario.name] = scenario
    
    def simulate_scenario(self, scenario_name: str) -> GrowthResult:
        """Simulate a specific growth scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenarios[scenario_name]
        
        if scenario.growth_type == GrowthType.CONSTANT:
            result = self._simulate_constant_growth(scenario)
        elif scenario.growth_type == GrowthType.VARIABLE:
            result = self._simulate_variable_growth(scenario)
        elif scenario.growth_type == GrowthType.CYCLICAL:
            result = self._simulate_cyclical_growth(scenario)
        else:  # STOCHASTIC
            result = self._simulate_stochastic_growth(scenario)
        
        self.results[scenario_name] = result
        return result
    
    def simulate_all_scenarios(self) -> Dict[str, GrowthResult]:
        """Simulate all added scenarios"""
        results = {}
        for scenario_name in self.scenarios:
            results[scenario_name] = self.simulate_scenario(scenario_name)
        return results
    
    def _simulate_constant_growth(self, scenario: GrowthScenario) -> GrowthResult:
        """Simulate constant growth rate"""
        values = [scenario.initial_amount]
        real_values = [scenario.initial_amount]
        contributions = [scenario.initial_amount]
        
        current_value = scenario.initial_amount
        total_contributions = scenario.initial_amount
        annual_contribution = scenario.annual_contribution
        
        for year in range(1, scenario.time_horizon + 1):
            # Apply growth
            growth = current_value * scenario.annual_return
            current_value += growth
            
            # Add contribution
            current_value += annual_contribution
            total_contributions += annual_contribution
            
            # Apply taxes if applicable
            if scenario.tax_treatment == TaxTreatment.TAXABLE:
                tax = growth * scenario.tax_rate
                current_value -= tax
            
            # Calculate real value (inflation-adjusted)
            real_value = current_value / ((1 + scenario.inflation_rate) ** year)
            
            values.append(current_value)
            real_values.append(real_value)
            contributions.append(annual_contribution)
            
            # Grow contribution
            annual_contribution *= (1 + scenario.contribution_growth)
        
        return self._create_result(scenario, values, real_values, contributions, total_contributions)
    
    def _simulate_variable_growth(self, scenario: GrowthScenario) -> GrowthResult:
        """Simulate variable growth rates"""
        # Create variable growth pattern
        growth_rates = self._generate_variable_rates(scenario)
        
        values = [scenario.initial_amount]
        real_values = [scenario.initial_amount]
        contributions = [scenario.initial_amount]
        
        current_value = scenario.initial_amount
        total_contributions = scenario.initial_amount
        annual_contribution = scenario.annual_contribution
        
        for year in range(1, scenario.time_horizon + 1):
            # Apply variable growth
            growth_rate = growth_rates[year - 1]
            growth = current_value * growth_rate
            current_value += growth
            
            # Add contribution
            current_value += annual_contribution
            total_contributions += annual_contribution
            
            # Apply taxes if applicable
            if scenario.tax_treatment == TaxTreatment.TAXABLE:
                tax = growth * scenario.tax_rate
                current_value -= tax
            
            # Calculate real value
            real_value = current_value / ((1 + scenario.inflation_rate) ** year)
            
            values.append(current_value)
            real_values.append(real_value)
            contributions.append(annual_contribution)
            
            annual_contribution *= (1 + scenario.contribution_growth)
        
        return self._create_result(scenario, values, real_values, contributions, total_contributions)
    
    def _simulate_cyclical_growth(self, scenario: GrowthScenario) -> GrowthResult:
        """Simulate cyclical growth patterns"""
        cycle_length = 7  # 7-year economic cycle
        
        values = [scenario.initial_amount]
        real_values = [scenario.initial_amount]
        contributions = [scenario.initial_amount]
        
        current_value = scenario.initial_amount
        total_contributions = scenario.initial_amount
        annual_contribution = scenario.annual_contribution
        
        for year in range(1, scenario.time_horizon + 1):
            # Calculate cyclical growth rate
            cycle_position = (year - 1) % cycle_length
            cycle_factor = math.sin(2 * math.pi * cycle_position / cycle_length)
            growth_rate = scenario.annual_return * (1 + 0.3 * cycle_factor)
            
            growth = current_value * growth_rate
            current_value += growth
            
            # Add contribution
            current_value += annual_contribution
            total_contributions += annual_contribution
            
            # Apply taxes
            if scenario.tax_treatment == TaxTreatment.TAXABLE:
                tax = growth * scenario.tax_rate
                current_value -= tax
            
            real_value = current_value / ((1 + scenario.inflation_rate) ** year)
            
            values.append(current_value)
            real_values.append(real_value)
            contributions.append(annual_contribution)
            
            annual_contribution *= (1 + scenario.contribution_growth)
        
        return self._create_result(scenario, values, real_values, contributions, total_contributions)
    
    def _simulate_stochastic_growth(self, scenario: GrowthScenario) -> GrowthResult:
        """Simulate stochastic growth with volatility"""
        np.random.seed(42)  # For reproducible results
        
        values = [scenario.initial_amount]
        real_values = [scenario.initial_amount]
        contributions = [scenario.initial_amount]
        
        current_value = scenario.initial_amount
        total_contributions = scenario.initial_amount
        annual_contribution = scenario.annual_contribution
        
        for year in range(1, scenario.time_horizon + 1):
            # Generate random return
            random_return = np.random.normal(scenario.annual_return, scenario.volatility)
            growth = current_value * random_return
            current_value += growth
            
            # Add contribution
            current_value += annual_contribution
            total_contributions += annual_contribution
            
            # Apply taxes
            if scenario.tax_treatment == TaxTreatment.TAXABLE and growth > 0:
                tax = growth * scenario.tax_rate
                current_value -= tax
            
            real_value = current_value / ((1 + scenario.inflation_rate) ** year)
            
            values.append(max(0, current_value))  # Prevent negative values
            real_values.append(max(0, real_value))
            contributions.append(annual_contribution)
            
            annual_contribution *= (1 + scenario.contribution_growth)
        
        return self._create_result(scenario, values, real_values, contributions, total_contributions)
    
    def _generate_variable_rates(self, scenario: GrowthScenario) -> List[float]:
        """Generate variable growth rates"""
        rates = []
        base_rate = scenario.annual_return
        
        for year in range(scenario.time_horizon):
            # Create some variation around the base rate
            variation = 0.3 * base_rate * math.sin(year * 0.5)
            rate = base_rate + variation
            rates.append(max(rate, -0.5))  # Prevent extreme negative returns
        
        return rates
    
    def _create_result(self, scenario: GrowthScenario, values: List[float], 
                      real_values: List[float], contributions: List[float],
                      total_contributions: float) -> GrowthResult:
        """Create growth result object"""
        final_value = values[-1]
        real_final_value = real_values[-1]
        total_growth = final_value - total_contributions
        
        # Calculate effective annual return
        if scenario.initial_amount > 0:
            effective_return = (final_value / scenario.initial_amount) ** (1 / scenario.time_horizon) - 1
        else:
            effective_return = 0.0
        
        # Calculate years to double
        if scenario.annual_return > 0:
            years_to_double = math.log(2) / math.log(1 + scenario.annual_return)
        else:
            years_to_double = float('inf')
        
        # Calculate years to reach $1 million
        years_to_million = None
        for i, value in enumerate(values):
            if value >= 1_000_000:
                years_to_million = i
                break
        
        return GrowthResult(
            scenario_name=scenario.name,
            final_value=final_value,
            real_value=real_final_value,
            total_contributions=total_contributions,
            total_growth=total_growth,
            annual_values=values,
            annual_real_values=real_values,
            annual_contributions=contributions,
            effective_annual_return=effective_return,
            years_to_double=years_to_double,
            years_to_million=years_to_million
        )
    
    def compare_scenarios(self, scenario_names: List[str]) -> pd.DataFrame:
        """Compare multiple scenarios"""
        comparison_data = []
        
        for name in scenario_names:
            if name in self.results:
                result = self.results[name]
                comparison_data.append({
                    'Scenario': name,
                    'Final Value': result.final_value,
                    'Real Value': result.real_value,
                    'Total Growth': result.total_growth,
                    'Effective Return': result.effective_annual_return,
                    'Years to Double': result.years_to_double,
                    'Years to Million': result.years_to_million
                })
        
        return pd.DataFrame(comparison_data)
    
    def calculate_rule_of_72(self, annual_return: float) -> float:
        """Calculate years to double using Rule of 72"""
        if annual_return <= 0:
            return float('inf')
        return 72 / (annual_return * 100)
    
    def calculate_future_value(self, present_value: float, rate: float, 
                             periods: int, payment: float = 0) -> float:
        """Calculate future value with optional periodic payments"""
        if rate == 0:
            return present_value + payment * periods
        
        fv_pv = present_value * (1 + rate) ** periods
        fv_pmt = payment * (((1 + rate) ** periods - 1) / rate)
        
        return fv_pv + fv_pmt
    
    def calculate_required_return(self, present_value: float, future_value: float,
                                periods: int) -> float:
        """Calculate required return to reach target"""
        if present_value <= 0 or future_value <= 0 or periods <= 0:
            return 0.0
        
        return (future_value / present_value) ** (1 / periods) - 1