"""
Lifecycle Wealth Accumulation Models

Models that adapt wealth accumulation strategies based on life stages,
age, goals, and changing financial circumstances over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

class LifeStage(Enum):
    """Life stages for wealth accumulation"""
    YOUNG_PROFESSIONAL = "young_professional"  # 22-30
    EARLY_CAREER = "early_career"              # 30-40
    MID_CAREER = "mid_career"                  # 40-50
    PRE_RETIREMENT = "pre_retirement"          # 50-65
    RETIREMENT = "retirement"                  # 65+

class AccumulationGoal(Enum):
    """Types of accumulation goals"""
    RETIREMENT = "retirement"
    HOME_PURCHASE = "home_purchase"
    EDUCATION = "education"
    EMERGENCY_FUND = "emergency_fund"
    WEALTH_BUILDING = "wealth_building"
    LEGACY = "legacy"

class RiskProfile(Enum):
    """Risk tolerance profiles"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"

@dataclass
class LifecycleParameters:
    """Parameters for lifecycle accumulation"""
    current_age: int
    retirement_age: int = 65
    life_expectancy: int = 85
    current_income: float = 50000
    income_growth_rate: float = 0.03
    savings_rate: float = 0.15
    risk_profile: RiskProfile = RiskProfile.MODERATE
    goals: List[AccumulationGoal] = None
    family_size: int = 1
    home_ownership: bool = False

@dataclass
class LifecycleResult:
    """Results from lifecycle accumulation model"""
    total_accumulation: float
    retirement_readiness: float
    goal_achievement: Dict[str, bool]
    stage_breakdown: Dict[str, Dict]
    recommended_adjustments: List[str]
    wealth_trajectory: List[Dict]
    risk_adjusted_returns: Dict[str, float]

class LifecycleAccumulator:
    """
    Comprehensive lifecycle wealth accumulation model
    """
    
    def __init__(self):
        self.parameters = None
        self.results = {}
        
        # Stage-specific parameters
        self.stage_parameters = {
            LifeStage.YOUNG_PROFESSIONAL: {
                'recommended_savings_rate': 0.20,
                'risk_tolerance': 0.9,
                'liquidity_needs': 0.1,
                'growth_focus': 0.8
            },
            LifeStage.EARLY_CAREER: {
                'recommended_savings_rate': 0.15,
                'risk_tolerance': 0.8,
                'liquidity_needs': 0.15,
                'growth_focus': 0.7
            },
            LifeStage.MID_CAREER: {
                'recommended_savings_rate': 0.18,
                'risk_tolerance': 0.6,
                'liquidity_needs': 0.2,
                'growth_focus': 0.6
            },
            LifeStage.PRE_RETIREMENT: {
                'recommended_savings_rate': 0.25,
                'risk_tolerance': 0.4,
                'liquidity_needs': 0.3,
                'growth_focus': 0.4
            },
            LifeStage.RETIREMENT: {
                'recommended_savings_rate': -0.04,  # Withdrawal phase
                'risk_tolerance': 0.3,
                'liquidity_needs': 0.4,
                'growth_focus': 0.3
            }
        }
        
        # Risk profile parameters
        self.risk_parameters = {
            RiskProfile.CONSERVATIVE: {
                'expected_return': 0.05,
                'volatility': 0.08,
                'equity_allocation': 0.3
            },
            RiskProfile.MODERATE: {
                'expected_return': 0.07,
                'volatility': 0.12,
                'equity_allocation': 0.6
            },
            RiskProfile.AGGRESSIVE: {
                'expected_return': 0.09,
                'volatility': 0.16,
                'equity_allocation': 0.8
            },
            RiskProfile.VERY_AGGRESSIVE: {
                'expected_return': 0.11,
                'volatility': 0.20,
                'equity_allocation': 0.95
            }
        }
    
    def set_parameters(self, params: LifecycleParameters) -> None:
        """Set lifecycle parameters"""
        self.parameters = params
        if params.goals is None:
            params.goals = [AccumulationGoal.RETIREMENT]
    
    def simulate_lifecycle(self, scenario_name: str = "base") -> LifecycleResult:
        """Simulate wealth accumulation over lifecycle"""
        if self.parameters is None:
            raise ValueError("Parameters must be set before simulation")
        
        wealth_trajectory = []
        stage_breakdown = {}
        goal_achievement = {}
        
        current_age = self.parameters.current_age
        current_wealth = 0
        current_income = self.parameters.current_income
        
        # Simulate year by year
        for age in range(current_age, self.parameters.life_expectancy + 1):
            life_stage = self._determine_life_stage(age)
            stage_params = self.stage_parameters[life_stage]
            
            # Calculate income
            years_from_start = age - current_age
            if age < self.parameters.retirement_age:
                income = current_income * ((1 + self.parameters.income_growth_rate) ** years_from_start)
            else:
                # Retirement income (simplified)
                income = current_wealth * 0.04  # 4% rule
            
            # Calculate savings/withdrawal
            if age < self.parameters.retirement_age:
                # Accumulation phase
                savings_rate = self._calculate_dynamic_savings_rate(age, life_stage)
                annual_savings = income * savings_rate
            else:
                # Withdrawal phase
                annual_savings = -income  # Withdrawing for living expenses
            
            # Calculate investment returns
            risk_adjusted_return = self._calculate_risk_adjusted_return(age, life_stage)
            investment_return = current_wealth * risk_adjusted_return
            
            # Update wealth
            current_wealth += annual_savings + investment_return
            current_wealth = max(0, current_wealth)  # Prevent negative wealth
            
            # Record trajectory
            wealth_trajectory.append({
                'age': age,
                'wealth': current_wealth,
                'income': income,
                'savings': annual_savings,
                'returns': investment_return,
                'life_stage': life_stage.value
            })
            
            # Update stage breakdown
            if life_stage not in stage_breakdown:
                stage_breakdown[life_stage] = {
                    'start_age': age,
                    'start_wealth': current_wealth,
                    'total_savings': 0,
                    'total_returns': 0
                }
            
            stage_breakdown[life_stage]['total_savings'] += annual_savings
            stage_breakdown[life_stage]['total_returns'] += investment_return
            stage_breakdown[life_stage]['end_wealth'] = current_wealth
        
        # Evaluate goal achievement
        goal_achievement = self._evaluate_goals(wealth_trajectory)
        
        # Calculate retirement readiness
        retirement_readiness = self._calculate_retirement_readiness(wealth_trajectory)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(wealth_trajectory, goal_achievement)
        
        # Calculate risk-adjusted returns by stage
        risk_adjusted_returns = self._calculate_stage_returns(stage_breakdown)
        
        result = LifecycleResult(
            total_accumulation=current_wealth,
            retirement_readiness=retirement_readiness,
            goal_achievement=goal_achievement,
            stage_breakdown={stage.value: data for stage, data in stage_breakdown.items()},
            recommended_adjustments=recommendations,
            wealth_trajectory=wealth_trajectory,
            risk_adjusted_returns=risk_adjusted_returns
        )
        
        self.results[scenario_name] = result
        return result
    
    def _determine_life_stage(self, age: int) -> LifeStage:
        """Determine life stage based on age"""
        if age < 30:
            return LifeStage.YOUNG_PROFESSIONAL
        elif age < 40:
            return LifeStage.EARLY_CAREER
        elif age < 50:
            return LifeStage.MID_CAREER
        elif age < self.parameters.retirement_age:
            return LifeStage.PRE_RETIREMENT
        else:
            return LifeStage.RETIREMENT
    
    def _calculate_dynamic_savings_rate(self, age: int, life_stage: LifeStage) -> float:
        """Calculate dynamic savings rate based on age and life stage"""
        base_rate = self.parameters.savings_rate
        recommended_rate = self.stage_parameters[life_stage]['recommended_savings_rate']
        
        # Blend base rate with recommended rate
        blend_factor = 0.7  # 70% weight to recommended rate
        dynamic_rate = base_rate * (1 - blend_factor) + recommended_rate * blend_factor
        
        # Adjust for family size
        family_adjustment = 1 - (self.parameters.family_size - 1) * 0.05
        dynamic_rate *= family_adjustment
        
        # Adjust for home ownership
        if self.parameters.home_ownership and life_stage in [LifeStage.EARLY_CAREER, LifeStage.MID_CAREER]:
            dynamic_rate *= 0.9  # Slightly lower due to mortgage payments
        
        return max(0.05, min(0.5, dynamic_rate))  # Bound between 5% and 50%
    
    def _calculate_risk_adjusted_return(self, age: int, life_stage: LifeStage) -> float:
        """Calculate risk-adjusted return based on age and risk profile"""
        base_risk_params = self.risk_parameters[self.parameters.risk_profile]
        stage_params = self.stage_parameters[life_stage]
        
        # Adjust expected return based on life stage risk tolerance
        risk_adjustment = stage_params['risk_tolerance']
        expected_return = base_risk_params['expected_return'] * risk_adjustment
        
        # Add some randomness (simplified)
        volatility = base_risk_params['volatility'] * risk_adjustment
        np.random.seed(age)  # Deterministic randomness based on age
        actual_return = np.random.normal(expected_return, volatility)
        
        return actual_return
    
    def _evaluate_goals(self, wealth_trajectory: List[Dict]) -> Dict[str, bool]:
        """Evaluate achievement of financial goals"""
        goal_achievement = {}
        
        for goal in self.parameters.goals:
            if goal == AccumulationGoal.RETIREMENT:
                # Need 10-12x final working income
                final_working_income = self.parameters.current_income * (
                    (1 + self.parameters.income_growth_rate) ** 
                    (self.parameters.retirement_age - self.parameters.current_age)
                )
                retirement_target = final_working_income * 10
                retirement_wealth = next(
                    (entry['wealth'] for entry in wealth_trajectory 
                     if entry['age'] == self.parameters.retirement_age), 0
                )
                goal_achievement[goal.value] = retirement_wealth >= retirement_target
                
            elif goal == AccumulationGoal.EMERGENCY_FUND:
                # Need 6 months of expenses
                emergency_target = self.parameters.current_income * 0.5  # 6 months
                early_wealth = next(
                    (entry['wealth'] for entry in wealth_trajectory 
                     if entry['age'] <= self.parameters.current_age + 5), 0
                )
                goal_achievement[goal.value] = early_wealth >= emergency_target
                
            elif goal == AccumulationGoal.HOME_PURCHASE:
                # Need 20% down payment (assume 4x income home price)
                home_price = self.parameters.current_income * 4
                down_payment = home_price * 0.2
                home_purchase_age = min(35, self.parameters.retirement_age - 10)
                wealth_at_purchase = next(
                    (entry['wealth'] for entry in wealth_trajectory 
                     if entry['age'] == home_purchase_age), 0
                )
                goal_achievement[goal.value] = wealth_at_purchase >= down_payment
                
            else:
                # Default to achieved for other goals
                goal_achievement[goal.value] = True
        
        return goal_achievement
    
    def _calculate_retirement_readiness(self, wealth_trajectory: List[Dict]) -> float:
        """Calculate retirement readiness score (0-1)"""
        retirement_entry = next(
            (entry for entry in wealth_trajectory 
             if entry['age'] == self.parameters.retirement_age), None
        )
        
        if retirement_entry is None:
            return 0.0
        
        retirement_wealth = retirement_entry['wealth']
        
        # Calculate target retirement wealth (10x final income)
        final_working_income = self.parameters.current_income * (
            (1 + self.parameters.income_growth_rate) ** 
            (self.parameters.retirement_age - self.parameters.current_age)
        )
        target_wealth = final_working_income * 10
        
        # Calculate readiness score
        readiness = min(1.0, retirement_wealth / target_wealth)
        return readiness
    
    def _generate_recommendations(self, wealth_trajectory: List[Dict], 
                                goal_achievement: Dict[str, bool]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Check retirement readiness
        retirement_wealth = next(
            (entry['wealth'] for entry in wealth_trajectory 
             if entry['age'] == self.parameters.retirement_age), 0
        )
        
        final_working_income = self.parameters.current_income * (
            (1 + self.parameters.income_growth_rate) ** 
            (self.parameters.retirement_age - self.parameters.current_age)
        )
        target_wealth = final_working_income * 10
        
        if retirement_wealth < target_wealth * 0.8:
            recommendations.append("Increase savings rate by 2-3% to improve retirement readiness")
        
        # Check emergency fund
        if AccumulationGoal.EMERGENCY_FUND in self.parameters.goals:
            if not goal_achievement.get('emergency_fund', False):
                recommendations.append("Build emergency fund of 6 months expenses before investing")
        
        # Check risk profile alignment
        current_stage = self._determine_life_stage(self.parameters.current_age)
        stage_risk_tolerance = self.stage_parameters[current_stage]['risk_tolerance']
        
        if self.parameters.risk_profile == RiskProfile.CONSERVATIVE and stage_risk_tolerance > 0.7:
            recommendations.append("Consider more aggressive investment approach given your age")
        elif self.parameters.risk_profile == RiskProfile.AGGRESSIVE and stage_risk_tolerance < 0.5:
            recommendations.append("Consider reducing investment risk as you approach retirement")
        
        # Check savings rate
        recommended_rate = self.stage_parameters[current_stage]['recommended_savings_rate']
        if self.parameters.savings_rate < recommended_rate * 0.8:
            recommendations.append(f"Increase savings rate to at least {recommended_rate:.1%}")
        
        # Home ownership recommendation
        if not self.parameters.home_ownership and self.parameters.current_age < 40:
            recommendations.append("Consider home ownership for wealth building and stability")
        
        return recommendations
    
    def _calculate_stage_returns(self, stage_breakdown: Dict) -> Dict[str, float]:
        """Calculate average returns by life stage"""
        stage_returns = {}
        
        for stage, data in stage_breakdown.items():
            if 'total_returns' in data and 'start_wealth' in data:
                avg_wealth = (data['start_wealth'] + data['end_wealth']) / 2
                if avg_wealth > 0:
                    stage_returns[stage.value] = data['total_returns'] / avg_wealth
                else:
                    stage_returns[stage.value] = 0.0
        
        return stage_returns
    
    def optimize_lifecycle_strategy(self) -> Dict[str, Any]:
        """Optimize lifecycle strategy through scenario analysis"""
        if self.parameters is None:
            raise ValueError("Parameters must be set before optimization")
        
        optimization_results = {}
        
        # Test different savings rates
        savings_rates = [0.10, 0.15, 0.20, 0.25, 0.30]
        original_rate = self.parameters.savings_rate
        
        for rate in savings_rates:
            self.parameters.savings_rate = rate
            result = self.simulate_lifecycle(f"savings_rate_{rate:.0%}")
            optimization_results[f"savings_rate_{rate:.0%}"] = {
                'final_wealth': result.total_accumulation,
                'retirement_readiness': result.retirement_readiness,
                'savings_rate': rate
            }
        
        # Restore original rate
        self.parameters.savings_rate = original_rate
        
        # Test different risk profiles
        risk_profiles = [RiskProfile.CONSERVATIVE, RiskProfile.MODERATE, 
                        RiskProfile.AGGRESSIVE, RiskProfile.VERY_AGGRESSIVE]
        original_risk = self.parameters.risk_profile
        
        for risk in risk_profiles:
            self.parameters.risk_profile = risk
            result = self.simulate_lifecycle(f"risk_{risk.value}")
            optimization_results[f"risk_{risk.value}"] = {
                'final_wealth': result.total_accumulation,
                'retirement_readiness': result.retirement_readiness,
                'risk_profile': risk.value
            }
        
        # Restore original risk profile
        self.parameters.risk_profile = original_risk
        
        # Find optimal combination
        best_scenario = max(optimization_results.items(), 
                          key=lambda x: x[1]['retirement_readiness'])
        
        return {
            'optimization_results': optimization_results,
            'best_scenario': best_scenario[0],
            'best_readiness': best_scenario[1]['retirement_readiness'],
            'recommendations': self._generate_optimization_recommendations(optimization_results)
        }
    
    def _generate_optimization_recommendations(self, optimization_results: Dict) -> List[str]:
        """Generate recommendations based on optimization results"""
        recommendations = []
        
        # Find best savings rate
        savings_results = {k: v for k, v in optimization_results.items() if 'savings_rate' in k}
        if savings_results:
            best_savings = max(savings_results.items(), key=lambda x: x[1]['retirement_readiness'])
            recommendations.append(f"Optimal savings rate: {best_savings[1]['savings_rate']:.0%}")
        
        # Find best risk profile
        risk_results = {k: v for k, v in optimization_results.items() if 'risk_' in k}
        if risk_results:
            best_risk = max(risk_results.items(), key=lambda x: x[1]['retirement_readiness'])
            recommendations.append(f"Optimal risk profile: {best_risk[1]['risk_profile']}")
        
        return recommendations
    
    def compare_scenarios(self, scenario_names: List[str]) -> pd.DataFrame:
        """Compare multiple lifecycle scenarios"""
        comparison_data = []
        
        for name in scenario_names:
            if name in self.results:
                result = self.results[name]
                comparison_data.append({
                    'Scenario': name,
                    'Final Wealth': result.total_accumulation,
                    'Retirement Readiness': result.retirement_readiness,
                    'Goals Achieved': sum(result.goal_achievement.values()),
                    'Total Goals': len(result.goal_achievement),
                    'Recommendations': len(result.recommended_adjustments)
                })
        
        return pd.DataFrame(comparison_data)
    
    def calculate_replacement_ratio(self, scenario_name: str = "base") -> float:
        """Calculate income replacement ratio in retirement"""
        if scenario_name not in self.results:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        result = self.results[scenario_name]
        
        # Find retirement wealth
        retirement_entry = next(
            (entry for entry in result.wealth_trajectory 
             if entry['age'] == self.parameters.retirement_age), None
        )
        
        if retirement_entry is None:
            return 0.0
        
        retirement_wealth = retirement_entry['wealth']
        
        # Calculate final working income
        final_working_income = self.parameters.current_income * (
            (1 + self.parameters.income_growth_rate) ** 
            (self.parameters.retirement_age - self.parameters.current_age)
        )
        
        # Calculate replacement ratio using 4% rule
        retirement_income = retirement_wealth * 0.04
        replacement_ratio = retirement_income / final_working_income
        
        return replacement_ratio