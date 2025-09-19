"""
Future Scenarios Analyzer

Classes for modeling different potential futures of AI's impact on wealth creation,
distribution, and economic structures, including optimistic, pessimistic, and
realistic scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
import matplotlib.pyplot as plt

class ScenarioType(Enum):
    """Types of future scenarios"""
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    REALISTIC = "realistic"
    DYSTOPIAN = "dystopian"
    UTOPIAN = "utopian"
    REGULATORY_HEAVY = "regulatory_heavy"
    LAISSEZ_FAIRE = "laissez_faire"
    TECHNOLOGICAL_SINGULARITY = "technological_singularity"

class AIProgressRate(Enum):
    """Rate of AI technological progress"""
    SLOW = "slow"
    MODERATE = "moderate"
    RAPID = "rapid"
    EXPONENTIAL = "exponential"
    STAGNANT = "stagnant"

class SocietalResponse(Enum):
    """How society responds to AI changes"""
    ADAPTIVE = "adaptive"
    RESISTANT = "resistant"
    PROACTIVE = "proactive"
    REACTIVE = "reactive"
    FRAGMENTED = "fragmented"

class WealthDistributionOutcome(Enum):
    """Possible wealth distribution outcomes"""
    EXTREME_CONCENTRATION = "extreme_concentration"
    MODERATE_CONCENTRATION = "moderate_concentration"
    STABLE_INEQUALITY = "stable_inequality"
    REDUCED_INEQUALITY = "reduced_inequality"
    WIDESPREAD_PROSPERITY = "widespread_prosperity"

@dataclass
class ScenarioParameters:
    """Parameters defining a future scenario"""
    name: str
    scenario_type: ScenarioType
    ai_progress_rate: AIProgressRate
    societal_response: SocietalResponse
    
    # Economic parameters
    ai_productivity_multiplier: float  # How much AI increases productivity
    job_displacement_rate: float  # Rate of job displacement
    new_job_creation_rate: float  # Rate of new job creation
    wealth_concentration_tendency: float  # Tendency for wealth to concentrate
    
    # Technology parameters
    ai_capability_growth_rate: float  # Annual growth in AI capabilities
    automation_penetration_rate: float  # Rate of automation adoption
    digital_divide_factor: float  # How much digital divide affects outcomes
    
    # Policy parameters
    ubi_implementation: float  # 0-1, extent of UBI implementation
    wealth_redistribution_rate: float  # Rate of wealth redistribution
    ai_regulation_strength: float  # 0-1, strength of AI regulation
    education_adaptation_rate: float  # How quickly education adapts
    
    # Social parameters
    social_cohesion_level: float  # 0-1, level of social cohesion
    innovation_culture_strength: float  # 0-1, strength of innovation culture
    inequality_tolerance: float  # 0-1, society's tolerance for inequality
    
    # Time parameters
    transition_speed: float  # How quickly changes occur
    adaptation_lag: int  # Years of lag in adaptation

@dataclass
class ScenarioOutcome:
    """Outcome of a scenario simulation"""
    scenario_name: str
    final_year: int
    
    # Economic outcomes
    total_wealth_created: float
    wealth_distribution_gini: float
    unemployment_rate: float
    median_income: float
    gdp_growth_rate: float
    
    # Social outcomes
    social_stability_index: float
    quality_of_life_index: float
    innovation_index: float
    education_effectiveness: float
    
    # Technology outcomes
    ai_adoption_level: float
    automation_level: float
    digital_inclusion_rate: float
    
    # Inequality outcomes
    top_1_percent_wealth_share: float
    bottom_50_percent_wealth_share: float
    intergenerational_mobility: float
    
    # Risk factors
    systemic_risk_level: float
    social_unrest_probability: float
    economic_instability_risk: float

class FutureScenarioAnalyzer:
    """Analyze different future scenarios for AI's impact on wealth"""
    
    def __init__(self):
        self.scenarios = {}
        self.simulation_results = {}
        
        # Base parameters for comparison
        self.baseline_parameters = {
            'current_gini': 0.85,  # Current wealth inequality
            'current_unemployment': 0.05,  # 5% unemployment
            'current_median_income': 50000,  # $50k median income
            'current_gdp_growth': 0.02,  # 2% GDP growth
            'current_ai_adoption': 0.1,  # 10% AI adoption
            'current_automation': 0.2,  # 20% automation level
        }
    
    def add_scenario(self, scenario: ScenarioParameters) -> None:
        """Add a scenario for analysis"""
        self.scenarios[scenario.name] = scenario
    
    def simulate_scenario(self, scenario_name: str, time_horizon: int = 20,
                         random_seed: int = 42) -> ScenarioOutcome:
        """
        Simulate a specific scenario
        
        Args:
            scenario_name: Name of scenario to simulate
            time_horizon: Number of years to simulate
            random_seed: Random seed for reproducibility
            
        Returns:
            ScenarioOutcome with simulation results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        np.random.seed(random_seed)
        scenario = self.scenarios[scenario_name]
        
        # Initialize state variables
        state = self._initialize_scenario_state(scenario)
        
        # Simulate year by year
        yearly_results = []
        for year in range(time_horizon):
            year_result = self._simulate_scenario_year(scenario, state, year)
            yearly_results.append(year_result)
            
            # Update state for next year
            self._update_scenario_state(scenario, state, year_result, year)
        
        # Calculate final outcome
        outcome = self._calculate_scenario_outcome(scenario, yearly_results, time_horizon)
        
        # Store results
        self.simulation_results[scenario_name] = {
            'outcome': outcome,
            'yearly_results': yearly_results,
            'final_state': state
        }
        
        return outcome
    
    def _initialize_scenario_state(self, scenario: ScenarioParameters) -> Dict:
        """Initialize the state for scenario simulation"""
        return {
            # Economic state
            'total_wealth': 100000000000000,  # $100 trillion global wealth
            'wealth_distribution': np.random.lognormal(10, 2, 10000),  # Initial distribution
            'unemployment_rate': self.baseline_parameters['current_unemployment'],
            'median_income': self.baseline_parameters['current_median_income'],
            'gdp': 20000000000000,  # $20 trillion GDP
            
            # Technology state
            'ai_capability_level': 0.1,  # 10% of theoretical maximum
            'automation_level': self.baseline_parameters['current_automation'],
            'ai_adoption_rate': self.baseline_parameters['current_ai_adoption'],
            
            # Social state
            'social_cohesion': scenario.social_cohesion_level,
            'education_quality': 0.6,  # 60% effectiveness
            'innovation_culture': scenario.innovation_culture_strength,
            
            # Policy state
            'ubi_level': 0,  # No UBI initially
            'redistribution_rate': 0.3,  # 30% redistribution through taxes
            'regulation_level': 0.5,  # Moderate regulation
            
            # Population segments
            'high_skill_workers': 0.2,  # 20% high-skill
            'medium_skill_workers': 0.6,  # 60% medium-skill
            'low_skill_workers': 0.2,   # 20% low-skill
            
            # AI ownership concentration
            'ai_ownership_concentration': 0.9,  # 90% owned by top 1%
        }
    
    def _simulate_scenario_year(self, scenario: ScenarioParameters, 
                               state: Dict, year: int) -> Dict:
        """Simulate one year of the scenario"""
        
        # AI capability growth
        ai_growth = self._simulate_ai_progress(scenario, state, year)
        
        # Economic impacts
        economic_impacts = self._simulate_economic_impacts(scenario, state, year)
        
        # Labor market changes
        labor_changes = self._simulate_labor_market_changes(scenario, state, year)
        
        # Wealth distribution changes
        wealth_changes = self._simulate_wealth_distribution_changes(scenario, state, year)
        
        # Social and political responses
        social_responses = self._simulate_social_responses(scenario, state, year)
        
        # Policy interventions
        policy_interventions = self._simulate_policy_interventions(scenario, state, year)
        
        return {
            'year': year,
            'ai_growth': ai_growth,
            'economic_impacts': economic_impacts,
            'labor_changes': labor_changes,
            'wealth_changes': wealth_changes,
            'social_responses': social_responses,
            'policy_interventions': policy_interventions
        }
    
    def _simulate_ai_progress(self, scenario: ScenarioParameters, 
                             state: Dict, year: int) -> Dict:
        """Simulate AI technological progress"""
        
        # Base growth rate based on scenario
        growth_rates = {
            AIProgressRate.STAGNANT: 0.01,
            AIProgressRate.SLOW: 0.05,
            AIProgressRate.MODERATE: 0.15,
            AIProgressRate.RAPID: 0.30,
            AIProgressRate.EXPONENTIAL: 0.50
        }
        
        base_growth = growth_rates[scenario.ai_progress_rate]
        
        # Adjust for current capability level (diminishing returns)
        capability_factor = 1 - state['ai_capability_level'] * 0.5
        
        # Adjust for investment and innovation culture
        investment_factor = 1 + state['innovation_culture'] * 0.5
        
        # Adjust for regulation (can slow or accelerate depending on type)
        regulation_factor = 1 - scenario.ai_regulation_strength * 0.2
        
        # Random variation
        random_factor = np.random.uniform(0.8, 1.2)
        
        actual_growth = base_growth * capability_factor * investment_factor * regulation_factor * random_factor
        
        # Update AI capability
        new_capability = min(1.0, state['ai_capability_level'] * (1 + actual_growth))
        capability_increase = new_capability - state['ai_capability_level']
        
        # AI adoption follows capability with lag
        adoption_growth = capability_increase * 0.5  # Adoption lags capability
        new_adoption = min(1.0, state['ai_adoption_rate'] + adoption_growth)
        
        return {
            'capability_growth': actual_growth,
            'new_capability_level': new_capability,
            'capability_increase': capability_increase,
            'adoption_growth': adoption_growth,
            'new_adoption_rate': new_adoption
        }
    
    def _simulate_economic_impacts(self, scenario: ScenarioParameters,
                                  state: Dict, year: int) -> Dict:
        """Simulate economic impacts of AI progress"""
        
        # Productivity gains from AI
        ai_productivity_gain = (state['ai_adoption_rate'] * 
                               scenario.ai_productivity_multiplier * 
                               state['ai_capability_level'])
        
        # GDP growth from productivity
        productivity_gdp_growth = ai_productivity_gain * 0.5  # 50% of productivity gain translates to GDP
        
        # New wealth creation from AI
        ai_wealth_creation = state['gdp'] * ai_productivity_gain * 0.1  # 10% of productivity gain becomes new wealth
        
        # Automation cost savings
        automation_savings = (state['automation_level'] * 
                             state['gdp'] * 0.3 *  # 30% of GDP is labor costs
                             0.2)  # 20% cost reduction from automation
        
        # Innovation spillovers
        innovation_multiplier = 1 + state['innovation_culture'] * 0.3
        total_wealth_creation = (ai_wealth_creation + automation_savings) * innovation_multiplier
        
        # Market concentration effects
        concentration_factor = scenario.wealth_concentration_tendency
        concentrated_wealth = total_wealth_creation * concentration_factor
        distributed_wealth = total_wealth_creation * (1 - concentration_factor)
        
        return {
            'ai_productivity_gain': ai_productivity_gain,
            'productivity_gdp_growth': productivity_gdp_growth,
            'ai_wealth_creation': ai_wealth_creation,
            'automation_savings': automation_savings,
            'total_wealth_creation': total_wealth_creation,
            'concentrated_wealth': concentrated_wealth,
            'distributed_wealth': distributed_wealth
        }
    
    def _simulate_labor_market_changes(self, scenario: ScenarioParameters,
                                      state: Dict, year: int) -> Dict:
        """Simulate changes in labor markets"""
        
        # Job displacement from automation
        automation_displacement = (scenario.job_displacement_rate * 
                                  state['automation_level'] * 
                                  (1 - state['unemployment_rate']))  # Can't displace unemployed
        
        # Differential impact by skill level
        low_skill_displacement = automation_displacement * 2.0  # Low-skill jobs more vulnerable
        medium_skill_displacement = automation_displacement * 1.0
        high_skill_displacement = automation_displacement * 0.3  # High-skill jobs less vulnerable
        
        # New job creation from AI and new industries
        ai_job_creation = (scenario.new_job_creation_rate * 
                          state['ai_adoption_rate'] * 
                          state['innovation_culture'])
        
        # Skill premium changes
        skill_premium_increase = state['ai_capability_level'] * 0.2  # AI increases skill premium
        
        # Retraining and adaptation
        adaptation_rate = (scenario.education_adaptation_rate * 
                          state['education_quality'] * 
                          (1 - scenario.adaptation_lag / 10))  # Lag reduces effectiveness
        
        # Net employment change
        total_displacement = (low_skill_displacement * state['low_skill_workers'] +
                             medium_skill_displacement * state['medium_skill_workers'] +
                             high_skill_displacement * state['high_skill_workers'])
        
        net_employment_change = ai_job_creation - total_displacement
        
        # Update unemployment
        new_unemployment = max(0, min(0.5, state['unemployment_rate'] - net_employment_change))
        
        return {
            'automation_displacement': automation_displacement,
            'low_skill_displacement': low_skill_displacement,
            'medium_skill_displacement': medium_skill_displacement,
            'high_skill_displacement': high_skill_displacement,
            'ai_job_creation': ai_job_creation,
            'skill_premium_increase': skill_premium_increase,
            'adaptation_rate': adaptation_rate,
            'total_displacement': total_displacement,
            'net_employment_change': net_employment_change,
            'new_unemployment_rate': new_unemployment
        }
    
    def _simulate_wealth_distribution_changes(self, scenario: ScenarioParameters,
                                            state: Dict, year: int) -> Dict:
        """Simulate changes in wealth distribution"""
        
        # AI ownership concentration effect
        ai_wealth_share_top1 = state['ai_ownership_concentration']
        
        # Returns to AI capital vs human labor
        ai_capital_returns = 0.15 + state['ai_capability_level'] * 0.10  # 15-25% returns
        human_labor_returns = 0.03 - state['automation_level'] * 0.02   # Decreasing returns to labor
        
        # Wealth concentration from AI returns
        ai_concentration_effect = (ai_capital_returns - human_labor_returns) * ai_wealth_share_top1
        
        # Network effects and winner-take-all dynamics
        network_concentration = state['ai_adoption_rate'] * 0.1  # Network effects increase concentration
        
        # Policy redistribution effects
        redistribution_effect = (scenario.wealth_redistribution_rate * 
                               state['redistribution_rate'] * 
                               -0.05)  # Negative because it reduces concentration
        
        # UBI effects
        ubi_effect = scenario.ubi_implementation * -0.03  # UBI reduces inequality
        
        # Total concentration change
        total_concentration_change = (ai_concentration_effect + 
                                    network_concentration + 
                                    redistribution_effect + 
                                    ubi_effect)
        
        # Update wealth distribution
        current_gini = self._calculate_gini(state['wealth_distribution'])
        new_gini = max(0.2, min(0.95, current_gini + total_concentration_change))
        
        # Calculate wealth shares
        top_1_percent_share = self._calculate_top_percentile_share(state['wealth_distribution'], 0.01)
        bottom_50_percent_share = self._calculate_bottom_percentile_share(state['wealth_distribution'], 0.5)
        
        return {
            'ai_concentration_effect': ai_concentration_effect,
            'network_concentration': network_concentration,
            'redistribution_effect': redistribution_effect,
            'ubi_effect': ubi_effect,
            'total_concentration_change': total_concentration_change,
            'new_gini': new_gini,
            'top_1_percent_share': top_1_percent_share,
            'bottom_50_percent_share': bottom_50_percent_share
        }
    
    def _simulate_social_responses(self, scenario: ScenarioParameters,
                                  state: Dict, year: int) -> Dict:
        """Simulate social and political responses to changes"""
        
        # Social tension from inequality
        inequality_tension = max(0, (self._calculate_gini(state['wealth_distribution']) - 
                                   scenario.inequality_tolerance) * 2)
        
        # Unemployment tension
        unemployment_tension = max(0, (state['unemployment_rate'] - 0.08) * 5)  # Tension above 8% unemployment
        
        # Adaptation stress
        adaptation_stress = (scenario.transition_speed * 
                           (1 - state['education_quality']) * 
                           (1 - state['social_cohesion']))
        
        # Total social stress
        total_social_stress = inequality_tension + unemployment_tension + adaptation_stress
        
        # Social cohesion response
        cohesion_change = -total_social_stress * 0.1  # Stress reduces cohesion
        if scenario.societal_response == SocietalResponse.ADAPTIVE:
            cohesion_change *= 0.5  # Adaptive societies maintain cohesion better
        elif scenario.societal_response == SocietalResponse.RESISTANT:
            cohesion_change *= 1.5  # Resistant societies lose cohesion faster
        
        new_social_cohesion = max(0.1, min(1.0, state['social_cohesion'] + cohesion_change))
        
        # Political pressure for intervention
        intervention_pressure = total_social_stress * 0.3
        
        # Probability of social unrest
        unrest_probability = min(0.8, total_social_stress * 0.2)
        
        return {
            'inequality_tension': inequality_tension,
            'unemployment_tension': unemployment_tension,
            'adaptation_stress': adaptation_stress,
            'total_social_stress': total_social_stress,
            'cohesion_change': cohesion_change,
            'new_social_cohesion': new_social_cohesion,
            'intervention_pressure': intervention_pressure,
            'unrest_probability': unrest_probability
        }
    
    def _simulate_policy_interventions(self, scenario: ScenarioParameters,
                                     state: Dict, year: int) -> Dict:
        """Simulate policy interventions in response to changes"""
        
        # UBI implementation pressure
        ubi_pressure = (state['unemployment_rate'] * 2 + 
                       self._calculate_gini(state['wealth_distribution']) - 0.5)
        
        # UBI implementation
        if scenario.ubi_implementation > 0:
            ubi_growth = min(0.1, ubi_pressure * 0.05)  # Gradual implementation
            new_ubi_level = min(scenario.ubi_implementation, state['ubi_level'] + ubi_growth)
        else:
            new_ubi_level = state['ubi_level']
        
        # Education system adaptation
        education_pressure = (state['unemployment_rate'] + 
                            (1 - state['ai_adoption_rate']) * 0.5)  # Pressure from unemployment and AI gap
        
        education_improvement = (education_pressure * 
                               scenario.education_adaptation_rate * 
                               0.02)  # 2% max improvement per year
        
        new_education_quality = min(1.0, state['education_quality'] + education_improvement)
        
        # Regulation changes
        regulation_pressure = (state['ai_capability_level'] * 0.5 + 
                             self._calculate_gini(state['wealth_distribution']) * 0.3)
        
        if scenario.ai_regulation_strength > state['regulation_level']:
            regulation_increase = min(0.05, regulation_pressure * 0.02)
            new_regulation_level = min(scenario.ai_regulation_strength, 
                                     state['regulation_level'] + regulation_increase)
        else:
            new_regulation_level = state['regulation_level']
        
        # Redistribution policy changes
        redistribution_pressure = self._calculate_gini(state['wealth_distribution']) - 0.6
        
        if redistribution_pressure > 0:
            redistribution_increase = min(0.02, redistribution_pressure * 0.1)
            new_redistribution_rate = min(0.7, state['redistribution_rate'] + redistribution_increase)
        else:
            new_redistribution_rate = state['redistribution_rate']
        
        return {
            'ubi_pressure': ubi_pressure,
            'new_ubi_level': new_ubi_level,
            'education_pressure': education_pressure,
            'education_improvement': education_improvement,
            'new_education_quality': new_education_quality,
            'regulation_pressure': regulation_pressure,
            'new_regulation_level': new_regulation_level,
            'redistribution_pressure': redistribution_pressure,
            'new_redistribution_rate': new_redistribution_rate
        }
    
    def _update_scenario_state(self, scenario: ScenarioParameters, state: Dict,
                              year_result: Dict, year: int) -> None:
        """Update scenario state based on year results"""
        
        # Update AI progress
        ai_growth = year_result['ai_growth']
        state['ai_capability_level'] = ai_growth['new_capability_level']
        state['ai_adoption_rate'] = ai_growth['new_adoption_rate']
        
        # Update economic state
        economic_impacts = year_result['economic_impacts']
        state['gdp'] *= (1 + economic_impacts['productivity_gdp_growth'])
        state['total_wealth'] += economic_impacts['total_wealth_creation']
        
        # Update labor market
        labor_changes = year_result['labor_changes']
        state['unemployment_rate'] = labor_changes['new_unemployment_rate']
        
        # Update wealth distribution
        wealth_changes = year_result['wealth_changes']
        state['wealth_distribution'] = self._update_wealth_distribution(
            state['wealth_distribution'], wealth_changes['new_gini']
        )
        
        # Update social state
        social_responses = year_result['social_responses']
        state['social_cohesion'] = social_responses['new_social_cohesion']
        
        # Update policy state
        policy_interventions = year_result['policy_interventions']
        state['ubi_level'] = policy_interventions['new_ubi_level']
        state['education_quality'] = policy_interventions['new_education_quality']
        state['regulation_level'] = policy_interventions['new_regulation_level']
        state['redistribution_rate'] = policy_interventions['new_redistribution_rate']
        
        # Update automation level
        automation_growth = state['ai_capability_level'] * scenario.automation_penetration_rate * 0.1
        state['automation_level'] = min(0.9, state['automation_level'] + automation_growth)
    
    def _calculate_scenario_outcome(self, scenario: ScenarioParameters,
                                   yearly_results: List[Dict], 
                                   time_horizon: int) -> ScenarioOutcome:
        """Calculate final outcome of scenario simulation"""
        
        final_year_result = yearly_results[-1]
        
        # Extract final values
        final_state = self.simulation_results.get(scenario.name, {}).get('final_state', {})
        
        # Economic outcomes
        total_wealth_created = sum(yr['economic_impacts']['total_wealth_creation'] 
                                 for yr in yearly_results)
        wealth_distribution_gini = final_year_result['wealth_changes']['new_gini']
        unemployment_rate = final_year_result['labor_changes']['new_unemployment_rate']
        median_income = self.baseline_parameters['current_median_income'] * 1.5  # Simplified
        gdp_growth_rate = np.mean([yr['economic_impacts']['productivity_gdp_growth'] 
                                  for yr in yearly_results])
        
        # Social outcomes
        social_stability_index = 1 - np.mean([yr['social_responses']['total_social_stress'] 
                                            for yr in yearly_results])
        quality_of_life_index = (1 - unemployment_rate) * 0.4 + (1 - wealth_distribution_gini) * 0.6
        innovation_index = final_state.get('innovation_culture', 0.5)
        education_effectiveness = final_state.get('education_quality', 0.6)
        
        # Technology outcomes
        ai_adoption_level = final_state.get('ai_adoption_rate', 0.1)
        automation_level = final_state.get('automation_level', 0.2)
        digital_inclusion_rate = 1 - scenario.digital_divide_factor  # Simplified
        
        # Inequality outcomes
        top_1_percent_wealth_share = final_year_result['wealth_changes']['top_1_percent_share']
        bottom_50_percent_wealth_share = final_year_result['wealth_changes']['bottom_50_percent_share']
        intergenerational_mobility = max(0.1, 1 - wealth_distribution_gini)  # Simplified relationship
        
        # Risk factors
        systemic_risk_level = wealth_distribution_gini * 0.5 + unemployment_rate * 0.3
        social_unrest_probability = np.mean([yr['social_responses']['unrest_probability'] 
                                           for yr in yearly_results])
        economic_instability_risk = max(0, unemployment_rate - 0.1) + max(0, wealth_distribution_gini - 0.8)
        
        return ScenarioOutcome(
            scenario_name=scenario.name,
            final_year=time_horizon - 1,
            total_wealth_created=total_wealth_created,
            wealth_distribution_gini=wealth_distribution_gini,
            unemployment_rate=unemployment_rate,
            median_income=median_income,
            gdp_growth_rate=gdp_growth_rate,
            social_stability_index=max(0, min(1, social_stability_index)),
            quality_of_life_index=max(0, min(1, quality_of_life_index)),
            innovation_index=innovation_index,
            education_effectiveness=education_effectiveness,
            ai_adoption_level=ai_adoption_level,
            automation_level=automation_level,
            digital_inclusion_rate=digital_inclusion_rate,
            top_1_percent_wealth_share=top_1_percent_wealth_share,
            bottom_50_percent_wealth_share=bottom_50_percent_wealth_share,
            intergenerational_mobility=intergenerational_mobility,
            systemic_risk_level=max(0, min(1, systemic_risk_level)),
            social_unrest_probability=max(0, min(1, social_unrest_probability)),
            economic_instability_risk=max(0, min(1, economic_instability_risk))
        )
    
    def compare_scenarios(self, scenario_names: List[str]) -> Dict:
        """Compare multiple scenarios"""
        if not all(name in self.simulation_results for name in scenario_names):
            missing = [name for name in scenario_names if name not in self.simulation_results]
            raise ValueError(f"Scenarios not simulated: {missing}")
        
        comparison = {
            'scenarios': scenario_names,
            'outcomes': {},
            'rankings': {},
            'trade_offs': {}
        }
        
        # Collect outcomes
        outcomes = {}
        for name in scenario_names:
            outcomes[name] = self.simulation_results[name]['outcome']
        
        comparison['outcomes'] = outcomes
        
        # Rank scenarios by different criteria
        criteria = [
            'total_wealth_created', 'wealth_distribution_gini', 'unemployment_rate',
            'social_stability_index', 'quality_of_life_index', 'innovation_index'
        ]
        
        rankings = {}
        for criterion in criteria:
            values = [(name, getattr(outcomes[name], criterion)) for name in scenario_names]
            
            # Sort based on criterion (lower is better for gini, unemployment, risk)
            if criterion in ['wealth_distribution_gini', 'unemployment_rate', 'systemic_risk_level']:
                values.sort(key=lambda x: x[1])  # Lower is better
            else:
                values.sort(key=lambda x: x[1], reverse=True)  # Higher is better
            
            rankings[criterion] = [name for name, _ in values]
        
        comparison['rankings'] = rankings
        
        # Analyze trade-offs
        trade_offs = self._analyze_scenario_tradeoffs(outcomes)
        comparison['trade_offs'] = trade_offs
        
        return comparison
    
    def _analyze_scenario_tradeoffs(self, outcomes: Dict[str, ScenarioOutcome]) -> Dict:
        """Analyze trade-offs between different outcomes"""
        trade_offs = {}
        
        # Wealth creation vs inequality trade-off
        wealth_inequality_tradeoff = {}
        for name, outcome in outcomes.items():
            wealth_inequality_tradeoff[name] = {
                'wealth_created': outcome.total_wealth_created,
                'inequality': outcome.wealth_distribution_gini,
                'efficiency_equity_ratio': outcome.total_wealth_created / max(0.1, outcome.wealth_distribution_gini)
            }
        
        trade_offs['wealth_inequality'] = wealth_inequality_tradeoff
        
        # Innovation vs stability trade-off
        innovation_stability_tradeoff = {}
        for name, outcome in outcomes.items():
            innovation_stability_tradeoff[name] = {
                'innovation': outcome.innovation_index,
                'stability': outcome.social_stability_index,
                'innovation_stability_balance': (outcome.innovation_index + outcome.social_stability_index) / 2
            }
        
        trade_offs['innovation_stability'] = innovation_stability_tradeoff
        
        # Employment vs automation trade-off
        employment_automation_tradeoff = {}
        for name, outcome in outcomes.items():
            employment_automation_tradeoff[name] = {
                'employment_rate': 1 - outcome.unemployment_rate,
                'automation_level': outcome.automation_level,
                'human_machine_balance': (1 - outcome.unemployment_rate) * (1 - outcome.automation_level)
            }
        
        trade_offs['employment_automation'] = employment_automation_tradeoff
        
        return trade_offs
    
    # Helper methods
    def _calculate_gini(self, wealth_distribution: np.ndarray) -> float:
        """Calculate Gini coefficient"""
        if len(wealth_distribution) == 0:
            return 0
        
        sorted_wealth = np.sort(wealth_distribution)
        n = len(sorted_wealth)
        
        if np.sum(sorted_wealth) == 0:
            return 0
        
        index_sum = np.sum((np.arange(1, n + 1) * sorted_wealth))
        total_sum = np.sum(sorted_wealth)
        
        gini = (2 * index_sum) / (n * total_sum) - (n + 1) / n
        return max(0, min(1, gini))
    
    def _calculate_top_percentile_share(self, wealth_distribution: np.ndarray, 
                                       percentile: float) -> float:
        """Calculate wealth share of top percentile"""
        if len(wealth_distribution) == 0:
            return 0
        
        sorted_wealth = np.sort(wealth_distribution)[::-1]  # Descending order
        top_count = max(1, int(len(sorted_wealth) * percentile))
        top_wealth = np.sum(sorted_wealth[:top_count])
        total_wealth = np.sum(sorted_wealth)
        
        return top_wealth / max(1, total_wealth)
    
    def _calculate_bottom_percentile_share(self, wealth_distribution: np.ndarray,
                                         percentile: float) -> float:
        """Calculate wealth share of bottom percentile"""
        if len(wealth_distribution) == 0:
            return 0
        
        sorted_wealth = np.sort(wealth_distribution)  # Ascending order
        bottom_count = max(1, int(len(sorted_wealth) * percentile))
        bottom_wealth = np.sum(sorted_wealth[:bottom_count])
        total_wealth = np.sum(sorted_wealth)
        
        return bottom_wealth / max(1, total_wealth)
    
    def _update_wealth_distribution(self, current_distribution: np.ndarray,
                                   target_gini: float) -> np.ndarray:
        """Update wealth distribution to match target Gini coefficient"""
        # Simplified approach: adjust distribution to match target Gini
        current_gini = self._calculate_gini(current_distribution)
        
        if abs(current_gini - target_gini) < 0.01:
            return current_distribution
        
        # Generate new distribution with target Gini (simplified)
        # This is a placeholder - in practice, you'd use more sophisticated methods
        if target_gini > current_gini:
            # Increase inequality
            factor = 1 + (target_gini - current_gini)
            new_distribution = current_distribution * np.random.lognormal(0, factor, len(current_distribution))
        else:
            # Decrease inequality
            factor = 1 - (current_gini - target_gini)
            new_distribution = current_distribution * np.random.uniform(factor, 1/factor, len(current_distribution))
        
        return new_distribution
    
    def create_predefined_scenarios(self) -> List[ScenarioParameters]:
        """Create a set of predefined scenarios for analysis"""
        scenarios = []
        
        # Optimistic scenario
        optimistic = ScenarioParameters(
            name="AI_Optimistic",
            scenario_type=ScenarioType.OPTIMISTIC,
            ai_progress_rate=AIProgressRate.RAPID,
            societal_response=SocietalResponse.ADAPTIVE,
            ai_productivity_multiplier=2.5,
            job_displacement_rate=0.03,
            new_job_creation_rate=0.05,
            wealth_concentration_tendency=0.6,
            ai_capability_growth_rate=0.25,
            automation_penetration_rate=0.8,
            digital_divide_factor=0.2,
            ubi_implementation=0.7,
            wealth_redistribution_rate=0.4,
            ai_regulation_strength=0.6,
            education_adaptation_rate=0.8,
            social_cohesion_level=0.8,
            innovation_culture_strength=0.9,
            inequality_tolerance=0.6,
            transition_speed=0.7,
            adaptation_lag=2
        )
        scenarios.append(optimistic)
        
        # Pessimistic scenario
        pessimistic = ScenarioParameters(
            name="AI_Pessimistic",
            scenario_type=ScenarioType.PESSIMISTIC,
            ai_progress_rate=AIProgressRate.RAPID,
            societal_response=SocietalResponse.RESISTANT,
            ai_productivity_multiplier=1.8,
            job_displacement_rate=0.08,
            new_job_creation_rate=0.02,
            wealth_concentration_tendency=0.9,
            ai_capability_growth_rate=0.20,
            automation_penetration_rate=0.9,
            digital_divide_factor=0.6,
            ubi_implementation=0.1,
            wealth_redistribution_rate=0.2,
            ai_regulation_strength=0.3,
            education_adaptation_rate=0.3,
            social_cohesion_level=0.4,
            innovation_culture_strength=0.6,
            inequality_tolerance=0.3,
            transition_speed=0.9,
            adaptation_lag=5
        )
        scenarios.append(pessimistic)
        
        # Realistic scenario
        realistic = ScenarioParameters(
            name="AI_Realistic",
            scenario_type=ScenarioType.REALISTIC,
            ai_progress_rate=AIProgressRate.MODERATE,
            societal_response=SocietalResponse.REACTIVE,
            ai_productivity_multiplier=2.0,
            job_displacement_rate=0.05,
            new_job_creation_rate=0.03,
            wealth_concentration_tendency=0.75,
            ai_capability_growth_rate=0.15,
            automation_penetration_rate=0.7,
            digital_divide_factor=0.4,
            ubi_implementation=0.3,
            wealth_redistribution_rate=0.3,
            ai_regulation_strength=0.5,
            education_adaptation_rate=0.5,
            social_cohesion_level=0.6,
            innovation_culture_strength=0.7,
            inequality_tolerance=0.5,
            transition_speed=0.6,
            adaptation_lag=3
        )
        scenarios.append(realistic)
        
        # Regulatory Heavy scenario
        regulatory_heavy = ScenarioParameters(
            name="AI_Regulatory_Heavy",
            scenario_type=ScenarioType.REGULATORY_HEAVY,
            ai_progress_rate=AIProgressRate.SLOW,
            societal_response=SocietalResponse.PROACTIVE,
            ai_productivity_multiplier=1.5,
            job_displacement_rate=0.03,
            new_job_creation_rate=0.04,
            wealth_concentration_tendency=0.5,
            ai_capability_growth_rate=0.10,
            automation_penetration_rate=0.5,
            digital_divide_factor=0.2,
            ubi_implementation=0.8,
            wealth_redistribution_rate=0.6,
            ai_regulation_strength=0.9,
            education_adaptation_rate=0.9,
            social_cohesion_level=0.8,
            innovation_culture_strength=0.5,
            inequality_tolerance=0.7,
            transition_speed=0.4,
            adaptation_lag=1
        )
        scenarios.append(regulatory_heavy)
        
        # Laissez-faire scenario
        laissez_faire = ScenarioParameters(
            name="AI_Laissez_Faire",
            scenario_type=ScenarioType.LAISSEZ_FAIRE,
            ai_progress_rate=AIProgressRate.EXPONENTIAL,
            societal_response=SocietalResponse.FRAGMENTED,
            ai_productivity_multiplier=3.0,
            job_displacement_rate=0.10,
            new_job_creation_rate=0.06,
            wealth_concentration_tendency=0.95,
            ai_capability_growth_rate=0.35,
            automation_penetration_rate=0.95,
            digital_divide_factor=0.8,
            ubi_implementation=0.0,
            wealth_redistribution_rate=0.1,
            ai_regulation_strength=0.1,
            education_adaptation_rate=0.2,
            social_cohesion_level=0.3,
            innovation_culture_strength=0.95,
            inequality_tolerance=0.2,
            transition_speed=0.95,
            adaptation_lag=7
        )
        scenarios.append(laissez_faire)
        
        return scenarios
    
    def run_scenario_analysis(self, time_horizon: int = 20) -> Dict:
        """Run complete scenario analysis with predefined scenarios"""
        
        # Create and add predefined scenarios
        predefined_scenarios = self.create_predefined_scenarios()
        for scenario in predefined_scenarios:
            self.add_scenario(scenario)
        
        # Simulate all scenarios
        results = {}
        for scenario_name in self.scenarios.keys():
            outcome = self.simulate_scenario(scenario_name, time_horizon)
            results[scenario_name] = outcome
        
        # Compare scenarios
        scenario_names = list(self.scenarios.keys())
        comparison = self.compare_scenarios(scenario_names)
        
        # Generate summary
        summary = self._generate_analysis_summary(results, comparison)
        
        return {
            'individual_results': results,
            'comparison': comparison,
            'summary': summary
        }
    
    def _generate_analysis_summary(self, results: Dict[str, ScenarioOutcome],
                                  comparison: Dict) -> Dict:
        """Generate summary of scenario analysis"""
        
        # Best and worst scenarios by different criteria
        best_scenarios = {}
        worst_scenarios = {}
        
        criteria = ['total_wealth_created', 'quality_of_life_index', 'social_stability_index']
        
        for criterion in criteria:
            values = [(name, getattr(outcome, criterion)) for name, outcome in results.items()]
            values.sort(key=lambda x: x[1], reverse=True)
            
            best_scenarios[criterion] = values[0][0]
            worst_scenarios[criterion] = values[-1][0]
        
        # Risk analysis
        high_risk_scenarios = []
        for name, outcome in results.items():
            if (outcome.systemic_risk_level > 0.7 or 
                outcome.social_unrest_probability > 0.6 or
                outcome.economic_instability_risk > 0.6):
                high_risk_scenarios.append(name)
        
        # Key insights
        insights = []
        
        # Wealth vs inequality insight
        wealth_inequality_correlation = np.corrcoef(
            [outcome.total_wealth_created for outcome in results.values()],
            [outcome.wealth_distribution_gini for outcome in results.values()]
        )[0, 1]
        
        if wealth_inequality_correlation > 0.5:
            insights.append("Strong positive correlation between wealth creation and inequality")
        elif wealth_inequality_correlation < -0.5:
            insights.append("Strong negative correlation between wealth creation and inequality")
        
        # Automation vs employment insight
        automation_employment_correlation = np.corrcoef(
            [outcome.automation_level for outcome in results.values()],
            [1 - outcome.unemployment_rate for outcome in results.values()]
        )[0, 1]
        
        if automation_employment_correlation < -0.5:
            insights.append("High automation strongly associated with unemployment")
        
        return {
            'best_scenarios': best_scenarios,
            'worst_scenarios': worst_scenarios,
            'high_risk_scenarios': high_risk_scenarios,
            'key_insights': insights,
            'wealth_inequality_correlation': wealth_inequality_correlation,
            'automation_employment_correlation': automation_employment_correlation
        }