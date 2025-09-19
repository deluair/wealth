"""
Scenario Analysis for Wealth Accumulation

Models for analyzing wealth accumulation under various economic scenarios,
life events, and market conditions with Monte Carlo simulation capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
from scipy import stats

class ScenarioType(Enum):
    """Types of scenarios to analyze"""
    ECONOMIC_CYCLE = "economic_cycle"
    MARKET_CRASH = "market_crash"
    INFLATION_SHOCK = "inflation_shock"
    LIFE_EVENT = "life_event"
    CAREER_CHANGE = "career_change"
    MONTE_CARLO = "monte_carlo"

class EconomicCondition(Enum):
    """Economic conditions"""
    RECESSION = "recession"
    RECOVERY = "recovery"
    EXPANSION = "expansion"
    PEAK = "peak"

class LifeEvent(Enum):
    """Major life events"""
    MARRIAGE = "marriage"
    DIVORCE = "divorce"
    CHILD_BIRTH = "child_birth"
    HOME_PURCHASE = "home_purchase"
    JOB_LOSS = "job_loss"
    ILLNESS = "illness"
    INHERITANCE = "inheritance"
    EDUCATION = "education"

@dataclass
class AccumulationScenario:
    """Parameters for wealth accumulation scenario"""
    name: str
    scenario_type: ScenarioType
    base_parameters: Dict[str, Any]
    scenario_events: List[Dict[str, Any]]
    time_horizon: int
    monte_carlo_runs: int = 1000
    confidence_levels: List[float] = None

@dataclass
class ScenarioResult:
    """Results from scenario analysis"""
    scenario_name: str
    base_case_result: float
    scenario_results: List[float]
    percentile_results: Dict[str, float]
    probability_of_success: float
    expected_shortfall: float
    scenario_statistics: Dict[str, float]
    detailed_paths: List[List[float]]

class ScenarioAnalyzer:
    """
    Comprehensive scenario analyzer for wealth accumulation
    """
    
    def __init__(self):
        self.scenarios = {}
        self.results = {}
        self.base_model = None
    
    def set_base_model(self, model: Any) -> None:
        """Set the base accumulation model"""
        self.base_model = model
    
    def add_scenario(self, scenario: AccumulationScenario) -> None:
        """Add a scenario for analysis"""
        self.scenarios[scenario.name] = scenario
    
    def analyze_scenario(self, scenario_name: str) -> ScenarioResult:
        """Analyze a specific scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenarios[scenario_name]
        
        if scenario.scenario_type == ScenarioType.ECONOMIC_CYCLE:
            result = self._analyze_economic_cycle(scenario)
        elif scenario.scenario_type == ScenarioType.MARKET_CRASH:
            result = self._analyze_market_crash(scenario)
        elif scenario.scenario_type == ScenarioType.INFLATION_SHOCK:
            result = self._analyze_inflation_shock(scenario)
        elif scenario.scenario_type == ScenarioType.LIFE_EVENT:
            result = self._analyze_life_event(scenario)
        elif scenario.scenario_type == ScenarioType.CAREER_CHANGE:
            result = self._analyze_career_change(scenario)
        else:  # MONTE_CARLO
            result = self._analyze_monte_carlo(scenario)
        
        self.results[scenario_name] = result
        return result
    
    def _analyze_economic_cycle(self, scenario: AccumulationScenario) -> ScenarioResult:
        """Analyze economic cycle scenarios"""
        base_result = self._run_base_case(scenario)
        scenario_results = []
        detailed_paths = []
        
        # Define economic cycle parameters
        cycle_patterns = {
            EconomicCondition.RECESSION: {
                'duration': 2,
                'return_multiplier': -0.3,
                'volatility_multiplier': 2.0,
                'employment_impact': -0.2
            },
            EconomicCondition.RECOVERY: {
                'duration': 3,
                'return_multiplier': 0.15,
                'volatility_multiplier': 1.5,
                'employment_impact': 0.1
            },
            EconomicCondition.EXPANSION: {
                'duration': 5,
                'return_multiplier': 0.12,
                'volatility_multiplier': 1.0,
                'employment_impact': 0.05
            },
            EconomicCondition.PEAK: {
                'duration': 2,
                'return_multiplier': 0.08,
                'volatility_multiplier': 1.2,
                'employment_impact': 0.0
            }
        }
        
        # Run multiple economic cycle scenarios
        for run in range(scenario.monte_carlo_runs):
            path = self._simulate_economic_cycle_path(scenario, cycle_patterns)
            scenario_results.append(path[-1])
            detailed_paths.append(path)
        
        return self._create_scenario_result(scenario, base_result, scenario_results, detailed_paths)
    
    def _analyze_market_crash(self, scenario: AccumulationScenario) -> ScenarioResult:
        """Analyze market crash scenarios"""
        base_result = self._run_base_case(scenario)
        scenario_results = []
        detailed_paths = []
        
        # Market crash parameters
        crash_scenarios = [
            {'year': 5, 'magnitude': -0.4, 'recovery_years': 3},
            {'year': 10, 'magnitude': -0.5, 'recovery_years': 4},
            {'year': 15, 'magnitude': -0.3, 'recovery_years': 2},
            {'year': 20, 'magnitude': -0.6, 'recovery_years': 5}
        ]
        
        for crash in crash_scenarios:
            for run in range(scenario.monte_carlo_runs // len(crash_scenarios)):
                path = self._simulate_market_crash_path(scenario, crash)
                scenario_results.append(path[-1])
                detailed_paths.append(path)
        
        return self._create_scenario_result(scenario, base_result, scenario_results, detailed_paths)
    
    def _analyze_inflation_shock(self, scenario: AccumulationScenario) -> ScenarioResult:
        """Analyze inflation shock scenarios"""
        base_result = self._run_base_case(scenario)
        scenario_results = []
        detailed_paths = []
        
        # Inflation shock scenarios
        inflation_shocks = [
            {'start_year': 3, 'duration': 5, 'peak_inflation': 0.08},
            {'start_year': 7, 'duration': 3, 'peak_inflation': 0.12},
            {'start_year': 12, 'duration': 4, 'peak_inflation': 0.06},
            {'start_year': 18, 'duration': 6, 'peak_inflation': 0.10}
        ]
        
        for shock in inflation_shocks:
            for run in range(scenario.monte_carlo_runs // len(inflation_shocks)):
                path = self._simulate_inflation_shock_path(scenario, shock)
                scenario_results.append(path[-1])
                detailed_paths.append(path)
        
        return self._create_scenario_result(scenario, base_result, scenario_results, detailed_paths)
    
    def _analyze_life_event(self, scenario: AccumulationScenario) -> ScenarioResult:
        """Analyze life event scenarios"""
        base_result = self._run_base_case(scenario)
        scenario_results = []
        detailed_paths = []
        
        # Life event impacts
        life_event_impacts = {
            LifeEvent.MARRIAGE: {'income_change': 0.8, 'expense_change': 1.6, 'duration': 1},
            LifeEvent.DIVORCE: {'income_change': -0.3, 'expense_change': 0.5, 'duration': 2},
            LifeEvent.CHILD_BIRTH: {'income_change': 0.0, 'expense_change': 0.25, 'duration': 18},
            LifeEvent.HOME_PURCHASE: {'income_change': 0.0, 'expense_change': 0.3, 'duration': 30},
            LifeEvent.JOB_LOSS: {'income_change': -1.0, 'expense_change': -0.2, 'duration': 1},
            LifeEvent.ILLNESS: {'income_change': -0.5, 'expense_change': 0.4, 'duration': 2},
            LifeEvent.INHERITANCE: {'income_change': 2.0, 'expense_change': 0.0, 'duration': 1},
            LifeEvent.EDUCATION: {'income_change': -0.8, 'expense_change': 0.3, 'duration': 4}
        }
        
        for event_data in scenario.scenario_events:
            event_type = LifeEvent(event_data['event_type'])
            event_year = event_data.get('year', 10)
            
            for run in range(scenario.monte_carlo_runs // len(scenario.scenario_events)):
                path = self._simulate_life_event_path(scenario, event_type, event_year, 
                                                    life_event_impacts[event_type])
                scenario_results.append(path[-1])
                detailed_paths.append(path)
        
        return self._create_scenario_result(scenario, base_result, scenario_results, detailed_paths)
    
    def _analyze_career_change(self, scenario: AccumulationScenario) -> ScenarioResult:
        """Analyze career change scenarios"""
        base_result = self._run_base_case(scenario)
        scenario_results = []
        detailed_paths = []
        
        # Career change scenarios
        career_changes = [
            {'year': 5, 'income_change': 0.3, 'transition_cost': 0.1, 'transition_duration': 1},
            {'year': 10, 'income_change': -0.2, 'transition_cost': 0.05, 'transition_duration': 2},
            {'year': 15, 'income_change': 0.5, 'transition_cost': 0.15, 'transition_duration': 1},
            {'year': 20, 'income_change': -0.1, 'transition_cost': 0.0, 'transition_duration': 0}
        ]
        
        for change in career_changes:
            for run in range(scenario.monte_carlo_runs // len(career_changes)):
                path = self._simulate_career_change_path(scenario, change)
                scenario_results.append(path[-1])
                detailed_paths.append(path)
        
        return self._create_scenario_result(scenario, base_result, scenario_results, detailed_paths)
    
    def _analyze_monte_carlo(self, scenario: AccumulationScenario) -> ScenarioResult:
        """Run Monte Carlo simulation"""
        base_result = self._run_base_case(scenario)
        scenario_results = []
        detailed_paths = []
        
        np.random.seed(42)  # For reproducible results
        
        for run in range(scenario.monte_carlo_runs):
            path = self._simulate_monte_carlo_path(scenario)
            scenario_results.append(path[-1])
            detailed_paths.append(path)
        
        return self._create_scenario_result(scenario, base_result, scenario_results, detailed_paths)
    
    def _simulate_economic_cycle_path(self, scenario: AccumulationScenario, 
                                    cycle_patterns: Dict) -> List[float]:
        """Simulate wealth path through economic cycles"""
        path = [scenario.base_parameters.get('initial_amount', 10000)]
        current_value = path[0]
        
        # Generate economic cycle sequence
        cycle_sequence = self._generate_economic_cycle_sequence(scenario.time_horizon)
        
        for year in range(1, scenario.time_horizon + 1):
            current_cycle = cycle_sequence[year - 1]
            cycle_params = cycle_patterns[current_cycle]
            
            # Apply economic cycle effects
            base_return = scenario.base_parameters.get('annual_return', 0.07)
            adjusted_return = base_return * (1 + cycle_params['return_multiplier'])
            
            # Add volatility
            volatility = scenario.base_parameters.get('volatility', 0.15) * cycle_params['volatility_multiplier']
            actual_return = np.random.normal(adjusted_return, volatility)
            
            # Apply return
            current_value *= (1 + actual_return)
            
            # Add contributions (adjusted for employment impact)
            contribution = scenario.base_parameters.get('annual_contribution', 5000)
            employment_factor = 1 + cycle_params['employment_impact']
            adjusted_contribution = contribution * employment_factor
            current_value += adjusted_contribution
            
            path.append(current_value)
        
        return path
    
    def _simulate_market_crash_path(self, scenario: AccumulationScenario, 
                                  crash_params: Dict) -> List[float]:
        """Simulate wealth path with market crash"""
        path = [scenario.base_parameters.get('initial_amount', 10000)]
        current_value = path[0]
        
        crash_year = crash_params['year']
        crash_magnitude = crash_params['magnitude']
        recovery_years = crash_params['recovery_years']
        
        for year in range(1, scenario.time_horizon + 1):
            base_return = scenario.base_parameters.get('annual_return', 0.07)
            
            if year == crash_year:
                # Market crash
                actual_return = crash_magnitude
            elif crash_year < year <= crash_year + recovery_years:
                # Recovery period
                recovery_boost = abs(crash_magnitude) / recovery_years * 1.5
                actual_return = base_return + recovery_boost
            else:
                # Normal market conditions
                volatility = scenario.base_parameters.get('volatility', 0.15)
                actual_return = np.random.normal(base_return, volatility)
            
            current_value *= (1 + actual_return)
            
            # Add contributions
            contribution = scenario.base_parameters.get('annual_contribution', 5000)
            current_value += contribution
            
            path.append(current_value)
        
        return path
    
    def _simulate_inflation_shock_path(self, scenario: AccumulationScenario,
                                     shock_params: Dict) -> List[float]:
        """Simulate wealth path with inflation shock"""
        path = [scenario.base_parameters.get('initial_amount', 10000)]
        current_value = path[0]
        
        start_year = shock_params['start_year']
        duration = shock_params['duration']
        peak_inflation = shock_params['peak_inflation']
        base_inflation = scenario.base_parameters.get('inflation_rate', 0.03)
        
        for year in range(1, scenario.time_horizon + 1):
            # Calculate inflation rate
            if start_year <= year < start_year + duration:
                # Inflation shock period
                shock_progress = (year - start_year) / duration
                if shock_progress <= 0.5:
                    # Rising inflation
                    inflation_rate = base_inflation + (peak_inflation - base_inflation) * (shock_progress * 2)
                else:
                    # Falling inflation
                    inflation_rate = peak_inflation - (peak_inflation - base_inflation) * ((shock_progress - 0.5) * 2)
            else:
                inflation_rate = base_inflation
            
            # Adjust returns for inflation
            base_return = scenario.base_parameters.get('annual_return', 0.07)
            real_return = (1 + base_return) / (1 + inflation_rate) - 1
            
            # Add volatility
            volatility = scenario.base_parameters.get('volatility', 0.15)
            actual_return = np.random.normal(real_return, volatility)
            
            current_value *= (1 + actual_return)
            
            # Add contributions (adjusted for inflation)
            contribution = scenario.base_parameters.get('annual_contribution', 5000)
            real_contribution = contribution / (1 + inflation_rate)
            current_value += real_contribution
            
            path.append(current_value)
        
        return path
    
    def _simulate_life_event_path(self, scenario: AccumulationScenario, event_type: LifeEvent,
                                event_year: int, event_impact: Dict) -> List[float]:
        """Simulate wealth path with life event"""
        path = [scenario.base_parameters.get('initial_amount', 10000)]
        current_value = path[0]
        
        for year in range(1, scenario.time_horizon + 1):
            # Base return
            base_return = scenario.base_parameters.get('annual_return', 0.07)
            volatility = scenario.base_parameters.get('volatility', 0.15)
            actual_return = np.random.normal(base_return, volatility)
            
            current_value *= (1 + actual_return)
            
            # Calculate contribution adjustment
            base_contribution = scenario.base_parameters.get('annual_contribution', 5000)
            
            if event_year <= year < event_year + event_impact['duration']:
                # Life event is active
                income_factor = 1 + event_impact['income_change']
                expense_factor = 1 + event_impact['expense_change']
                
                # Adjust contribution based on income and expense changes
                adjusted_contribution = base_contribution * income_factor / expense_factor
            else:
                adjusted_contribution = base_contribution
            
            current_value += adjusted_contribution
            path.append(current_value)
        
        return path
    
    def _simulate_career_change_path(self, scenario: AccumulationScenario,
                                   change_params: Dict) -> List[float]:
        """Simulate wealth path with career change"""
        path = [scenario.base_parameters.get('initial_amount', 10000)]
        current_value = path[0]
        
        change_year = change_params['year']
        income_change = change_params['income_change']
        transition_cost = change_params['transition_cost']
        transition_duration = change_params['transition_duration']
        
        for year in range(1, scenario.time_horizon + 1):
            # Base return
            base_return = scenario.base_parameters.get('annual_return', 0.07)
            volatility = scenario.base_parameters.get('volatility', 0.15)
            actual_return = np.random.normal(base_return, volatility)
            
            current_value *= (1 + actual_return)
            
            # Calculate contribution
            base_contribution = scenario.base_parameters.get('annual_contribution', 5000)
            
            if year == change_year:
                # Transition year - pay transition costs
                transition_cost_amount = current_value * transition_cost
                current_value -= transition_cost_amount
                adjusted_contribution = base_contribution * (1 - transition_cost)
            elif change_year < year <= change_year + transition_duration:
                # Transition period - reduced income
                adjusted_contribution = base_contribution * 0.5
            elif year > change_year + transition_duration:
                # Post-transition - new income level
                adjusted_contribution = base_contribution * (1 + income_change)
            else:
                adjusted_contribution = base_contribution
            
            current_value += adjusted_contribution
            path.append(current_value)
        
        return path
    
    def _simulate_monte_carlo_path(self, scenario: AccumulationScenario) -> List[float]:
        """Simulate single Monte Carlo path"""
        path = [scenario.base_parameters.get('initial_amount', 10000)]
        current_value = path[0]
        
        for year in range(1, scenario.time_horizon + 1):
            # Random return
            base_return = scenario.base_parameters.get('annual_return', 0.07)
            volatility = scenario.base_parameters.get('volatility', 0.15)
            actual_return = np.random.normal(base_return, volatility)
            
            current_value *= (1 + actual_return)
            
            # Random contribution variation
            base_contribution = scenario.base_parameters.get('annual_contribution', 5000)
            contribution_volatility = scenario.base_parameters.get('contribution_volatility', 0.1)
            contribution_factor = np.random.normal(1.0, contribution_volatility)
            adjusted_contribution = base_contribution * max(0, contribution_factor)
            
            current_value += adjusted_contribution
            path.append(current_value)
        
        return path
    
    def _generate_economic_cycle_sequence(self, years: int) -> List[EconomicCondition]:
        """Generate sequence of economic conditions"""
        sequence = []
        current_condition = EconomicCondition.EXPANSION
        years_in_condition = 0
        
        # Typical cycle durations
        durations = {
            EconomicCondition.RECESSION: 2,
            EconomicCondition.RECOVERY: 3,
            EconomicCondition.EXPANSION: 5,
            EconomicCondition.PEAK: 2
        }
        
        # Transition probabilities
        transitions = {
            EconomicCondition.RECESSION: EconomicCondition.RECOVERY,
            EconomicCondition.RECOVERY: EconomicCondition.EXPANSION,
            EconomicCondition.EXPANSION: EconomicCondition.PEAK,
            EconomicCondition.PEAK: EconomicCondition.RECESSION
        }
        
        for year in range(years):
            sequence.append(current_condition)
            years_in_condition += 1
            
            # Check for transition
            if years_in_condition >= durations[current_condition]:
                current_condition = transitions[current_condition]
                years_in_condition = 0
        
        return sequence
    
    def _run_base_case(self, scenario: AccumulationScenario) -> float:
        """Run base case scenario"""
        initial_amount = scenario.base_parameters.get('initial_amount', 10000)
        annual_return = scenario.base_parameters.get('annual_return', 0.07)
        annual_contribution = scenario.base_parameters.get('annual_contribution', 5000)
        
        current_value = initial_amount
        for year in range(scenario.time_horizon):
            current_value *= (1 + annual_return)
            current_value += annual_contribution
        
        return current_value
    
    def _create_scenario_result(self, scenario: AccumulationScenario, base_result: float,
                              scenario_results: List[float], detailed_paths: List[List[float]]) -> ScenarioResult:
        """Create scenario result object"""
        scenario_results = [max(0, result) for result in scenario_results]  # Ensure non-negative
        
        # Calculate percentiles
        confidence_levels = scenario.confidence_levels or [0.05, 0.25, 0.5, 0.75, 0.95]
        percentile_results = {}
        for level in confidence_levels:
            percentile_results[f"P{int(level*100)}"] = np.percentile(scenario_results, level * 100)
        
        # Calculate probability of success (beating base case)
        success_count = sum(1 for result in scenario_results if result >= base_result)
        probability_of_success = success_count / len(scenario_results)
        
        # Calculate expected shortfall (average of worst 5% outcomes)
        worst_5_percent = np.percentile(scenario_results, 5)
        shortfall_results = [result for result in scenario_results if result <= worst_5_percent]
        expected_shortfall = np.mean(shortfall_results) if shortfall_results else 0
        
        # Calculate statistics
        scenario_statistics = {
            'mean': np.mean(scenario_results),
            'std': np.std(scenario_results),
            'min': np.min(scenario_results),
            'max': np.max(scenario_results),
            'skewness': stats.skew(scenario_results),
            'kurtosis': stats.kurtosis(scenario_results)
        }
        
        return ScenarioResult(
            scenario_name=scenario.name,
            base_case_result=base_result,
            scenario_results=scenario_results,
            percentile_results=percentile_results,
            probability_of_success=probability_of_success,
            expected_shortfall=expected_shortfall,
            scenario_statistics=scenario_statistics,
            detailed_paths=detailed_paths
        )
    
    def compare_scenarios(self, scenario_names: List[str]) -> pd.DataFrame:
        """Compare multiple scenarios"""
        comparison_data = []
        
        for name in scenario_names:
            if name in self.results:
                result = self.results[name]
                comparison_data.append({
                    'Scenario': name,
                    'Base Case': result.base_case_result,
                    'Mean Result': result.scenario_statistics['mean'],
                    'P5': result.percentile_results.get('P5', 0),
                    'P50': result.percentile_results.get('P50', 0),
                    'P95': result.percentile_results.get('P95', 0),
                    'Success Probability': result.probability_of_success,
                    'Expected Shortfall': result.expected_shortfall,
                    'Volatility': result.scenario_statistics['std']
                })
        
        return pd.DataFrame(comparison_data)
    
    def calculate_value_at_risk(self, scenario_name: str, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk for a scenario"""
        if scenario_name not in self.results:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        result = self.results[scenario_name]
        return np.percentile(result.scenario_results, confidence_level * 100)
    
    def calculate_conditional_value_at_risk(self, scenario_name: str, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if scenario_name not in self.results:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        result = self.results[scenario_name]
        var = self.calculate_value_at_risk(scenario_name, confidence_level)
        tail_results = [r for r in result.scenario_results if r <= var]
        
        return np.mean(tail_results) if tail_results else 0