"""
Wealth Distribution Analyzer

Main analysis engine for wealth distribution patterns, inequality metrics,
and socioeconomic modeling using various statistical approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from .inequality_metrics import GiniCoefficient, ParetoDistribution, LorenzCurve

@dataclass
class PopulationSegment:
    """Represents a segment of the population with wealth characteristics"""
    name: str
    size: int  # Number of individuals
    mean_wealth: float
    wealth_std: float
    min_wealth: float = 0
    max_wealth: float = float('inf')

@dataclass
class DistributionParams:
    """Parameters for wealth distribution analysis"""
    population_size: int = 100000
    segments: List[PopulationSegment] = None
    time_horizon: int = 50
    mobility_rate: float = 0.1  # Rate of movement between wealth classes
    inheritance_rate: float = 0.8  # Wealth retention across generations
    economic_growth_rate: float = 0.03
    inflation_rate: float = 0.02

class WealthDistributionAnalyzer:
    """
    Comprehensive wealth distribution analysis and simulation engine
    """
    
    def __init__(self, params: Optional[DistributionParams] = None):
        self.params = params or DistributionParams()
        
        # Default population segments if not provided
        if self.params.segments is None:
            self.params.segments = self._create_default_segments()
        
        self.gini_calculator = GiniCoefficient()
        self.pareto_analyzer = ParetoDistribution()
        self.lorenz_calculator = LorenzCurve()
        
    def _create_default_segments(self) -> List[PopulationSegment]:
        """Create default population segments based on real-world wealth distribution"""
        return [
            PopulationSegment("Bottom 50%", 50000, 5000, 3000, 0, 50000),
            PopulationSegment("Middle 40%", 40000, 150000, 75000, 50000, 500000),
            PopulationSegment("Top 9%", 9000, 800000, 400000, 500000, 5000000),
            PopulationSegment("Top 1%", 1000, 5000000, 3000000, 5000000, float('inf'))
        ]
    
    def generate_wealth_distribution(self, random_seed: int = 42) -> np.ndarray:
        """Generate synthetic wealth distribution based on population segments"""
        np.random.seed(random_seed)
        
        all_wealth = []
        
        for segment in self.params.segments:
            # Generate wealth for this segment using log-normal distribution
            # to ensure realistic wealth distribution patterns
            
            # Convert to log-normal parameters
            mean_log = np.log(segment.mean_wealth**2 / np.sqrt(segment.wealth_std**2 + segment.mean_wealth**2))
            std_log = np.sqrt(np.log(1 + segment.wealth_std**2 / segment.mean_wealth**2))
            
            # Generate wealth values
            segment_wealth = np.random.lognormal(mean_log, std_log, segment.size)
            
            # Apply bounds
            segment_wealth = np.clip(segment_wealth, segment.min_wealth, segment.max_wealth)
            
            all_wealth.extend(segment_wealth)
        
        return np.array(all_wealth)
    
    def analyze_inequality_metrics(self, wealth_data: np.ndarray) -> Dict:
        """Calculate comprehensive inequality metrics"""
        
        # Sort wealth data
        sorted_wealth = np.sort(wealth_data)
        
        # Basic statistics
        total_wealth = np.sum(sorted_wealth)
        mean_wealth = np.mean(sorted_wealth)
        median_wealth = np.median(sorted_wealth)
        
        # Gini coefficient
        gini = self.gini_calculator.calculate(sorted_wealth)
        
        # Pareto analysis
        pareto_results = self.pareto_analyzer.fit_and_analyze(sorted_wealth)
        
        # Wealth concentration ratios
        concentration_metrics = self._calculate_concentration_ratios(sorted_wealth)
        
        # Percentile analysis
        percentiles = self._calculate_wealth_percentiles(sorted_wealth)
        
        # Lorenz curve data
        lorenz_data = self.lorenz_calculator.calculate(sorted_wealth)
        
        return {
            'basic_stats': {
                'total_wealth': total_wealth,
                'mean_wealth': mean_wealth,
                'median_wealth': median_wealth,
                'wealth_ratio_mean_median': mean_wealth / median_wealth if median_wealth > 0 else 0,
                'population_size': len(wealth_data)
            },
            'inequality_metrics': {
                'gini_coefficient': gini,
                'pareto_alpha': pareto_results['alpha'],
                'pareto_threshold': pareto_results['threshold'],
                'pareto_fit_quality': pareto_results['fit_quality']
            },
            'concentration_ratios': concentration_metrics,
            'percentiles': percentiles,
            'lorenz_curve': lorenz_data
        }
    
    def _calculate_concentration_ratios(self, sorted_wealth: np.ndarray) -> Dict:
        """Calculate wealth concentration ratios (e.g., top 1%, top 10%)"""
        total_wealth = np.sum(sorted_wealth)
        n = len(sorted_wealth)
        
        ratios = {}
        
        # Common concentration ratios
        for pct in [1, 5, 10, 20, 50]:
            threshold_idx = int(n * (100 - pct) / 100)
            top_wealth = np.sum(sorted_wealth[threshold_idx:])
            ratios[f'top_{pct}_percent'] = top_wealth / total_wealth if total_wealth > 0 else 0
        
        # Bottom percentages
        for pct in [10, 20, 50]:
            threshold_idx = int(n * pct / 100)
            bottom_wealth = np.sum(sorted_wealth[:threshold_idx])
            ratios[f'bottom_{pct}_percent'] = bottom_wealth / total_wealth if total_wealth > 0 else 0
        
        # 90/10 ratio (wealth of 90th percentile vs 10th percentile)
        p90_idx = int(n * 0.9)
        p10_idx = int(n * 0.1)
        if sorted_wealth[p10_idx] > 0:
            ratios['p90_p10_ratio'] = sorted_wealth[p90_idx] / sorted_wealth[p10_idx]
        else:
            ratios['p90_p10_ratio'] = float('inf')
        
        return ratios
    
    def _calculate_wealth_percentiles(self, sorted_wealth: np.ndarray) -> Dict:
        """Calculate wealth values at various percentiles"""
        percentiles = {}
        
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]:
            percentiles[f'p{p}'] = np.percentile(sorted_wealth, p)
        
        return percentiles
    
    def simulate_wealth_evolution(self, initial_wealth: np.ndarray, years: int = 50) -> Dict:
        """Simulate how wealth distribution evolves over time"""
        
        wealth_history = [initial_wealth.copy()]
        inequality_history = []
        
        current_wealth = initial_wealth.copy()
        
        for year in range(years):
            # Economic growth
            growth_factor = 1 + self.params.economic_growth_rate
            current_wealth *= growth_factor
            
            # Individual wealth changes (investment returns, income, etc.)
            individual_returns = self._simulate_individual_returns(current_wealth)
            current_wealth *= individual_returns
            
            # Social mobility (wealth transfers between individuals)
            if self.params.mobility_rate > 0:
                current_wealth = self._apply_social_mobility(current_wealth)
            
            # Inheritance effects (generational wealth transfer)
            if year % 25 == 0 and year > 0:  # Every generation
                current_wealth = self._apply_inheritance_effects(current_wealth)
            
            # Economic shocks
            if np.random.random() < 0.1:  # 10% chance of economic shock per year
                shock_magnitude = np.random.uniform(0.8, 1.2)
                current_wealth *= shock_magnitude
            
            # Ensure no negative wealth
            current_wealth = np.maximum(current_wealth, 0)
            
            # Store results
            wealth_history.append(current_wealth.copy())
            
            # Calculate inequality metrics for this year
            inequality_metrics = self.analyze_inequality_metrics(current_wealth)
            inequality_history.append({
                'year': year + 1,
                'gini': inequality_metrics['inequality_metrics']['gini_coefficient'],
                'top_1_percent': inequality_metrics['concentration_ratios']['top_1_percent'],
                'top_10_percent': inequality_metrics['concentration_ratios']['top_10_percent'],
                'bottom_50_percent': inequality_metrics['concentration_ratios']['bottom_50_percent'],
                'mean_wealth': inequality_metrics['basic_stats']['mean_wealth'],
                'median_wealth': inequality_metrics['basic_stats']['median_wealth']
            })
        
        return {
            'wealth_history': wealth_history,
            'inequality_evolution': pd.DataFrame(inequality_history),
            'final_distribution': current_wealth,
            'initial_vs_final_comparison': self._compare_distributions(initial_wealth, current_wealth)
        }
    
    def _simulate_individual_returns(self, wealth: np.ndarray) -> np.ndarray:
        """Simulate individual wealth returns based on wealth level"""
        returns = np.ones_like(wealth)
        
        # Wealth-dependent returns (richer individuals often have better investment opportunities)
        for i, w in enumerate(wealth):
            if w < 10000:  # Low wealth - limited investment options
                returns[i] = np.random.normal(1.02, 0.05)  # 2% return, 5% volatility
            elif w < 100000:  # Middle wealth
                returns[i] = np.random.normal(1.05, 0.08)  # 5% return, 8% volatility
            elif w < 1000000:  # High wealth
                returns[i] = np.random.normal(1.07, 0.12)  # 7% return, 12% volatility
            else:  # Ultra-high wealth - access to exclusive investments
                returns[i] = np.random.normal(1.10, 0.15)  # 10% return, 15% volatility
        
        return returns
    
    def _apply_social_mobility(self, wealth: np.ndarray) -> np.ndarray:
        """Apply social mobility effects (wealth redistribution)"""
        n = len(wealth)
        mobility_count = int(n * self.params.mobility_rate)
        
        if mobility_count < 2:
            return wealth
        
        # Select random individuals for wealth transfer
        indices = np.random.choice(n, mobility_count, replace=False)
        
        # Simple wealth redistribution
        total_mobile_wealth = np.sum(wealth[indices])
        
        # Redistribute with some randomness
        new_distribution = np.random.dirichlet(np.ones(mobility_count))
        wealth[indices] = total_mobile_wealth * new_distribution
        
        return wealth
    
    def _apply_inheritance_effects(self, wealth: np.ndarray) -> np.ndarray:
        """Apply intergenerational wealth transfer effects"""
        n = len(wealth)
        
        # Simulate inheritance patterns
        for i in range(n):
            # Wealth retention rate (some wealth is lost to taxes, consumption, etc.)
            retention_rate = self.params.inheritance_rate
            
            # Add some randomness to inheritance
            inheritance_factor = np.random.normal(retention_rate, 0.1)
            inheritance_factor = np.clip(inheritance_factor, 0.3, 1.2)
            
            wealth[i] *= inheritance_factor
        
        return wealth
    
    def _compare_distributions(self, initial: np.ndarray, final: np.ndarray) -> Dict:
        """Compare initial and final wealth distributions"""
        initial_metrics = self.analyze_inequality_metrics(initial)
        final_metrics = self.analyze_inequality_metrics(final)
        
        return {
            'gini_change': final_metrics['inequality_metrics']['gini_coefficient'] - 
                          initial_metrics['inequality_metrics']['gini_coefficient'],
            'mean_wealth_growth': final_metrics['basic_stats']['mean_wealth'] / 
                                 initial_metrics['basic_stats']['mean_wealth'] - 1,
            'median_wealth_growth': final_metrics['basic_stats']['median_wealth'] / 
                                   initial_metrics['basic_stats']['median_wealth'] - 1,
            'top_1_percent_change': final_metrics['concentration_ratios']['top_1_percent'] - 
                                   initial_metrics['concentration_ratios']['top_1_percent'],
            'bottom_50_percent_change': final_metrics['concentration_ratios']['bottom_50_percent'] - 
                                       initial_metrics['concentration_ratios']['bottom_50_percent']
        }
    
    def analyze_wealth_mobility(self, wealth_history: List[np.ndarray]) -> Dict:
        """Analyze wealth mobility patterns over time"""
        if len(wealth_history) < 2:
            return {}
        
        initial_wealth = wealth_history[0]
        final_wealth = wealth_history[-1]
        
        # Create wealth quintiles based on initial distribution
        initial_quintiles = np.percentile(initial_wealth, [20, 40, 60, 80])
        
        # Classify individuals into initial quintiles
        initial_classes = np.digitize(initial_wealth, initial_quintiles)
        final_classes = np.digitize(final_wealth, np.percentile(final_wealth, [20, 40, 60, 80]))
        
        # Create transition matrix
        transition_matrix = np.zeros((5, 5))
        
        for i in range(len(initial_wealth)):
            initial_class = min(initial_classes[i], 4)
            final_class = min(final_classes[i], 4)
            transition_matrix[initial_class, final_class] += 1
        
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        # Calculate mobility metrics
        mobility_metrics = {
            'transition_matrix': transition_matrix,
            'upward_mobility': np.sum(np.triu(transition_matrix, k=1)),
            'downward_mobility': np.sum(np.tril(transition_matrix, k=-1)),
            'persistence': np.trace(transition_matrix) / 5,  # Diagonal average
            'perfect_mobility_index': 1 - np.trace(transition_matrix) / 5
        }
        
        return mobility_metrics
    
    def simulate_policy_interventions(self, initial_wealth: np.ndarray, 
                                    interventions: Dict) -> Dict:
        """Simulate the effects of various policy interventions on wealth distribution"""
        
        results = {}
        
        for policy_name, policy_params in interventions.items():
            modified_wealth = initial_wealth.copy()
            
            if policy_name == "progressive_taxation":
                modified_wealth = self._apply_progressive_tax(modified_wealth, policy_params)
            elif policy_name == "universal_basic_income":
                modified_wealth = self._apply_ubi(modified_wealth, policy_params)
            elif policy_name == "wealth_tax":
                modified_wealth = self._apply_wealth_tax(modified_wealth, policy_params)
            elif policy_name == "education_investment":
                modified_wealth = self._apply_education_investment(modified_wealth, policy_params)
            
            # Analyze the results
            policy_metrics = self.analyze_inequality_metrics(modified_wealth)
            original_metrics = self.analyze_inequality_metrics(initial_wealth)
            
            results[policy_name] = {
                'modified_distribution': modified_wealth,
                'metrics': policy_metrics,
                'gini_change': policy_metrics['inequality_metrics']['gini_coefficient'] - 
                              original_metrics['inequality_metrics']['gini_coefficient'],
                'policy_params': policy_params
            }
        
        return results
    
    def _apply_progressive_tax(self, wealth: np.ndarray, params: Dict) -> np.ndarray:
        """Apply progressive taxation"""
        tax_brackets = params.get('brackets', [(50000, 0.1), (200000, 0.2), (1000000, 0.3)])
        
        modified_wealth = wealth.copy()
        total_tax_collected = 0
        
        for i, w in enumerate(wealth):
            tax_owed = 0
            remaining_wealth = w
            
            for bracket_threshold, tax_rate in tax_brackets:
                if remaining_wealth > bracket_threshold:
                    taxable_amount = min(remaining_wealth, bracket_threshold)
                    tax_owed += taxable_amount * tax_rate
                    remaining_wealth -= taxable_amount
                else:
                    tax_owed += remaining_wealth * tax_rate
                    break
            
            modified_wealth[i] = w - tax_owed
            total_tax_collected += tax_owed
        
        # Redistribute collected taxes (simplified)
        if params.get('redistribute', True):
            per_person_redistribution = total_tax_collected / len(wealth)
            modified_wealth += per_person_redistribution
        
        return modified_wealth
    
    def _apply_ubi(self, wealth: np.ndarray, params: Dict) -> np.ndarray:
        """Apply Universal Basic Income"""
        ubi_amount = params.get('amount', 12000)  # Annual UBI
        
        return wealth + ubi_amount
    
    def _apply_wealth_tax(self, wealth: np.ndarray, params: Dict) -> np.ndarray:
        """Apply wealth tax"""
        threshold = params.get('threshold', 1000000)
        tax_rate = params.get('rate', 0.02)
        
        modified_wealth = wealth.copy()
        
        for i, w in enumerate(wealth):
            if w > threshold:
                tax_owed = (w - threshold) * tax_rate
                modified_wealth[i] = w - tax_owed
        
        return modified_wealth
    
    def _apply_education_investment(self, wealth: np.ndarray, params: Dict) -> np.ndarray:
        """Apply education investment effects"""
        investment_amount = params.get('amount', 5000)
        effectiveness = params.get('effectiveness', 0.1)  # 10% wealth boost
        
        # Target lower wealth individuals for education investment
        threshold = np.percentile(wealth, 50)  # Bottom 50%
        
        modified_wealth = wealth.copy()
        
        for i, w in enumerate(wealth):
            if w < threshold:
                # Education investment leads to future wealth increase
                modified_wealth[i] = w * (1 + effectiveness)
        
        return modified_wealth