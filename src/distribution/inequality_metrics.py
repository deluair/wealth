"""
Inequality Metrics

Specialized classes for calculating various wealth inequality metrics
including Gini coefficient, Pareto distribution analysis, and Lorenz curves.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

class GiniCoefficient:
    """Calculate and analyze Gini coefficient for wealth inequality"""
    
    def calculate(self, wealth_data: np.ndarray) -> float:
        """
        Calculate Gini coefficient
        
        Args:
            wealth_data: Array of wealth values
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        # Sort the data
        sorted_wealth = np.sort(wealth_data)
        n = len(sorted_wealth)
        
        if n == 0:
            return 0.0
        
        # Calculate cumulative wealth
        cumulative_wealth = np.cumsum(sorted_wealth)
        total_wealth = cumulative_wealth[-1]
        
        if total_wealth == 0:
            return 0.0
        
        # Calculate Gini coefficient using the formula:
        # G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n + 1) / n
        index_sum = np.sum((np.arange(1, n + 1) * sorted_wealth))
        gini = (2 * index_sum) / (n * total_wealth) - (n + 1) / n
        
        return max(0.0, min(1.0, gini))  # Clamp between 0 and 1
    
    def calculate_by_groups(self, wealth_data: np.ndarray, 
                           group_labels: np.ndarray) -> Dict[str, float]:
        """Calculate Gini coefficient for different groups"""
        results = {}
        
        unique_groups = np.unique(group_labels)
        
        for group in unique_groups:
            group_wealth = wealth_data[group_labels == group]
            results[str(group)] = self.calculate(group_wealth)
        
        return results
    
    def decompose_gini(self, wealth_data: np.ndarray, 
                      group_labels: np.ndarray) -> Dict:
        """
        Decompose Gini coefficient into within-group and between-group components
        """
        total_gini = self.calculate(wealth_data)
        
        unique_groups = np.unique(group_labels)
        n_total = len(wealth_data)
        total_wealth = np.sum(wealth_data)
        
        within_group_component = 0
        between_group_component = 0
        
        group_stats = {}
        
        for group in unique_groups:
            group_mask = group_labels == group
            group_wealth = wealth_data[group_mask]
            
            n_group = len(group_wealth)
            group_total_wealth = np.sum(group_wealth)
            group_mean_wealth = np.mean(group_wealth)
            
            # Group statistics
            group_stats[str(group)] = {
                'size': n_group,
                'total_wealth': group_total_wealth,
                'mean_wealth': group_mean_wealth,
                'wealth_share': group_total_wealth / total_wealth,
                'population_share': n_group / n_total,
                'gini': self.calculate(group_wealth)
            }
            
            # Within-group contribution
            group_gini = self.calculate(group_wealth)
            weight = (n_group / n_total) * (group_total_wealth / total_wealth)
            within_group_component += weight * group_gini
        
        # Between-group component
        between_group_component = total_gini - within_group_component
        
        return {
            'total_gini': total_gini,
            'within_group_component': within_group_component,
            'between_group_component': between_group_component,
            'group_statistics': group_stats
        }

class ParetoDistribution:
    """Analyze wealth distribution using Pareto distribution"""
    
    def fit_pareto(self, wealth_data: np.ndarray, 
                   threshold_percentile: float = 90) -> Dict:
        """
        Fit Pareto distribution to the upper tail of wealth distribution
        
        Args:
            wealth_data: Array of wealth values
            threshold_percentile: Percentile above which to fit Pareto
            
        Returns:
            Dictionary with Pareto parameters and fit statistics
        """
        # Determine threshold
        threshold = np.percentile(wealth_data, threshold_percentile)
        
        # Extract tail data
        tail_data = wealth_data[wealth_data >= threshold]
        
        if len(tail_data) < 10:  # Need sufficient data points
            return {
                'alpha': None,
                'threshold': threshold,
                'fit_quality': 0,
                'tail_size': len(tail_data),
                'error': 'Insufficient data points in tail'
            }
        
        # Fit Pareto distribution using Maximum Likelihood Estimation
        # For Pareto Type I: alpha = n / sum(ln(x_i / x_min))
        log_ratios = np.log(tail_data / threshold)
        alpha = len(tail_data) / np.sum(log_ratios)
        
        # Calculate fit quality using Kolmogorov-Smirnov test
        theoretical_cdf = 1 - (threshold / tail_data) ** alpha
        empirical_cdf = np.arange(1, len(tail_data) + 1) / len(tail_data)
        
        # Sort for proper comparison
        sort_indices = np.argsort(tail_data)
        theoretical_cdf_sorted = theoretical_cdf[sort_indices]
        
        ks_statistic = np.max(np.abs(theoretical_cdf_sorted - empirical_cdf))
        
        return {
            'alpha': alpha,
            'threshold': threshold,
            'threshold_percentile': threshold_percentile,
            'tail_size': len(tail_data),
            'ks_statistic': ks_statistic,
            'fit_quality': 1 - ks_statistic,  # Higher is better
            'pareto_coefficient': 1 / alpha if alpha > 0 else None
        }
    
    def fit_and_analyze(self, wealth_data: np.ndarray) -> Dict:
        """Comprehensive Pareto analysis with multiple thresholds"""
        results = {}
        
        # Try different threshold percentiles
        thresholds = [80, 85, 90, 95, 99]
        
        best_fit = None
        best_quality = 0
        
        for threshold_pct in thresholds:
            fit_result = self.fit_pareto(wealth_data, threshold_pct)
            
            if fit_result.get('fit_quality', 0) > best_quality:
                best_quality = fit_result['fit_quality']
                best_fit = fit_result
            
            results[f'threshold_{threshold_pct}'] = fit_result
        
        # Add best fit summary
        results['best_fit'] = best_fit
        
        # Calculate Pareto principle metrics (80-20 rule variations)
        results['pareto_principles'] = self._calculate_pareto_principles(wealth_data)
        
        return results
    
    def _calculate_pareto_principles(self, wealth_data: np.ndarray) -> Dict:
        """Calculate various Pareto principle metrics"""
        sorted_wealth = np.sort(wealth_data)
        total_wealth = np.sum(sorted_wealth)
        n = len(sorted_wealth)
        
        principles = {}
        
        # Classic 80-20: What percentage of population holds 80% of wealth?
        cumulative_wealth = np.cumsum(sorted_wealth[::-1])  # Reverse for top-down
        wealth_80_idx = np.argmax(cumulative_wealth >= 0.8 * total_wealth)
        principles['population_holding_80pct_wealth'] = (wealth_80_idx + 1) / n
        
        # What percentage of wealth is held by top 20%?
        top_20_idx = int(n * 0.8)  # Bottom 80% index
        top_20_wealth = np.sum(sorted_wealth[top_20_idx:])
        principles['wealth_held_by_top_20pct'] = top_20_wealth / total_wealth
        
        # Other variations
        for pop_pct in [1, 5, 10, 20]:
            threshold_idx = int(n * (100 - pop_pct) / 100)
            top_wealth = np.sum(sorted_wealth[threshold_idx:])
            principles[f'wealth_held_by_top_{pop_pct}pct'] = top_wealth / total_wealth
        
        return principles
    
    def generate_pareto_samples(self, alpha: float, threshold: float, 
                               size: int) -> np.ndarray:
        """Generate random samples from Pareto distribution"""
        # Use inverse transform sampling
        u = np.random.uniform(0, 1, size)
        samples = threshold * (1 - u) ** (-1/alpha)
        return samples

class LorenzCurve:
    """Calculate and visualize Lorenz curves for wealth distribution"""
    
    def calculate(self, wealth_data: np.ndarray) -> Dict:
        """
        Calculate Lorenz curve coordinates
        
        Args:
            wealth_data: Array of wealth values
            
        Returns:
            Dictionary with Lorenz curve data
        """
        # Sort wealth data
        sorted_wealth = np.sort(wealth_data)
        n = len(sorted_wealth)
        
        if n == 0:
            return {'population_percentiles': [], 'wealth_percentiles': []}
        
        # Calculate cumulative wealth
        cumulative_wealth = np.cumsum(sorted_wealth)
        total_wealth = cumulative_wealth[-1]
        
        if total_wealth == 0:
            return {'population_percentiles': [], 'wealth_percentiles': []}
        
        # Population percentiles (x-axis)
        population_percentiles = np.arange(0, n + 1) / n
        
        # Wealth percentiles (y-axis)
        wealth_percentiles = np.concatenate([[0], cumulative_wealth / total_wealth])
        
        return {
            'population_percentiles': population_percentiles,
            'wealth_percentiles': wealth_percentiles,
            'total_wealth': total_wealth,
            'sample_size': n
        }
    
    def compare_distributions(self, distributions: Dict[str, np.ndarray]) -> Dict:
        """Compare Lorenz curves for multiple distributions"""
        results = {}
        
        for name, wealth_data in distributions.items():
            results[name] = self.calculate(wealth_data)
        
        return results
    
    def calculate_area_between_curves(self, curve1: Dict, curve2: Dict) -> float:
        """Calculate area between two Lorenz curves"""
        # Interpolate curves to common grid
        common_x = np.linspace(0, 1, 1000)
        
        y1 = np.interp(common_x, curve1['population_percentiles'], 
                      curve1['wealth_percentiles'])
        y2 = np.interp(common_x, curve2['population_percentiles'], 
                      curve2['wealth_percentiles'])
        
        # Calculate area using trapezoidal rule
        area = np.trapz(np.abs(y1 - y2), common_x)
        
        return area

class WealthConcentrationMetrics:
    """Calculate various wealth concentration and inequality metrics"""
    
    def __init__(self):
        self.gini_calc = GiniCoefficient()
        self.pareto_calc = ParetoDistribution()
        self.lorenz_calc = LorenzCurve()
    
    def calculate_all_metrics(self, wealth_data: np.ndarray) -> Dict:
        """Calculate comprehensive set of inequality metrics"""
        
        sorted_wealth = np.sort(wealth_data)
        n = len(sorted_wealth)
        total_wealth = np.sum(sorted_wealth)
        
        if n == 0 or total_wealth == 0:
            return self._empty_metrics()
        
        metrics = {}
        
        # Basic statistics
        metrics['basic_stats'] = {
            'mean': np.mean(sorted_wealth),
            'median': np.median(sorted_wealth),
            'std': np.std(sorted_wealth),
            'min': np.min(sorted_wealth),
            'max': np.max(sorted_wealth),
            'total_wealth': total_wealth,
            'sample_size': n
        }
        
        # Gini coefficient
        metrics['gini_coefficient'] = self.gini_calc.calculate(sorted_wealth)
        
        # Pareto analysis
        metrics['pareto_analysis'] = self.pareto_calc.fit_and_analyze(sorted_wealth)
        
        # Concentration ratios
        metrics['concentration_ratios'] = self._calculate_concentration_ratios(sorted_wealth)
        
        # Percentile ratios
        metrics['percentile_ratios'] = self._calculate_percentile_ratios(sorted_wealth)
        
        # Atkinson indices
        metrics['atkinson_indices'] = self._calculate_atkinson_indices(sorted_wealth)
        
        # Theil indices
        metrics['theil_indices'] = self._calculate_theil_indices(sorted_wealth)
        
        # Hoover index (Robin Hood index)
        metrics['hoover_index'] = self._calculate_hoover_index(sorted_wealth)
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure for edge cases"""
        return {
            'basic_stats': {},
            'gini_coefficient': 0,
            'pareto_analysis': {},
            'concentration_ratios': {},
            'percentile_ratios': {},
            'atkinson_indices': {},
            'theil_indices': {},
            'hoover_index': 0
        }
    
    def _calculate_concentration_ratios(self, sorted_wealth: np.ndarray) -> Dict:
        """Calculate wealth concentration ratios"""
        total_wealth = np.sum(sorted_wealth)
        n = len(sorted_wealth)
        
        ratios = {}
        
        # Top percentages
        for pct in [0.1, 0.5, 1, 5, 10, 20]:
            threshold_idx = int(n * (1 - pct/100))
            top_wealth = np.sum(sorted_wealth[threshold_idx:])
            ratios[f'top_{pct}_percent'] = top_wealth / total_wealth
        
        # Bottom percentages
        for pct in [10, 20, 40, 50]:
            threshold_idx = int(n * pct/100)
            bottom_wealth = np.sum(sorted_wealth[:threshold_idx])
            ratios[f'bottom_{pct}_percent'] = bottom_wealth / total_wealth
        
        return ratios
    
    def _calculate_percentile_ratios(self, sorted_wealth: np.ndarray) -> Dict:
        """Calculate ratios between different percentiles"""
        ratios = {}
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        p_values = np.percentile(sorted_wealth, percentiles)
        
        # Common ratios
        if p_values[1] > 0:  # p25
            ratios['p75_p25'] = p_values[4] / p_values[1]  # p75/p25
        
        if p_values[0] > 0:  # p10
            ratios['p90_p10'] = p_values[5] / p_values[0]  # p90/p10
        
        if p_values[2] > 0:  # p50 (median)
            ratios['p95_p50'] = p_values[6] / p_values[2]  # p95/p50
            ratios['p99_p50'] = p_values[7] / p_values[2]  # p99/p50
        
        return ratios
    
    def _calculate_atkinson_indices(self, sorted_wealth: np.ndarray) -> Dict:
        """Calculate Atkinson inequality indices for different epsilon values"""
        mean_wealth = np.mean(sorted_wealth)
        
        if mean_wealth == 0:
            return {}
        
        indices = {}
        
        # Different epsilon values (inequality aversion parameters)
        epsilons = [0.5, 1.0, 1.5, 2.0]
        
        for eps in epsilons:
            if eps == 1.0:
                # Special case for epsilon = 1
                log_mean = np.mean(np.log(np.maximum(sorted_wealth, 1e-10)))
                equally_distributed_equivalent = np.exp(log_mean)
            else:
                # General case
                power_mean = np.mean(sorted_wealth ** (1 - eps))
                equally_distributed_equivalent = power_mean ** (1 / (1 - eps))
            
            atkinson = 1 - equally_distributed_equivalent / mean_wealth
            indices[f'epsilon_{eps}'] = max(0, min(1, atkinson))
        
        return indices
    
    def _calculate_theil_indices(self, sorted_wealth: np.ndarray) -> Dict:
        """Calculate Theil inequality indices"""
        mean_wealth = np.mean(sorted_wealth)
        total_wealth = np.sum(sorted_wealth)
        n = len(sorted_wealth)
        
        if mean_wealth == 0 or total_wealth == 0:
            return {'theil_t': 0, 'theil_l': 0}
        
        # Theil T index (mean log deviation)
        # T = (1/n) * sum(ln(mean/x_i))
        log_ratios = np.log(mean_wealth / np.maximum(sorted_wealth, 1e-10))
        theil_t = np.mean(log_ratios)
        
        # Theil L index (generalized entropy with alpha=0)
        # L = (1/n) * sum((x_i/mean) * ln(x_i/mean))
        wealth_ratios = sorted_wealth / mean_wealth
        # Handle zero wealth values
        wealth_ratios = np.maximum(wealth_ratios, 1e-10)
        theil_l = np.mean(wealth_ratios * np.log(wealth_ratios))
        
        return {
            'theil_t': max(0, theil_t),
            'theil_l': max(0, theil_l)
        }
    
    def _calculate_hoover_index(self, sorted_wealth: np.ndarray) -> float:
        """Calculate Hoover index (Robin Hood index)"""
        n = len(sorted_wealth)
        total_wealth = np.sum(sorted_wealth)
        
        if total_wealth == 0:
            return 0
        
        # Equal share would be total_wealth / n
        equal_share = total_wealth / n
        
        # Sum of absolute deviations from equal share
        deviations = np.abs(sorted_wealth - equal_share)
        hoover = np.sum(deviations) / (2 * total_wealth)
        
        return hoover