"""
Social Mobility Analysis

Classes for analyzing wealth mobility patterns, intergenerational wealth transfer,
and social class transitions over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@dataclass
class MobilityMetrics:
    """Container for social mobility metrics"""
    transition_matrix: np.ndarray
    mobility_index: float
    immobility_ratio: float
    upward_mobility: float
    downward_mobility: float
    persistence_rates: Dict[str, float]
    correlation_coefficient: float

class SocialMobilityAnalyzer:
    """Analyze social mobility patterns in wealth distribution"""
    
    def __init__(self, n_classes: int = 5):
        """
        Initialize mobility analyzer
        
        Args:
            n_classes: Number of social classes to define (default: quintiles)
        """
        self.n_classes = n_classes
        self.class_labels = [f"Class_{i+1}" for i in range(n_classes)]
        
    def define_wealth_classes(self, wealth_data: np.ndarray, 
                             method: str = 'percentile') -> Tuple[np.ndarray, Dict]:
        """
        Define wealth classes based on different methods
        
        Args:
            wealth_data: Array of wealth values
            method: 'percentile', 'kmeans', or 'fixed_thresholds'
            
        Returns:
            Tuple of (class_assignments, class_boundaries)
        """
        if method == 'percentile':
            return self._percentile_classes(wealth_data)
        elif method == 'kmeans':
            return self._kmeans_classes(wealth_data)
        elif method == 'fixed_thresholds':
            return self._fixed_threshold_classes(wealth_data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _percentile_classes(self, wealth_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Define classes based on percentiles"""
        percentiles = np.linspace(0, 100, self.n_classes + 1)
        boundaries = np.percentile(wealth_data, percentiles)
        
        # Assign classes
        class_assignments = np.digitize(wealth_data, boundaries[1:-1])
        
        class_info = {
            'method': 'percentile',
            'boundaries': boundaries,
            'class_names': self.class_labels,
            'percentiles': percentiles
        }
        
        return class_assignments, class_info
    
    def _kmeans_classes(self, wealth_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Define classes using K-means clustering"""
        # Reshape for sklearn
        wealth_reshaped = wealth_data.reshape(-1, 1)
        
        # Standardize data
        scaler = StandardScaler()
        wealth_scaled = scaler.fit_transform(wealth_reshaped)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=self.n_classes, random_state=42, n_init=10)
        class_assignments = kmeans.fit_predict(wealth_scaled)
        
        # Get cluster centers in original scale
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled).flatten()
        
        # Sort classes by wealth level
        sort_indices = np.argsort(centers_original)
        class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sort_indices)}
        class_assignments = np.array([class_mapping[cls] for cls in class_assignments])
        
        class_info = {
            'method': 'kmeans',
            'centers': np.sort(centers_original),
            'class_names': self.class_labels,
            'inertia': kmeans.inertia_
        }
        
        return class_assignments, class_info
    
    def _fixed_threshold_classes(self, wealth_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Define classes using fixed wealth thresholds"""
        # Define thresholds based on common wealth brackets
        max_wealth = np.max(wealth_data)
        
        if max_wealth <= 100000:
            thresholds = [0, 25000, 50000, 75000, 100000, np.inf]
        elif max_wealth <= 1000000:
            thresholds = [0, 100000, 250000, 500000, 1000000, np.inf]
        else:
            thresholds = [0, 250000, 500000, 1000000, 5000000, np.inf]
        
        # Adjust number of classes if needed
        thresholds = thresholds[:self.n_classes + 1]
        if len(thresholds) < self.n_classes + 1:
            thresholds.append(np.inf)
        
        class_assignments = np.digitize(wealth_data, thresholds[1:-1])
        
        class_info = {
            'method': 'fixed_thresholds',
            'thresholds': thresholds,
            'class_names': self.class_labels
        }
        
        return class_assignments, class_info
    
    def calculate_transition_matrix(self, initial_classes: np.ndarray, 
                                  final_classes: np.ndarray) -> np.ndarray:
        """
        Calculate transition matrix between time periods
        
        Args:
            initial_classes: Class assignments at time t
            final_classes: Class assignments at time t+1
            
        Returns:
            Transition matrix where element (i,j) is probability of moving from class i to j
        """
        # Create transition matrix
        transition_matrix = np.zeros((self.n_classes, self.n_classes))
        
        for i in range(self.n_classes):
            initial_mask = initial_classes == i
            if np.sum(initial_mask) > 0:
                for j in range(self.n_classes):
                    final_mask = final_classes == j
                    transitions = np.sum(initial_mask & final_mask)
                    transition_matrix[i, j] = transitions / np.sum(initial_mask)
        
        return transition_matrix
    
    def analyze_mobility(self, initial_wealth: np.ndarray, 
                        final_wealth: np.ndarray,
                        class_method: str = 'percentile') -> MobilityMetrics:
        """
        Comprehensive mobility analysis between two time periods
        
        Args:
            initial_wealth: Wealth at time t
            final_wealth: Wealth at time t+1
            class_method: Method for defining wealth classes
            
        Returns:
            MobilityMetrics object with comprehensive mobility statistics
        """
        # Define classes for both periods
        initial_classes, _ = self.define_wealth_classes(initial_wealth, class_method)
        final_classes, _ = self.define_wealth_classes(final_wealth, class_method)
        
        # Calculate transition matrix
        transition_matrix = self.calculate_transition_matrix(initial_classes, final_classes)
        
        # Calculate mobility metrics
        mobility_index = self._calculate_mobility_index(transition_matrix)
        immobility_ratio = self._calculate_immobility_ratio(transition_matrix)
        upward_mobility, downward_mobility = self._calculate_directional_mobility(transition_matrix)
        persistence_rates = self._calculate_persistence_rates(transition_matrix)
        
        # Calculate intergenerational correlation
        correlation_coefficient = np.corrcoef(initial_wealth, final_wealth)[0, 1]
        
        return MobilityMetrics(
            transition_matrix=transition_matrix,
            mobility_index=mobility_index,
            immobility_ratio=immobility_ratio,
            upward_mobility=upward_mobility,
            downward_mobility=downward_mobility,
            persistence_rates=persistence_rates,
            correlation_coefficient=correlation_coefficient
        )
    
    def _calculate_mobility_index(self, transition_matrix: np.ndarray) -> float:
        """Calculate overall mobility index (1 - trace of transition matrix)"""
        diagonal_sum = np.trace(transition_matrix)
        return 1 - (diagonal_sum / self.n_classes)
    
    def _calculate_immobility_ratio(self, transition_matrix: np.ndarray) -> float:
        """Calculate immobility ratio (proportion staying in same class)"""
        return np.trace(transition_matrix) / self.n_classes
    
    def _calculate_directional_mobility(self, transition_matrix: np.ndarray) -> Tuple[float, float]:
        """Calculate upward and downward mobility rates"""
        upward_mobility = 0
        downward_mobility = 0
        
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if j > i:  # Upward movement
                    upward_mobility += transition_matrix[i, j]
                elif j < i:  # Downward movement
                    downward_mobility += transition_matrix[i, j]
        
        # Normalize by number of classes
        upward_mobility /= self.n_classes
        downward_mobility /= self.n_classes
        
        return upward_mobility, downward_mobility
    
    def _calculate_persistence_rates(self, transition_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate persistence rates for each class"""
        persistence_rates = {}
        
        for i in range(self.n_classes):
            persistence_rates[self.class_labels[i]] = transition_matrix[i, i]
        
        return persistence_rates
    
    def simulate_mobility_over_time(self, initial_wealth: np.ndarray,
                                  transition_matrix: np.ndarray,
                                  n_periods: int) -> Dict:
        """
        Simulate wealth class evolution over multiple periods
        
        Args:
            initial_wealth: Starting wealth distribution
            transition_matrix: Transition probabilities between classes
            n_periods: Number of time periods to simulate
            
        Returns:
            Dictionary with simulation results
        """
        # Define initial classes
        initial_classes, class_info = self.define_wealth_classes(initial_wealth)
        
        # Initialize results
        class_evolution = np.zeros((n_periods + 1, self.n_classes))
        class_evolution[0] = np.bincount(initial_classes, minlength=self.n_classes)
        
        # Simulate evolution
        current_distribution = class_evolution[0].copy()
        
        for period in range(1, n_periods + 1):
            # Apply transition matrix
            new_distribution = current_distribution @ transition_matrix
            class_evolution[period] = new_distribution
            current_distribution = new_distribution
        
        # Calculate steady state
        steady_state = self._calculate_steady_state(transition_matrix)
        
        return {
            'class_evolution': class_evolution,
            'periods': np.arange(n_periods + 1),
            'steady_state': steady_state,
            'class_info': class_info,
            'convergence_rate': self._calculate_convergence_rate(class_evolution, steady_state)
        }
    
    def _calculate_steady_state(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Calculate steady-state distribution"""
        # Find eigenvector corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        
        # Find index of eigenvalue closest to 1
        steady_state_idx = np.argmin(np.abs(eigenvalues - 1))
        steady_state = np.real(eigenvectors[:, steady_state_idx])
        
        # Normalize to sum to 1
        steady_state = np.abs(steady_state) / np.sum(np.abs(steady_state))
        
        return steady_state
    
    def _calculate_convergence_rate(self, class_evolution: np.ndarray, 
                                  steady_state: np.ndarray) -> float:
        """Calculate rate of convergence to steady state"""
        if len(class_evolution) < 2:
            return 0
        
        # Calculate distances from steady state
        distances = []
        for period_dist in class_evolution:
            # Normalize distribution
            normalized_dist = period_dist / np.sum(period_dist) if np.sum(period_dist) > 0 else period_dist
            distance = np.linalg.norm(normalized_dist - steady_state)
            distances.append(distance)
        
        # Fit exponential decay to estimate convergence rate
        if len(distances) > 2:
            periods = np.arange(len(distances))
            # Avoid log(0) by adding small epsilon
            log_distances = np.log(np.maximum(distances, 1e-10))
            
            # Linear regression on log scale
            slope, _, _, _, _ = stats.linregress(periods, log_distances)
            convergence_rate = -slope  # Negative slope indicates decay
        else:
            convergence_rate = 0
        
        return max(0, convergence_rate)

class IntergenerationalMobility:
    """Analyze intergenerational wealth mobility patterns"""
    
    def __init__(self):
        self.mobility_analyzer = SocialMobilityAnalyzer()
    
    def analyze_parent_child_mobility(self, parent_wealth: np.ndarray,
                                    child_wealth: np.ndarray) -> Dict:
        """
        Analyze mobility between parent and child generations
        
        Args:
            parent_wealth: Wealth of parent generation
            child_wealth: Wealth of child generation
            
        Returns:
            Dictionary with intergenerational mobility analysis
        """
        # Basic correlation analysis
        correlation = np.corrcoef(parent_wealth, child_wealth)[0, 1]
        
        # Rank correlation (more robust to outliers)
        rank_correlation = stats.spearmanr(parent_wealth, child_wealth)[0]
        
        # Mobility analysis using class transitions
        mobility_metrics = self.mobility_analyzer.analyze_mobility(
            parent_wealth, child_wealth
        )
        
        # Calculate intergenerational elasticity
        elasticity = self._calculate_intergenerational_elasticity(
            parent_wealth, child_wealth
        )
        
        # Analyze mobility by parent wealth quintiles
        quintile_analysis = self._analyze_by_parent_quintiles(
            parent_wealth, child_wealth
        )
        
        return {
            'correlation': correlation,
            'rank_correlation': rank_correlation,
            'mobility_metrics': mobility_metrics,
            'intergenerational_elasticity': elasticity,
            'quintile_analysis': quintile_analysis,
            'mobility_summary': self._summarize_mobility(mobility_metrics, elasticity)
        }
    
    def _calculate_intergenerational_elasticity(self, parent_wealth: np.ndarray,
                                              child_wealth: np.ndarray) -> Dict:
        """
        Calculate intergenerational elasticity of wealth
        
        Lower elasticity indicates higher mobility
        """
        # Handle zero and negative values by adding small constant
        min_wealth = 1000  # Minimum wealth for log transformation
        
        parent_log = np.log(np.maximum(parent_wealth, min_wealth))
        child_log = np.log(np.maximum(child_wealth, min_wealth))
        
        # Linear regression: log(child_wealth) = alpha + beta * log(parent_wealth)
        slope, intercept, r_value, p_value, std_err = stats.linregress(parent_log, child_log)
        
        return {
            'elasticity': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'standard_error': std_err,
            'mobility_interpretation': self._interpret_elasticity(slope)
        }
    
    def _interpret_elasticity(self, elasticity: float) -> str:
        """Interpret intergenerational elasticity value"""
        if elasticity < 0.2:
            return "Very high mobility"
        elif elasticity < 0.4:
            return "High mobility"
        elif elasticity < 0.6:
            return "Moderate mobility"
        elif elasticity < 0.8:
            return "Low mobility"
        else:
            return "Very low mobility"
    
    def _analyze_by_parent_quintiles(self, parent_wealth: np.ndarray,
                                   child_wealth: np.ndarray) -> Dict:
        """Analyze child outcomes by parent wealth quintiles"""
        # Define parent quintiles
        parent_quintiles = pd.qcut(parent_wealth, q=5, labels=False)
        
        quintile_analysis = {}
        
        for quintile in range(5):
            mask = parent_quintiles == quintile
            child_wealth_quintile = child_wealth[mask]
            
            if len(child_wealth_quintile) > 0:
                # Define child quintiles based on overall distribution
                child_quintiles = pd.qcut(child_wealth, q=5, labels=False)
                child_quintiles_subset = child_quintiles[mask]
                
                # Calculate transition probabilities
                transition_probs = np.bincount(child_quintiles_subset, minlength=5) / len(child_quintiles_subset)
                
                quintile_analysis[f'parent_quintile_{quintile + 1}'] = {
                    'parent_wealth_range': (np.min(parent_wealth[mask]), np.max(parent_wealth[mask])),
                    'child_mean_wealth': np.mean(child_wealth_quintile),
                    'child_median_wealth': np.median(child_wealth_quintile),
                    'child_quintile_distribution': transition_probs,
                    'upward_mobility_rate': np.sum(transition_probs[quintile + 1:]) if quintile < 4 else 0,
                    'downward_mobility_rate': np.sum(transition_probs[:quintile]) if quintile > 0 else 0,
                    'persistence_rate': transition_probs[quintile]
                }
        
        return quintile_analysis
    
    def _summarize_mobility(self, mobility_metrics: MobilityMetrics, 
                          elasticity: Dict) -> Dict:
        """Create summary of mobility patterns"""
        return {
            'overall_mobility_level': elasticity['mobility_interpretation'],
            'class_persistence_average': np.mean(list(mobility_metrics.persistence_rates.values())),
            'upward_mobility_rate': mobility_metrics.upward_mobility,
            'downward_mobility_rate': mobility_metrics.downward_mobility,
            'most_persistent_class': max(mobility_metrics.persistence_rates.items(), key=lambda x: x[1]),
            'least_persistent_class': min(mobility_metrics.persistence_rates.items(), key=lambda x: x[1])
        }
    
    def simulate_generational_wealth_transfer(self, initial_wealth: np.ndarray,
                                            n_generations: int,
                                            inheritance_rate: float = 0.7,
                                            mobility_factor: float = 0.3) -> Dict:
        """
        Simulate wealth transfer across multiple generations
        
        Args:
            initial_wealth: Starting wealth distribution
            n_generations: Number of generations to simulate
            inheritance_rate: Proportion of wealth inherited
            mobility_factor: Factor controlling social mobility
            
        Returns:
            Dictionary with simulation results
        """
        generations = [initial_wealth.copy()]
        mobility_metrics_over_time = []
        
        for gen in range(1, n_generations + 1):
            previous_wealth = generations[-1]
            
            # Simulate inheritance
            inherited_wealth = previous_wealth * inheritance_rate
            
            # Add mobility/random factors
            mobility_noise = np.random.normal(0, mobility_factor * np.std(previous_wealth), 
                                            len(previous_wealth))
            
            # Economic growth factor
            growth_factor = np.random.normal(1.02, 0.05)  # 2% average growth with volatility
            
            # Calculate new generation wealth
            new_wealth = (inherited_wealth + mobility_noise) * growth_factor
            new_wealth = np.maximum(new_wealth, 0)  # Ensure non-negative wealth
            
            generations.append(new_wealth)
            
            # Calculate mobility metrics between generations
            if gen > 0:
                mobility = self.analyze_parent_child_mobility(previous_wealth, new_wealth)
                mobility_metrics_over_time.append(mobility)
        
        return {
            'generations': generations,
            'mobility_over_time': mobility_metrics_over_time,
            'wealth_evolution': self._analyze_wealth_evolution(generations),
            'inequality_evolution': self._analyze_inequality_evolution(generations)
        }
    
    def _analyze_wealth_evolution(self, generations: List[np.ndarray]) -> Dict:
        """Analyze how wealth statistics evolve across generations"""
        evolution = {
            'mean_wealth': [],
            'median_wealth': [],
            'std_wealth': [],
            'total_wealth': []
        }
        
        for gen_wealth in generations:
            evolution['mean_wealth'].append(np.mean(gen_wealth))
            evolution['median_wealth'].append(np.median(gen_wealth))
            evolution['std_wealth'].append(np.std(gen_wealth))
            evolution['total_wealth'].append(np.sum(gen_wealth))
        
        return evolution
    
    def _analyze_inequality_evolution(self, generations: List[np.ndarray]) -> Dict:
        """Analyze how inequality evolves across generations"""
        from .inequality_metrics import GiniCoefficient
        
        gini_calc = GiniCoefficient()
        
        evolution = {
            'gini_coefficients': [],
            'top_10_percent_share': [],
            'bottom_50_percent_share': []
        }
        
        for gen_wealth in generations:
            # Gini coefficient
            gini = gini_calc.calculate(gen_wealth)
            evolution['gini_coefficients'].append(gini)
            
            # Wealth shares
            sorted_wealth = np.sort(gen_wealth)
            total_wealth = np.sum(sorted_wealth)
            
            if total_wealth > 0:
                # Top 10%
                top_10_idx = int(len(sorted_wealth) * 0.9)
                top_10_share = np.sum(sorted_wealth[top_10_idx:]) / total_wealth
                evolution['top_10_percent_share'].append(top_10_share)
                
                # Bottom 50%
                bottom_50_idx = int(len(sorted_wealth) * 0.5)
                bottom_50_share = np.sum(sorted_wealth[:bottom_50_idx]) / total_wealth
                evolution['bottom_50_percent_share'].append(bottom_50_share)
            else:
                evolution['top_10_percent_share'].append(0)
                evolution['bottom_50_percent_share'].append(0)
        
        return evolution