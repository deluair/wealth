"""
Production Analysis

Classes for analyzing wealth creation through production processes,
including factor inputs, productivity analysis, and production functions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, minimize_scalar
from scipy import stats
import matplotlib.pyplot as plt

@dataclass
class ProductionFactor:
    """Represents a factor of production"""
    name: str
    quantity: float
    cost_per_unit: float
    productivity: float
    elasticity: float  # Output elasticity with respect to this factor
    depreciation_rate: float = 0.0

@dataclass
class ProductionTechnology:
    """Represents production technology characteristics"""
    name: str
    total_factor_productivity: float
    returns_to_scale: float  # >1 increasing, =1 constant, <1 decreasing
    technology_level: float
    innovation_rate: float
    adoption_cost: float

class ProductionAnalyzer:
    """Analyze wealth creation through production processes"""
    
    def __init__(self):
        self.production_functions = {
            'cobb_douglas': self._cobb_douglas,
            'ces': self._ces_production,
            'leontief': self._leontief,
            'translog': self._translog
        }
    
    def analyze_production_system(self, factors: List[ProductionFactor],
                                technology: ProductionTechnology,
                                production_function: str = 'cobb_douglas') -> Dict:
        """
        Comprehensive analysis of a production system
        
        Args:
            factors: List of production factors
            technology: Production technology specification
            production_function: Type of production function to use
            
        Returns:
            Dictionary with production analysis results
        """
        analysis = {}
        
        # Basic production metrics
        analysis['basic_metrics'] = self._calculate_basic_metrics(factors, technology)
        
        # Production function analysis
        analysis['production_function'] = self._analyze_production_function(
            factors, technology, production_function
        )
        
        # Factor productivity analysis
        analysis['factor_productivity'] = self._analyze_factor_productivity(factors)
        
        # Cost analysis
        analysis['cost_analysis'] = self._analyze_costs(factors, technology)
        
        # Efficiency analysis
        analysis['efficiency_analysis'] = self._analyze_efficiency(factors, technology)
        
        # Returns to scale analysis
        analysis['returns_to_scale'] = self._analyze_returns_to_scale(
            factors, technology, production_function
        )
        
        # Optimization analysis
        analysis['optimization'] = self._optimize_production(
            factors, technology, production_function
        )
        
        return analysis
    
    def _calculate_basic_metrics(self, factors: List[ProductionFactor],
                               technology: ProductionTechnology) -> Dict:
        """Calculate basic production metrics"""
        total_cost = sum(factor.quantity * factor.cost_per_unit for factor in factors)
        total_quantity = sum(factor.quantity for factor in factors)
        
        # Weighted average productivity
        weighted_productivity = sum(
            factor.quantity * factor.productivity for factor in factors
        ) / total_quantity if total_quantity > 0 else 0
        
        return {
            'total_factors': len(factors),
            'total_input_cost': total_cost,
            'total_input_quantity': total_quantity,
            'average_factor_cost': total_cost / len(factors) if factors else 0,
            'weighted_average_productivity': weighted_productivity,
            'technology_level': technology.technology_level,
            'total_factor_productivity': technology.total_factor_productivity
        }
    
    def _analyze_production_function(self, factors: List[ProductionFactor],
                                   technology: ProductionTechnology,
                                   function_type: str) -> Dict:
        """Analyze the production function"""
        if function_type not in self.production_functions:
            raise ValueError(f"Unknown production function: {function_type}")
        
        # Prepare factor inputs
        factor_quantities = [factor.quantity for factor in factors]
        factor_elasticities = [factor.elasticity for factor in factors]
        
        # Calculate output
        output = self.production_functions[function_type](
            factor_quantities, factor_elasticities, technology
        )
        
        # Calculate marginal products
        marginal_products = self._calculate_marginal_products(
            factors, technology, function_type
        )
        
        # Calculate average products
        average_products = [
            output / factor.quantity if factor.quantity > 0 else 0
            for factor in factors
        ]
        
        return {
            'function_type': function_type,
            'total_output': output,
            'output_per_unit_cost': output / sum(f.quantity * f.cost_per_unit for f in factors) if factors else 0,
            'marginal_products': marginal_products,
            'average_products': average_products,
            'factor_elasticities': factor_elasticities
        }
    
    def _cobb_douglas(self, quantities: List[float], elasticities: List[float],
                     technology: ProductionTechnology) -> float:
        """Cobb-Douglas production function: Y = A * ∏(X_i^α_i)"""
        if not quantities or any(q <= 0 for q in quantities):
            return 0
        
        output = technology.total_factor_productivity
        for quantity, elasticity in zip(quantities, elasticities):
            output *= (quantity ** elasticity)
        
        return output
    
    def _ces_production(self, quantities: List[float], elasticities: List[float],
                       technology: ProductionTechnology, rho: float = -0.5) -> float:
        """CES (Constant Elasticity of Substitution) production function"""
        if not quantities:
            return 0
        
        # CES: Y = A * (∑(α_i * X_i^ρ))^(1/ρ)
        ces_sum = sum(
            elasticity * (quantity ** rho) if quantity > 0 else 0
            for quantity, elasticity in zip(quantities, elasticities)
        )
        
        if ces_sum <= 0:
            return 0
        
        return technology.total_factor_productivity * (ces_sum ** (1/rho))
    
    def _leontief(self, quantities: List[float], elasticities: List[float],
                 technology: ProductionTechnology) -> float:
        """Leontief (fixed proportions) production function"""
        if not quantities:
            return 0
        
        # Leontief: Y = A * min(X_i/a_i) where a_i are technical coefficients
        # Using elasticities as inverse of technical coefficients
        ratios = [
            quantity * elasticity if elasticity > 0 else 0
            for quantity, elasticity in zip(quantities, elasticities)
        ]
        
        return technology.total_factor_productivity * min(ratios) if ratios else 0
    
    def _translog(self, quantities: List[float], elasticities: List[float],
                 technology: ProductionTechnology) -> float:
        """Translog production function (simplified version)"""
        if not quantities or any(q <= 0 for q in quantities):
            return 0
        
        # Simplified translog: ln(Y) = ln(A) + ∑(α_i * ln(X_i)) + 0.5 * ∑∑(β_ij * ln(X_i) * ln(X_j))
        log_quantities = [np.log(q) for q in quantities]
        
        # Linear terms
        linear_sum = sum(
            elasticity * log_q
            for elasticity, log_q in zip(elasticities, log_quantities)
        )
        
        # Quadratic terms (simplified - only diagonal terms)
        quadratic_sum = 0.1 * sum(log_q ** 2 for log_q in log_quantities)
        
        log_output = np.log(technology.total_factor_productivity) + linear_sum + quadratic_sum
        
        return np.exp(log_output)
    
    def _calculate_marginal_products(self, factors: List[ProductionFactor],
                                   technology: ProductionTechnology,
                                   function_type: str) -> List[float]:
        """Calculate marginal products for each factor"""
        marginal_products = []
        
        for i, factor in enumerate(factors):
            # Calculate marginal product using numerical differentiation
            delta = 0.01 * factor.quantity  # Small change
            
            # Original quantities
            original_quantities = [f.quantity for f in factors]
            elasticities = [f.elasticity for f in factors]
            
            # Calculate output with original quantities
            original_output = self.production_functions[function_type](
                original_quantities, elasticities, technology
            )
            
            # Calculate output with increased quantity for factor i
            modified_quantities = original_quantities.copy()
            modified_quantities[i] += delta
            
            modified_output = self.production_functions[function_type](
                modified_quantities, elasticities, technology
            )
            
            # Marginal product
            marginal_product = (modified_output - original_output) / delta
            marginal_products.append(marginal_product)
        
        return marginal_products
    
    def _analyze_factor_productivity(self, factors: List[ProductionFactor]) -> Dict:
        """Analyze productivity of individual factors"""
        productivity_analysis = {}
        
        # Basic productivity statistics
        productivities = [factor.productivity for factor in factors]
        productivity_analysis['statistics'] = {
            'mean_productivity': np.mean(productivities),
            'median_productivity': np.median(productivities),
            'std_productivity': np.std(productivities),
            'min_productivity': np.min(productivities),
            'max_productivity': np.max(productivities)
        }
        
        # Productivity distribution
        productivity_analysis['distribution'] = {
            'gini_coefficient': self._calculate_gini(productivities),
            'coefficient_of_variation': np.std(productivities) / np.mean(productivities) if np.mean(productivities) > 0 else 0
        }
        
        # Factor-specific analysis
        factor_analysis = {}
        for factor in factors:
            factor_analysis[factor.name] = {
                'productivity': factor.productivity,
                'quantity': factor.quantity,
                'total_productive_capacity': factor.productivity * factor.quantity,
                'cost_per_unit': factor.cost_per_unit,
                'productivity_per_cost': factor.productivity / factor.cost_per_unit if factor.cost_per_unit > 0 else 0
            }
        
        productivity_analysis['factor_details'] = factor_analysis
        
        return productivity_analysis
    
    def _analyze_costs(self, factors: List[ProductionFactor],
                      technology: ProductionTechnology) -> Dict:
        """Analyze production costs"""
        cost_analysis = {}
        
        # Variable costs
        variable_costs = sum(factor.quantity * factor.cost_per_unit for factor in factors)
        
        # Fixed costs (technology adoption and depreciation)
        depreciation_costs = sum(
            factor.quantity * factor.cost_per_unit * factor.depreciation_rate
            for factor in factors
        )
        
        fixed_costs = technology.adoption_cost + depreciation_costs
        total_costs = variable_costs + fixed_costs
        
        cost_analysis['cost_breakdown'] = {
            'variable_costs': variable_costs,
            'fixed_costs': fixed_costs,
            'depreciation_costs': depreciation_costs,
            'total_costs': total_costs
        }
        
        # Cost per factor
        factor_costs = {}
        for factor in factors:
            factor_cost = factor.quantity * factor.cost_per_unit
            factor_costs[factor.name] = {
                'total_cost': factor_cost,
                'cost_share': factor_cost / variable_costs if variable_costs > 0 else 0,
                'cost_per_unit': factor.cost_per_unit,
                'quantity': factor.quantity
            }
        
        cost_analysis['factor_costs'] = factor_costs
        
        # Cost efficiency metrics
        total_quantity = sum(factor.quantity for factor in factors)
        cost_analysis['efficiency_metrics'] = {
            'cost_per_unit_input': total_costs / total_quantity if total_quantity > 0 else 0,
            'variable_cost_ratio': variable_costs / total_costs if total_costs > 0 else 0,
            'fixed_cost_ratio': fixed_costs / total_costs if total_costs > 0 else 0
        }
        
        return cost_analysis
    
    def _analyze_efficiency(self, factors: List[ProductionFactor],
                          technology: ProductionTechnology) -> Dict:
        """Analyze production efficiency"""
        efficiency_analysis = {}
        
        # Technical efficiency (actual vs. potential output)
        # Simplified: based on factor productivities and technology level
        potential_output = sum(
            factor.quantity * factor.productivity * technology.technology_level
            for factor in factors
        )
        
        actual_output = sum(
            factor.quantity * factor.productivity
            for factor in factors
        )
        
        technical_efficiency = actual_output / potential_output if potential_output > 0 else 0
        
        # Allocative efficiency (optimal factor mix)
        # Simplified: based on cost-productivity ratios
        productivity_cost_ratios = [
            factor.productivity / factor.cost_per_unit if factor.cost_per_unit > 0 else 0
            for factor in factors
        ]
        
        # Coefficient of variation of productivity-cost ratios (lower is better)
        mean_ratio = np.mean(productivity_cost_ratios)
        allocative_efficiency = 1 - (np.std(productivity_cost_ratios) / mean_ratio) if mean_ratio > 0 else 0
        allocative_efficiency = max(0, min(1, allocative_efficiency))
        
        # Overall efficiency
        overall_efficiency = technical_efficiency * allocative_efficiency
        
        efficiency_analysis['efficiency_measures'] = {
            'technical_efficiency': technical_efficiency,
            'allocative_efficiency': allocative_efficiency,
            'overall_efficiency': overall_efficiency,
            'potential_output': potential_output,
            'actual_output': actual_output
        }
        
        # Efficiency by factor
        factor_efficiencies = {}
        for factor in factors:
            factor_efficiency = (factor.productivity * technology.technology_level) / factor.productivity if factor.productivity > 0 else 0
            factor_efficiencies[factor.name] = {
                'efficiency': min(1.0, factor_efficiency),
                'productivity_cost_ratio': factor.productivity / factor.cost_per_unit if factor.cost_per_unit > 0 else 0
            }
        
        efficiency_analysis['factor_efficiencies'] = factor_efficiencies
        
        return efficiency_analysis
    
    def _analyze_returns_to_scale(self, factors: List[ProductionFactor],
                                technology: ProductionTechnology,
                                function_type: str) -> Dict:
        """Analyze returns to scale"""
        returns_analysis = {}
        
        # Calculate output at current scale
        original_quantities = [factor.quantity for factor in factors]
        elasticities = [factor.elasticity for factor in factors]
        
        original_output = self.production_functions[function_type](
            original_quantities, elasticities, technology
        )
        
        # Test different scales
        scale_factors = [0.5, 0.8, 1.2, 1.5, 2.0]
        scale_results = []
        
        for scale in scale_factors:
            scaled_quantities = [q * scale for q in original_quantities]
            scaled_output = self.production_functions[function_type](
                scaled_quantities, elasticities, technology
            )
            
            output_ratio = scaled_output / original_output if original_output > 0 else 0
            returns_to_scale = output_ratio / scale if scale > 0 else 0
            
            scale_results.append({
                'scale_factor': scale,
                'output_ratio': output_ratio,
                'returns_to_scale': returns_to_scale
            })
        
        returns_analysis['scale_analysis'] = scale_results
        
        # Classify returns to scale
        # Use elasticity sum for Cobb-Douglas, or empirical measurement for others
        if function_type == 'cobb_douglas':
            elasticity_sum = sum(elasticities)
            if elasticity_sum > 1.05:
                scale_type = "Increasing returns to scale"
            elif elasticity_sum < 0.95:
                scale_type = "Decreasing returns to scale"
            else:
                scale_type = "Constant returns to scale"
            
            returns_analysis['scale_classification'] = {
                'type': scale_type,
                'elasticity_sum': elasticity_sum
            }
        else:
            # Use empirical measurement
            avg_returns = np.mean([r['returns_to_scale'] for r in scale_results])
            if avg_returns > 1.05:
                scale_type = "Increasing returns to scale"
            elif avg_returns < 0.95:
                scale_type = "Decreasing returns to scale"
            else:
                scale_type = "Constant returns to scale"
            
            returns_analysis['scale_classification'] = {
                'type': scale_type,
                'average_returns': avg_returns
            }
        
        return returns_analysis
    
    def _optimize_production(self, factors: List[ProductionFactor],
                           technology: ProductionTechnology,
                           function_type: str) -> Dict:
        """Optimize production given constraints"""
        optimization_results = {}
        
        # Cost minimization for given output level
        target_output = self.production_functions[function_type](
            [factor.quantity for factor in factors],
            [factor.elasticity for factor in factors],
            technology
        )
        
        cost_minimization = self._minimize_cost_for_output(
            factors, technology, function_type, target_output
        )
        
        optimization_results['cost_minimization'] = cost_minimization
        
        # Profit maximization (assuming output price)
        output_price = 100  # Assumed output price
        profit_maximization = self._maximize_profit(
            factors, technology, function_type, output_price
        )
        
        optimization_results['profit_maximization'] = profit_maximization
        
        # Efficiency optimization
        efficiency_optimization = self._optimize_efficiency(
            factors, technology, function_type
        )
        
        optimization_results['efficiency_optimization'] = efficiency_optimization
        
        return optimization_results
    
    def _minimize_cost_for_output(self, factors: List[ProductionFactor],
                                technology: ProductionTechnology,
                                function_type: str,
                                target_output: float) -> Dict:
        """Minimize cost for a given output level"""
        
        def objective(quantities):
            """Cost function to minimize"""
            return sum(q * factor.cost_per_unit for q, factor in zip(quantities, factors))
        
        def constraint(quantities):
            """Output constraint"""
            elasticities = [factor.elasticity for factor in factors]
            actual_output = self.production_functions[function_type](
                quantities, elasticities, technology
            )
            return actual_output - target_output
        
        # Initial guess
        initial_quantities = [factor.quantity for factor in factors]
        
        # Bounds (quantities must be positive)
        bounds = [(0.1, None) for _ in factors]
        
        # Constraint
        constraints = {'type': 'eq', 'fun': constraint}
        
        try:
            result = minimize(
                objective,
                initial_quantities,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_quantities = result.x
                optimal_cost = result.fun
                
                return {
                    'success': True,
                    'optimal_quantities': optimal_quantities,
                    'optimal_cost': optimal_cost,
                    'cost_savings': objective(initial_quantities) - optimal_cost,
                    'quantity_changes': [
                        (opt - orig) / orig if orig > 0 else 0
                        for opt, orig in zip(optimal_quantities, initial_quantities)
                    ]
                }
            else:
                return {'success': False, 'message': result.message}
        
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def _maximize_profit(self, factors: List[ProductionFactor],
                        technology: ProductionTechnology,
                        function_type: str,
                        output_price: float) -> Dict:
        """Maximize profit given output price"""
        
        def objective(quantities):
            """Negative profit function (to minimize)"""
            elasticities = [factor.elasticity for factor in factors]
            output = self.production_functions[function_type](
                quantities, elasticities, technology
            )
            revenue = output * output_price
            cost = sum(q * factor.cost_per_unit for q, factor in zip(quantities, factors))
            return -(revenue - cost)  # Negative for minimization
        
        # Initial guess
        initial_quantities = [factor.quantity for factor in factors]
        
        # Bounds (quantities must be positive)
        bounds = [(0.1, None) for _ in factors]
        
        try:
            result = minimize(
                objective,
                initial_quantities,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if result.success:
                optimal_quantities = result.x
                optimal_profit = -result.fun
                
                # Calculate current profit
                current_output = self.production_functions[function_type](
                    initial_quantities,
                    [factor.elasticity for factor in factors],
                    technology
                )
                current_cost = sum(q * factor.cost_per_unit for q, factor in zip(initial_quantities, factors))
                current_profit = current_output * output_price - current_cost
                
                return {
                    'success': True,
                    'optimal_quantities': optimal_quantities,
                    'optimal_profit': optimal_profit,
                    'profit_improvement': optimal_profit - current_profit,
                    'quantity_changes': [
                        (opt - orig) / orig if orig > 0 else 0
                        for opt, orig in zip(optimal_quantities, initial_quantities)
                    ]
                }
            else:
                return {'success': False, 'message': result.message}
        
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def _optimize_efficiency(self, factors: List[ProductionFactor],
                           technology: ProductionTechnology,
                           function_type: str) -> Dict:
        """Optimize for maximum efficiency"""
        
        def objective(quantities):
            """Negative efficiency function (to minimize)"""
            elasticities = [factor.elasticity for factor in factors]
            output = self.production_functions[function_type](
                quantities, elasticities, technology
            )
            total_input = sum(quantities)
            efficiency = output / total_input if total_input > 0 else 0
            return -efficiency  # Negative for minimization
        
        # Initial guess
        initial_quantities = [factor.quantity for factor in factors]
        
        # Bounds (quantities must be positive)
        bounds = [(0.1, None) for _ in factors]
        
        try:
            result = minimize(
                objective,
                initial_quantities,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if result.success:
                optimal_quantities = result.x
                optimal_efficiency = -result.fun
                
                # Calculate current efficiency
                current_output = self.production_functions[function_type](
                    initial_quantities,
                    [factor.elasticity for factor in factors],
                    technology
                )
                current_efficiency = current_output / sum(initial_quantities)
                
                return {
                    'success': True,
                    'optimal_quantities': optimal_quantities,
                    'optimal_efficiency': optimal_efficiency,
                    'efficiency_improvement': optimal_efficiency - current_efficiency,
                    'quantity_changes': [
                        (opt - orig) / orig if orig > 0 else 0
                        for opt, orig in zip(optimal_quantities, initial_quantities)
                    ]
                }
            else:
                return {'success': False, 'message': result.message}
        
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def simulate_production_scenarios(self, base_factors: List[ProductionFactor],
                                    base_technology: ProductionTechnology,
                                    scenarios: List[Dict]) -> Dict:
        """
        Simulate different production scenarios
        
        Args:
            base_factors: Base factor configuration
            base_technology: Base technology configuration
            scenarios: List of scenario modifications
            
        Returns:
            Dictionary with scenario analysis results
        """
        scenario_results = {}
        
        # Analyze base scenario
        base_analysis = self.analyze_production_system(
            base_factors, base_technology
        )
        scenario_results['base_scenario'] = base_analysis
        
        # Analyze each scenario
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'scenario_{i+1}')
            
            # Modify factors
            modified_factors = []
            for factor in base_factors:
                modified_factor = ProductionFactor(
                    name=factor.name,
                    quantity=factor.quantity * scenario.get('factor_quantity_multiplier', {}).get(factor.name, 1.0),
                    cost_per_unit=factor.cost_per_unit * scenario.get('factor_cost_multiplier', {}).get(factor.name, 1.0),
                    productivity=factor.productivity * scenario.get('factor_productivity_multiplier', {}).get(factor.name, 1.0),
                    elasticity=factor.elasticity,
                    depreciation_rate=factor.depreciation_rate
                )
                modified_factors.append(modified_factor)
            
            # Modify technology
            modified_technology = ProductionTechnology(
                name=base_technology.name,
                total_factor_productivity=base_technology.total_factor_productivity * scenario.get('tfp_multiplier', 1.0),
                returns_to_scale=base_technology.returns_to_scale,
                technology_level=base_technology.technology_level * scenario.get('technology_multiplier', 1.0),
                innovation_rate=base_technology.innovation_rate,
                adoption_cost=base_technology.adoption_cost * scenario.get('adoption_cost_multiplier', 1.0)
            )
            
            # Analyze modified scenario
            scenario_analysis = self.analyze_production_system(
                modified_factors, modified_technology
            )
            
            # Compare with base scenario
            comparison = self._compare_scenarios(base_analysis, scenario_analysis)
            scenario_analysis['comparison_with_base'] = comparison
            
            scenario_results[scenario_name] = scenario_analysis
        
        return scenario_results
    
    def _compare_scenarios(self, base_analysis: Dict, scenario_analysis: Dict) -> Dict:
        """Compare two production scenarios"""
        comparison = {}
        
        # Output comparison
        base_output = base_analysis['production_function']['total_output']
        scenario_output = scenario_analysis['production_function']['total_output']
        
        comparison['output_change'] = {
            'absolute': scenario_output - base_output,
            'relative': (scenario_output - base_output) / base_output if base_output > 0 else 0
        }
        
        # Cost comparison
        base_cost = base_analysis['cost_analysis']['cost_breakdown']['total_costs']
        scenario_cost = scenario_analysis['cost_analysis']['cost_breakdown']['total_costs']
        
        comparison['cost_change'] = {
            'absolute': scenario_cost - base_cost,
            'relative': (scenario_cost - base_cost) / base_cost if base_cost > 0 else 0
        }
        
        # Efficiency comparison
        base_efficiency = base_analysis['efficiency_analysis']['efficiency_measures']['overall_efficiency']
        scenario_efficiency = scenario_analysis['efficiency_analysis']['efficiency_measures']['overall_efficiency']
        
        comparison['efficiency_change'] = {
            'absolute': scenario_efficiency - base_efficiency,
            'relative': (scenario_efficiency - base_efficiency) / base_efficiency if base_efficiency > 0 else 0
        }
        
        # Productivity comparison
        base_productivity = base_analysis['basic_metrics']['weighted_average_productivity']
        scenario_productivity = scenario_analysis['basic_metrics']['weighted_average_productivity']
        
        comparison['productivity_change'] = {
            'absolute': scenario_productivity - base_productivity,
            'relative': (scenario_productivity - base_productivity) / base_productivity if base_productivity > 0 else 0
        }
        
        return comparison
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient"""
        if not values or len(values) == 0:
            return 0
        
        values = np.array(values)
        values = values[values >= 0]
        
        if len(values) == 0 or np.sum(values) == 0:
            return 0
        
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        index_sum = np.sum((np.arange(1, n + 1) * sorted_values))
        total_sum = np.sum(sorted_values)
        
        gini = (2 * index_sum) / (n * total_sum) - (n + 1) / n
        return max(0, min(1, gini))