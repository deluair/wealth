"""
Consumption Analysis

Classes for analyzing wealth distribution through consumption patterns,
including consumer behavior, demand analysis, and consumption-driven wealth flows.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ConsumerSegment(Enum):
    """Consumer segment classifications"""
    LOW_INCOME = "low_income"
    MIDDLE_INCOME = "middle_income"
    HIGH_INCOME = "high_income"
    ULTRA_HIGH_NET_WORTH = "ultra_high_net_worth"

class ConsumptionCategory(Enum):
    """Categories of consumption"""
    NECESSITIES = "necessities"
    DISCRETIONARY = "discretionary"
    LUXURY = "luxury"
    INVESTMENT = "investment"
    SERVICES = "services"

@dataclass
class Consumer:
    """Represents a consumer in the economy"""
    id: str
    segment: ConsumerSegment
    income: float
    wealth: float
    age: int
    location: str
    preferences: Dict[ConsumptionCategory, float]  # Preference weights
    price_sensitivity: float  # 0-1, higher means more sensitive
    brand_loyalty: float  # 0-1, higher means more loyal
    consumption_propensity: float  # Marginal propensity to consume

@dataclass
class Product:
    """Represents a product or service"""
    id: str
    name: str
    category: ConsumptionCategory
    base_price: float
    quality_level: float  # 0-1
    brand_strength: float  # 0-1
    necessity_level: float  # 0-1, higher means more necessary
    income_elasticity: float  # Income elasticity of demand
    price_elasticity: float  # Price elasticity of demand
    substitute_availability: float  # 0-1, availability of substitutes

@dataclass
class ConsumptionTransaction:
    """Represents a consumption transaction"""
    consumer_id: str
    product_id: str
    quantity: float
    price_paid: float
    timestamp: float
    satisfaction_level: float  # 0-1
    repeat_purchase: bool

class ConsumptionAnalyzer:
    """Analyze wealth flows through consumption patterns"""
    
    def __init__(self):
        self.consumers = {}
        self.products = {}
        self.transactions = []
        self.market_conditions = {
            'inflation_rate': 0.02,
            'unemployment_rate': 0.05,
            'consumer_confidence': 0.7,
            'interest_rate': 0.03
        }
    
    def add_consumers(self, consumers: List[Consumer]) -> None:
        """Add consumers to the analysis"""
        for consumer in consumers:
            self.consumers[consumer.id] = consumer
    
    def add_products(self, products: List[Product]) -> None:
        """Add products to the analysis"""
        for product in products:
            self.products[product.id] = product
    
    def simulate_consumption_patterns(self, time_periods: int = 12,
                                    random_seed: int = 42) -> Dict:
        """
        Simulate consumption patterns over time
        
        Args:
            time_periods: Number of time periods to simulate
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results
        """
        np.random.seed(random_seed)
        simulation_results = {}
        
        # Initialize tracking variables
        period_results = []
        consumer_spending_history = {consumer_id: [] for consumer_id in self.consumers.keys()}
        product_sales_history = {product_id: [] for product_id in self.products.keys()}
        
        for period in range(time_periods):
            period_transactions = []
            period_spending = {}
            period_sales = {}
            
            # Simulate consumption for each consumer
            for consumer_id, consumer in self.consumers.items():
                consumer_transactions = self._simulate_consumer_period(consumer, period)
                period_transactions.extend(consumer_transactions)
                
                # Track consumer spending
                consumer_spending = sum(t.price_paid * t.quantity for t in consumer_transactions)
                consumer_spending_history[consumer_id].append(consumer_spending)
                period_spending[consumer_id] = consumer_spending
            
            # Track product sales
            for product_id in self.products.keys():
                product_transactions = [t for t in period_transactions if t.product_id == product_id]
                product_sales = sum(t.price_paid * t.quantity for t in product_transactions)
                product_sales_history[product_id].append(product_sales)
                period_sales[product_id] = product_sales
            
            # Calculate period metrics
            period_metrics = self._calculate_period_metrics(period_transactions, period)
            period_results.append(period_metrics)
            
            # Update market conditions (simple evolution)
            self._update_market_conditions(period)
        
        # Compile results
        simulation_results['period_results'] = period_results
        simulation_results['consumer_spending_history'] = consumer_spending_history
        simulation_results['product_sales_history'] = product_sales_history
        simulation_results['summary_statistics'] = self._calculate_simulation_summary(period_results)
        
        # Store transactions for further analysis
        self.transactions.extend([t for period_txns in [self._simulate_consumer_period(consumer, p) 
                                                       for p in range(time_periods) 
                                                       for consumer in self.consumers.values()] 
                                 for t in period_txns])
        
        return simulation_results
    
    def _simulate_consumer_period(self, consumer: Consumer, period: int) -> List[ConsumptionTransaction]:
        """Simulate consumption for a single consumer in one period"""
        transactions = []
        
        # Calculate available spending budget
        # Base budget from income, adjusted by consumption propensity and market conditions
        base_budget = consumer.income * consumer.consumption_propensity
        
        # Adjust for market conditions
        confidence_adjustment = self.market_conditions['consumer_confidence']
        inflation_adjustment = 1 + self.market_conditions['inflation_rate']
        budget = base_budget * confidence_adjustment / inflation_adjustment
        
        # Add wealth effect (small portion of wealth can be consumed)
        wealth_consumption = consumer.wealth * 0.01  # 1% of wealth per period
        budget += wealth_consumption
        
        remaining_budget = budget
        
        # Prioritize consumption by category (necessities first)
        category_priorities = [
            ConsumptionCategory.NECESSITIES,
            ConsumptionCategory.SERVICES,
            ConsumptionCategory.DISCRETIONARY,
            ConsumptionCategory.LUXURY,
            ConsumptionCategory.INVESTMENT
        ]
        
        for category in category_priorities:
            if remaining_budget <= 0:
                break
            
            # Get products in this category
            category_products = [p for p in self.products.values() if p.category == category]
            
            if not category_products:
                continue
            
            # Calculate category budget allocation
            category_preference = consumer.preferences.get(category, 0.1)
            category_budget = budget * category_preference
            category_budget = min(category_budget, remaining_budget)
            
            # Select and purchase products in this category
            category_transactions = self._simulate_category_consumption(
                consumer, category_products, category_budget, period
            )
            
            transactions.extend(category_transactions)
            spent_in_category = sum(t.price_paid * t.quantity for t in category_transactions)
            remaining_budget -= spent_in_category
        
        return transactions
    
    def _simulate_category_consumption(self, consumer: Consumer, products: List[Product],
                                     budget: float, period: int) -> List[ConsumptionTransaction]:
        """Simulate consumption within a specific category"""
        transactions = []
        remaining_budget = budget
        
        # Sort products by utility (considering price, quality, brand, etc.)
        product_utilities = []
        for product in products:
            utility = self._calculate_product_utility(consumer, product)
            product_utilities.append((product, utility))
        
        # Sort by utility (descending)
        product_utilities.sort(key=lambda x: x[1], reverse=True)
        
        for product, utility in product_utilities:
            if remaining_budget <= 0:
                break
            
            # Calculate demand for this product
            demand_quantity = self._calculate_product_demand(consumer, product, remaining_budget)
            
            if demand_quantity > 0:
                # Calculate actual price (may include discounts, taxes, etc.)
                actual_price = self._calculate_actual_price(product, consumer, period)
                
                # Check if consumer can afford and wants to buy
                total_cost = actual_price * demand_quantity
                
                if total_cost <= remaining_budget:
                    # Create transaction
                    satisfaction = self._calculate_satisfaction(consumer, product, actual_price)
                    
                    transaction = ConsumptionTransaction(
                        consumer_id=consumer.id,
                        product_id=product.id,
                        quantity=demand_quantity,
                        price_paid=actual_price,
                        timestamp=period,
                        satisfaction_level=satisfaction,
                        repeat_purchase=self._is_repeat_purchase(consumer.id, product.id)
                    )
                    
                    transactions.append(transaction)
                    remaining_budget -= total_cost
        
        return transactions
    
    def _calculate_product_utility(self, consumer: Consumer, product: Product) -> float:
        """Calculate utility of a product for a consumer"""
        utility = 0.0
        
        # Base utility from quality
        utility += product.quality_level * 10
        
        # Brand utility (adjusted by consumer's brand loyalty)
        utility += product.brand_strength * consumer.brand_loyalty * 5
        
        # Necessity utility (higher for necessities)
        utility += product.necessity_level * 8
        
        # Price utility (negative, adjusted by price sensitivity)
        price_penalty = product.base_price * consumer.price_sensitivity * 0.01
        utility -= price_penalty
        
        # Income appropriateness (products should match income level)
        income_match = self._calculate_income_match(consumer, product)
        utility += income_match * 3
        
        # Add some randomness
        utility += np.random.normal(0, 1)
        
        return max(0, utility)
    
    def _calculate_income_match(self, consumer: Consumer, product: Product) -> float:
        """Calculate how well a product matches consumer's income level"""
        # Normalize income to 0-1 scale (assuming max income of 1M)
        normalized_income = min(consumer.income / 1000000, 1.0)
        
        # Normalize price to 0-1 scale (assuming max price of 10K)
        normalized_price = min(product.base_price / 10000, 1.0)
        
        # Calculate match (closer values = better match)
        match_score = 1 - abs(normalized_income - normalized_price)
        
        return match_score
    
    def _calculate_product_demand(self, consumer: Consumer, product: Product,
                                available_budget: float) -> float:
        """Calculate demand quantity for a product"""
        # Base demand (simplified)
        base_demand = 1.0
        
        # Adjust for income elasticity
        income_factor = (consumer.income / 50000) ** product.income_elasticity  # Assuming median income of 50K
        base_demand *= income_factor
        
        # Adjust for price elasticity
        price_factor = (product.base_price / 100) ** product.price_elasticity  # Assuming base price of 100
        base_demand *= price_factor
        
        # Adjust for necessity level
        necessity_factor = 0.5 + product.necessity_level * 0.5
        base_demand *= necessity_factor
        
        # Budget constraint
        max_affordable = available_budget / product.base_price
        demand = min(base_demand, max_affordable)
        
        # Add randomness
        demand *= np.random.uniform(0.8, 1.2)
        
        return max(0, demand)
    
    def _calculate_actual_price(self, product: Product, consumer: Consumer, period: int) -> float:
        """Calculate actual price including discounts, taxes, etc."""
        base_price = product.base_price
        
        # Inflation adjustment
        inflation_factor = (1 + self.market_conditions['inflation_rate']) ** period
        adjusted_price = base_price * inflation_factor
        
        # Consumer segment discounts
        if consumer.segment == ConsumerSegment.LOW_INCOME:
            adjusted_price *= 0.95  # 5% discount for low income
        elif consumer.segment == ConsumerSegment.ULTRA_HIGH_NET_WORTH:
            adjusted_price *= 1.1  # Premium pricing for ultra high net worth
        
        # Brand loyalty discount
        if consumer.brand_loyalty > 0.8:
            adjusted_price *= 0.98  # 2% loyalty discount
        
        # Random price variation
        adjusted_price *= np.random.uniform(0.95, 1.05)
        
        return adjusted_price
    
    def _calculate_satisfaction(self, consumer: Consumer, product: Product, price_paid: float) -> float:
        """Calculate consumer satisfaction with a purchase"""
        satisfaction = 0.5  # Base satisfaction
        
        # Quality satisfaction
        satisfaction += product.quality_level * 0.3
        
        # Price satisfaction (lower price = higher satisfaction)
        expected_price = product.base_price
        price_satisfaction = max(0, (expected_price - price_paid) / expected_price)
        satisfaction += price_satisfaction * 0.2
        
        # Brand satisfaction
        satisfaction += product.brand_strength * consumer.brand_loyalty * 0.2
        
        # Income appropriateness satisfaction
        income_match = self._calculate_income_match(consumer, product)
        satisfaction += income_match * 0.1
        
        # Add randomness
        satisfaction += np.random.normal(0, 0.1)
        
        return max(0, min(1, satisfaction))
    
    def _is_repeat_purchase(self, consumer_id: str, product_id: str) -> bool:
        """Check if this is a repeat purchase"""
        # Check existing transactions
        for transaction in self.transactions:
            if transaction.consumer_id == consumer_id and transaction.product_id == product_id:
                return True
        return False
    
    def _calculate_period_metrics(self, transactions: List[ConsumptionTransaction], period: int) -> Dict:
        """Calculate metrics for a single period"""
        if not transactions:
            return {
                'period': period,
                'total_transactions': 0,
                'total_spending': 0,
                'average_transaction_value': 0,
                'average_satisfaction': 0
            }
        
        total_spending = sum(t.price_paid * t.quantity for t in transactions)
        average_transaction_value = total_spending / len(transactions)
        average_satisfaction = np.mean([t.satisfaction_level for t in transactions])
        
        # Spending by consumer segment
        spending_by_segment = {}
        for segment in ConsumerSegment:
            segment_transactions = [
                t for t in transactions 
                if self.consumers[t.consumer_id].segment == segment
            ]
            segment_spending = sum(t.price_paid * t.quantity for t in segment_transactions)
            spending_by_segment[segment.value] = segment_spending
        
        # Spending by category
        spending_by_category = {}
        for category in ConsumptionCategory:
            category_transactions = [
                t for t in transactions 
                if self.products[t.product_id].category == category
            ]
            category_spending = sum(t.price_paid * t.quantity for t in category_transactions)
            spending_by_category[category.value] = category_spending
        
        return {
            'period': period,
            'total_transactions': len(transactions),
            'total_spending': total_spending,
            'average_transaction_value': average_transaction_value,
            'average_satisfaction': average_satisfaction,
            'spending_by_segment': spending_by_segment,
            'spending_by_category': spending_by_category
        }
    
    def _update_market_conditions(self, period: int) -> None:
        """Update market conditions over time"""
        # Simple market evolution
        self.market_conditions['consumer_confidence'] += np.random.normal(0, 0.02)
        self.market_conditions['consumer_confidence'] = max(0.3, min(1.0, self.market_conditions['consumer_confidence']))
        
        self.market_conditions['inflation_rate'] += np.random.normal(0, 0.001)
        self.market_conditions['inflation_rate'] = max(0, min(0.1, self.market_conditions['inflation_rate']))
    
    def _calculate_simulation_summary(self, period_results: List[Dict]) -> Dict:
        """Calculate summary statistics for the entire simulation"""
        if not period_results:
            return {}
        
        total_spendings = [p['total_spending'] for p in period_results]
        total_transactions = [p['total_transactions'] for p in period_results]
        average_satisfactions = [p['average_satisfaction'] for p in period_results]
        
        return {
            'total_periods': len(period_results),
            'total_spending_all_periods': sum(total_spendings),
            'average_spending_per_period': np.mean(total_spendings),
            'spending_growth_rate': self._calculate_growth_rate(total_spendings),
            'total_transactions_all_periods': sum(total_transactions),
            'average_transactions_per_period': np.mean(total_transactions),
            'overall_average_satisfaction': np.mean(average_satisfactions),
            'satisfaction_trend': self._calculate_trend(average_satisfactions)
        }
    
    def analyze_consumption_patterns(self) -> Dict:
        """Analyze consumption patterns from transaction data"""
        if not self.transactions:
            return {'error': 'No transaction data available. Run simulation first.'}
        
        analysis = {}
        
        # Consumer behavior analysis
        analysis['consumer_behavior'] = self._analyze_consumer_behavior()
        
        # Product performance analysis
        analysis['product_performance'] = self._analyze_product_performance()
        
        # Market segmentation analysis
        analysis['market_segmentation'] = self._analyze_market_segmentation()
        
        # Demand elasticity analysis
        analysis['demand_elasticity'] = self._analyze_demand_elasticity()
        
        # Wealth distribution impact
        analysis['wealth_distribution_impact'] = self._analyze_wealth_distribution_impact()
        
        # Consumption inequality
        analysis['consumption_inequality'] = self._analyze_consumption_inequality()
        
        return analysis
    
    def _analyze_consumer_behavior(self) -> Dict:
        """Analyze consumer behavior patterns"""
        behavior_analysis = {}
        
        # Spending patterns by consumer segment
        spending_by_segment = {}
        for segment in ConsumerSegment:
            segment_consumers = [c for c in self.consumers.values() if c.segment == segment]
            segment_transactions = [t for t in self.transactions if t.consumer_id in [c.id for c in segment_consumers]]
            
            if segment_transactions:
                total_spending = sum(t.price_paid * t.quantity for t in segment_transactions)
                avg_transaction_value = total_spending / len(segment_transactions)
                avg_satisfaction = np.mean([t.satisfaction_level for t in segment_transactions])
                
                spending_by_segment[segment.value] = {
                    'total_spending': total_spending,
                    'transaction_count': len(segment_transactions),
                    'average_transaction_value': avg_transaction_value,
                    'average_satisfaction': avg_satisfaction
                }
        
        behavior_analysis['spending_by_segment'] = spending_by_segment
        
        # Repeat purchase analysis
        repeat_purchases = [t for t in self.transactions if t.repeat_purchase]
        behavior_analysis['repeat_purchase_rate'] = len(repeat_purchases) / len(self.transactions) if self.transactions else 0
        
        # Price sensitivity analysis
        price_sensitive_consumers = [c for c in self.consumers.values() if c.price_sensitivity > 0.7]
        behavior_analysis['price_sensitive_share'] = len(price_sensitive_consumers) / len(self.consumers) if self.consumers else 0
        
        return behavior_analysis
    
    def _analyze_product_performance(self) -> Dict:
        """Analyze product performance"""
        performance_analysis = {}
        
        # Sales by product
        sales_by_product = {}
        for product_id, product in self.products.items():
            product_transactions = [t for t in self.transactions if t.product_id == product_id]
            
            if product_transactions:
                total_sales = sum(t.price_paid * t.quantity for t in product_transactions)
                total_quantity = sum(t.quantity for t in product_transactions)
                avg_satisfaction = np.mean([t.satisfaction_level for t in product_transactions])
                
                sales_by_product[product_id] = {
                    'total_sales': total_sales,
                    'total_quantity': total_quantity,
                    'transaction_count': len(product_transactions),
                    'average_satisfaction': avg_satisfaction,
                    'category': product.category.value
                }
        
        performance_analysis['sales_by_product'] = sales_by_product
        
        # Top performing products
        if sales_by_product:
            sorted_products = sorted(sales_by_product.items(), key=lambda x: x[1]['total_sales'], reverse=True)
            performance_analysis['top_products'] = sorted_products[:5]
        
        # Sales by category
        sales_by_category = {}
        for category in ConsumptionCategory:
            category_transactions = [t for t in self.transactions if self.products[t.product_id].category == category]
            
            if category_transactions:
                total_sales = sum(t.price_paid * t.quantity for t in category_transactions)
                sales_by_category[category.value] = total_sales
        
        performance_analysis['sales_by_category'] = sales_by_category
        
        return performance_analysis
    
    def _analyze_market_segmentation(self) -> Dict:
        """Analyze market segmentation effectiveness"""
        segmentation_analysis = {}
        
        # Consumer distribution
        segment_distribution = {}
        for segment in ConsumerSegment:
            segment_count = len([c for c in self.consumers.values() if c.segment == segment])
            segment_distribution[segment.value] = segment_count
        
        segmentation_analysis['segment_distribution'] = segment_distribution
        
        # Spending power by segment
        spending_power = {}
        for segment in ConsumerSegment:
            segment_consumers = [c for c in self.consumers.values() if c.segment == segment]
            if segment_consumers:
                avg_income = np.mean([c.income for c in segment_consumers])
                avg_wealth = np.mean([c.wealth for c in segment_consumers])
                spending_power[segment.value] = {
                    'average_income': avg_income,
                    'average_wealth': avg_wealth
                }
        
        segmentation_analysis['spending_power'] = spending_power
        
        return segmentation_analysis
    
    def _analyze_demand_elasticity(self) -> Dict:
        """Analyze demand elasticity patterns"""
        elasticity_analysis = {}
        
        # Price elasticity by product category
        category_elasticities = {}
        for category in ConsumptionCategory:
            category_products = [p for p in self.products.values() if p.category == category]
            if category_products:
                avg_price_elasticity = np.mean([p.price_elasticity for p in category_products])
                avg_income_elasticity = np.mean([p.income_elasticity for p in category_products])
                
                category_elasticities[category.value] = {
                    'average_price_elasticity': avg_price_elasticity,
                    'average_income_elasticity': avg_income_elasticity
                }
        
        elasticity_analysis['category_elasticities'] = category_elasticities
        
        return elasticity_analysis
    
    def _analyze_wealth_distribution_impact(self) -> Dict:
        """Analyze impact of consumption on wealth distribution"""
        wealth_impact = {}
        
        # Consumption as percentage of wealth by segment
        consumption_wealth_ratios = {}
        for segment in ConsumerSegment:
            segment_consumers = [c for c in self.consumers.values() if c.segment == segment]
            segment_transactions = [t for t in self.transactions if t.consumer_id in [c.id for c in segment_consumers]]
            
            if segment_consumers and segment_transactions:
                total_consumption = sum(t.price_paid * t.quantity for t in segment_transactions)
                total_wealth = sum(c.wealth for c in segment_consumers)
                
                if total_wealth > 0:
                    consumption_wealth_ratio = total_consumption / total_wealth
                    consumption_wealth_ratios[segment.value] = consumption_wealth_ratio
        
        wealth_impact['consumption_wealth_ratios'] = consumption_wealth_ratios
        
        # Wealth transfer through consumption
        # This represents how consumption redistributes wealth from consumers to producers
        total_consumption_value = sum(t.price_paid * t.quantity for t in self.transactions)
        wealth_impact['total_wealth_transfer'] = total_consumption_value
        
        return wealth_impact
    
    def _analyze_consumption_inequality(self) -> Dict:
        """Analyze inequality in consumption patterns"""
        inequality_analysis = {}
        
        # Calculate consumption by consumer
        consumer_consumption = {}
        for consumer_id in self.consumers.keys():
            consumer_transactions = [t for t in self.transactions if t.consumer_id == consumer_id]
            total_consumption = sum(t.price_paid * t.quantity for t in consumer_transactions)
            consumer_consumption[consumer_id] = total_consumption
        
        consumption_values = list(consumer_consumption.values())
        
        if consumption_values:
            # Gini coefficient for consumption
            gini_coefficient = self._calculate_gini_coefficient(consumption_values)
            inequality_analysis['consumption_gini'] = gini_coefficient
            
            # Consumption percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            consumption_percentiles = {}
            for p in percentiles:
                consumption_percentiles[f'p{p}'] = np.percentile(consumption_values, p)
            
            inequality_analysis['consumption_percentiles'] = consumption_percentiles
            
            # Top 10% consumption share
            sorted_consumption = sorted(consumption_values, reverse=True)
            top_10_percent_count = max(1, len(sorted_consumption) // 10)
            top_10_percent_consumption = sum(sorted_consumption[:top_10_percent_count])
            total_consumption = sum(consumption_values)
            
            inequality_analysis['top_10_percent_share'] = top_10_percent_consumption / total_consumption if total_consumption > 0 else 0
        
        return inequality_analysis
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate compound annual growth rate"""
        if len(values) < 2 or values[0] == 0:
            return 0
        
        periods = len(values) - 1
        growth_rate = (values[-1] / values[0]) ** (1/periods) - 1
        return growth_rate
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
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
    
    def generate_consumption_scenarios(self, scenarios: List[Dict]) -> Dict:
        """
        Generate different consumption scenarios
        
        Args:
            scenarios: List of scenario parameters
            
        Returns:
            Dictionary with scenario results
        """
        scenario_results = {}
        
        # Store original state
        original_market_conditions = self.market_conditions.copy()
        original_consumers = {cid: Consumer(**consumer.__dict__) for cid, consumer in self.consumers.items()}
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'scenario_{i+1}')
            
            # Apply scenario modifications
            self._apply_scenario_modifications(scenario)
            
            # Run simulation
            simulation_results = self.simulate_consumption_patterns(
                time_periods=scenario.get('time_periods', 12),
                random_seed=scenario.get('random_seed', 42 + i)
            )
            
            # Analyze results
            analysis_results = self.analyze_consumption_patterns()
            
            scenario_results[scenario_name] = {
                'scenario_parameters': scenario,
                'simulation_results': simulation_results,
                'analysis_results': analysis_results
            }
            
            # Reset to original state
            self.market_conditions = original_market_conditions.copy()
            self.consumers = {cid: Consumer(**consumer.__dict__) for cid, consumer in original_consumers.items()}
            self.transactions = []  # Clear transactions for next scenario
        
        return scenario_results
    
    def _apply_scenario_modifications(self, scenario: Dict) -> None:
        """Apply modifications for a specific scenario"""
        # Modify market conditions
        if 'market_conditions' in scenario:
            for key, value in scenario['market_conditions'].items():
                if key in self.market_conditions:
                    self.market_conditions[key] = value
        
        # Modify consumer characteristics
        if 'consumer_modifications' in scenario:
            modifications = scenario['consumer_modifications']
            
            for consumer_id, consumer in self.consumers.items():
                # Apply income changes
                if 'income_multiplier' in modifications:
                    consumer.income *= modifications['income_multiplier']
                
                # Apply wealth changes
                if 'wealth_multiplier' in modifications:
                    consumer.wealth *= modifications['wealth_multiplier']
                
                # Apply preference changes
                if 'preference_changes' in modifications:
                    for category, change in modifications['preference_changes'].items():
                        if hasattr(ConsumptionCategory, category.upper()):
                            cat_enum = ConsumptionCategory[category.upper()]
                            if cat_enum in consumer.preferences:
                                consumer.preferences[cat_enum] *= change
        
        # Modify product characteristics
        if 'product_modifications' in scenario:
            modifications = scenario['product_modifications']
            
            for product_id, product in self.products.items():
                # Apply price changes
                if 'price_multiplier' in modifications:
                    product.base_price *= modifications['price_multiplier']
                
                # Apply quality changes
                if 'quality_multiplier' in modifications:
                    product.quality_level *= modifications['quality_multiplier']
                    product.quality_level = min(1.0, product.quality_level)  # Cap at 1.0