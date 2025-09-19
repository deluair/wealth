"""
AI-Driven Wealth Creation

Classes for analyzing new wealth creation opportunities arising from AI,
including digital assets, platform economics, and AI-enabled business models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class AIOpportunityType(Enum):
    """Types of AI-driven wealth opportunities"""
    AI_SERVICES = "ai_services"
    DATA_MONETIZATION = "data_monetization"
    PLATFORM_BUSINESS = "platform_business"
    AUTOMATION_SOLUTIONS = "automation_solutions"
    AI_RESEARCH = "ai_research"
    DIGITAL_PRODUCTS = "digital_products"
    AI_CONSULTING = "ai_consulting"
    ALGORITHMIC_TRADING = "algorithmic_trading"

class DigitalAssetType(Enum):
    """Types of digital assets"""
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DATA_ASSETS = "data_assets"
    ALGORITHMS = "algorithms"
    DIGITAL_PLATFORMS = "digital_platforms"
    AI_MODELS = "ai_models"
    DIGITAL_CONTENT = "digital_content"
    VIRTUAL_ASSETS = "virtual_assets"
    NETWORK_EFFECTS = "network_effects"

class MarketMaturity(Enum):
    """Market maturity levels"""
    EMERGING = "emerging"
    GROWTH = "growth"
    MATURE = "mature"
    DECLINING = "declining"

@dataclass
class AIOpportunity:
    """Represents an AI-driven wealth creation opportunity"""
    id: str
    name: str
    opportunity_type: AIOpportunityType
    market_size: float  # Total addressable market
    growth_rate: float  # Annual growth rate
    entry_barrier: float  # 0-1, difficulty of entry
    capital_requirement: float  # Initial capital needed
    time_to_profitability: int  # Months
    scalability_factor: float  # How quickly it can scale
    network_effects: float  # 0-1, strength of network effects
    ai_dependency: float  # 0-1, how dependent on AI technology
    competitive_advantage_duration: int  # Months of sustainable advantage
    
@dataclass
class DigitalAsset:
    """Represents a digital asset that can create wealth"""
    id: str
    name: str
    asset_type: DigitalAssetType
    creation_cost: float
    maintenance_cost_annual: float
    revenue_potential_annual: float
    depreciation_rate: float  # Annual depreciation
    network_value_multiplier: float  # How network size affects value
    data_value_coefficient: float  # How data quantity affects value
    ai_enhancement_factor: float  # How AI improves the asset
    transferability: float  # 0-1, how easily it can be sold/transferred
    
@dataclass
class AIEntrepreneur:
    """Represents an entrepreneur in the AI space"""
    id: str
    name: str
    technical_skills: float  # 0-1
    business_skills: float  # 0-1
    ai_expertise: float  # 0-1
    network_strength: float  # 0-1
    risk_tolerance: float  # 0-1
    capital_access: float  # Available capital
    previous_experience: int  # Years of relevant experience
    innovation_capability: float  # 0-1
    
@dataclass
class AIMarket:
    """Represents a market for AI opportunities"""
    id: str
    name: str
    size: float
    growth_rate: float
    maturity: MarketMaturity
    competition_intensity: float  # 0-1
    regulation_level: float  # 0-1
    technology_adoption_rate: float  # 0-1
    customer_willingness_to_pay: float  # 0-1

class AIWealthCreator:
    """Analyze AI-driven wealth creation opportunities and outcomes"""
    
    def __init__(self):
        self.opportunities = {}
        self.digital_assets = {}
        self.entrepreneurs = {}
        self.markets = {}
        self.global_ai_trends = {
            'ai_investment_growth': 0.25,  # 25% annual growth in AI investment
            'ai_adoption_rate': 0.15,      # 15% annual increase in adoption
            'ai_talent_shortage': 0.8,     # 80% of companies report talent shortage
            'ai_productivity_multiplier': 1.3  # AI increases productivity by 30%
        }
    
    def add_opportunities(self, opportunities: List[AIOpportunity]) -> None:
        """Add AI opportunities to the analysis"""
        for opportunity in opportunities:
            self.opportunities[opportunity.id] = opportunity
    
    def add_digital_assets(self, assets: List[DigitalAsset]) -> None:
        """Add digital assets to the analysis"""
        for asset in assets:
            self.digital_assets[asset.id] = asset
    
    def add_entrepreneurs(self, entrepreneurs: List[AIEntrepreneur]) -> None:
        """Add entrepreneurs to the analysis"""
        for entrepreneur in entrepreneurs:
            self.entrepreneurs[entrepreneur.id] = entrepreneur
    
    def add_markets(self, markets: List[AIMarket]) -> None:
        """Add markets to the analysis"""
        for market in markets:
            self.markets[market.id] = market
    
    def simulate_ai_wealth_creation(self, time_horizon: int = 10,
                                  random_seed: int = 42) -> Dict:
        """
        Simulate AI-driven wealth creation over time
        
        Args:
            time_horizon: Number of years to simulate
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results
        """
        np.random.seed(random_seed)
        
        results = {
            'yearly_results': [],
            'opportunity_outcomes': {},
            'entrepreneur_journeys': {},
            'digital_asset_valuations': {},
            'market_evolution': {}
        }
        
        # Initialize tracking
        for opp_id in self.opportunities.keys():
            results['opportunity_outcomes'][opp_id] = []
        
        for ent_id in self.entrepreneurs.keys():
            results['entrepreneur_journeys'][ent_id] = []
        
        for asset_id in self.digital_assets.keys():
            results['digital_asset_valuations'][asset_id] = []
        
        for market_id in self.markets.keys():
            results['market_evolution'][market_id] = []
        
        # Simulate each year
        for year in range(time_horizon):
            year_results = self._simulate_year_ai_wealth(year)
            results['yearly_results'].append(year_results)
            
            # Track individual components
            self._update_opportunity_outcomes(results['opportunity_outcomes'], year)
            self._update_entrepreneur_journeys(results['entrepreneur_journeys'], year)
            self._update_digital_asset_valuations(results['digital_asset_valuations'], year)
            self._update_market_evolution(results['market_evolution'], year)
            
            # Update global trends
            self._update_global_ai_trends(year)
        
        # Calculate summary statistics
        results['summary'] = self._calculate_ai_wealth_summary(results)
        
        return results
    
    def _simulate_year_ai_wealth(self, year: int) -> Dict:
        """Simulate AI wealth creation for a single year"""
        
        # Calculate total wealth created this year
        total_wealth_created = 0
        wealth_by_type = {}
        
        # Simulate opportunity outcomes
        opportunity_results = {}
        for opp_id, opportunity in self.opportunities.items():
            outcome = self._simulate_opportunity_outcome(opportunity, year)
            opportunity_results[opp_id] = outcome
            total_wealth_created += outcome['wealth_created']
            
            opp_type = opportunity.opportunity_type.value
            if opp_type not in wealth_by_type:
                wealth_by_type[opp_type] = 0
            wealth_by_type[opp_type] += outcome['wealth_created']
        
        # Simulate digital asset value creation
        digital_asset_results = {}
        for asset_id, asset in self.digital_assets.items():
            valuation = self._calculate_digital_asset_value(asset, year)
            digital_asset_results[asset_id] = valuation
            total_wealth_created += valuation['value_increase']
        
        # Simulate entrepreneur success
        entrepreneur_results = {}
        for ent_id, entrepreneur in self.entrepreneurs.items():
            success_metrics = self._calculate_entrepreneur_success(entrepreneur, year)
            entrepreneur_results[ent_id] = success_metrics
        
        # Calculate market dynamics
        market_results = {}
        for market_id, market in self.markets.items():
            market_dynamics = self._calculate_market_dynamics(market, year)
            market_results[market_id] = market_dynamics
        
        # Calculate wealth distribution effects
        wealth_distribution = self._analyze_ai_wealth_distribution(
            total_wealth_created, opportunity_results, entrepreneur_results
        )
        
        return {
            'year': year,
            'total_wealth_created': total_wealth_created,
            'wealth_by_type': wealth_by_type,
            'opportunity_results': opportunity_results,
            'digital_asset_results': digital_asset_results,
            'entrepreneur_results': entrepreneur_results,
            'market_results': market_results,
            'wealth_distribution': wealth_distribution,
            'ai_adoption_level': self._calculate_ai_adoption_level(year),
            'innovation_index': self._calculate_innovation_index(year)
        }
    
    def _simulate_opportunity_outcome(self, opportunity: AIOpportunity, year: int) -> Dict:
        """Simulate the outcome of an AI opportunity"""
        
        # Base success probability
        success_probability = 0.3  # 30% base success rate for AI ventures
        
        # Adjust for opportunity characteristics
        success_probability *= (1 - opportunity.entry_barrier * 0.5)  # Lower barriers = higher success
        success_probability *= (1 + opportunity.scalability_factor * 0.3)  # Higher scalability = higher success
        success_probability *= (1 + opportunity.network_effects * 0.4)  # Network effects boost success
        
        # Adjust for market conditions
        success_probability *= (1 + self.global_ai_trends['ai_adoption_rate'])
        
        # Adjust for time (early years are harder)
        time_factor = min(1.0, (year + 1) / 3)  # Ramp up over 3 years
        success_probability *= time_factor
        
        # Random success determination
        is_successful = np.random.random() < success_probability
        
        if is_successful:
            # Calculate wealth created
            base_wealth = opportunity.market_size * 0.01  # Capture 1% of market
            
            # Scale based on opportunity characteristics
            wealth_multiplier = 1.0
            wealth_multiplier *= (1 + opportunity.scalability_factor)
            wealth_multiplier *= (1 + opportunity.network_effects * 2)  # Network effects are powerful
            wealth_multiplier *= (1 + opportunity.growth_rate)
            
            # Apply AI productivity multiplier
            wealth_multiplier *= self.global_ai_trends['ai_productivity_multiplier']
            
            wealth_created = base_wealth * wealth_multiplier
            
            # Add some randomness
            wealth_created *= np.random.uniform(0.5, 2.0)
        else:
            wealth_created = 0
        
        return {
            'is_successful': is_successful,
            'wealth_created': wealth_created,
            'success_probability': success_probability,
            'market_share_captured': wealth_created / opportunity.market_size if opportunity.market_size > 0 else 0
        }
    
    def _calculate_digital_asset_value(self, asset: DigitalAsset, year: int) -> Dict:
        """Calculate the value of a digital asset"""
        
        # Base value calculation
        base_value = asset.revenue_potential_annual - asset.maintenance_cost_annual
        
        # Apply depreciation
        depreciated_value = base_value * ((1 - asset.depreciation_rate) ** year)
        
        # Apply network effects (assume network grows over time)
        network_size_factor = 1 + year * 0.2  # Network grows 20% per year
        network_value = depreciated_value * (asset.network_value_multiplier * network_size_factor)
        
        # Apply data value (assume data accumulates over time)
        data_factor = 1 + year * 0.15  # Data value grows 15% per year
        data_enhanced_value = network_value * (1 + asset.data_value_coefficient * data_factor)
        
        # Apply AI enhancement
        ai_enhancement = 1 + asset.ai_enhancement_factor * self.global_ai_trends['ai_productivity_multiplier']
        final_value = data_enhanced_value * ai_enhancement
        
        # Calculate value increase from previous year
        if year == 0:
            previous_value = asset.creation_cost
        else:
            # Simplified: assume previous value was 90% of current (for growth calculation)
            previous_value = final_value * 0.9
        
        value_increase = final_value - previous_value
        
        return {
            'current_value': final_value,
            'value_increase': max(0, value_increase),
            'network_contribution': network_value - depreciated_value,
            'data_contribution': data_enhanced_value - network_value,
            'ai_contribution': final_value - data_enhanced_value
        }
    
    def _calculate_entrepreneur_success(self, entrepreneur: AIEntrepreneur, year: int) -> Dict:
        """Calculate entrepreneur success metrics"""
        
        # Base success score
        success_score = 0.0
        
        # Skill contributions
        success_score += entrepreneur.technical_skills * 0.25
        success_score += entrepreneur.business_skills * 0.25
        success_score += entrepreneur.ai_expertise * 0.20
        success_score += entrepreneur.innovation_capability * 0.15
        success_score += entrepreneur.network_strength * 0.15
        
        # Experience bonus
        experience_bonus = min(0.2, entrepreneur.previous_experience * 0.02)
        success_score += experience_bonus
        
        # Capital access impact
        capital_factor = min(1.0, entrepreneur.capital_access / 1000000)  # Normalize to $1M
        success_score *= (0.5 + capital_factor * 0.5)  # Capital access affects success
        
        # Market timing (AI market is hot in early years, then normalizes)
        timing_factor = max(0.8, 1.2 - year * 0.05)  # Decreases over time
        success_score *= timing_factor
        
        # Calculate wealth generated
        if success_score > 0.6:  # High success threshold
            wealth_generated = entrepreneur.capital_access * success_score * 5  # 5x return for successful entrepreneurs
        elif success_score > 0.4:  # Moderate success
            wealth_generated = entrepreneur.capital_access * success_score * 2  # 2x return
        else:
            wealth_generated = entrepreneur.capital_access * success_score * 0.5  # Partial loss
        
        # Add randomness
        wealth_generated *= np.random.uniform(0.3, 3.0)
        
        return {
            'success_score': success_score,
            'wealth_generated': wealth_generated,
            'capital_multiplier': wealth_generated / entrepreneur.capital_access if entrepreneur.capital_access > 0 else 0,
            'risk_adjusted_return': wealth_generated * entrepreneur.risk_tolerance
        }
    
    def _calculate_market_dynamics(self, market: AIMarket, year: int) -> Dict:
        """Calculate market dynamics and evolution"""
        
        # Market size evolution
        current_size = market.size * ((1 + market.growth_rate) ** year)
        
        # Competition evolution
        # Competition typically increases over time in growing markets
        competition_increase = year * 0.05 if market.maturity == MarketMaturity.GROWTH else year * 0.02
        current_competition = min(1.0, market.competition_intensity + competition_increase)
        
        # Technology adoption evolution
        adoption_increase = year * 0.1 * market.technology_adoption_rate
        current_adoption = min(1.0, market.technology_adoption_rate + adoption_increase)
        
        # Customer willingness to pay (may decrease as market matures)
        willingness_factor = 1.0 - (year * 0.02) if market.maturity == MarketMaturity.MATURE else 1.0
        current_willingness = market.customer_willingness_to_pay * willingness_factor
        
        # Market attractiveness score
        attractiveness = (
            (current_size / 1000000) * 0.3 +  # Size factor (normalized to millions)
            market.growth_rate * 0.3 +
            (1 - current_competition) * 0.2 +
            current_adoption * 0.1 +
            current_willingness * 0.1
        )
        
        return {
            'current_size': current_size,
            'size_growth': current_size - market.size,
            'competition_level': current_competition,
            'adoption_level': current_adoption,
            'customer_willingness': current_willingness,
            'market_attractiveness': attractiveness
        }
    
    def _analyze_ai_wealth_distribution(self, total_wealth: float,
                                      opportunity_results: Dict,
                                      entrepreneur_results: Dict) -> Dict:
        """Analyze how AI-created wealth is distributed"""
        
        # Calculate wealth concentration
        entrepreneur_wealth = [r['wealth_generated'] for r in entrepreneur_results.values()]
        
        if entrepreneur_wealth:
            # Gini coefficient for entrepreneur wealth
            gini = self._calculate_gini_coefficient(entrepreneur_wealth)
            
            # Top 10% share
            sorted_wealth = sorted(entrepreneur_wealth, reverse=True)
            top_10_percent_count = max(1, len(sorted_wealth) // 10)
            top_10_percent_wealth = sum(sorted_wealth[:top_10_percent_count])
            top_10_percent_share = top_10_percent_wealth / sum(entrepreneur_wealth) if sum(entrepreneur_wealth) > 0 else 0
            
            # Successful vs unsuccessful split
            successful_entrepreneurs = [w for w in entrepreneur_wealth if w > 0]
            success_rate = len(successful_entrepreneurs) / len(entrepreneur_wealth)
            
        else:
            gini = 0
            top_10_percent_share = 0
            success_rate = 0
        
        # Wealth by opportunity type
        opportunity_wealth_distribution = {}
        for opp_id, result in opportunity_results.items():
            opp_type = self.opportunities[opp_id].opportunity_type.value
            if opp_type not in opportunity_wealth_distribution:
                opportunity_wealth_distribution[opp_type] = 0
            opportunity_wealth_distribution[opp_type] += result['wealth_created']
        
        return {
            'total_wealth_created': total_wealth,
            'wealth_gini_coefficient': gini,
            'top_10_percent_share': top_10_percent_share,
            'entrepreneur_success_rate': success_rate,
            'wealth_by_opportunity_type': opportunity_wealth_distribution,
            'average_wealth_per_entrepreneur': np.mean(entrepreneur_wealth) if entrepreneur_wealth else 0,
            'median_wealth_per_entrepreneur': np.median(entrepreneur_wealth) if entrepreneur_wealth else 0
        }
    
    def _calculate_ai_adoption_level(self, year: int) -> float:
        """Calculate overall AI adoption level"""
        base_adoption = 0.1  # 10% base adoption
        growth_rate = self.global_ai_trends['ai_adoption_rate']
        
        # S-curve adoption
        adoption_level = 1 / (1 + np.exp(-0.3 * (year - 5)))
        
        return min(0.95, base_adoption + adoption_level * 0.8)  # Cap at 95%
    
    def _calculate_innovation_index(self, year: int) -> float:
        """Calculate innovation index based on AI developments"""
        base_innovation = 0.5
        
        # Innovation accelerates with AI investment
        investment_factor = (1 + self.global_ai_trends['ai_investment_growth']) ** year
        
        # But faces diminishing returns
        diminishing_returns = 1 / (1 + year * 0.1)
        
        innovation_index = base_innovation * investment_factor * diminishing_returns
        
        return min(1.0, innovation_index)
    
    def _update_opportunity_outcomes(self, outcomes: Dict, year: int) -> None:
        """Update opportunity outcome tracking"""
        for opp_id, opportunity in self.opportunities.items():
            outcome = self._simulate_opportunity_outcome(opportunity, year)
            outcomes[opp_id].append(outcome)
    
    def _update_entrepreneur_journeys(self, journeys: Dict, year: int) -> None:
        """Update entrepreneur journey tracking"""
        for ent_id, entrepreneur in self.entrepreneurs.items():
            success_metrics = self._calculate_entrepreneur_success(entrepreneur, year)
            journeys[ent_id].append(success_metrics)
    
    def _update_digital_asset_valuations(self, valuations: Dict, year: int) -> None:
        """Update digital asset valuation tracking"""
        for asset_id, asset in self.digital_assets.items():
            valuation = self._calculate_digital_asset_value(asset, year)
            valuations[asset_id].append(valuation)
    
    def _update_market_evolution(self, evolution: Dict, year: int) -> None:
        """Update market evolution tracking"""
        for market_id, market in self.markets.items():
            dynamics = self._calculate_market_dynamics(market, year)
            evolution[market_id].append(dynamics)
    
    def _update_global_ai_trends(self, year: int) -> None:
        """Update global AI trends over time"""
        # AI investment growth may slow down over time
        self.global_ai_trends['ai_investment_growth'] *= 0.98  # Slight decrease each year
        
        # AI adoption rate increases but with diminishing returns
        self.global_ai_trends['ai_adoption_rate'] *= 1.02  # Slight increase
        
        # Talent shortage may improve over time as more people get trained
        self.global_ai_trends['ai_talent_shortage'] *= 0.95  # Gradual improvement
        
        # Productivity multiplier may increase with better AI
        self.global_ai_trends['ai_productivity_multiplier'] *= 1.01  # Gradual improvement
    
    def _calculate_ai_wealth_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics for AI wealth creation simulation"""
        yearly_results = results['yearly_results']
        
        if not yearly_results:
            return {}
        
        # Extract time series data
        total_wealth_series = [r['total_wealth_created'] for r in yearly_results]
        gini_series = [r['wealth_distribution']['wealth_gini_coefficient'] for r in yearly_results]
        success_rate_series = [r['wealth_distribution']['entrepreneur_success_rate'] for r in yearly_results]
        ai_adoption_series = [r['ai_adoption_level'] for r in yearly_results]
        
        # Calculate summary statistics
        summary = {
            'total_simulation_years': len(yearly_results),
            'cumulative_wealth_created': sum(total_wealth_series),
            'average_annual_wealth_creation': np.mean(total_wealth_series),
            'wealth_creation_growth_rate': self._calculate_growth_rate(total_wealth_series),
            'final_wealth_inequality': gini_series[-1] if gini_series else 0,
            'inequality_trend': self._calculate_trend(gini_series),
            'final_success_rate': success_rate_series[-1] if success_rate_series else 0,
            'success_rate_trend': self._calculate_trend(success_rate_series),
            'final_ai_adoption': ai_adoption_series[-1] if ai_adoption_series else 0,
            'ai_adoption_growth_rate': self._calculate_growth_rate(ai_adoption_series)
        }
        
        # Analyze top performing opportunities
        summary['top_opportunities'] = self._identify_top_opportunities(results)
        
        # Analyze top performing entrepreneurs
        summary['top_entrepreneurs'] = self._identify_top_entrepreneurs(results)
        
        # Analyze most valuable digital assets
        summary['top_digital_assets'] = self._identify_top_digital_assets(results)
        
        return summary
    
    def _identify_top_opportunities(self, results: Dict) -> List[Dict]:
        """Identify top performing opportunities"""
        opportunity_performance = {}
        
        for opp_id, outcomes in results['opportunity_outcomes'].items():
            total_wealth = sum(outcome['wealth_created'] for outcome in outcomes)
            success_rate = sum(1 for outcome in outcomes if outcome['is_successful']) / len(outcomes)
            
            opportunity_performance[opp_id] = {
                'opportunity_id': opp_id,
                'opportunity_name': self.opportunities[opp_id].name,
                'total_wealth_created': total_wealth,
                'success_rate': success_rate,
                'opportunity_type': self.opportunities[opp_id].opportunity_type.value
            }
        
        # Sort by total wealth created
        top_opportunities = sorted(
            opportunity_performance.values(),
            key=lambda x: x['total_wealth_created'],
            reverse=True
        )
        
        return top_opportunities[:5]  # Top 5
    
    def _identify_top_entrepreneurs(self, results: Dict) -> List[Dict]:
        """Identify top performing entrepreneurs"""
        entrepreneur_performance = {}
        
        for ent_id, journey in results['entrepreneur_journeys'].items():
            total_wealth = sum(step['wealth_generated'] for step in journey)
            avg_success_score = np.mean([step['success_score'] for step in journey])
            
            entrepreneur_performance[ent_id] = {
                'entrepreneur_id': ent_id,
                'entrepreneur_name': self.entrepreneurs[ent_id].name,
                'total_wealth_generated': total_wealth,
                'average_success_score': avg_success_score,
                'final_capital_multiplier': journey[-1]['capital_multiplier'] if journey else 0
            }
        
        # Sort by total wealth generated
        top_entrepreneurs = sorted(
            entrepreneur_performance.values(),
            key=lambda x: x['total_wealth_generated'],
            reverse=True
        )
        
        return top_entrepreneurs[:5]  # Top 5
    
    def _identify_top_digital_assets(self, results: Dict) -> List[Dict]:
        """Identify most valuable digital assets"""
        asset_performance = {}
        
        for asset_id, valuations in results['digital_asset_valuations'].items():
            final_value = valuations[-1]['current_value'] if valuations else 0
            total_value_increase = sum(val['value_increase'] for val in valuations)
            
            asset_performance[asset_id] = {
                'asset_id': asset_id,
                'asset_name': self.digital_assets[asset_id].name,
                'final_value': final_value,
                'total_value_increase': total_value_increase,
                'asset_type': self.digital_assets[asset_id].asset_type.value
            }
        
        # Sort by final value
        top_assets = sorted(
            asset_performance.values(),
            key=lambda x: x['final_value'],
            reverse=True
        )
        
        return top_assets[:5]  # Top 5
    
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
    
    def analyze_ai_wealth_scenarios(self, scenarios: List[Dict],
                                  time_horizon: int = 10) -> Dict:
        """
        Analyze different AI wealth creation scenarios
        
        Args:
            scenarios: List of scenario parameters
            time_horizon: Number of years to simulate
            
        Returns:
            Dictionary with scenario comparison results
        """
        scenario_results = {}
        
        # Store original state
        original_trends = self.global_ai_trends.copy()
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'scenario_{i+1}')
            
            # Apply scenario modifications
            self._apply_ai_scenario_modifications(scenario)
            
            # Run simulation
            simulation_results = self.simulate_ai_wealth_creation(
                time_horizon=time_horizon,
                random_seed=42 + i
            )
            
            scenario_results[scenario_name] = {
                'scenario_parameters': scenario,
                'simulation_results': simulation_results
            }
            
            # Reset to original state
            self.global_ai_trends = original_trends.copy()
        
        # Comparative analysis
        comparative_analysis = self._compare_ai_scenarios(scenario_results)
        
        return {
            'individual_scenarios': scenario_results,
            'comparative_analysis': comparative_analysis
        }
    
    def _apply_ai_scenario_modifications(self, scenario: Dict) -> None:
        """Apply modifications for a specific AI scenario"""
        
        # Modify global AI trends
        if 'ai_trends' in scenario:
            for key, value in scenario['ai_trends'].items():
                if key in self.global_ai_trends:
                    self.global_ai_trends[key] = value
        
        # Modify opportunity characteristics
        if 'opportunity_modifications' in scenario:
            modifications = scenario['opportunity_modifications']
            
            for opp_id, opportunity in self.opportunities.items():
                if 'market_size_multiplier' in modifications:
                    opportunity.market_size *= modifications['market_size_multiplier']
                
                if 'growth_rate_adjustment' in modifications:
                    opportunity.growth_rate += modifications['growth_rate_adjustment']
                
                if 'entry_barrier_adjustment' in modifications:
                    opportunity.entry_barrier += modifications['entry_barrier_adjustment']
                    opportunity.entry_barrier = max(0, min(1, opportunity.entry_barrier))
        
        # Modify entrepreneur characteristics
        if 'entrepreneur_modifications' in scenario:
            modifications = scenario['entrepreneur_modifications']
            
            for ent_id, entrepreneur in self.entrepreneurs.items():
                if 'skill_boost' in modifications:
                    entrepreneur.technical_skills = min(1.0, entrepreneur.technical_skills + modifications['skill_boost'])
                    entrepreneur.ai_expertise = min(1.0, entrepreneur.ai_expertise + modifications['skill_boost'])
                
                if 'capital_multiplier' in modifications:
                    entrepreneur.capital_access *= modifications['capital_multiplier']
    
    def _compare_ai_scenarios(self, scenario_results: Dict) -> Dict:
        """Compare AI wealth creation scenarios"""
        comparison = {
            'wealth_creation_comparison': {},
            'inequality_comparison': {},
            'success_rate_comparison': {},
            'ai_adoption_comparison': {}
        }
        
        for scenario_name, results in scenario_results.items():
            summary = results['simulation_results']['summary']
            
            comparison['wealth_creation_comparison'][scenario_name] = {
                'cumulative_wealth': summary.get('cumulative_wealth_created', 0),
                'growth_rate': summary.get('wealth_creation_growth_rate', 0)
            }
            
            comparison['inequality_comparison'][scenario_name] = {
                'final_gini': summary.get('final_wealth_inequality', 0),
                'trend': summary.get('inequality_trend', 'stable')
            }
            
            comparison['success_rate_comparison'][scenario_name] = {
                'final_rate': summary.get('final_success_rate', 0),
                'trend': summary.get('success_rate_trend', 'stable')
            }
            
            comparison['ai_adoption_comparison'][scenario_name] = {
                'final_adoption': summary.get('final_ai_adoption', 0),
                'growth_rate': summary.get('ai_adoption_growth_rate', 0)
            }
        
        return comparison
    
    def create_ai_opportunities_dataset(self) -> List[AIOpportunity]:
        """Create a realistic dataset of AI opportunities"""
        opportunities = [
            AIOpportunity(
                "ai_healthcare", "AI Healthcare Diagnostics", AIOpportunityType.AI_SERVICES,
                500000000, 0.25, 0.7, 5000000, 18, 3.0, 0.6, 0.9, 36
            ),
            
            AIOpportunity(
                "autonomous_vehicles", "Autonomous Vehicle Platform", AIOpportunityType.PLATFORM_BUSINESS,
                2000000000, 0.30, 0.9, 50000000, 60, 5.0, 0.8, 0.95, 120
            ),
            
            AIOpportunity(
                "ai_trading", "AI Trading Algorithms", AIOpportunityType.ALGORITHMIC_TRADING,
                100000000, 0.15, 0.8, 1000000, 6, 2.0, 0.3, 0.85, 24
            ),
            
            AIOpportunity(
                "data_analytics", "Enterprise Data Analytics", AIOpportunityType.DATA_MONETIZATION,
                300000000, 0.20, 0.5, 2000000, 12, 2.5, 0.5, 0.8, 30
            ),
            
            AIOpportunity(
                "ai_content", "AI Content Generation", AIOpportunityType.DIGITAL_PRODUCTS,
                150000000, 0.35, 0.3, 500000, 9, 4.0, 0.7, 0.9, 18
            ),
            
            AIOpportunity(
                "robotics_automation", "Industrial Robotics", AIOpportunityType.AUTOMATION_SOLUTIONS,
                800000000, 0.18, 0.8, 10000000, 36, 1.5, 0.4, 0.7, 48
            ),
            
            AIOpportunity(
                "ai_education", "AI-Powered Education", AIOpportunityType.AI_SERVICES,
                200000000, 0.28, 0.4, 1500000, 15, 3.5, 0.6, 0.8, 24
            ),
            
            AIOpportunity(
                "ai_consulting", "AI Strategy Consulting", AIOpportunityType.AI_CONSULTING,
                50000000, 0.22, 0.2, 100000, 3, 1.2, 0.2, 0.6, 12
            )
        ]
        
        return opportunities
    
    def create_digital_assets_dataset(self) -> List[DigitalAsset]:
        """Create a realistic dataset of digital assets"""
        assets = [
            DigitalAsset(
                "ai_model_nlp", "Advanced NLP Model", DigitalAssetType.AI_MODELS,
                2000000, 200000, 1000000, 0.15, 1.5, 2.0, 1.8, 0.7
            ),
            
            DigitalAsset(
                "customer_data", "Customer Behavior Dataset", DigitalAssetType.DATA_ASSETS,
                500000, 100000, 800000, 0.05, 2.0, 3.0, 1.2, 0.4
            ),
            
            DigitalAsset(
                "trading_algorithm", "Quantitative Trading Algorithm", DigitalAssetType.ALGORITHMS,
                1000000, 150000, 2000000, 0.20, 0.8, 1.5, 2.5, 0.8
            ),
            
            DigitalAsset(
                "platform_network", "Digital Platform Network", DigitalAssetType.DIGITAL_PLATFORMS,
                5000000, 500000, 3000000, 0.10, 3.0, 1.0, 1.5, 0.3
            ),
            
            DigitalAsset(
                "ip_portfolio", "AI Patent Portfolio", DigitalAssetType.INTELLECTUAL_PROPERTY,
                3000000, 300000, 1500000, 0.08, 0.5, 0.8, 1.3, 0.9
            ),
            
            DigitalAsset(
                "content_library", "AI-Generated Content Library", DigitalAssetType.DIGITAL_CONTENT,
                800000, 80000, 600000, 0.25, 1.2, 1.8, 2.0, 0.6
            )
        ]
        
        return assets