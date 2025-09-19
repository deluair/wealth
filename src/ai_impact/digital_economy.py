"""
Digital Economy Simulator

Classes for modeling the transformation of traditional economic structures
through AI and digital technologies, including platform economics, digital currencies,
and new forms of value creation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import networkx as nx

class DigitalEconomyType(Enum):
    """Types of digital economy models"""
    PLATFORM_ECONOMY = "platform_economy"
    SHARING_ECONOMY = "sharing_economy"
    CREATOR_ECONOMY = "creator_economy"
    TOKEN_ECONOMY = "token_economy"
    DATA_ECONOMY = "data_economy"
    SUBSCRIPTION_ECONOMY = "subscription_economy"
    GIG_ECONOMY = "gig_economy"
    METAVERSE_ECONOMY = "metaverse_economy"

class ParticipantType(Enum):
    """Types of participants in digital economy"""
    PLATFORM_OWNER = "platform_owner"
    CONTENT_CREATOR = "content_creator"
    SERVICE_PROVIDER = "service_provider"
    CONSUMER = "consumer"
    INVESTOR = "investor"
    DEVELOPER = "developer"
    DATA_PROVIDER = "data_provider"
    INTERMEDIARY = "intermediary"

class ValueCreationMechanism(Enum):
    """Mechanisms of value creation in digital economy"""
    NETWORK_EFFECTS = "network_effects"
    DATA_MONETIZATION = "data_monetization"
    ALGORITHMIC_OPTIMIZATION = "algorithmic_optimization"
    AUTOMATION = "automation"
    PERSONALIZATION = "personalization"
    MATCHING = "matching"
    AGGREGATION = "aggregation"
    DISINTERMEDIATION = "disintermediation"

@dataclass
class DigitalParticipant:
    """Represents a participant in the digital economy"""
    id: str
    name: str
    participant_type: ParticipantType
    digital_skills: float  # 0-1
    capital: float
    network_size: int
    reputation_score: float  # 0-1
    data_assets: float  # Value of data owned
    platform_dependency: float  # 0-1, how dependent on platforms
    innovation_capability: float  # 0-1
    risk_tolerance: float  # 0-1
    
@dataclass
class DigitalPlatform:
    """Represents a digital platform"""
    id: str
    name: str
    platform_type: DigitalEconomyType
    user_base_size: int
    revenue_model: str  # "commission", "subscription", "advertising", "freemium"
    commission_rate: float  # 0-1
    network_effect_strength: float  # 0-1
    data_collection_capability: float  # 0-1
    ai_integration_level: float  # 0-1
    market_power: float  # 0-1
    switching_costs: float  # 0-1, how hard it is for users to leave
    
@dataclass
class DigitalTransaction:
    """Represents a transaction in the digital economy"""
    id: str
    platform_id: str
    provider_id: str
    consumer_id: str
    transaction_value: float
    platform_fee: float
    data_generated: float  # Value of data generated from transaction
    timestamp: int
    transaction_type: str
    
@dataclass
class EconomicIndicator:
    """Economic indicators for digital economy"""
    gdp_contribution: float
    employment_impact: float
    productivity_gain: float
    inequality_change: float
    innovation_index: float
    market_concentration: float
    consumer_surplus: float
    platform_power_index: float

class DigitalEconomySimulator:
    """Simulate the evolution and impact of digital economy"""
    
    def __init__(self):
        self.participants = {}
        self.platforms = {}
        self.transactions = []
        self.economic_indicators = []
        self.network_graph = nx.Graph()
        
        # Global digital economy parameters
        self.global_parameters = {
            'digital_adoption_rate': 0.15,  # 15% annual increase
            'ai_integration_speed': 0.20,   # 20% annual increase
            'platform_concentration_tendency': 0.8,  # Tendency toward winner-take-all
            'data_value_multiplier': 1.5,   # How much data increases in value
            'network_effect_power': 2.0,    # Power law for network effects
            'automation_displacement_rate': 0.05,  # 5% jobs displaced annually
            'new_job_creation_rate': 0.03   # 3% new jobs created annually
        }
    
    def add_participants(self, participants: List[DigitalParticipant]) -> None:
        """Add participants to the digital economy"""
        for participant in participants:
            self.participants[participant.id] = participant
            self.network_graph.add_node(participant.id, 
                                      participant_type=participant.participant_type.value)
    
    def add_platforms(self, platforms: List[DigitalPlatform]) -> None:
        """Add platforms to the digital economy"""
        for platform in platforms:
            self.platforms[platform.id] = platform
    
    def simulate_digital_economy_evolution(self, time_horizon: int = 10,
                                         random_seed: int = 42) -> Dict:
        """
        Simulate the evolution of digital economy
        
        Args:
            time_horizon: Number of years to simulate
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results
        """
        np.random.seed(random_seed)
        
        results = {
            'yearly_results': [],
            'participant_evolution': {},
            'platform_evolution': {},
            'transaction_history': [],
            'economic_indicators': [],
            'network_evolution': []
        }
        
        # Initialize tracking
        for participant_id in self.participants.keys():
            results['participant_evolution'][participant_id] = []
        
        for platform_id in self.platforms.keys():
            results['platform_evolution'][platform_id] = []
        
        # Simulate each year
        for year in range(time_horizon):
            year_results = self._simulate_year_digital_economy(year)
            results['yearly_results'].append(year_results)
            
            # Update tracking
            self._update_participant_evolution(results['participant_evolution'], year)
            self._update_platform_evolution(results['platform_evolution'], year)
            self._record_transactions(results['transaction_history'], year)
            self._calculate_economic_indicators(results['economic_indicators'], year)
            self._analyze_network_evolution(results['network_evolution'], year)
            
            # Update global parameters
            self._update_global_digital_parameters(year)
        
        # Calculate summary
        results['summary'] = self._calculate_digital_economy_summary(results)
        
        return results
    
    def _simulate_year_digital_economy(self, year: int) -> Dict:
        """Simulate digital economy for a single year"""
        
        # Simulate platform growth and competition
        platform_results = self._simulate_platform_dynamics(year)
        
        # Simulate participant interactions and transactions
        transaction_results = self._simulate_digital_transactions(year)
        
        # Simulate value creation mechanisms
        value_creation_results = self._simulate_value_creation(year)
        
        # Simulate market concentration and power dynamics
        market_dynamics = self._simulate_market_dynamics(year)
        
        # Simulate employment and labor market effects
        labor_effects = self._simulate_labor_market_effects(year)
        
        # Calculate wealth distribution effects
        wealth_distribution = self._analyze_digital_wealth_distribution(year)
        
        # Simulate regulatory and policy impacts
        regulatory_effects = self._simulate_regulatory_effects(year)
        
        return {
            'year': year,
            'platform_results': platform_results,
            'transaction_results': transaction_results,
            'value_creation_results': value_creation_results,
            'market_dynamics': market_dynamics,
            'labor_effects': labor_effects,
            'wealth_distribution': wealth_distribution,
            'regulatory_effects': regulatory_effects,
            'total_digital_gdp': self._calculate_digital_gdp(year),
            'digital_adoption_level': self._calculate_digital_adoption(year)
        }
    
    def _simulate_platform_dynamics(self, year: int) -> Dict:
        """Simulate platform growth, competition, and evolution"""
        platform_results = {}
        
        for platform_id, platform in self.platforms.items():
            # User base growth with network effects
            network_effect = platform.network_effect_strength * (platform.user_base_size / 1000000) ** 0.5
            growth_rate = 0.1 + network_effect * 0.3  # Base 10% + network effects
            
            # Market saturation effects
            saturation_factor = max(0.1, 1 - (platform.user_base_size / 100000000))  # Slow down at 100M users
            actual_growth_rate = growth_rate * saturation_factor
            
            # Competition effects
            competition_pressure = self._calculate_competition_pressure(platform)
            actual_growth_rate *= (1 - competition_pressure * 0.3)
            
            # AI integration benefits
            ai_boost = platform.ai_integration_level * 0.2
            actual_growth_rate *= (1 + ai_boost)
            
            # Update user base
            new_users = int(platform.user_base_size * actual_growth_rate)
            platform.user_base_size += new_users
            
            # Revenue calculation
            revenue_per_user = self._calculate_revenue_per_user(platform, year)
            total_revenue = platform.user_base_size * revenue_per_user
            
            # Market power evolution
            market_share = self._calculate_market_share(platform)
            platform.market_power = min(1.0, market_share * 1.5)  # Market power grows with share
            
            # Data asset accumulation
            data_generated = new_users * 1000 + platform.user_base_size * 100  # Data points
            data_value = data_generated * self.global_parameters['data_value_multiplier']
            
            platform_results[platform_id] = {
                'user_growth': new_users,
                'total_users': platform.user_base_size,
                'revenue': total_revenue,
                'revenue_per_user': revenue_per_user,
                'market_power': platform.market_power,
                'data_value_generated': data_value,
                'competition_pressure': competition_pressure,
                'network_effect_strength': network_effect
            }
        
        return platform_results
    
    def _simulate_digital_transactions(self, year: int) -> Dict:
        """Simulate transactions between participants on platforms"""
        total_transactions = 0
        total_transaction_value = 0
        total_platform_fees = 0
        transactions_by_type = {}
        
        for platform_id, platform in self.platforms.items():
            # Calculate transaction volume based on user base and activity
            transactions_per_user_per_year = self._calculate_transaction_frequency(platform)
            platform_transactions = int(platform.user_base_size * transactions_per_user_per_year)
            
            # Average transaction value
            avg_transaction_value = self._calculate_avg_transaction_value(platform, year)
            
            # Platform fees
            platform_fee_rate = platform.commission_rate
            
            for _ in range(platform_transactions):
                # Generate transaction
                transaction_value = np.random.exponential(avg_transaction_value)
                platform_fee = transaction_value * platform_fee_rate
                
                # Create transaction record
                transaction = DigitalTransaction(
                    id=f"txn_{year}_{total_transactions}",
                    platform_id=platform_id,
                    provider_id=f"provider_{np.random.randint(1000)}",
                    consumer_id=f"consumer_{np.random.randint(1000)}",
                    transaction_value=transaction_value,
                    platform_fee=platform_fee,
                    data_generated=transaction_value * 0.1,  # Data value proportional to transaction
                    timestamp=year,
                    transaction_type=platform.platform_type.value
                )
                
                self.transactions.append(transaction)
                
                # Update totals
                total_transactions += 1
                total_transaction_value += transaction_value
                total_platform_fees += platform_fee
                
                # Update by type
                tx_type = platform.platform_type.value
                if tx_type not in transactions_by_type:
                    transactions_by_type[tx_type] = {'count': 0, 'value': 0}
                transactions_by_type[tx_type]['count'] += 1
                transactions_by_type[tx_type]['value'] += transaction_value
        
        return {
            'total_transactions': total_transactions,
            'total_transaction_value': total_transaction_value,
            'total_platform_fees': total_platform_fees,
            'transactions_by_type': transactions_by_type,
            'average_transaction_value': total_transaction_value / max(1, total_transactions),
            'platform_fee_percentage': total_platform_fees / max(1, total_transaction_value)
        }
    
    def _simulate_value_creation(self, year: int) -> Dict:
        """Simulate various value creation mechanisms in digital economy"""
        value_creation = {}
        
        # Network effects value creation
        network_value = 0
        for platform in self.platforms.values():
            # Metcalfe's law: value proportional to n^2
            network_contribution = (platform.user_base_size ** 2) * platform.network_effect_strength * 0.000001
            network_value += network_contribution
        
        # Data monetization value
        data_value = 0
        for participant in self.participants.values():
            if participant.data_assets > 0:
                # Data value grows with AI capabilities
                ai_multiplier = 1 + self.global_parameters['ai_integration_speed'] * year
                data_contribution = participant.data_assets * ai_multiplier
                data_value += data_contribution
        
        # Algorithmic optimization value
        algo_value = 0
        for platform in self.platforms.values():
            # AI-driven optimization creates value
            optimization_value = platform.user_base_size * platform.ai_integration_level * 10
            algo_value += optimization_value
        
        # Automation value creation
        automation_value = 0
        total_participants = len(self.participants)
        for participant in self.participants.values():
            if participant.participant_type in [ParticipantType.SERVICE_PROVIDER, ParticipantType.CONTENT_CREATOR]:
                # Automation increases productivity
                productivity_gain = participant.digital_skills * 0.3 * participant.capital
                automation_value += productivity_gain
        
        value_creation = {
            'network_effects_value': network_value,
            'data_monetization_value': data_value,
            'algorithmic_optimization_value': algo_value,
            'automation_value': automation_value,
            'total_value_created': network_value + data_value + algo_value + automation_value
        }
        
        return value_creation
    
    def _simulate_market_dynamics(self, year: int) -> Dict:
        """Simulate market concentration and competitive dynamics"""
        
        # Calculate market concentration (HHI)
        total_market_size = sum(p.user_base_size for p in self.platforms.values())
        if total_market_size == 0:
            hhi = 0
        else:
            market_shares = [(p.user_base_size / total_market_size) ** 2 for p in self.platforms.values()]
            hhi = sum(market_shares)
        
        # Platform power concentration
        platform_powers = [p.market_power for p in self.platforms.values()]
        avg_platform_power = np.mean(platform_powers) if platform_powers else 0
        max_platform_power = max(platform_powers) if platform_powers else 0
        
        # Switching costs analysis
        avg_switching_costs = np.mean([p.switching_costs for p in self.platforms.values()])
        
        # Competition intensity
        num_platforms = len(self.platforms)
        competition_intensity = max(0, 1 - hhi) * min(1, num_platforms / 10)
        
        # Market entry barriers
        avg_network_effects = np.mean([p.network_effect_strength for p in self.platforms.values()])
        entry_barriers = avg_network_effects * 0.5 + avg_switching_costs * 0.3 + hhi * 0.2
        
        return {
            'herfindahl_index': hhi,
            'average_platform_power': avg_platform_power,
            'max_platform_power': max_platform_power,
            'average_switching_costs': avg_switching_costs,
            'competition_intensity': competition_intensity,
            'market_entry_barriers': entry_barriers,
            'number_of_platforms': num_platforms
        }
    
    def _simulate_labor_market_effects(self, year: int) -> Dict:
        """Simulate effects on labor markets and employment"""
        
        # Count participants by type
        participant_counts = {}
        for participant in self.participants.values():
            p_type = participant.participant_type.value
            participant_counts[p_type] = participant_counts.get(p_type, 0) + 1
        
        # Calculate job displacement due to automation
        automation_rate = self.global_parameters['automation_displacement_rate']
        jobs_displaced = 0
        
        for participant in self.participants.values():
            if participant.participant_type == ParticipantType.SERVICE_PROVIDER:
                # Higher digital skills = less likely to be displaced
                displacement_probability = automation_rate * (1 - participant.digital_skills)
                if np.random.random() < displacement_probability:
                    jobs_displaced += 1
        
        # Calculate new job creation
        job_creation_rate = self.global_parameters['new_job_creation_rate']
        new_jobs_created = int(len(self.participants) * job_creation_rate)
        
        # Gig economy growth
        gig_workers = participant_counts.get('service_provider', 0)
        gig_growth_rate = 0.15  # 15% annual growth
        new_gig_workers = int(gig_workers * gig_growth_rate)
        
        # Income effects
        total_income = 0
        income_by_type = {}
        
        for participant in self.participants.values():
            # Calculate income based on participant type and digital skills
            base_income = self._calculate_participant_income(participant, year)
            total_income += base_income
            
            p_type = participant.participant_type.value
            if p_type not in income_by_type:
                income_by_type[p_type] = 0
            income_by_type[p_type] += base_income
        
        # Skills premium calculation
        high_skill_participants = [p for p in self.participants.values() if p.digital_skills > 0.7]
        low_skill_participants = [p for p in self.participants.values() if p.digital_skills < 0.3]
        
        if high_skill_participants and low_skill_participants:
            high_skill_avg_income = np.mean([self._calculate_participant_income(p, year) for p in high_skill_participants])
            low_skill_avg_income = np.mean([self._calculate_participant_income(p, year) for p in low_skill_participants])
            skills_premium = high_skill_avg_income / max(1, low_skill_avg_income)
        else:
            skills_premium = 1.0
        
        return {
            'jobs_displaced': jobs_displaced,
            'new_jobs_created': new_jobs_created,
            'net_job_change': new_jobs_created - jobs_displaced,
            'gig_workers': gig_workers,
            'new_gig_workers': new_gig_workers,
            'total_income': total_income,
            'income_by_participant_type': income_by_type,
            'digital_skills_premium': skills_premium,
            'participant_counts': participant_counts
        }
    
    def _analyze_digital_wealth_distribution(self, year: int) -> Dict:
        """Analyze wealth distribution in the digital economy"""
        
        # Calculate wealth for each participant
        participant_wealth = []
        platform_owner_wealth = []
        creator_wealth = []
        service_provider_wealth = []
        consumer_wealth = []
        
        for participant in self.participants.values():
            wealth = self._calculate_participant_wealth(participant, year)
            participant_wealth.append(wealth)
            
            if participant.participant_type == ParticipantType.PLATFORM_OWNER:
                platform_owner_wealth.append(wealth)
            elif participant.participant_type == ParticipantType.CONTENT_CREATOR:
                creator_wealth.append(wealth)
            elif participant.participant_type == ParticipantType.SERVICE_PROVIDER:
                service_provider_wealth.append(wealth)
            elif participant.participant_type == ParticipantType.CONSUMER:
                consumer_wealth.append(wealth)
        
        # Calculate Gini coefficient
        gini = self._calculate_gini_coefficient(participant_wealth)
        
        # Calculate wealth concentration
        if participant_wealth:
            sorted_wealth = sorted(participant_wealth, reverse=True)
            top_1_percent = max(1, len(sorted_wealth) // 100)
            top_10_percent = max(1, len(sorted_wealth) // 10)
            
            top_1_percent_wealth = sum(sorted_wealth[:top_1_percent])
            top_10_percent_wealth = sum(sorted_wealth[:top_10_percent])
            total_wealth = sum(participant_wealth)
            
            top_1_percent_share = top_1_percent_wealth / max(1, total_wealth)
            top_10_percent_share = top_10_percent_wealth / max(1, total_wealth)
        else:
            top_1_percent_share = 0
            top_10_percent_share = 0
            total_wealth = 0
        
        # Platform vs participant wealth
        platform_wealth = sum(platform_owner_wealth)
        participant_wealth_total = sum(creator_wealth + service_provider_wealth)
        
        platform_wealth_share = platform_wealth / max(1, total_wealth) if total_wealth > 0 else 0
        
        return {
            'total_wealth': total_wealth,
            'wealth_gini_coefficient': gini,
            'top_1_percent_share': top_1_percent_share,
            'top_10_percent_share': top_10_percent_share,
            'platform_wealth_share': platform_wealth_share,
            'average_wealth_by_type': {
                'platform_owner': np.mean(platform_owner_wealth) if platform_owner_wealth else 0,
                'content_creator': np.mean(creator_wealth) if creator_wealth else 0,
                'service_provider': np.mean(service_provider_wealth) if service_provider_wealth else 0,
                'consumer': np.mean(consumer_wealth) if consumer_wealth else 0
            },
            'wealth_counts_by_type': {
                'platform_owner': len(platform_owner_wealth),
                'content_creator': len(creator_wealth),
                'service_provider': len(service_provider_wealth),
                'consumer': len(consumer_wealth)
            }
        }
    
    def _simulate_regulatory_effects(self, year: int) -> Dict:
        """Simulate regulatory and policy impacts on digital economy"""
        
        # Antitrust pressure based on market concentration
        hhi = self._calculate_market_hhi()
        antitrust_pressure = max(0, (hhi - 0.25) * 2)  # Pressure increases above 0.25 HHI
        
        # Data privacy regulations impact
        data_regulation_impact = min(1.0, year * 0.1)  # Increases over time
        
        # Platform liability regulations
        platform_liability = min(1.0, year * 0.05)  # Gradual increase
        
        # Tax policy effects
        digital_tax_rate = min(0.3, year * 0.02)  # Digital services tax
        
        # Regulatory compliance costs
        total_compliance_cost = 0
        for platform in self.platforms.values():
            # Compliance cost proportional to size and market power
            compliance_cost = platform.user_base_size * platform.market_power * 0.1
            total_compliance_cost += compliance_cost
        
        # Innovation impact of regulation
        innovation_dampening = min(0.5, antitrust_pressure * 0.3 + data_regulation_impact * 0.2)
        
        return {
            'antitrust_pressure': antitrust_pressure,
            'data_regulation_impact': data_regulation_impact,
            'platform_liability_level': platform_liability,
            'digital_tax_rate': digital_tax_rate,
            'total_compliance_costs': total_compliance_cost,
            'innovation_dampening_effect': innovation_dampening
        }
    
    def _calculate_digital_gdp(self, year: int) -> float:
        """Calculate digital economy contribution to GDP"""
        
        # Platform revenues
        platform_gdp = 0
        for platform in self.platforms.values():
            revenue_per_user = self._calculate_revenue_per_user(platform, year)
            platform_revenue = platform.user_base_size * revenue_per_user
            platform_gdp += platform_revenue
        
        # Transaction value
        recent_transactions = [t for t in self.transactions if t.timestamp == year]
        transaction_gdp = sum(t.transaction_value for t in recent_transactions)
        
        # Value creation from digital assets
        digital_asset_gdp = sum(p.data_assets for p in self.participants.values())
        
        # AI productivity gains
        ai_productivity_gdp = len(self.participants) * 50000 * self.global_parameters['ai_integration_speed'] * year
        
        total_digital_gdp = platform_gdp + transaction_gdp + digital_asset_gdp + ai_productivity_gdp
        
        return total_digital_gdp
    
    def _calculate_digital_adoption(self, year: int) -> float:
        """Calculate overall digital adoption level"""
        base_adoption = 0.2  # 20% base adoption
        growth_rate = self.global_parameters['digital_adoption_rate']
        
        # S-curve adoption with network effects
        adoption_level = 1 / (1 + np.exp(-0.4 * (year - 3)))
        
        return min(0.95, base_adoption + adoption_level * 0.7)
    
    # Helper methods
    def _calculate_competition_pressure(self, platform: DigitalPlatform) -> float:
        """Calculate competitive pressure on a platform"""
        same_type_platforms = [p for p in self.platforms.values() 
                             if p.platform_type == platform.platform_type and p.id != platform.id]
        
        if not same_type_platforms:
            return 0
        
        # Competition based on relative size and market power
        total_competitor_size = sum(p.user_base_size for p in same_type_platforms)
        relative_size = platform.user_base_size / max(1, total_competitor_size)
        
        # Higher relative size = less pressure
        pressure = max(0, 1 - relative_size)
        
        return min(1.0, pressure)
    
    def _calculate_revenue_per_user(self, platform: DigitalPlatform, year: int) -> float:
        """Calculate revenue per user for a platform"""
        base_revenue = 100  # $100 base annual revenue per user
        
        # Adjust based on platform type
        type_multipliers = {
            DigitalEconomyType.PLATFORM_ECONOMY: 1.5,
            DigitalEconomyType.SUBSCRIPTION_ECONOMY: 2.0,
            DigitalEconomyType.CREATOR_ECONOMY: 0.8,
            DigitalEconomyType.SHARING_ECONOMY: 1.2,
            DigitalEconomyType.GIG_ECONOMY: 1.0,
            DigitalEconomyType.TOKEN_ECONOMY: 1.8,
            DigitalEconomyType.DATA_ECONOMY: 2.5,
            DigitalEconomyType.METAVERSE_ECONOMY: 1.3
        }
        
        type_multiplier = type_multipliers.get(platform.platform_type, 1.0)
        
        # AI integration increases revenue per user
        ai_multiplier = 1 + platform.ai_integration_level * 0.5
        
        # Market power allows higher pricing
        power_multiplier = 1 + platform.market_power * 0.3
        
        # Time trend (generally increasing)
        time_multiplier = 1 + year * 0.05
        
        revenue_per_user = base_revenue * type_multiplier * ai_multiplier * power_multiplier * time_multiplier
        
        return revenue_per_user
    
    def _calculate_market_share(self, platform: DigitalPlatform) -> float:
        """Calculate market share for a platform"""
        same_type_platforms = [p for p in self.platforms.values() 
                             if p.platform_type == platform.platform_type]
        
        if not same_type_platforms:
            return 1.0
        
        total_users = sum(p.user_base_size for p in same_type_platforms)
        
        if total_users == 0:
            return 1.0 / len(same_type_platforms)
        
        return platform.user_base_size / total_users
    
    def _calculate_transaction_frequency(self, platform: DigitalPlatform) -> float:
        """Calculate annual transactions per user for a platform"""
        base_frequency = 10  # 10 transactions per user per year
        
        # Adjust based on platform type
        type_frequencies = {
            DigitalEconomyType.PLATFORM_ECONOMY: 15,
            DigitalEconomyType.SHARING_ECONOMY: 25,
            DigitalEconomyType.CREATOR_ECONOMY: 5,
            DigitalEconomyType.GIG_ECONOMY: 50,
            DigitalEconomyType.SUBSCRIPTION_ECONOMY: 1,
            DigitalEconomyType.TOKEN_ECONOMY: 100,
            DigitalEconomyType.DATA_ECONOMY: 365,
            DigitalEconomyType.METAVERSE_ECONOMY: 200
        }
        
        return type_frequencies.get(platform.platform_type, base_frequency)
    
    def _calculate_avg_transaction_value(self, platform: DigitalPlatform, year: int) -> float:
        """Calculate average transaction value for a platform"""
        base_value = 50  # $50 base transaction value
        
        # Adjust based on platform type
        type_values = {
            DigitalEconomyType.PLATFORM_ECONOMY: 100,
            DigitalEconomyType.SHARING_ECONOMY: 30,
            DigitalEconomyType.CREATOR_ECONOMY: 20,
            DigitalEconomyType.GIG_ECONOMY: 40,
            DigitalEconomyType.SUBSCRIPTION_ECONOMY: 15,
            DigitalEconomyType.TOKEN_ECONOMY: 500,
            DigitalEconomyType.DATA_ECONOMY: 1,
            DigitalEconomyType.METAVERSE_ECONOMY: 25
        }
        
        type_value = type_values.get(platform.platform_type, base_value)
        
        # Inflation and growth over time
        time_multiplier = 1 + year * 0.03
        
        return type_value * time_multiplier
    
    def _calculate_participant_income(self, participant: DigitalParticipant, year: int) -> float:
        """Calculate annual income for a participant"""
        base_income = 50000  # $50k base income
        
        # Adjust based on participant type
        type_multipliers = {
            ParticipantType.PLATFORM_OWNER: 10.0,
            ParticipantType.CONTENT_CREATOR: 0.8,
            ParticipantType.SERVICE_PROVIDER: 1.0,
            ParticipantType.CONSUMER: 0.0,  # Consumers don't earn from platform
            ParticipantType.INVESTOR: 5.0,
            ParticipantType.DEVELOPER: 2.0,
            ParticipantType.DATA_PROVIDER: 1.5,
            ParticipantType.INTERMEDIARY: 1.2
        }
        
        type_multiplier = type_multipliers.get(participant.participant_type, 1.0)
        
        # Digital skills premium
        skills_multiplier = 1 + participant.digital_skills * 1.5
        
        # Network effects
        network_multiplier = 1 + (participant.network_size / 10000) * 0.5
        
        # Reputation premium
        reputation_multiplier = 1 + participant.reputation_score * 0.3
        
        # Capital returns
        capital_returns = participant.capital * 0.08  # 8% return on capital
        
        income = (base_income * type_multiplier * skills_multiplier * 
                 network_multiplier * reputation_multiplier) + capital_returns
        
        return income
    
    def _calculate_participant_wealth(self, participant: DigitalParticipant, year: int) -> float:
        """Calculate total wealth for a participant"""
        # Base wealth from capital
        wealth = participant.capital
        
        # Add accumulated income (simplified)
        annual_income = self._calculate_participant_income(participant, year)
        accumulated_income = annual_income * (year + 1) * 0.5  # Simplified accumulation
        
        # Add data asset value
        wealth += participant.data_assets
        
        # Platform owners get additional value from platform equity
        if participant.participant_type == ParticipantType.PLATFORM_OWNER:
            # Find platforms owned by this participant (simplified assumption)
            platform_value = 0
            for platform in self.platforms.values():
                # Assume platform value is 10x annual revenue
                revenue_per_user = self._calculate_revenue_per_user(platform, year)
                platform_revenue = platform.user_base_size * revenue_per_user
                platform_value += platform_revenue * 10
            
            # Assume participant owns a fraction based on their capital
            ownership_fraction = min(1.0, participant.capital / 10000000)  # $10M for full ownership
            wealth += platform_value * ownership_fraction
        
        total_wealth = wealth + accumulated_income
        
        return max(0, total_wealth)
    
    def _calculate_market_hhi(self) -> float:
        """Calculate Herfindahl-Hirschman Index for market concentration"""
        if not self.platforms:
            return 0
        
        total_market_size = sum(p.user_base_size for p in self.platforms.values())
        
        if total_market_size == 0:
            return 0
        
        market_shares = [(p.user_base_size / total_market_size) ** 2 for p in self.platforms.values()]
        hhi = sum(market_shares)
        
        return hhi
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for wealth distribution"""
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
    
    # Update methods for tracking
    def _update_participant_evolution(self, evolution: Dict, year: int) -> None:
        """Update participant evolution tracking"""
        for participant_id, participant in self.participants.items():
            wealth = self._calculate_participant_wealth(participant, year)
            income = self._calculate_participant_income(participant, year)
            
            evolution[participant_id].append({
                'year': year,
                'wealth': wealth,
                'income': income,
                'digital_skills': participant.digital_skills,
                'network_size': participant.network_size,
                'reputation_score': participant.reputation_score
            })
    
    def _update_platform_evolution(self, evolution: Dict, year: int) -> None:
        """Update platform evolution tracking"""
        for platform_id, platform in self.platforms.items():
            market_share = self._calculate_market_share(platform)
            revenue_per_user = self._calculate_revenue_per_user(platform, year)
            
            evolution[platform_id].append({
                'year': year,
                'user_base_size': platform.user_base_size,
                'market_share': market_share,
                'market_power': platform.market_power,
                'revenue_per_user': revenue_per_user,
                'ai_integration_level': platform.ai_integration_level
            })
    
    def _record_transactions(self, transaction_history: List, year: int) -> None:
        """Record transaction summary for the year"""
        year_transactions = [t for t in self.transactions if t.timestamp == year]
        
        if year_transactions:
            total_value = sum(t.transaction_value for t in year_transactions)
            total_fees = sum(t.platform_fee for t in year_transactions)
            avg_value = total_value / len(year_transactions)
            
            transaction_history.append({
                'year': year,
                'transaction_count': len(year_transactions),
                'total_value': total_value,
                'total_fees': total_fees,
                'average_value': avg_value
            })
    
    def _calculate_economic_indicators(self, indicators: List, year: int) -> None:
        """Calculate and record economic indicators"""
        digital_gdp = self._calculate_digital_gdp(year)
        
        # Employment impact (simplified)
        total_participants = len(self.participants)
        employment_impact = total_participants / 1000000  # Per million population
        
        # Productivity gain from AI
        productivity_gain = self.global_parameters['ai_integration_speed'] * year * 0.1
        
        # Inequality change
        wealth_values = [self._calculate_participant_wealth(p, year) for p in self.participants.values()]
        inequality_change = self._calculate_gini_coefficient(wealth_values)
        
        # Innovation index
        innovation_index = min(1.0, sum(p.ai_integration_level for p in self.platforms.values()) / max(1, len(self.platforms)))
        
        # Market concentration
        market_concentration = self._calculate_market_hhi()
        
        # Consumer surplus (simplified)
        consumer_surplus = sum(p.user_base_size for p in self.platforms.values()) * 100  # $100 per user
        
        # Platform power index
        platform_power_index = np.mean([p.market_power for p in self.platforms.values()]) if self.platforms else 0
        
        indicator = EconomicIndicator(
            gdp_contribution=digital_gdp,
            employment_impact=employment_impact,
            productivity_gain=productivity_gain,
            inequality_change=inequality_change,
            innovation_index=innovation_index,
            market_concentration=market_concentration,
            consumer_surplus=consumer_surplus,
            platform_power_index=platform_power_index
        )
        
        indicators.append(indicator)
    
    def _analyze_network_evolution(self, network_evolution: List, year: int) -> None:
        """Analyze network structure evolution"""
        # Update network connections based on transactions
        year_transactions = [t for t in self.transactions if t.timestamp == year]
        
        for transaction in year_transactions:
            # Add edge between provider and consumer
            if (transaction.provider_id in self.network_graph.nodes and 
                transaction.consumer_id in self.network_graph.nodes):
                
                if self.network_graph.has_edge(transaction.provider_id, transaction.consumer_id):
                    # Increase edge weight
                    self.network_graph[transaction.provider_id][transaction.consumer_id]['weight'] += 1
                else:
                    # Add new edge
                    self.network_graph.add_edge(transaction.provider_id, transaction.consumer_id, weight=1)
        
        # Calculate network metrics
        if len(self.network_graph.nodes) > 0:
            density = nx.density(self.network_graph)
            
            if len(self.network_graph.nodes) > 1:
                avg_clustering = nx.average_clustering(self.network_graph)
                
                # Calculate centrality for largest connected component
                largest_cc = max(nx.connected_components(self.network_graph), key=len)
                subgraph = self.network_graph.subgraph(largest_cc)
                
                if len(subgraph.nodes) > 1:
                    centrality = nx.degree_centrality(subgraph)
                    avg_centrality = np.mean(list(centrality.values()))
                else:
                    avg_centrality = 0
            else:
                avg_clustering = 0
                avg_centrality = 0
        else:
            density = 0
            avg_clustering = 0
            avg_centrality = 0
        
        network_evolution.append({
            'year': year,
            'network_density': density,
            'average_clustering': avg_clustering,
            'average_centrality': avg_centrality,
            'total_nodes': len(self.network_graph.nodes),
            'total_edges': len(self.network_graph.edges)
        })
    
    def _update_global_digital_parameters(self, year: int) -> None:
        """Update global digital economy parameters over time"""
        # Digital adoption rate may slow down as market matures
        self.global_parameters['digital_adoption_rate'] *= 0.98
        
        # AI integration speed may accelerate initially then slow down
        if year < 5:
            self.global_parameters['ai_integration_speed'] *= 1.05
        else:
            self.global_parameters['ai_integration_speed'] *= 0.98
        
        # Platform concentration tendency may increase with network effects
        self.global_parameters['platform_concentration_tendency'] = min(0.95, 
            self.global_parameters['platform_concentration_tendency'] * 1.01)
        
        # Data value multiplier increases with AI capabilities
        self.global_parameters['data_value_multiplier'] *= 1.02
    
    def _calculate_digital_economy_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics for digital economy simulation"""
        yearly_results = results['yearly_results']
        
        if not yearly_results:
            return {}
        
        # Extract time series
        gdp_series = [r['total_digital_gdp'] for r in yearly_results]
        adoption_series = [r['digital_adoption_level'] for r in yearly_results]
        
        # Platform evolution
        platform_growth = {}
        for platform_id, evolution in results['platform_evolution'].items():
            initial_users = evolution[0]['user_base_size'] if evolution else 0
            final_users = evolution[-1]['user_base_size'] if evolution else 0
            growth_rate = (final_users / max(1, initial_users)) ** (1/len(evolution)) - 1 if evolution else 0
            platform_growth[platform_id] = growth_rate
        
        # Economic indicators
        final_indicators = results['economic_indicators'][-1] if results['economic_indicators'] else None
        
        summary = {
            'simulation_years': len(yearly_results),
            'final_digital_gdp': gdp_series[-1] if gdp_series else 0,
            'digital_gdp_growth_rate': self._calculate_growth_rate(gdp_series),
            'final_digital_adoption': adoption_series[-1] if adoption_series else 0,
            'platform_growth_rates': platform_growth,
            'final_economic_indicators': {
                'gdp_contribution': final_indicators.gdp_contribution if final_indicators else 0,
                'employment_impact': final_indicators.employment_impact if final_indicators else 0,
                'productivity_gain': final_indicators.productivity_gain if final_indicators else 0,
                'inequality_change': final_indicators.inequality_change if final_indicators else 0,
                'innovation_index': final_indicators.innovation_index if final_indicators else 0,
                'market_concentration': final_indicators.market_concentration if final_indicators else 0,
                'consumer_surplus': final_indicators.consumer_surplus if final_indicators else 0,
                'platform_power_index': final_indicators.platform_power_index if final_indicators else 0
            } if final_indicators else {}
        }
        
        return summary
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate compound annual growth rate"""
        if len(values) < 2 or values[0] == 0:
            return 0
        
        periods = len(values) - 1
        growth_rate = (values[-1] / values[0]) ** (1/periods) - 1
        return growth_rate