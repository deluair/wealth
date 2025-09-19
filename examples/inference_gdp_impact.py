#!/usr/bin/env python3
"""
Inference Management to GDP Impact Model

This model demonstrates how AI inference capabilities translate into GDP impact
through various economic channels and multiplier effects.

The pathway shows:
1. Computing Infrastructure → AI Inference Capacity
2. Inference Services → Business Process Automation
3. Productivity Gains → Economic Output Increase
4. Innovation Acceleration → New Market Creation
5. Economic Integration → GDP Growth

Author: Wealth Analysis Framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import our wealth creation framework
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class InferenceGDPModel:
    """
    Models the economic impact pathway from AI inference to GDP growth.
    """
    
    def __init__(self):
        self.economic_sectors = {
            'manufacturing': {
                'ai_adoption_rate': 0.65,
                'productivity_multiplier': 1.25,
                'gdp_contribution': 0.20,
                'automation_potential': 0.70
            },
            'services': {
                'ai_adoption_rate': 0.45,
                'productivity_multiplier': 1.15,
                'gdp_contribution': 0.65,
                'automation_potential': 0.40
            },
            'finance': {
                'ai_adoption_rate': 0.80,
                'productivity_multiplier': 1.35,
                'gdp_contribution': 0.08,
                'automation_potential': 0.60
            },
            'healthcare': {
                'ai_adoption_rate': 0.30,
                'productivity_multiplier': 1.20,
                'gdp_contribution': 0.07,
                'automation_potential': 0.30
            }
        }
        
        self.inference_stages = {
            'compute_infrastructure': {
                'name': 'AI Compute Infrastructure',
                'flops_per_dollar': 1e12,  # FLOPS per dollar invested
                'utilization_rate': 0.70,
                'efficiency_improvement': 0.15,  # Annual improvement
                'cost_per_inference': 0.001,  # Cost per inference operation
                'scaling_factor': 1.0
            },
            'inference_services': {
                'name': 'AI Inference Services',
                'inferences_per_second': 1000,
                'service_value_multiplier': 0.01,  # Realistic $0.01 per inference
                'market_penetration': 0.25,
                'growth_rate': 0.40,  # Annual growth
                'scaling_factor': 10.0
            },
            'business_automation': {
                'name': 'Business Process Automation',
                'processes_automated': 50,
                'cost_savings_per_process': 50000,  # Annual savings
                'productivity_gain': 0.20,
                'adoption_rate': 0.35,
                'scaling_factor': 100.0
            },
            'productivity_enhancement': {
                'name': 'Economic Productivity Enhancement',
                'worker_productivity_gain': 0.15,
                'process_efficiency_gain': 0.25,
                'innovation_acceleration': 0.30,
                'market_expansion': 0.20,
                'scaling_factor': 1000.0
            },
            'new_market_creation': {
                'name': 'New Market and Industry Creation',
                'new_business_models': 25,
                'market_value_per_model': 1000000,  # Value per new business model
                'job_creation_multiplier': 2.5,
                'innovation_spillover': 0.40,
                'scaling_factor': 10000.0
            },
            'gdp_integration': {
                'name': 'GDP Economic Integration',
                'direct_gdp_contribution': 0.02,  # 2% direct contribution
                'indirect_multiplier': 1.5,  # Realistic 1.5x indirect multiplier
                'long_term_growth_rate': 0.05,  # Additional annual growth
                'employment_impact': 0.03,  # Net employment change
                'scaling_factor': 100000.0
            }
        }
        
        self.initialize_economic_parameters()
    
    def initialize_economic_parameters(self):
        """Initialize economic modeling parameters."""
        self.base_gdp = 25e12  # $25 trillion base GDP
        self.ai_investment_ratio = 0.03  # 3% of GDP invested in AI
        self.productivity_elasticity = 0.7  # GDP elasticity to productivity
        self.innovation_multiplier = 1.8  # Innovation impact multiplier
        
    def calculate_inference_capacity(self, compute_investment: float) -> Dict:
        """Calculate AI inference capacity from compute investment."""
        stage = self.inference_stages['compute_infrastructure']
        
        # Calculate raw compute capacity
        total_flops = compute_investment * stage['flops_per_dollar']
        effective_flops = total_flops * stage['utilization_rate']
        
        # Inference capacity (assuming 1e9 FLOPS per inference)
        inference_capacity = effective_flops / 1e9
        
        # Cost efficiency
        cost_per_inference = stage['cost_per_inference']
        annual_inferences = inference_capacity * 365 * 24 * 3600  # Per second to annual
        
        return {
            'compute_investment': compute_investment,
            'total_flops': total_flops,
            'effective_flops': effective_flops,
            'inference_capacity_per_second': inference_capacity,
            'annual_inference_capacity': annual_inferences,
            'cost_per_inference': cost_per_inference,
            'annual_operating_cost': annual_inferences * cost_per_inference
        }
    
    def model_service_value_creation(self, inference_capacity: Dict) -> Dict:
        """Model value creation from inference services."""
        stage = self.inference_stages['inference_services']
        
        # Service capacity
        service_capacity = inference_capacity['inference_capacity_per_second'] * stage['market_penetration']
        
        # Value creation
        service_value = service_capacity * stage['service_value_multiplier']
        annual_service_value = service_value * 365 * 24 * 3600
        
        # Market dynamics
        market_size = annual_service_value * (1 + stage['growth_rate'])
        
        return {
            'service_capacity_per_second': service_capacity,
            'service_value_per_second': service_value,
            'annual_service_value': annual_service_value,
            'projected_market_size': market_size,
            'growth_rate': stage['growth_rate'],
            'market_penetration': stage['market_penetration']
        }
    
    def calculate_automation_impact(self, service_value: Dict) -> Dict:
        """Calculate business automation impact."""
        stage = self.inference_stages['business_automation']
        
        # Automation capacity
        automation_capacity = service_value['annual_service_value'] / 1e6  # Scale factor
        processes_automated = min(automation_capacity * stage['processes_automated'], 
                                stage['processes_automated'] * 1000)  # Cap at reasonable level
        
        # Cost savings
        annual_cost_savings = processes_automated * stage['cost_savings_per_process']
        
        # Productivity gains
        productivity_value = annual_cost_savings * (1 + stage['productivity_gain'])
        
        return {
            'processes_automated': processes_automated,
            'annual_cost_savings': annual_cost_savings,
            'productivity_value': productivity_value,
            'productivity_gain': stage['productivity_gain'],
            'adoption_rate': stage['adoption_rate']
        }
    
    def model_productivity_enhancement(self, automation_impact: Dict) -> Dict:
        """Model economy-wide productivity enhancement."""
        stage = self.inference_stages['productivity_enhancement']
        
        # Base productivity gains
        base_productivity_gain = automation_impact['productivity_value']
        
        # Enhanced productivity across sectors
        sector_impacts = {}
        total_productivity_gain = 0
        
        for sector, params in self.economic_sectors.items():
            sector_gain = (base_productivity_gain * params['gdp_contribution'] * 
                          params['ai_adoption_rate'] * params['productivity_multiplier'])
            sector_impacts[sector] = sector_gain
            total_productivity_gain += sector_gain
        
        # Innovation acceleration
        innovation_value = total_productivity_gain * stage['innovation_acceleration']
        
        # Market expansion
        market_expansion_value = total_productivity_gain * stage['market_expansion']
        
        return {
            'base_productivity_gain': base_productivity_gain,
            'sector_impacts': sector_impacts,
            'total_productivity_gain': total_productivity_gain,
            'innovation_value': innovation_value,
            'market_expansion_value': market_expansion_value,
            'combined_enhancement': total_productivity_gain + innovation_value + market_expansion_value
        }
    
    def calculate_new_market_creation(self, productivity_enhancement: Dict) -> Dict:
        """Calculate new market and industry creation."""
        stage = self.inference_stages['new_market_creation']
        
        # New business models
        innovation_driver = productivity_enhancement['innovation_value']
        new_business_models = min(innovation_driver / 1e8 * stage['new_business_models'], 
                                stage['new_business_models'] * 10)  # Cap at reasonable level
        
        # Market value creation
        new_market_value = new_business_models * stage['market_value_per_model']
        
        # Job creation
        jobs_created = new_market_value / 100000 * stage['job_creation_multiplier']  # Jobs per $100k
        
        # Innovation spillover
        spillover_value = new_market_value * stage['innovation_spillover']
        
        return {
            'new_business_models': new_business_models,
            'new_market_value': new_market_value,
            'jobs_created': jobs_created,
            'spillover_value': spillover_value,
            'total_new_value': new_market_value + spillover_value
        }
    
    def calculate_gdp_impact(self, new_market_creation: Dict, 
                           productivity_enhancement: Dict) -> Dict:
        """Calculate final GDP impact."""
        stage = self.inference_stages['gdp_integration']
        
        # Direct GDP contribution
        direct_contribution = (productivity_enhancement['combined_enhancement'] + 
                             new_market_creation['total_new_value'])
        
        # Indirect multiplier effects
        indirect_contribution = direct_contribution * stage['indirect_multiplier']
        
        # Total GDP impact
        total_gdp_impact = direct_contribution + indirect_contribution
        
        # As percentage of base GDP
        gdp_percentage_impact = total_gdp_impact / self.base_gdp
        
        # Long-term growth impact
        long_term_annual_growth = gdp_percentage_impact * stage['long_term_growth_rate']
        
        # Employment impact
        employment_change = total_gdp_impact / 75000 * stage['employment_impact']  # Jobs per $75k GDP
        
        return {
            'direct_gdp_contribution': direct_contribution,
            'indirect_gdp_contribution': indirect_contribution,
            'total_gdp_impact': total_gdp_impact,
            'gdp_percentage_impact': gdp_percentage_impact,
            'long_term_annual_growth': long_term_annual_growth,
            'employment_change': employment_change,
            'gdp_multiplier': total_gdp_impact / direct_contribution if direct_contribution > 0 else 0
        }
    
    def simulate_complete_inference_pathway(self, initial_compute_investment: float) -> Dict:
        """Simulate the complete inference-to-GDP pathway."""
        
        # Stage 1: Compute Infrastructure
        inference_capacity = self.calculate_inference_capacity(initial_compute_investment)
        
        # Stage 2: Inference Services
        service_value = self.model_service_value_creation(inference_capacity)
        
        # Stage 3: Business Automation
        automation_impact = self.calculate_automation_impact(service_value)
        
        # Stage 4: Productivity Enhancement
        productivity_enhancement = self.model_productivity_enhancement(automation_impact)
        
        # Stage 5: New Market Creation
        new_market_creation = self.calculate_new_market_creation(productivity_enhancement)
        
        # Stage 6: GDP Impact
        gdp_impact = self.calculate_gdp_impact(new_market_creation, productivity_enhancement)
        
        return {
            'initial_investment': initial_compute_investment,
            'inference_capacity': inference_capacity,
            'service_value': service_value,
            'automation_impact': automation_impact,
            'productivity_enhancement': productivity_enhancement,
            'new_market_creation': new_market_creation,
            'gdp_impact': gdp_impact
        }
    
    def create_inference_gdp_visualization(self, pathway_results: Dict) -> go.Figure:
        """Create comprehensive visualization of inference-to-GDP pathway."""
        
        # Extract key metrics for visualization
        stages = [
            'Compute Investment',
            'Inference Capacity',
            'Service Value',
            'Automation Impact',
            'Productivity Enhancement',
            'New Market Creation',
            'GDP Impact'
        ]
        
        values = [
            pathway_results['initial_investment'],
            pathway_results['inference_capacity']['annual_inference_capacity'] / 1e9,  # Billions
            pathway_results['service_value']['annual_service_value'] / 1e6,  # Millions
            pathway_results['automation_impact']['productivity_value'] / 1e6,  # Millions
            pathway_results['productivity_enhancement']['combined_enhancement'] / 1e9,  # Billions
            pathway_results['new_market_creation']['total_new_value'] / 1e9,  # Billions
            pathway_results['gdp_impact']['total_gdp_impact'] / 1e9  # Billions
        ]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Inference-to-GDP Value Chain',
                'Sector-wise Productivity Impact',
                'Economic Multiplier Effects',
                'ROI and Growth Projections'
            ],
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "waterfall"}, {"secondary_y": True}]]
        )
        
        # 1. Value Chain Flow
        fig.add_trace(
            go.Scatter(
                x=list(range(len(stages))),
                y=values,
                mode='lines+markers',
                name='Value Creation ($B)',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Add logarithmic scale line
        log_values = [np.log10(max(v, 1)) for v in values]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(stages))),
                y=log_values,
                mode='lines+markers',
                name='Log Scale',
                line=dict(color='red', width=2, dash='dash'),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Sector Impact
        sector_names = list(self.economic_sectors.keys())
        sector_impacts = [pathway_results['productivity_enhancement']['sector_impacts'][sector] / 1e9 
                         for sector in sector_names]
        
        fig.add_trace(
            go.Bar(
                x=sector_names,
                y=sector_impacts,
                name='Sector Impact ($B)',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # 3. Economic Multiplier Waterfall
        waterfall_values = [
            pathway_results['gdp_impact']['direct_gdp_contribution'] / 1e9,
            pathway_results['gdp_impact']['indirect_gdp_contribution'] / 1e9,
            -pathway_results['initial_investment'] / 1e9  # Investment as negative
        ]
        
        fig.add_trace(
            go.Waterfall(
                x=['Direct Impact', 'Indirect Multiplier', 'Initial Investment'],
                y=waterfall_values,
                name='GDP Impact Breakdown'
            ),
            row=2, col=1
        )
        
        # 4. ROI and Growth
        roi = (pathway_results['gdp_impact']['total_gdp_impact'] / 
               pathway_results['initial_investment'] - 1) * 100
        
        years = list(range(1, 11))
        cumulative_growth = [pathway_results['gdp_impact']['long_term_annual_growth'] * year * 100 
                           for year in years]
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=cumulative_growth,
                mode='lines+markers',
                name='Cumulative GDP Growth (%)',
                line=dict(color='purple', width=3)
            ),
            row=2, col=2
        )
        
        # Add ROI as horizontal line
        fig.add_trace(
            go.Scatter(
                x=years,
                y=[roi] * len(years),
                mode='lines',
                name=f'Total ROI: {roi:.1f}%',
                line=dict(color='orange', width=2, dash='dot'),
                yaxis='y2'
            ),
            row=2, col=2, secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title='AI Inference to GDP Impact Analysis',
            height=800,
            showlegend=True
        )
        
        # Update x-axis labels for value chain
        fig.update_xaxes(
            ticktext=[s[:12] + '...' if len(s) > 12 else s for s in stages],
            tickvals=list(range(len(stages))),
            row=1, col=1
        )
        
        return fig
    
    def generate_policy_recommendations(self, pathway_results: Dict) -> Dict:
        """Generate policy recommendations based on the analysis."""
        
        gdp_impact = pathway_results['gdp_impact']
        roi = (gdp_impact['total_gdp_impact'] / pathway_results['initial_investment'] - 1) * 100
        
        recommendations = {
            'investment_policy': [
                f"Increase AI compute investment - current ROI: {roi:.1f}%",
                f"Target {gdp_impact['gdp_percentage_impact']:.2%} GDP growth potential",
                "Prioritize sectors with highest AI adoption rates",
                "Establish public-private partnerships for infrastructure"
            ],
            'regulatory_framework': [
                "Develop AI governance standards for inference systems",
                "Create data sharing protocols for economic modeling",
                "Establish AI safety requirements for critical applications",
                "Implement privacy protections for inference data"
            ],
            'economic_strategy': [
                f"Plan for {gdp_impact['employment_change']:,.0f} job market changes",
                f"Leverage {gdp_impact['gdp_multiplier']:.1f}x economic multiplier",
                "Invest in workforce retraining programs",
                "Support new market creation initiatives"
            ],
            'key_metrics': {
                'total_roi': f"{roi:.1f}%",
                'gdp_impact': f"{gdp_impact['gdp_percentage_impact']:.2%}",
                'employment_change': f"{gdp_impact['employment_change']:,.0f} jobs",
                'economic_multiplier': f"{gdp_impact['gdp_multiplier']:.1f}x",
                'long_term_growth': f"{gdp_impact['long_term_annual_growth']:.2%} annually"
            }
        }
        
        return recommendations


def run_inference_gdp_analysis():
    """Run the complete inference-to-GDP impact analysis."""
    
    print("🧠 AI Inference to GDP Impact Analysis")
    print("=" * 60)
    
    # Initialize the model
    model = InferenceGDPModel()
    
    # Test different investment levels
    investment_levels = [1e9, 10e9, 100e9, 1e12]  # $1B to $1T
    
    for investment in investment_levels:
        print(f"\n💰 Analysis for ${investment/1e9:.0f}B compute investment:")
        print("-" * 50)
        
        # Run simulation
        results = model.simulate_complete_inference_pathway(investment)
        
        # Display key results
        gdp_impact = results['gdp_impact']['total_gdp_impact']
        gdp_percentage = results['gdp_impact']['gdp_percentage_impact']
        roi = (gdp_impact / investment - 1) * 100
        
        print(f"Total GDP Impact: ${gdp_impact/1e9:.1f}B ({gdp_percentage:.2%})")
        print(f"Economic ROI: {roi:.1f}%")
        print(f"Jobs Impact: {results['gdp_impact']['employment_change']:,.0f}")
        print(f"Economic Multiplier: {results['gdp_impact']['gdp_multiplier']:.1f}x")
    
    # Detailed analysis for $100B investment
    print(f"\n📊 Detailed Analysis for $100B Investment:")
    print("=" * 60)
    
    detailed_results = model.simulate_complete_inference_pathway(100e9)
    
    # Show pathway breakdown
    print("\nInference-to-GDP Pathway:")
    print(f"1. Compute Investment: ${detailed_results['initial_investment']/1e9:.1f}B")
    print(f"2. Annual Inference Capacity: {detailed_results['inference_capacity']['annual_inference_capacity']/1e12:.1f}T inferences")
    print(f"3. Service Value: ${detailed_results['service_value']['annual_service_value']/1e9:.1f}B")
    print(f"4. Automation Savings: ${detailed_results['automation_impact']['productivity_value']/1e9:.1f}B")
    print(f"5. Productivity Enhancement: ${detailed_results['productivity_enhancement']['combined_enhancement']/1e9:.1f}B")
    print(f"6. New Market Value: ${detailed_results['new_market_creation']['total_new_value']/1e9:.1f}B")
    print(f"7. Total GDP Impact: ${detailed_results['gdp_impact']['total_gdp_impact']/1e9:.1f}B")
    
    # Generate recommendations
    recommendations = model.generate_policy_recommendations(detailed_results)
    
    print("\n🎯 Policy Recommendations:")
    print("-" * 30)
    
    for category, recs in recommendations.items():
        if category != 'key_metrics':
            print(f"\n{category.replace('_', ' ').title()}:")
            for rec in recs:
                print(f"  • {rec}")
    
    print(f"\n📈 Key Economic Metrics:")
    for metric, value in recommendations['key_metrics'].items():
        print(f"  • {metric.replace('_', ' ').title()}: {value}")
    
    # Create visualization
    fig = model.create_inference_gdp_visualization(detailed_results)
    
    # Save visualization
    output_path = os.path.join(os.path.dirname(__file__), 'inference_gdp_analysis.html')
    fig.write_html(output_path)
    print(f"\n📊 Visualization saved to: {output_path}")
    
    return detailed_results, model, fig


if __name__ == "__main__":
    results, model, fig = run_inference_gdp_analysis()
    fig.show()