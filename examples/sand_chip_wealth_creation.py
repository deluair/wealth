#!/usr/bin/env python3
"""
Sand-to-Chip-to-AI Wealth Creation Example

This example demonstrates the complete wealth creation pathway from raw sand
to semiconductor chips to AI inference systems and their GDP impact.

The model shows how value is created and multiplied through each stage:
1. Raw Materials (Sand/Silicon) → Basic Materials
2. Silicon Purification → Semiconductor Grade Materials  
3. Wafer Manufacturing → Semiconductor Substrates
4. Chip Fabrication → Processing Units
5. System Integration → Computing Hardware
6. Software Development → AI/ML Platforms
7. Application Development → AI Services
8. Economic Integration → GDP Impact through Inference Management

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

from wealth_creation.sources import WealthSource, SourceParameters
from value_chain.analyzer import ValueChainAnalyzer
from ai_impact.wealth_creation import AIWealthCreator


class SandChipWealthCreator:
    """
    Models the complete wealth creation pathway from sand to AI-driven GDP impact.
    """
    
    def __init__(self):
        self.stages = {
            'raw_materials': {
                'name': 'Raw Sand/Silicon',
                'value_multiplier': 1.0,
                'cost_base': 0.10,  # $0.10 per kg of sand
                'processing_time': 1,  # days
                'capital_intensity': 0.1,
                'labor_intensity': 0.3,
                'technology_factor': 1.0
            },
            'purification': {
                'name': 'Silicon Purification',
                'value_multiplier': 2.5,  # Realistic 150% markup
                'cost_base': 5.0,
                'processing_time': 7,
                'capital_intensity': 0.7,
                'labor_intensity': 0.2,
                'technology_factor': 1.5
            },
            'wafer_manufacturing': {
                'name': 'Wafer Manufacturing',
                'value_multiplier': 3.0,  # Realistic 200% markup
                'cost_base': 20.0,
                'processing_time': 14,
                'capital_intensity': 0.8,
                'labor_intensity': 0.15,
                'technology_factor': 2.0
            },
            'chip_fabrication': {
                'name': 'Chip Fabrication',
                'value_multiplier': 4.0,  # Realistic 300% markup for high-tech
                'cost_base': 1000.0,
                'processing_time': 90,
                'capital_intensity': 0.9,
                'labor_intensity': 0.1,
                'technology_factor': 5.0
            },
            'system_integration': {
                'name': 'Computing Systems',
                'value_multiplier': 2.2,  # Realistic 120% markup for assembly
                'cost_base': 5000.0,
                'processing_time': 30,
                'capital_intensity': 0.6,
                'labor_intensity': 0.3,
                'technology_factor': 3.0
            },
            'software_platform': {
                'name': 'AI/ML Software',
                'value_multiplier': 3.5,  # Realistic 250% markup for software
                'cost_base': 10000.0,
                'processing_time': 180,
                'capital_intensity': 0.2,
                'labor_intensity': 0.7,
                'technology_factor': 10.0
            },
            'ai_applications': {
                'name': 'AI Applications',
                'value_multiplier': 2.8,  # Realistic 180% markup for services
                'cost_base': 50000.0,
                'processing_time': 365,
                'capital_intensity': 0.1,
                'labor_intensity': 0.8,
                'technology_factor': 20.0
            },
            'inference_management': {
                'name': 'Inference Management Systems',
                'value_multiplier': 1.8,  # Realistic 80% markup for enterprise
                'cost_base': 200000.0,
                'processing_time': 730,
                'capital_intensity': 0.3,
                'labor_intensity': 0.6,
                'technology_factor': 50.0
            },
            'gdp_impact': {
                'name': 'GDP Economic Impact',
                'value_multiplier': 1.4,  # Realistic 40% economic multiplier
                'cost_base': 1000000.0,
                'processing_time': 1825,  # 5 years
                'capital_intensity': 0.4,
                'labor_intensity': 0.5,
                'technology_factor': 100.0
            }
        }
        
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize the various analysis models."""
        self.value_chain = ValueChainAnalyzer()
        self.ai_creator = AIWealthCreator()
    
    def calculate_stage_wealth(self, stage_key: str, input_value: float, 
                             scale_factor: float = 1.0) -> Dict:
        """Calculate wealth creation for a specific stage."""
        stage = self.stages[stage_key]
        
        # Base calculations
        output_value = input_value * stage['value_multiplier'] * scale_factor
        value_added = output_value - input_value
        
        # Cost structure
        capital_cost = stage['cost_base'] * stage['capital_intensity'] * scale_factor
        labor_cost = stage['cost_base'] * stage['labor_intensity'] * scale_factor
        technology_cost = stage['cost_base'] * 0.1 * stage['technology_factor'] * scale_factor
        
        total_cost = capital_cost + labor_cost + technology_cost
        profit = value_added - total_cost
        profit_margin = profit / output_value if output_value > 0 else 0
        
        # ROI calculations
        roi = profit / total_cost if total_cost > 0 else 0
        
        # Time-adjusted returns
        daily_return = roi / stage['processing_time'] if stage['processing_time'] > 0 else 0
        annualized_return = daily_return * 365
        
        return {
            'stage_name': stage['name'],
            'input_value': input_value,
            'output_value': output_value,
            'value_added': value_added,
            'capital_cost': capital_cost,
            'labor_cost': labor_cost,
            'technology_cost': technology_cost,
            'total_cost': total_cost,
            'profit': profit,
            'profit_margin': profit_margin,
            'roi': roi,
            'processing_time': stage['processing_time'],
            'daily_return': daily_return,
            'annualized_return': annualized_return,
            'value_multiplier': stage['value_multiplier']
        }
    
    def simulate_complete_pathway(self, initial_investment: float = 1000.0,
                                scale_factor: float = 1.0) -> pd.DataFrame:
        """Simulate the complete sand-to-GDP wealth creation pathway."""
        
        results = []
        current_value = initial_investment
        cumulative_investment = 0
        cumulative_profit = 0
        
        for i, stage_key in enumerate(self.stages.keys()):
            stage_result = self.calculate_stage_wealth(stage_key, current_value, scale_factor)
            
            # Update cumulative metrics
            cumulative_investment += stage_result['total_cost']
            cumulative_profit += stage_result['profit']
            
            # Add cumulative metrics
            stage_result.update({
                'stage_number': i + 1,
                'cumulative_investment': cumulative_investment,
                'cumulative_profit': cumulative_profit,
                'cumulative_value': stage_result['output_value'],
                'total_multiplier': stage_result['output_value'] / initial_investment
            })
            
            results.append(stage_result)
            current_value = stage_result['output_value']
        
        return pd.DataFrame(results)
    
    def analyze_value_chain_bottlenecks(self, pathway_df: pd.DataFrame) -> Dict:
        """Identify bottlenecks and optimization opportunities in the value chain."""
        
        bottlenecks = {
            'lowest_roi_stage': pathway_df.loc[pathway_df['roi'].idxmin()]['stage_name'],
            'highest_cost_stage': pathway_df.loc[pathway_df['total_cost'].idxmax()]['stage_name'],
            'longest_processing_stage': pathway_df.loc[pathway_df['processing_time'].idxmax()]['stage_name'],
            'lowest_margin_stage': pathway_df.loc[pathway_df['profit_margin'].idxmin()]['stage_name']
        }
        
        opportunities = {
            'highest_value_add': pathway_df.loc[pathway_df['value_added'].idxmax()]['stage_name'],
            'best_roi_stage': pathway_df.loc[pathway_df['roi'].idxmax()]['stage_name'],
            'fastest_processing': pathway_df.loc[pathway_df['processing_time'].idxmin()]['stage_name'],
            'highest_margin': pathway_df.loc[pathway_df['profit_margin'].idxmax()]['stage_name']
        }
        
        return {'bottlenecks': bottlenecks, 'opportunities': opportunities}
    
    def create_wealth_pathway_visualization(self, pathway_df: pd.DataFrame) -> go.Figure:
        """Create comprehensive visualization of the wealth creation pathway."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Value Creation Pathway',
                'Profit Margins by Stage',
                'ROI Analysis',
                'Cost Structure',
                'Processing Time vs Value Added',
                'Cumulative Wealth Growth'
            ],
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"secondary_y": True}]]
        )
        
        # 1. Value Creation Pathway
        fig.add_trace(
            go.Scatter(
                x=pathway_df['stage_number'],
                y=pathway_df['output_value'],
                mode='lines+markers',
                name='Output Value',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pathway_df['stage_number'],
                y=pathway_df['total_cost'],
                mode='lines+markers',
                name='Total Cost',
                line=dict(color='red', width=2),
                marker=dict(size=6),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Profit Margins
        fig.add_trace(
            go.Bar(
                x=pathway_df['stage_name'],
                y=pathway_df['profit_margin'] * 100,
                name='Profit Margin %',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # 3. ROI Analysis
        fig.add_trace(
            go.Scatter(
                x=pathway_df['processing_time'],
                y=pathway_df['roi'] * 100,
                mode='markers',
                marker=dict(
                    size=pathway_df['value_added'] / pathway_df['value_added'].max() * 50,
                    color=pathway_df['stage_number'],
                    colorscale='viridis',
                    showscale=True
                ),
                text=pathway_df['stage_name'],
                name='ROI vs Processing Time'
            ),
            row=2, col=1
        )
        
        # 4. Cost Structure (Pie chart for last stage)
        last_stage = pathway_df.iloc[-1]
        fig.add_trace(
            go.Pie(
                labels=['Capital Cost', 'Labor Cost', 'Technology Cost'],
                values=[last_stage['capital_cost'], last_stage['labor_cost'], 
                       last_stage['technology_cost']],
                name='Cost Structure'
            ),
            row=2, col=2
        )
        
        # 5. Processing Time vs Value Added
        fig.add_trace(
            go.Scatter(
                x=pathway_df['processing_time'],
                y=pathway_df['value_added'],
                mode='markers+text',
                marker=dict(size=12, color='orange'),
                text=pathway_df['stage_number'],
                textposition='middle center',
                name='Value vs Time'
            ),
            row=3, col=1
        )
        
        # 6. Cumulative Wealth Growth
        fig.add_trace(
            go.Scatter(
                x=pathway_df['stage_number'],
                y=pathway_df['cumulative_value'],
                mode='lines+markers',
                name='Cumulative Value',
                line=dict(color='purple', width=3),
                fill='tonexty'
            ),
            row=3, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=pathway_df['stage_number'],
                y=pathway_df['cumulative_investment'],
                mode='lines+markers',
                name='Cumulative Investment',
                line=dict(color='red', width=2, dash='dash'),
                yaxis='y2'
            ),
            row=3, col=2, secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title='Sand-to-Chip-to-AI Wealth Creation Pathway Analysis',
            height=1200,
            showlegend=True
        )
        
        # Update x-axis labels
        stage_names = [name[:15] + '...' if len(name) > 15 else name 
                      for name in pathway_df['stage_name']]
        
        fig.update_xaxes(
            ticktext=stage_names,
            tickvals=pathway_df['stage_number'],
            row=1, col=2
        )
        
        return fig
    
    def generate_investment_recommendations(self, pathway_df: pd.DataFrame) -> Dict:
        """Generate investment recommendations based on the analysis."""
        
        analysis = self.analyze_value_chain_bottlenecks(pathway_df)
        
        recommendations = {
            'high_priority': [
                f"Focus investment on {analysis['opportunities']['best_roi_stage']} - highest ROI stage",
                f"Optimize {analysis['bottlenecks']['lowest_roi_stage']} - current bottleneck",
                f"Scale up {analysis['opportunities']['highest_value_add']} - maximum value creation"
            ],
            'medium_priority': [
                f"Reduce processing time in {analysis['bottlenecks']['longest_processing_stage']}",
                f"Improve margins in {analysis['bottlenecks']['lowest_margin_stage']}",
                "Consider vertical integration for cost optimization"
            ],
            'strategic_insights': [
                f"Total wealth multiplier: {pathway_df.iloc[-1]['total_multiplier']:.0f}x",
                f"Best stage ROI: {pathway_df['roi'].max():.1%}",
                f"Average processing time: {pathway_df['processing_time'].mean():.0f} days",
                f"Total value created: ${pathway_df.iloc[-1]['output_value']:,.0f}"
            ]
        }
        
        return recommendations


def run_sand_chip_analysis():
    """Run the complete sand-to-chip-to-AI wealth creation analysis."""
    
    print("🏭 Sand-to-Chip-to-AI Wealth Creation Analysis")
    print("=" * 60)
    
    # Initialize the creator
    creator = SandChipWealthCreator()
    
    # Run simulation with different investment levels
    investment_levels = [1000, 10000, 100000, 1000000]
    
    for investment in investment_levels:
        print(f"\n💰 Analysis for ${investment:,} initial investment:")
        print("-" * 50)
        
        # Simulate pathway
        pathway_df = creator.simulate_complete_pathway(
            initial_investment=investment,
            scale_factor=investment / 1000  # Scale factor based on investment
        )
        
        # Display key metrics
        final_value = pathway_df.iloc[-1]['cumulative_value']
        total_multiplier = final_value / investment
        total_profit = pathway_df['profit'].sum()
        
        print(f"Final Value: ${final_value:,.0f}")
        print(f"Total Multiplier: {total_multiplier:.1f}x")
        print(f"Total Profit: ${total_profit:,.0f}")
        print(f"Overall ROI: {(total_profit / investment) * 100:.1f}%")
        
        # Show top 3 value-creating stages
        top_stages = pathway_df.nlargest(3, 'value_added')[['stage_name', 'value_added', 'roi']]
        print("\nTop Value-Creating Stages:")
        for _, stage in top_stages.iterrows():
            print(f"  • {stage['stage_name']}: ${stage['value_added']:,.0f} ({stage['roi']:.1%} ROI)")
    
    # Generate detailed analysis for $100K investment
    print(f"\n📊 Detailed Analysis for $100,000 Investment:")
    print("=" * 60)
    
    pathway_df = creator.simulate_complete_pathway(initial_investment=100000, scale_factor=100)
    
    # Display complete pathway
    print("\nComplete Wealth Creation Pathway:")
    for _, stage in pathway_df.iterrows():
        print(f"{stage['stage_number']}. {stage['stage_name']}")
        print(f"   Input: ${stage['input_value']:,.0f} → Output: ${stage['output_value']:,.0f}")
        print(f"   Value Added: ${stage['value_added']:,.0f} | ROI: {stage['roi']:.1%}")
        print(f"   Processing Time: {stage['processing_time']} days")
        print()
    
    # Generate recommendations
    recommendations = creator.generate_investment_recommendations(pathway_df)
    
    print("🎯 Investment Recommendations:")
    print("-" * 30)
    print("\nHigh Priority:")
    for rec in recommendations['high_priority']:
        print(f"  • {rec}")
    
    print("\nMedium Priority:")
    for rec in recommendations['medium_priority']:
        print(f"  • {rec}")
    
    print("\nStrategic Insights:")
    for insight in recommendations['strategic_insights']:
        print(f"  • {insight}")
    
    # Create and save visualization
    fig = creator.create_wealth_pathway_visualization(pathway_df)
    
    # Save the visualization
    output_path = os.path.join(os.path.dirname(__file__), 'sand_chip_wealth_analysis.html')
    fig.write_html(output_path)
    print(f"\n📈 Visualization saved to: {output_path}")
    
    return pathway_df, creator, fig


if __name__ == "__main__":
    pathway_df, creator, fig = run_sand_chip_analysis()
    
    # Show the plot
    fig.show()