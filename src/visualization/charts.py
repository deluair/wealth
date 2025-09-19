"""
Wealth Analysis Chart Generation

Comprehensive chart generation utilities for creating various types of
visualizations for wealth analysis, including trajectories, distributions,
portfolios, and scenario comparisons.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt

class ChartType(Enum):
    """Types of charts available"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    AREA = "area"
    WATERFALL = "waterfall"
    SUNBURST = "sunburst"
    TREEMAP = "treemap"
    VIOLIN = "violin"

class ChartStyle(Enum):
    """Chart styling options"""
    PROFESSIONAL = "professional"
    MODERN = "modern"
    MINIMAL = "minimal"
    COLORFUL = "colorful"
    DARK = "dark"

@dataclass
class ChartConfig:
    """Configuration for chart generation"""
    width: int = 800
    height: int = 600
    title: str = ""
    subtitle: str = ""
    show_legend: bool = True
    show_grid: bool = True
    color_scheme: List[str] = None
    font_family: str = "Arial"
    font_size: int = 12
    background_color: str = "white"
    export_format: str = "png"

class WealthChartGenerator:
    """
    Comprehensive chart generator for wealth analysis
    """
    
    def __init__(self, style: ChartStyle = ChartStyle.PROFESSIONAL):
        self.style = style
        self.color_schemes = self._initialize_color_schemes()
        self.templates = self._initialize_templates()
    
    def _initialize_color_schemes(self) -> Dict[str, List[str]]:
        """Initialize color schemes for different chart styles"""
        return {
            ChartStyle.PROFESSIONAL: [
                '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#1B998B',
                '#84A59D', '#F28482', '#F6BD60', '#F7EDE2', '#84A98C'
            ],
            ChartStyle.MODERN: [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
            ],
            ChartStyle.MINIMAL: [
                '#2C3E50', '#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7',
                '#ECF0F1', '#3498DB', '#E74C3C', '#2ECC71', '#F39C12'
            ],
            ChartStyle.COLORFUL: px.colors.qualitative.Set3,
            ChartStyle.DARK: [
                '#FF6B9D', '#C44569', '#F8B500', '#FF3838', '#70A1FF',
                '#5352ED', '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3'
            ]
        }
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize Plotly templates for different styles"""
        return {
            ChartStyle.PROFESSIONAL: 'plotly_white',
            ChartStyle.MODERN: 'plotly',
            ChartStyle.MINIMAL: 'simple_white',
            ChartStyle.COLORFUL: 'plotly',
            ChartStyle.DARK: 'plotly_dark'
        }
    
    def create_wealth_trajectory_chart(self, 
                                     data: pd.DataFrame,
                                     scenarios: List[str] = None,
                                     config: ChartConfig = None) -> go.Figure:
        """Create wealth trajectory chart with multiple scenarios"""
        config = config or ChartConfig()
        
        fig = go.Figure()
        colors = self.color_schemes[self.style]
        
        # Add traces for each scenario
        scenarios = scenarios or [col for col in data.columns if col != 'Year']
        
        for i, scenario in enumerate(scenarios):
            if scenario in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['Year'] if 'Year' in data.columns else data.index,
                    y=data[scenario],
                    mode='lines+markers',
                    name=scenario,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{scenario}</b><br>' +
                                'Year: %{x}<br>' +
                                'Wealth: $%{y:,.0f}<br>' +
                                '<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=config.title or "Wealth Trajectory Analysis",
                x=0.5,
                font=dict(size=16, family=config.font_family)
            ),
            xaxis=dict(
                title="Years",
                showgrid=config.show_grid,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Wealth ($)",
                showgrid=config.show_grid,
                gridcolor='lightgray',
                tickformat='$,.0f'
            ),
            template=self.templates[self.style],
            hovermode='x unified',
            showlegend=config.show_legend,
            width=config.width,
            height=config.height,
            font=dict(family=config.font_family, size=config.font_size)
        )
        
        return fig
    
    def create_distribution_chart(self, 
                                data: pd.DataFrame,
                                chart_type: ChartType = ChartType.HISTOGRAM,
                                config: ChartConfig = None) -> go.Figure:
        """Create wealth distribution chart"""
        config = config or ChartConfig()
        
        if chart_type == ChartType.HISTOGRAM:
            fig = px.histogram(
                data, x='wealth', nbins=50,
                title=config.title or "Wealth Distribution",
                color_discrete_sequence=self.color_schemes[self.style]
            )
            
            fig.update_layout(
                xaxis_title="Wealth ($)",
                yaxis_title="Frequency",
                template=self.templates[self.style]
            )
            
        elif chart_type == ChartType.BOX:
            fig = px.box(
                data, y='wealth',
                title=config.title or "Wealth Distribution Box Plot",
                color_discrete_sequence=self.color_schemes[self.style]
            )
            
        elif chart_type == ChartType.VIOLIN:
            fig = go.Figure()
            fig.add_trace(go.Violin(
                y=data['wealth'],
                box_visible=True,
                line_color=self.color_schemes[self.style][0],
                fillcolor=self.color_schemes[self.style][0],
                opacity=0.6,
                name="Wealth Distribution"
            ))
            
            fig.update_layout(
                title=config.title or "Wealth Distribution (Violin Plot)",
                yaxis_title="Wealth ($)",
                template=self.templates[self.style]
            )
        
        return fig
    
    def create_portfolio_chart(self, 
                             allocation_data: pd.DataFrame,
                             chart_type: ChartType = ChartType.PIE,
                             config: ChartConfig = None) -> go.Figure:
        """Create portfolio allocation chart"""
        config = config or ChartConfig()
        
        if chart_type == ChartType.PIE:
            fig = px.pie(
                allocation_data, 
                values='weight', 
                names='asset',
                title=config.title or "Portfolio Allocation",
                color_discrete_sequence=self.color_schemes[self.style]
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>' +
                            'Weight: %{percent}<br>' +
                            'Value: $%{value:,.0f}<br>' +
                            '<extra></extra>'
            )
            
        elif chart_type == ChartType.TREEMAP:
            fig = px.treemap(
                allocation_data,
                path=['asset_class', 'asset'],
                values='weight',
                title=config.title or "Portfolio Allocation (Treemap)",
                color='return',
                color_continuous_scale='RdYlGn'
            )
            
        elif chart_type == ChartType.SUNBURST:
            fig = px.sunburst(
                allocation_data,
                path=['asset_class', 'asset'],
                values='weight',
                title=config.title or "Portfolio Allocation (Sunburst)",
                color='sharpe_ratio',
                color_continuous_scale='Viridis'
            )
        
        fig.update_layout(
            template=self.templates[self.style],
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_scenario_comparison_chart(self, 
                                       scenario_data: Dict[str, pd.DataFrame],
                                       config: ChartConfig = None) -> go.Figure:
        """Create scenario comparison chart"""
        config = config or ChartConfig()
        
        fig = go.Figure()
        colors = self.color_schemes[self.style]
        
        for i, (scenario_name, data) in enumerate(scenario_data.items()):
            fig.add_trace(go.Scatter(
                x=data['year'],
                y=data['wealth'],
                mode='lines',
                name=scenario_name,
                line=dict(color=colors[i % len(colors)], width=2),
                fill='tonexty' if i > 0 else None,
                fillcolor=f'rgba({colors[i % len(colors)][1:]}, 0.1)' if i > 0 else None
            ))
        
        fig.update_layout(
            title=config.title or "Scenario Comparison",
            xaxis_title="Years",
            yaxis_title="Wealth ($)",
            template=self.templates[self.style],
            hovermode='x unified',
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_efficient_frontier_chart(self, 
                                      frontier_data: pd.DataFrame,
                                      current_portfolio: Dict = None,
                                      config: ChartConfig = None) -> go.Figure:
        """Create efficient frontier chart"""
        config = config or ChartConfig()
        
        fig = go.Figure()
        
        # Add efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_data['risk'],
            y=frontier_data['return'],
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color=self.color_schemes[self.style][0], width=3),
            marker=dict(size=6),
            hovertemplate='Risk: %{x:.2%}<br>' +
                        'Return: %{y:.2%}<br>' +
                        '<extra></extra>'
        ))
        
        # Add current portfolio if provided
        if current_portfolio:
            fig.add_trace(go.Scatter(
                x=[current_portfolio['risk']],
                y=[current_portfolio['return']],
                mode='markers',
                name='Current Portfolio',
                marker=dict(
                    size=15,
                    color=self.color_schemes[self.style][1],
                    symbol='star'
                ),
                hovertemplate='<b>Current Portfolio</b><br>' +
                            'Risk: %{x:.2%}<br>' +
                            'Return: %{y:.2%}<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title=config.title or "Efficient Frontier",
            xaxis=dict(
                title="Risk (Standard Deviation)",
                tickformat='.1%'
            ),
            yaxis=dict(
                title="Expected Return",
                tickformat='.1%'
            ),
            template=self.templates[self.style],
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_risk_return_scatter(self, 
                                 assets_data: pd.DataFrame,
                                 config: ChartConfig = None) -> go.Figure:
        """Create risk-return scatter plot"""
        config = config or ChartConfig()
        
        fig = px.scatter(
            assets_data,
            x='risk',
            y='return',
            size='weight',
            color='sharpe_ratio',
            hover_name='asset',
            title=config.title or "Risk-Return Analysis",
            color_continuous_scale='Viridis',
            size_max=30
        )
        
        fig.update_layout(
            xaxis=dict(title="Risk (Standard Deviation)", tickformat='.1%'),
            yaxis=dict(title="Expected Return", tickformat='.1%'),
            template=self.templates[self.style],
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_correlation_heatmap(self, 
                                 correlation_matrix: pd.DataFrame,
                                 config: ChartConfig = None) -> go.Figure:
        """Create correlation heatmap"""
        config = config or ChartConfig()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title=config.title or "Asset Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(
            template=self.templates[self.style],
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_performance_attribution_chart(self, 
                                           attribution_data: pd.DataFrame,
                                           config: ChartConfig = None) -> go.Figure:
        """Create performance attribution waterfall chart"""
        config = config or ChartConfig()
        
        fig = go.Figure(go.Waterfall(
            name="Performance Attribution",
            orientation="v",
            measure=["relative"] * (len(attribution_data) - 1) + ["total"],
            x=attribution_data['factor'],
            textposition="outside",
            text=[f"{val:+.2%}" for val in attribution_data['contribution']],
            y=attribution_data['contribution'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": self.color_schemes[self.style][2]}},
            decreasing={"marker": {"color": self.color_schemes[self.style][3]}},
            totals={"marker": {"color": self.color_schemes[self.style][0]}}
        ))
        
        fig.update_layout(
            title=config.title or "Performance Attribution",
            xaxis_title="Attribution Factors",
            yaxis=dict(title="Contribution (%)", tickformat='.1%'),
            template=self.templates[self.style],
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_drawdown_chart(self, 
                            returns_data: pd.DataFrame,
                            config: ChartConfig = None) -> go.Figure:
        """Create drawdown chart"""
        config = config or ChartConfig()
        
        # Calculate cumulative returns and drawdown
        cumulative_returns = (1 + returns_data['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Returns', 'Drawdown'),
            vertical_spacing=0.1
        )
        
        # Add cumulative returns
        fig.add_trace(
            go.Scatter(
                x=returns_data['date'],
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color=self.color_schemes[self.style][0])
            ),
            row=1, col=1
        )
        
        # Add drawdown
        fig.add_trace(
            go.Scatter(
                x=returns_data['date'],
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color=self.color_schemes[self.style][3]),
                fillcolor=f'rgba({self.color_schemes[self.style][3][1:]}, 0.3)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=config.title or "Returns and Drawdown Analysis",
            template=self.templates[self.style],
            width=config.width,
            height=config.height,
            showlegend=False
        )
        
        fig.update_yaxes(tickformat='.1%', row=1, col=1)
        fig.update_yaxes(tickformat='.1%', row=2, col=1)
        
        return fig
    
    def create_monte_carlo_chart(self, 
                               simulation_results: np.ndarray,
                               percentiles: List[int] = [10, 25, 50, 75, 90],
                               config: ChartConfig = None) -> go.Figure:
        """Create Monte Carlo simulation results chart"""
        config = config or ChartConfig()
        
        fig = go.Figure()
        colors = self.color_schemes[self.style]
        
        # Calculate percentiles
        years = np.arange(simulation_results.shape[1])
        
        for i, percentile in enumerate(percentiles):
            values = np.percentile(simulation_results, percentile, axis=0)
            
            fig.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines',
                name=f'{percentile}th Percentile',
                line=dict(color=colors[i % len(colors)]),
                opacity=0.7
            ))
        
        # Add median with fill
        median = np.percentile(simulation_results, 50, axis=0)
        p25 = np.percentile(simulation_results, 25, axis=0)
        p75 = np.percentile(simulation_results, 75, axis=0)
        
        fig.add_trace(go.Scatter(
            x=years,
            y=p75,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=p25,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=f'rgba({colors[0][1:]}, 0.2)',
            name='25th-75th Percentile Range',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=config.title or "Monte Carlo Simulation Results",
            xaxis_title="Years",
            yaxis=dict(title="Wealth ($)", tickformat='$,.0f'),
            template=self.templates[self.style],
            width=config.width,
            height=config.height
        )
        
        return fig

# Convenience functions for quick chart creation
def create_wealth_trajectory_chart(data: pd.DataFrame, 
                                 style: ChartStyle = ChartStyle.PROFESSIONAL,
                                 **kwargs) -> go.Figure:
    """Quick function to create wealth trajectory chart"""
    generator = WealthChartGenerator(style)
    return generator.create_wealth_trajectory_chart(data, **kwargs)

def create_distribution_chart(data: pd.DataFrame,
                            chart_type: ChartType = ChartType.HISTOGRAM,
                            style: ChartStyle = ChartStyle.PROFESSIONAL,
                            **kwargs) -> go.Figure:
    """Quick function to create distribution chart"""
    generator = WealthChartGenerator(style)
    return generator.create_distribution_chart(data, chart_type, **kwargs)

def create_portfolio_chart(allocation_data: pd.DataFrame,
                         chart_type: ChartType = ChartType.PIE,
                         style: ChartStyle = ChartStyle.PROFESSIONAL,
                         **kwargs) -> go.Figure:
    """Quick function to create portfolio chart"""
    generator = WealthChartGenerator(style)
    return generator.create_portfolio_chart(allocation_data, chart_type, **kwargs)

def create_scenario_comparison_chart(scenario_data: Dict[str, pd.DataFrame],
                                   style: ChartStyle = ChartStyle.PROFESSIONAL,
                                   **kwargs) -> go.Figure:
    """Quick function to create scenario comparison chart"""
    generator = WealthChartGenerator(style)
    return generator.create_scenario_comparison_chart(scenario_data, **kwargs)