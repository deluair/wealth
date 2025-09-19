"""
Wealth Analysis Visualization Module

Comprehensive visualization tools for wealth creation, distribution, and management analysis.
Provides interactive dashboards, charts, and graphs for all wealth modeling components.
"""

from .dashboard import WealthDashboard, DashboardConfig
from .charts import (
    WealthChartGenerator, ChartType, ChartStyle,
    create_wealth_trajectory_chart, create_distribution_chart,
    create_portfolio_chart, create_scenario_comparison_chart
)

__version__ = "1.0.0"
__author__ = "Wealth Analysis Framework"

__all__ = [
    # Dashboard
    "WealthDashboard",
    "DashboardConfig",
    
    # Charts
    "WealthChartGenerator",
    "ChartType",
    "ChartStyle",
    "create_wealth_trajectory_chart",
    "create_distribution_chart",
    "create_portfolio_chart",
    "create_scenario_comparison_chart",
]