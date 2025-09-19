"""
Wealth Analysis Simulation Framework

A comprehensive toolkit for analyzing wealth dynamics, creation, distribution,
and the impact of AI on economic systems.
"""

__version__ = "1.0.0"
__author__ = "Wealth Analysis Team"

from .wealth_creation import WealthCreationSimulator
from .distribution import WealthDistributionAnalyzer
from .management import WealthManagementOptimizer
from .accumulation import WealthAccumulationModel
from .ai_impact import AIImpactAnalyzer
from .visualization import WealthDashboard

__all__ = [
    'WealthCreationSimulator',
    'WealthDistributionAnalyzer', 
    'WealthManagementOptimizer',
    'WealthAccumulationModel',
    'AIImpactAnalyzer',
    'WealthDashboard'
]