"""
Wealth Value Chain Analysis

This module provides comprehensive analysis of wealth value chains,
including production, distribution, and consumption cycles.
"""

from .analyzer import ValueChainAnalyzer
from .production import ProductionAnalyzer
from .distribution_chain import DistributionChainAnalyzer
from .consumption import ConsumptionAnalyzer

__all__ = [
    'ValueChainAnalyzer',
    'ProductionAnalyzer', 
    'DistributionChainAnalyzer',
    'ConsumptionAnalyzer'
]

__version__ = "1.0.0"
__author__ = "Wealth Analysis Team"