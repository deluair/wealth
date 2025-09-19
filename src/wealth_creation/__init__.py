"""
Wealth Creation Simulation Module

This module provides comprehensive simulations for various wealth creation pathways
including business ventures, investments, inheritance, and other economic activities.
"""

from .simulator import WealthCreationSimulator
from .sources import (
    BusinessVentureModel,
    InvestmentModel, 
    InheritanceModel,
    EmploymentModel,
    RealEstateModel
)

__all__ = [
    'WealthCreationSimulator',
    'BusinessVentureModel',
    'InvestmentModel',
    'InheritanceModel', 
    'EmploymentModel',
    'RealEstateModel'
]