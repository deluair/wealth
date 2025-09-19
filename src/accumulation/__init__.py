"""
Wealth Accumulation Models

This module provides comprehensive models for wealth accumulation through various
mechanisms including compound growth, systematic investing, and scenario analysis.
"""

from .compound_growth import CompoundGrowthModel, GrowthScenario
from .systematic_investing import SystematicInvestor, InvestmentPlan
from .scenario_analyzer import ScenarioAnalyzer, AccumulationScenario
from .lifecycle_models import LifecycleAccumulator, LifeStage

__version__ = "1.0.0"
__author__ = "Wealth Analysis Framework"

__all__ = [
    "CompoundGrowthModel",
    "GrowthScenario", 
    "SystematicInvestor",
    "InvestmentPlan",
    "ScenarioAnalyzer",
    "AccumulationScenario",
    "LifecycleAccumulator",
    "LifeStage"
]