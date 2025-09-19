"""
AI Impact Analysis

This module analyzes the impact of artificial intelligence on wealth creation,
distribution, and economic structures. It includes models for automation effects,
AI-driven productivity gains, job displacement, new wealth creation opportunities,
and the changing nature of work and capital.

Key Components:
- AI Automation Analyzer: Models job displacement and productivity gains
- AI Wealth Creation: Analyzes new wealth opportunities from AI
- Digital Economy Simulator: Models platform economics and digital assets
- Future Scenarios: Projects long-term AI impact on wealth distribution
"""

from .automation_analyzer import AutomationAnalyzer, JobCategory, AutomationRisk
from .wealth_creation import AIWealthCreator, AIOpportunity, DigitalAsset
from .digital_economy import DigitalEconomySimulator, DigitalPlatform, DigitalParticipant
from .future_scenarios import FutureScenarioAnalyzer, ScenarioParameters, ScenarioOutcome

__version__ = "1.0.0"
__author__ = "Wealth Analysis Team"

__all__ = [
    'AutomationAnalyzer',
    'JobCategory', 
    'AutomationRisk',
    'AIWealthCreator',
    'AIOpportunity',
    'DigitalAsset',
    'DigitalEconomySimulator',
    'DigitalPlatform',
    'DigitalParticipant',
    'FutureScenarioAnalyzer',
    'ScenarioParameters',
    'ScenarioOutcome'
]