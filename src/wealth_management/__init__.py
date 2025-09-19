"""
Wealth Management Module

This module provides comprehensive wealth management tools including:
- Portfolio optimization and asset allocation
- Risk management and assessment
- Investment strategy simulation
- Performance analysis and benchmarking
- Tax optimization strategies
- Estate planning considerations

The module integrates modern portfolio theory, behavioral finance insights,
and AI-driven optimization techniques to provide sophisticated wealth
management capabilities.
"""

from .portfolio_optimizer import (
    PortfolioOptimizer,
    Asset,
    Portfolio,
    OptimizationObjective,
    RiskModel,
    OptimizationConstraints
)

from .risk_manager import (
    RiskManager,
    RiskMetrics,
    RiskAssessment,
    RiskProfile,
    RiskScenario,
    StressTest
)

from .investment_strategies import (
    InvestmentStrategy,
    StrategyType,
    StrategyConfig,
    StrategyPerformance,
    RebalancingFrequency,
    FactorType
)

from .performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceReport,
    PerformanceMetrics,
    AttributionAnalysis,
    BenchmarkComparison,
    PerformancePeriod,
    BenchmarkType,
    AttributionType
)

__version__ = "1.0.0"
__author__ = "Wealth Management AI System"

__all__ = [
    # Portfolio Optimization
    "PortfolioOptimizer",
    "Asset",
    "Portfolio", 
    "OptimizationObjective",
    "RiskModel",
    "OptimizationConstraints",
    
    # Risk Management
    "RiskManager",
    "RiskMetrics",
    "RiskAssessment",
    "RiskProfile",
    "RiskScenario",
    "StressTest",
    
    # Investment Strategies
    "InvestmentStrategy",
    "StrategyType",
    "StrategyConfig",
    "StrategyPerformance",
    "RebalancingFrequency",
    "FactorType",
    
    # Performance Analysis
    "PerformanceAnalyzer",
    "PerformanceReport",
    "PerformanceMetrics",
    "AttributionAnalysis",
    "BenchmarkComparison",
    "PerformancePeriod",
    "BenchmarkType",
    "AttributionType"
]