"""
Comprehensive Wealth Analysis Dashboard

Interactive dashboard that integrates all wealth modeling components
with real-time visualizations, scenario analysis, and reporting.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# Import our wealth analysis modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wealth_creation.sources import *
from distribution.analyzer import *
from ai_impact.wealth_creation import *
from wealth_management.portfolio_optimizer import *
from accumulation.lifecycle_models import *

class DashboardTheme(Enum):
    """Dashboard visual themes"""
    LIGHT = "light"
    DARK = "dark"
    PROFESSIONAL = "professional"
    COLORFUL = "colorful"

class VisualizationType(Enum):
    """Types of visualizations"""
    LINE_CHART = "line"
    BAR_CHART = "bar"
    SCATTER_PLOT = "scatter"
    HEATMAP = "heatmap"
    PIE_CHART = "pie"
    AREA_CHART = "area"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box"

@dataclass
class DashboardConfig:
    """Configuration for the wealth dashboard"""
    title: str = "Wealth Analysis Dashboard"
    theme: DashboardTheme = DashboardTheme.PROFESSIONAL
    show_sidebar: bool = True
    auto_refresh: bool = False
    refresh_interval: int = 30
    default_currency: str = "USD"
    decimal_places: int = 2
    show_tooltips: bool = True
    enable_export: bool = True

@dataclass
class LifecycleParameters:
    """Parameters for lifecycle planning"""
    current_age: int
    retirement_age: int
    current_income: float
    savings_rate: float
    income_growth_rate: float
    family_size: int
    home_ownership: bool

class WealthDashboard:
    """
    Comprehensive interactive dashboard for wealth analysis
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.data_cache = {}
        self.models = {}
        self._initialize_models()
        self._setup_theme()
    
    def _initialize_models(self):
        """Initialize all wealth analysis models"""
        # Create default parameters for wealth creation models
        from wealth_creation.sources import SourceParameters
        
        default_params = SourceParameters(
            initial_investment=10000.0,
            time_horizon=10,
            risk_level=0.5,
            market_conditions=0.7
        )
        
        self.models = {
            'business_venture': BusinessVentureModel(default_params),
            'investment': InvestmentModel(default_params),
            'employment': EmploymentModel(default_params),
            'wealth_distribution_analyzer': WealthDistributionAnalyzer(),
            'portfolio_optimizer': PortfolioOptimizer(),
            'lifecycle_accumulator': LifecycleAccumulator()
        }
    
    def _setup_theme(self):
        """Setup dashboard theme and styling"""
        if self.config.theme == DashboardTheme.DARK:
            self.color_scheme = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'warning': '#d62728',
                'background': '#2e2e2e',
                'text': '#ffffff'
            }
        elif self.config.theme == DashboardTheme.PROFESSIONAL:
            self.color_scheme = {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'success': '#F18F01',
                'warning': '#C73E1D',
                'background': '#F5F5F5',
                'text': '#333333'
            }
        else:
            self.color_scheme = {
                'primary': '#3498db',
                'secondary': '#e74c3c',
                'success': '#2ecc71',
                'warning': '#f39c12',
                'background': '#ffffff',
                'text': '#2c3e50'
            }
    
    def run(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title=self.config.title,
            page_icon="💰",
            layout="wide",
            initial_sidebar_state="expanded" if self.config.show_sidebar else "collapsed"
        )
        
        st.title(self.config.title)
        
        # Sidebar navigation
        if self.config.show_sidebar:
            page = self._render_sidebar()
            st.session_state['page'] = page
        
        # Main content area
        self._render_main_content()
        
        # Footer
        self._render_footer()
    
    def _render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("Navigation")
        
        page = st.sidebar.selectbox(
            "Select Analysis",
            [
                "Overview",
                "Wealth Creation",
                "Wealth Distribution",
                "Economic Modeling",
                "AI Impact Analysis",
                "Portfolio Management",
                "Lifecycle Planning",
                "Scenario Comparison",
                "Reports"
            ]
        )
        
        # Global parameters
        st.sidebar.header("Global Parameters")
        
        self.global_params = {
            'time_horizon': st.sidebar.slider("Time Horizon (years)", 1, 50, 20),
            'initial_wealth': st.sidebar.number_input("Initial Wealth ($)", 0, 10000000, 100000),
            'risk_tolerance': st.sidebar.selectbox("Risk Tolerance", 
                                                 ["Conservative", "Moderate", "Aggressive"]),
            'inflation_rate': st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 2.5) / 100
        }
        
        return page
    
    def _render_main_content(self):
        """Render the main dashboard content"""
        page = st.session_state.get('page', 'Overview')
        
        if page == "Overview":
            self._render_overview()
        elif page == "Wealth Creation":
            self._render_wealth_creation()
        elif page == "Wealth Distribution":
            self._render_wealth_distribution()
        elif page == "Economic Modeling":
            self._render_economic_modeling()
        elif page == "AI Impact Analysis":
            self._render_ai_impact()
        elif page == "Portfolio Management":
            self._render_portfolio_management()
        elif page == "Lifecycle Planning":
            self._render_lifecycle_planning()
        elif page == "Scenario Comparison":
            self._render_scenario_comparison()
        elif page == "Reports":
            self._render_reports()
        else:
            # Default to overview if page not found
            self._render_overview()
    
    def _render_overview(self):
        """Render the overview dashboard"""
        st.header("Wealth Analysis Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Wealth", "$1,250,000", "12.5%")
        with col2:
            st.metric("Annual Return", "8.2%", "0.3%")
        with col3:
            st.metric("Risk Score", "6.5/10", "-0.5")
        with col4:
            st.metric("Retirement Readiness", "85%", "5%")
        
        # Wealth trajectory chart
        st.subheader("Wealth Trajectory")
        wealth_data = self._generate_sample_wealth_trajectory()
        fig = self._create_wealth_trajectory_chart(wealth_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Asset allocation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Asset Allocation")
            allocation_data = {
                'Asset Class': ['Stocks', 'Bonds', 'Real Estate', 'Cash', 'Alternatives'],
                'Allocation': [60, 25, 10, 3, 2]
            }
            fig = px.pie(allocation_data, values='Allocation', names='Asset Class',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Risk vs Return")
            risk_return_data = self._generate_risk_return_data()
            fig = px.scatter(risk_return_data, x='Risk', y='Return', 
                           size='Allocation', hover_name='Asset',
                           color='Sharpe_Ratio', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_wealth_creation(self):
        """Render wealth creation analysis"""
        st.header("Wealth Creation Analysis")
        
        # Source selection
        source_type = st.selectbox(
            "Select Wealth Source",
            ["Business Venture", "Investment Portfolio", "Employment", "Real Estate", "Inheritance"]
        )
        
        if source_type == "Business Venture":
            self._render_business_venture_analysis()
        elif source_type == "Investment Portfolio":
            self._render_investment_analysis()
        elif source_type == "Employment":
            self._render_employment_analysis()
        elif source_type == "Real Estate":
            self._render_real_estate_analysis()
        elif source_type == "Inheritance":
            self._render_inheritance_analysis()
    
    def _render_business_venture_analysis(self):
        """Render business venture analysis"""
        st.subheader("Business Venture Analysis")
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing"])
            initial_investment = st.number_input("Initial Investment ($)", 10000, 10000000, 500000)
            growth_rate = st.slider("Expected Growth Rate (%)", 0, 50, 15) / 100
        
        with col2:
            risk_level = st.slider("Risk Level (1-10)", 1, 10, 7)
            time_to_exit = st.slider("Time to Exit (years)", 1, 20, 5)
            success_probability = st.slider("Success Probability (%)", 10, 95, 70) / 100
        
        # Simulate business venture
        params = SourceParameters(
            initial_investment=initial_investment,
            time_horizon=time_to_exit,
            risk_level=risk_level / 10,
            market_conditions=0.7
        )
        
        # Create visualization
        venture_data = self._simulate_business_venture(params, industry, growth_rate, success_probability)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            fig = self._create_venture_projection_chart(venture_data)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = self._create_venture_risk_analysis(venture_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            expected_value = venture_data['expected_value']
            st.metric("Expected Value", f"${expected_value:,.0f}")
        with col2:
            roi = (expected_value - initial_investment) / initial_investment * 100
            st.metric("Expected ROI", f"{roi:.1f}%")
        with col3:
            st.metric("Break-even Time", f"{venture_data['breakeven_time']:.1f} years")
        with col4:
            st.metric("Risk-Adjusted Return", f"{venture_data['risk_adjusted_return']:.1f}%")
    
    def _render_wealth_distribution(self):
        """Render wealth distribution analysis"""
        st.header("Wealth Distribution Analysis")
        
        # Generate sample distribution data
        distribution_data = self._generate_wealth_distribution_data()
        
        # Gini coefficient and other metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Gini Coefficient", "0.68", "0.02")
        with col2:
            st.metric("Top 1% Share", "32.1%", "1.2%")
        with col3:
            st.metric("Median Wealth", "$121,700", "3.5%")
        
        # Distribution visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Wealth Distribution")
            fig = self._create_wealth_distribution_chart(distribution_data)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Lorenz Curve")
            fig = self._create_lorenz_curve(distribution_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Social mobility analysis
        st.subheader("Social Mobility Analysis")
        mobility_data = self._generate_mobility_data()
        fig = self._create_mobility_heatmap(mobility_data)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_economic_modeling(self):
        """Render economic modeling analysis"""
        st.header("Economic Modeling")
        
        # Model selection
        model_type = st.selectbox(
            "Select Economic Model",
            ["GDP Growth Impact", "Inflation Analysis", "Market Cycles", "Policy Impact"]
        )
        
        if model_type == "GDP Growth Impact":
            st.subheader("GDP Growth Impact on Wealth")
            
            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                gdp_growth = st.slider("GDP Growth Rate (%)", -5.0, 10.0, 2.5) / 100
                inflation_rate = st.slider("Inflation Rate (%)", 0.0, 10.0, 2.0) / 100
            
            with col2:
                employment_rate = st.slider("Employment Rate (%)", 80.0, 100.0, 95.0) / 100
                interest_rate = st.slider("Interest Rate (%)", 0.0, 15.0, 3.0) / 100
            
            # Generate economic impact data
            impact_data = self._generate_economic_impact_data(gdp_growth, inflation_rate, employment_rate, interest_rate)
            
            # Visualization
            fig = px.line(impact_data, x='year', y='wealth_impact', 
                         title="Economic Impact on Wealth Over Time")
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_type == "Inflation Analysis":
            st.subheader("Inflation Impact Analysis")
            st.info("Inflation analysis shows how purchasing power changes over time.")
            
        elif model_type == "Market Cycles":
            st.subheader("Market Cycle Analysis")
            st.info("Market cycle analysis helps understand wealth creation patterns.")
            
        elif model_type == "Policy Impact":
            st.subheader("Policy Impact Analysis")
            st.info("Policy impact analysis evaluates wealth effects of economic policies.")

    def _render_ai_impact(self):
        """Render AI impact analysis"""
        st.header("AI Impact Analysis")
        
        # AI impact categories
        impact_category = st.selectbox(
            "Select AI Impact Category",
            ["Automation Effects", "Digital Economy", "Future Scenarios", "Job Displacement"]
        )
        
        if impact_category == "Automation Effects":
            st.subheader("Automation Impact on Wealth Creation")
            
            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                automation_level = st.slider("Automation Level (%)", 0, 100, 30)
                industry_sector = st.selectbox("Industry Sector", 
                                             ["Manufacturing", "Services", "Technology", "Healthcare"])
            
            with col2:
                adaptation_rate = st.slider("Workforce Adaptation Rate (%)", 0, 100, 60)
                investment_in_ai = st.number_input("AI Investment ($)", 0, 10000000, 1000000)
            
            # Generate automation impact data
            automation_data = self._generate_automation_impact_data(automation_level, industry_sector, adaptation_rate)
            
            # Visualization
            fig = px.bar(automation_data, x='job_category', y='impact_score',
                        title="Automation Impact by Job Category")
            st.plotly_chart(fig, use_container_width=True)
            
        elif impact_category == "Digital Economy":
            st.subheader("Digital Economy Opportunities")
            st.info("Digital economy analysis shows new wealth creation opportunities.")
            
        elif impact_category == "Future Scenarios":
            st.subheader("AI Future Scenarios")
            st.info("Future scenario analysis for AI-driven wealth changes.")
            
        elif impact_category == "Job Displacement":
            st.subheader("Job Displacement Analysis")
            st.info("Analysis of job displacement and retraining opportunities.")
    
    def _render_portfolio_management(self):
        """Render portfolio management analysis"""
        st.header("Portfolio Management")
        
        # Portfolio parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Configuration")
            portfolio_value = st.number_input("Portfolio Value ($)", 10000, 50000000, 1000000)
            rebalancing_freq = st.selectbox("Rebalancing Frequency", 
                                          ["Monthly", "Quarterly", "Semi-Annual", "Annual"])
            objective = st.selectbox("Optimization Objective", 
                                   ["Max Sharpe", "Min Variance", "Max Return", "Risk Parity"])
        
        with col2:
            st.subheader("Risk Parameters")
            risk_tolerance = st.slider("Risk Tolerance", 1, 10, 6)
            max_drawdown = st.slider("Max Drawdown (%)", 5, 50, 20) / 100
            target_return = st.slider("Target Return (%)", 3, 20, 8) / 100
        
        # Generate efficient frontier
        st.subheader("Efficient Frontier")
        efficient_frontier_data = self._generate_efficient_frontier()
        fig = self._create_efficient_frontier_chart(efficient_frontier_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio composition
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Optimal Allocation")
            allocation_data = self._generate_optimal_allocation()
            fig = px.pie(allocation_data, values='Weight', names='Asset',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Risk Contribution")
            risk_contrib_data = self._generate_risk_contribution()
            fig = px.bar(risk_contrib_data, x='Asset', y='Risk_Contribution',
                        color='Risk_Contribution', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_lifecycle_planning(self):
        """Render lifecycle planning analysis"""
        st.header("Lifecycle Wealth Planning")
        
        # Lifecycle parameters
        col1, col2 = st.columns(2)
        
        with col1:
            current_age = st.slider("Current Age", 18, 80, 30)
            retirement_age = st.slider("Retirement Age", 50, 80, 65)
            current_income = st.number_input("Current Income ($)", 20000, 1000000, 75000)
            savings_rate = st.slider("Savings Rate (%)", 5, 50, 15) / 100
        
        with col2:
            income_growth = st.slider("Income Growth Rate (%)", 0, 10, 3) / 100
            family_size = st.slider("Family Size", 1, 8, 2)
            risk_profile = st.selectbox("Risk Profile", 
                                      ["Conservative", "Moderate", "Aggressive", "Very Aggressive"])
            home_ownership = st.checkbox("Home Ownership", True)
        
        # Lifecycle simulation
        lifecycle_params = LifecycleParameters(
            current_age=current_age,
            retirement_age=retirement_age,
            current_income=current_income,
            savings_rate=savings_rate,
            income_growth_rate=income_growth,
            family_size=family_size,
            home_ownership=home_ownership
        )
        
        # Generate lifecycle trajectory
        lifecycle_data = self._simulate_lifecycle(lifecycle_params)
        
        # Wealth trajectory over lifecycle
        st.subheader("Wealth Accumulation Trajectory")
        fig = self._create_lifecycle_trajectory_chart(lifecycle_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Retirement readiness
        col1, col2, col3 = st.columns(3)
        
        with col1:
            retirement_wealth = lifecycle_data['final_wealth']
            st.metric("Retirement Wealth", f"${retirement_wealth:,.0f}")
        with col2:
            replacement_ratio = lifecycle_data['replacement_ratio']
            st.metric("Income Replacement", f"{replacement_ratio:.1%}")
        with col3:
            readiness_score = lifecycle_data['readiness_score']
            st.metric("Readiness Score", f"{readiness_score:.1%}")
    
    def _render_scenario_comparison(self):
        """Render scenario comparison analysis"""
        st.header("Scenario Comparison")
        
        # Scenario selection
        scenarios = st.multiselect(
            "Select Scenarios to Compare",
            ["Base Case", "Bull Market", "Bear Market", "High Inflation", 
             "Economic Recession", "Technology Boom", "Market Crash"],
            default=["Base Case", "Bull Market", "Bear Market"]
        )
        
        if scenarios:
            # Generate scenario data
            scenario_data = self._generate_scenario_comparison(scenarios)
            
            # Comparison chart
            st.subheader("Wealth Trajectory Comparison")
            fig = self._create_scenario_comparison_chart(scenario_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Scenario metrics table
            st.subheader("Scenario Metrics")
            metrics_df = self._create_scenario_metrics_table(scenario_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Risk analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Return Distribution")
                fig = self._create_return_distribution_chart(scenario_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Risk-Return Profile")
                fig = self._create_risk_return_scatter(scenario_data)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_reports(self):
        """Render reports section"""
        st.header("Reports & Export")
        
        report_type = st.selectbox(
            "Select Report Type",
            ["Comprehensive Analysis", "Portfolio Summary", "Lifecycle Planning", 
             "Risk Assessment", "Performance Attribution"]
        )
        
        report_format = st.selectbox("Report Format", ["PDF", "HTML", "Excel", "PowerPoint"])
        
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                report_data = self._generate_report(report_type, report_format)
                st.success("Report generated successfully!")
                
                # Display report preview
                if report_format == "HTML":
                    st.components.v1.html(report_data, height=600, scrolling=True)
                else:
                    st.download_button(
                        label=f"Download {report_type} Report",
                        data=report_data,
                        file_name=f"wealth_analysis_report.{report_format.lower()}",
                        mime=f"application/{report_format.lower()}"
                    )
    
    def _render_footer(self):
        """Render dashboard footer"""
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Wealth Analysis Framework v1.0**")
        with col2:
            st.markdown("Last Updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
        with col3:
            if st.button("Refresh Data"):
                st.experimental_rerun()
    
    # Helper methods for generating sample data and charts
    def _generate_sample_wealth_trajectory(self) -> pd.DataFrame:
        """Generate sample wealth trajectory data"""
        years = np.arange(0, 21)
        base_wealth = 100000
        growth_rate = 0.08
        volatility = 0.15
        
        np.random.seed(42)
        returns = np.random.normal(growth_rate, volatility, len(years))
        wealth = [base_wealth]
        
        for i in range(1, len(years)):
            wealth.append(wealth[-1] * (1 + returns[i]))
        
        return pd.DataFrame({
            'Year': years,
            'Wealth': wealth,
            'Conservative': [base_wealth * (1.05 ** year) for year in years],
            'Aggressive': [base_wealth * (1.12 ** year) for year in years]
        })
    
    def _create_wealth_trajectory_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create wealth trajectory chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['Year'], y=data['Wealth'],
            mode='lines+markers', name='Current Portfolio',
            line=dict(color=self.color_scheme['primary'], width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['Year'], y=data['Conservative'],
            mode='lines', name='Conservative',
            line=dict(color=self.color_scheme['success'], dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=data['Year'], y=data['Aggressive'],
            mode='lines', name='Aggressive',
            line=dict(color=self.color_scheme['warning'], dash='dot')
        ))
        
        fig.update_layout(
            title="Wealth Trajectory Projection",
            xaxis_title="Years",
            yaxis_title="Wealth ($)",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    # Additional helper methods would be implemented here...
    # (Due to length constraints, showing structure only)
    
    def _generate_risk_return_data(self) -> pd.DataFrame:
        """Generate sample risk-return data"""
        return pd.DataFrame({
            'Asset': ['Stocks', 'Bonds', 'Real Estate', 'Commodities', 'Cash'],
            'Risk': [0.16, 0.05, 0.12, 0.20, 0.01],
            'Return': [0.10, 0.04, 0.08, 0.06, 0.02],
            'Allocation': [60, 25, 10, 3, 2],
            'Sharpe_Ratio': [0.625, 0.8, 0.667, 0.3, 2.0]
        })
    
    def _generate_wealth_distribution_data(self) -> pd.DataFrame:
        """Generate sample wealth distribution data"""
        np.random.seed(42)
        wealth = np.random.lognormal(10, 1.5, 10000)
        return pd.DataFrame({'wealth': wealth})
    
    def _create_wealth_distribution_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create wealth distribution chart"""
        fig = px.histogram(data, x='wealth', nbins=50, title="Wealth Distribution")
        return fig
    
    def _create_lorenz_curve(self, data: pd.DataFrame) -> go.Figure:
        """Create Lorenz curve"""
        sorted_wealth = np.sort(data['wealth'])
        cumulative_wealth = np.cumsum(sorted_wealth) / np.sum(sorted_wealth)
        population_percentile = np.arange(1, len(sorted_wealth) + 1) / len(sorted_wealth)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=population_percentile, y=cumulative_wealth, 
                                mode='lines', name='Lorenz Curve'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                name='Perfect Equality', line=dict(dash='dash')))
        fig.update_layout(title="Lorenz Curve", xaxis_title="Population Percentile", 
                         yaxis_title="Cumulative Wealth Share")
        return fig
    
    def _generate_mobility_data(self) -> pd.DataFrame:
        """Generate sample mobility data"""
        np.random.seed(42)
        return pd.DataFrame({
            'parent_income': np.random.normal(50000, 20000, 1000),
            'child_income': np.random.normal(55000, 25000, 1000)
        })
    
    def _create_mobility_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create mobility heatmap"""
        fig = px.density_heatmap(data, x='parent_income', y='child_income', 
                                title="Income Mobility Heatmap")
        return fig
    
    def _generate_economic_impact_data(self, gdp_growth, inflation_rate, employment_rate, interest_rate) -> pd.DataFrame:
        """Generate economic impact data"""
        years = np.arange(2024, 2034)
        wealth_impact = []
        base_wealth = 100000
        
        for year in years:
            impact = base_wealth * ((1 + gdp_growth - inflation_rate) ** (year - 2024))
            impact *= employment_rate  # Employment effect
            impact *= (1 + 0.1 / (1 + interest_rate))  # Interest rate effect
            wealth_impact.append(impact)
        
        return pd.DataFrame({'year': years, 'wealth_impact': wealth_impact})
    
    def _generate_automation_impact_data(self, automation_level, industry_sector, adaptation_rate) -> pd.DataFrame:
        """Generate automation impact data"""
        job_categories = ['Manual Labor', 'Administrative', 'Technical', 'Creative', 'Management']
        
        # Base impact scores (negative means job displacement)
        base_impacts = {'Manual Labor': -0.8, 'Administrative': -0.6, 'Technical': -0.3, 
                       'Creative': 0.2, 'Management': 0.1}
        
        impact_scores = []
        for category in job_categories:
            impact = base_impacts[category] * (automation_level / 100)
            impact *= (1 - adaptation_rate / 100)  # Adaptation reduces negative impact
            impact_scores.append(impact)
        
        return pd.DataFrame({'job_category': job_categories, 'impact_score': impact_scores})
    
    def _simulate_business_venture(self, params, industry, growth_rate, success_prob):
        """Simulate business venture outcomes"""
        return {
            'expected_value': params.initial_investment * (1 + growth_rate) ** params.time_horizon * success_prob,
            'breakeven_time': 2.5,
            'risk_adjusted_return': growth_rate * success_prob * 100
        }
    
    def _create_venture_projection_chart(self, data):
        """Create venture projection chart"""
        fig = go.Figure()
        # Sample implementation
        years = np.arange(0, 6)
        projections = [data['expected_value'] * (0.8 ** year) for year in years]
        fig.add_trace(go.Scatter(x=years, y=projections, mode='lines+markers', name='Projection'))
        fig.update_layout(title="Venture Value Projection", xaxis_title="Years", yaxis_title="Value ($)")
        return fig
    
    def _create_venture_risk_analysis(self, data):
        """Create venture risk analysis chart"""
        fig = go.Figure()
        # Sample implementation
        scenarios = ['Best Case', 'Expected', 'Worst Case']
        values = [data['expected_value'] * 1.5, data['expected_value'], data['expected_value'] * 0.3]
        fig.add_trace(go.Bar(x=scenarios, y=values, name='Scenarios'))
        fig.update_layout(title="Risk Scenario Analysis", xaxis_title="Scenario", yaxis_title="Value ($)")
        return fig
    
    def _generate_efficient_frontier(self) -> pd.DataFrame:
        """Generate efficient frontier data"""
        risks = np.linspace(0.05, 0.25, 20)
        returns = []
        for risk in risks:
            ret = 0.02 + risk * 0.4 - 0.5 * risk ** 2  # Sample efficient frontier
            returns.append(max(ret, 0.01))
        return pd.DataFrame({'Risk': risks, 'Return': returns})
    
    def _create_efficient_frontier_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create efficient frontier chart"""
        fig = px.line(data, x='Risk', y='Return', title="Efficient Frontier")
        return fig
    
    def _generate_optimal_allocation(self) -> pd.DataFrame:
        """Generate optimal allocation data"""
        return pd.DataFrame({
            'Asset': ['Stocks', 'Bonds', 'Real Estate', 'Commodities'],
            'Weight': [0.5, 0.3, 0.15, 0.05]
        })
    
    def _generate_risk_contribution(self) -> pd.DataFrame:
        """Generate risk contribution data"""
        return pd.DataFrame({
            'Asset': ['Stocks', 'Bonds', 'Real Estate', 'Commodities'],
            'Risk_Contribution': [0.65, 0.15, 0.15, 0.05]
        })
    
    def _simulate_lifecycle(self, params) -> dict:
        """Simulate lifecycle wealth accumulation"""
        years_to_retirement = params.retirement_age - params.current_age
        final_wealth = params.current_income * params.savings_rate * years_to_retirement * 1.5
        replacement_ratio = (final_wealth * 0.04) / params.current_income
        readiness_score = min(replacement_ratio / 0.8, 1.0)
        
        return {
            'final_wealth': final_wealth,
            'replacement_ratio': replacement_ratio,
            'readiness_score': readiness_score
        }
    
    def _create_lifecycle_trajectory_chart(self, data: dict) -> go.Figure:
        """Create lifecycle trajectory chart"""
        fig = go.Figure()
        years = np.arange(0, 36)  # 35 years to retirement
        wealth = [year * data['final_wealth'] / 35 for year in years]
        fig.add_trace(go.Scatter(x=years, y=wealth, mode='lines', name='Wealth Accumulation'))
        fig.update_layout(title="Lifecycle Wealth Trajectory", xaxis_title="Years", yaxis_title="Wealth ($)")
        return fig
    
    def _generate_scenario_comparison(self, scenarios: List[str]) -> pd.DataFrame:
        """Generate scenario comparison data"""
        years = np.arange(0, 21)
        data = {'Year': years}
        
        for scenario in scenarios:
            if scenario == "Base Case":
                returns = [100000 * (1.08 ** year) for year in years]
            elif scenario == "Bull Market":
                returns = [100000 * (1.12 ** year) for year in years]
            elif scenario == "Bear Market":
                returns = [100000 * (1.04 ** year) for year in years]
            else:
                returns = [100000 * (1.06 ** year) for year in years]
            data[scenario] = returns
        
        return pd.DataFrame(data)
    
    def _create_scenario_comparison_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create scenario comparison chart"""
        fig = go.Figure()
        for column in data.columns[1:]:  # Skip 'Year' column
            fig.add_trace(go.Scatter(x=data['Year'], y=data[column], mode='lines', name=column))
        fig.update_layout(title="Scenario Comparison", xaxis_title="Years", yaxis_title="Wealth ($)")
        return fig
    
    def _create_scenario_metrics_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create scenario metrics table"""
        metrics = []
        for column in data.columns[1:]:
            final_value = data[column].iloc[-1]
            cagr = (final_value / data[column].iloc[0]) ** (1/20) - 1
            volatility = data[column].pct_change().std() * np.sqrt(12)
            metrics.append({
                'Scenario': column,
                'Final Value': f"${final_value:,.0f}",
                'CAGR': f"{cagr:.1%}",
                'Volatility': f"{volatility:.1%}"
            })
        return pd.DataFrame(metrics)
    
    def _create_return_distribution_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create return distribution chart"""
        fig = go.Figure()
        for column in data.columns[1:]:
            returns = data[column].pct_change().dropna()
            fig.add_trace(go.Histogram(x=returns, name=column, opacity=0.7))
        fig.update_layout(title="Return Distribution", xaxis_title="Returns", yaxis_title="Frequency")
        return fig
    
    def _create_risk_return_scatter(self, data: pd.DataFrame) -> go.Figure:
        """Create risk-return scatter plot"""
        fig = go.Figure()
        for column in data.columns[1:]:
            returns = data[column].pct_change().dropna()
            avg_return = returns.mean()
            volatility = returns.std()
            fig.add_trace(go.Scatter(x=[volatility], y=[avg_return], mode='markers', 
                                   name=column, marker=dict(size=10)))
        fig.update_layout(title="Risk vs Return", xaxis_title="Risk (Volatility)", yaxis_title="Return")
        return fig
    
    def _generate_report(self, report_type: str, report_format: str) -> bytes:
        """Generate report data"""
        if report_format == "HTML":
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report_type}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2E86AB; border-bottom: 2px solid #2E86AB; }}
                    .metric {{ background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                    .chart-placeholder {{ background: #e8e8e8; height: 200px; text-align: center; 
                                        line-height: 200px; margin: 20px 0; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>{report_type}</h1>
                <div class="metric">
                    <h3>Portfolio Value: $1,250,000</h3>
                    <p>Total Return: +15.3% YTD</p>
                </div>
                <div class="metric">
                    <h3>Risk Metrics</h3>
                    <p>Volatility: 12.5% | Sharpe Ratio: 1.42 | Max Drawdown: -8.2%</p>
                </div>
                <div class="chart-placeholder">
                    [Wealth Trajectory Chart Placeholder]
                </div>
                <div class="chart-placeholder">
                    [Asset Allocation Chart Placeholder]
                </div>
                <p><em>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            </body>
            </html>
            """
            return html_content.encode('utf-8')
        
        elif report_format == "PDF":
            try:
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                from io import BytesIO
                
                # Create PDF buffer
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []
                
                # Title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    spaceAfter=30,
                    textColor=colors.HexColor('#2E86AB')
                )
                story.append(Paragraph(report_type, title_style))
                story.append(Spacer(1, 20))
                
                # Executive Summary
                story.append(Paragraph("Executive Summary", styles['Heading2']))
                summary_text = f"""
                This {report_type.lower()} provides a comprehensive analysis of your wealth portfolio 
                as of {pd.Timestamp.now().strftime('%B %d, %Y')}. The analysis includes performance 
                metrics, risk assessment, and strategic recommendations for wealth optimization.
                """
                story.append(Paragraph(summary_text, styles['Normal']))
                story.append(Spacer(1, 20))
                
                # Key Metrics Table
                story.append(Paragraph("Key Performance Metrics", styles['Heading2']))
                data = [
                    ['Metric', 'Value', 'Benchmark'],
                    ['Portfolio Value', '$1,250,000', 'N/A'],
                    ['Total Return (YTD)', '+15.3%', '+12.1%'],
                    ['Volatility', '12.5%', '15.2%'],
                    ['Sharpe Ratio', '1.42', '1.18'],
                    ['Max Drawdown', '-8.2%', '-12.5%'],
                ]
                
                table = Table(data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 20))
                
                # Risk Analysis
                story.append(Paragraph("Risk Analysis", styles['Heading2']))
                risk_text = """
                Your portfolio demonstrates strong risk-adjusted returns with a Sharpe ratio of 1.42, 
                significantly outperforming the benchmark. The maximum drawdown of -8.2% indicates 
                good downside protection during market volatility.
                """
                story.append(Paragraph(risk_text, styles['Normal']))
                story.append(Spacer(1, 20))
                
                # Recommendations
                story.append(Paragraph("Strategic Recommendations", styles['Heading2']))
                recommendations = [
                    "Consider rebalancing to maintain target asset allocation",
                    "Evaluate tax-loss harvesting opportunities",
                    "Review and update investment policy statement",
                    "Monitor correlation changes in portfolio holdings"
                ]
                
                for rec in recommendations:
                    story.append(Paragraph(f"• {rec}", styles['Normal']))
                
                story.append(Spacer(1, 30))
                
                # Footer
                footer_text = f"Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Wealth Analysis Framework v1.0"
                story.append(Paragraph(footer_text, styles['Normal']))
                
                # Build PDF
                doc.build(story)
                buffer.seek(0)
                return buffer.getvalue()
                
            except ImportError:
                # Fallback if reportlab is not installed
                error_msg = f"PDF generation requires reportlab library. Please install: pip install reportlab"
                return error_msg.encode('utf-8')
        
        elif report_format == "Excel":
            try:
                import openpyxl
                from io import BytesIO
                
                buffer = BytesIO()
                workbook = openpyxl.Workbook()
                sheet = workbook.active
                sheet.title = report_type
                
                # Add headers and data
                sheet['A1'] = report_type
                sheet['A1'].font = openpyxl.styles.Font(size=16, bold=True)
                
                sheet['A3'] = 'Metric'
                sheet['B3'] = 'Value'
                sheet['C3'] = 'Benchmark'
                
                data = [
                    ['Portfolio Value', '$1,250,000', 'N/A'],
                    ['Total Return (YTD)', '+15.3%', '+12.1%'],
                    ['Volatility', '12.5%', '15.2%'],
                    ['Sharpe Ratio', '1.42', '1.18'],
                    ['Max Drawdown', '-8.2%', '-12.5%'],
                ]
                
                for i, row in enumerate(data, start=4):
                    for j, value in enumerate(row, start=1):
                        sheet.cell(row=i, column=j, value=value)
                
                workbook.save(buffer)
                buffer.seek(0)
                return buffer.getvalue()
                
            except ImportError:
                error_msg = f"Excel generation requires openpyxl library. Please install: pip install openpyxl"
                return error_msg.encode('utf-8')
        
        else:
            # Default fallback
            content = f"Sample {report_type} report data in {report_format} format\nGenerated: {pd.Timestamp.now()}"
            return content.encode('utf-8')

# Streamlit app entry point
def main():
    """Main entry point for the dashboard"""
    config = DashboardConfig(
        title="Comprehensive Wealth Analysis Dashboard",
        theme=DashboardTheme.PROFESSIONAL,
        show_sidebar=True,
        enable_export=True
    )
    
    dashboard = WealthDashboard(config)
    dashboard.run()

if __name__ == "__main__":
    main()