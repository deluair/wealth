"""
Automation Impact Analyzer

Classes for analyzing the impact of AI and automation on employment,
productivity, and wealth distribution patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class JobCategory(Enum):
    """Categories of jobs based on automation susceptibility"""
    MANUAL_ROUTINE = "manual_routine"
    COGNITIVE_ROUTINE = "cognitive_routine"
    MANUAL_NON_ROUTINE = "manual_non_routine"
    COGNITIVE_NON_ROUTINE = "cognitive_non_routine"
    CREATIVE = "creative"
    INTERPERSONAL = "interpersonal"
    MANAGEMENT = "management"

class AutomationRisk(Enum):
    """Risk levels for automation"""
    LOW = "low"          # 0-30% probability
    MEDIUM = "medium"    # 30-70% probability
    HIGH = "high"        # 70-100% probability

class SkillType(Enum):
    """Types of skills"""
    TECHNICAL = "technical"
    COGNITIVE = "cognitive"
    SOCIAL = "social"
    CREATIVE = "creative"
    PHYSICAL = "physical"

@dataclass
class Job:
    """Represents a job or occupation"""
    id: str
    title: str
    category: JobCategory
    automation_risk: AutomationRisk
    automation_probability: float  # 0-1
    current_workers: int
    median_wage: float
    skill_requirements: Dict[SkillType, float]  # 0-1 for each skill
    education_level: int  # 1-5 scale
    experience_required: int  # Years
    geographic_mobility: float  # 0-1, ability to work remotely
    
@dataclass
class Worker:
    """Represents a worker in the economy"""
    id: str
    age: int
    education_level: int
    skills: Dict[SkillType, float]  # 0-1 proficiency in each skill
    current_job_id: Optional[str]
    income: float
    wealth: float
    adaptability: float  # 0-1, ability to learn new skills
    geographic_mobility: float  # 0-1, willingness to relocate
    
@dataclass
class AutomationTechnology:
    """Represents an automation technology"""
    id: str
    name: str
    maturity_level: float  # 0-1, current technological maturity
    cost_per_unit: float
    productivity_multiplier: float  # How much it increases productivity
    job_categories_affected: List[JobCategory]
    implementation_time: int  # Years to fully implement
    learning_curve_factor: float  # How quickly costs decrease with adoption

@dataclass
class AutomationScenario:
    """Represents a scenario for automation adoption"""
    name: str
    technology_adoption_rate: float  # 0-1, speed of adoption
    policy_support: float  # 0-1, government support for automation
    worker_retraining_investment: float  # Investment in retraining programs
    social_safety_net_strength: float  # 0-1, strength of unemployment benefits
    new_job_creation_rate: float  # Rate at which new jobs are created

class AutomationAnalyzer:
    """Analyze the impact of automation on employment and wealth"""
    
    def __init__(self):
        self.jobs = {}
        self.workers = {}
        self.technologies = {}
        self.economic_parameters = {
            'gdp_growth_rate': 0.03,
            'productivity_growth_rate': 0.02,
            'wage_growth_rate': 0.025,
            'unemployment_rate': 0.05,
            'labor_force_participation': 0.63
        }
        
    def add_jobs(self, jobs: List[Job]) -> None:
        """Add jobs to the analysis"""
        for job in jobs:
            self.jobs[job.id] = job
    
    def add_workers(self, workers: List[Worker]) -> None:
        """Add workers to the analysis"""
        for worker in workers:
            self.workers[worker.id] = worker
    
    def add_technologies(self, technologies: List[AutomationTechnology]) -> None:
        """Add automation technologies"""
        for tech in technologies:
            self.technologies[tech.id] = tech
    
    def simulate_automation_impact(self, scenario: AutomationScenario,
                                 time_horizon: int = 20,
                                 random_seed: int = 42) -> Dict:
        """
        Simulate the impact of automation over time
        
        Args:
            scenario: Automation scenario parameters
            time_horizon: Number of years to simulate
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results
        """
        np.random.seed(random_seed)
        
        # Initialize tracking variables
        results = {
            'yearly_results': [],
            'employment_transitions': [],
            'wage_changes': [],
            'productivity_gains': [],
            'wealth_distribution_changes': []
        }
        
        # Create copies of jobs and workers for simulation
        sim_jobs = {jid: Job(**job.__dict__) for jid, job in self.jobs.items()}
        sim_workers = {wid: Worker(**worker.__dict__) for wid, worker in self.workers.items()}
        
        for year in range(time_horizon):
            year_results = self._simulate_year(
                sim_jobs, sim_workers, scenario, year
            )
            
            results['yearly_results'].append(year_results)
            
            # Update jobs and workers based on automation
            self._update_jobs_and_workers(sim_jobs, sim_workers, scenario, year)
        
        # Calculate summary statistics
        results['summary'] = self._calculate_automation_summary(results)
        
        return results
    
    def _simulate_year(self, jobs: Dict[str, Job], workers: Dict[str, Worker],
                      scenario: AutomationScenario, year: int) -> Dict:
        """Simulate automation impact for a single year"""
        
        # Calculate automation adoption for this year
        automation_progress = self._calculate_automation_progress(scenario, year)
        
        # Determine job displacement
        displaced_workers = self._calculate_job_displacement(
            jobs, workers, automation_progress, scenario
        )
        
        # Calculate productivity gains
        productivity_gains = self._calculate_productivity_gains(
            jobs, automation_progress
        )
        
        # Simulate job creation
        new_jobs_created = self._simulate_job_creation(
            scenario, productivity_gains, year
        )
        
        # Calculate wage effects
        wage_effects = self._calculate_wage_effects(
            jobs, workers, automation_progress, productivity_gains
        )
        
        # Analyze wealth distribution changes
        wealth_changes = self._analyze_wealth_distribution_changes(
            workers, displaced_workers, wage_effects, productivity_gains
        )
        
        return {
            'year': year,
            'automation_progress': automation_progress,
            'displaced_workers': len(displaced_workers),
            'new_jobs_created': new_jobs_created,
            'productivity_gains': productivity_gains,
            'wage_effects': wage_effects,
            'wealth_changes': wealth_changes,
            'unemployment_rate': self._calculate_unemployment_rate(workers),
            'gini_coefficient': self._calculate_wealth_gini(workers)
        }
    
    def _calculate_automation_progress(self, scenario: AutomationScenario, year: int) -> Dict:
        """Calculate automation progress for each technology"""
        progress = {}
        
        for tech_id, tech in self.technologies.items():
            # Technology adoption follows S-curve
            base_adoption = scenario.technology_adoption_rate
            policy_boost = scenario.policy_support * 0.2
            
            # Account for technology maturity and learning curve
            maturity_factor = tech.maturity_level
            learning_factor = (1 + year * 0.1) ** tech.learning_curve_factor
            
            adoption_rate = base_adoption * maturity_factor * learning_factor * (1 + policy_boost)
            
            # S-curve adoption
            cumulative_adoption = 1 / (1 + np.exp(-0.5 * (year - 10) * adoption_rate))
            
            progress[tech_id] = {
                'cumulative_adoption': cumulative_adoption,
                'yearly_adoption': adoption_rate * (1 - cumulative_adoption),
                'cost_reduction': learning_factor * 0.1,  # Cost decreases with adoption
                'productivity_impact': tech.productivity_multiplier * cumulative_adoption
            }
        
        return progress
    
    def _calculate_job_displacement(self, jobs: Dict[str, Job], workers: Dict[str, Worker],
                                  automation_progress: Dict, scenario: AutomationScenario) -> List[str]:
        """Calculate which workers are displaced by automation"""
        displaced_workers = []
        
        for worker_id, worker in workers.items():
            if worker.current_job_id and worker.current_job_id in jobs:
                job = jobs[worker.current_job_id]
                
                # Calculate displacement probability
                base_risk = job.automation_probability
                
                # Adjust for relevant technologies
                tech_impact = 0
                for tech_id, progress in automation_progress.items():
                    tech = self.technologies[tech_id]
                    if job.category in tech.job_categories_affected:
                        tech_impact += progress['cumulative_adoption'] * 0.3
                
                # Adjust for worker characteristics
                skill_protection = self._calculate_skill_protection(worker, job)
                age_factor = max(0.5, 1 - (worker.age - 25) * 0.01)  # Older workers more at risk
                
                displacement_probability = min(0.9, base_risk + tech_impact - skill_protection) * age_factor
                
                # Random displacement based on probability
                if np.random.random() < displacement_probability * scenario.technology_adoption_rate:
                    displaced_workers.append(worker_id)
        
        return displaced_workers
    
    def _calculate_skill_protection(self, worker: Worker, job: Job) -> float:
        """Calculate how much worker's skills protect against automation"""
        protection = 0
        
        # High-level skills provide more protection
        for skill_type, proficiency in worker.skills.items():
            if skill_type in [SkillType.CREATIVE, SkillType.SOCIAL, SkillType.COGNITIVE]:
                protection += proficiency * 0.2
            elif skill_type == SkillType.TECHNICAL:
                protection += proficiency * 0.1  # Technical skills can be automated too
        
        # Education provides protection
        protection += (worker.education_level / 5) * 0.15
        
        # Adaptability provides protection
        protection += worker.adaptability * 0.1
        
        return min(0.5, protection)  # Cap protection at 50%
    
    def _calculate_productivity_gains(self, jobs: Dict[str, Job], automation_progress: Dict) -> Dict:
        """Calculate productivity gains from automation"""
        total_productivity_gain = 0
        sector_gains = {}
        
        for job_id, job in jobs.items():
            job_productivity_gain = 0
            
            # Calculate impact from relevant technologies
            for tech_id, progress in automation_progress.items():
                tech = self.technologies[tech_id]
                if job.category in tech.job_categories_affected:
                    job_productivity_gain += progress['productivity_impact'] * job.current_workers
            
            total_productivity_gain += job_productivity_gain
            
            # Group by job category for sector analysis
            category = job.category.value
            if category not in sector_gains:
                sector_gains[category] = 0
            sector_gains[category] += job_productivity_gain
        
        return {
            'total_gain': total_productivity_gain,
            'sector_gains': sector_gains,
            'average_gain_per_worker': total_productivity_gain / max(1, sum(j.current_workers for j in jobs.values()))
        }
    
    def _simulate_job_creation(self, scenario: AutomationScenario, 
                             productivity_gains: Dict, year: int) -> int:
        """Simulate creation of new jobs due to automation and economic growth"""
        
        # Base job creation from economic growth
        base_creation_rate = self.economic_parameters['gdp_growth_rate'] * 0.5
        
        # Additional jobs from productivity gains (some gains translate to new opportunities)
        productivity_job_creation = productivity_gains['total_gain'] * 0.1
        
        # Scenario-specific job creation rate
        scenario_multiplier = scenario.new_job_creation_rate
        
        # Technology-driven new job categories (AI specialists, robot maintenance, etc.)
        tech_job_creation = sum(
            tech.maturity_level * 100 for tech in self.technologies.values()
        ) * scenario_multiplier
        
        total_new_jobs = int(
            (base_creation_rate + productivity_job_creation + tech_job_creation) * 
            (1 + year * 0.02)  # Accelerating job creation over time
        )
        
        return max(0, total_new_jobs)
    
    def _calculate_wage_effects(self, jobs: Dict[str, Job], workers: Dict[str, Worker],
                              automation_progress: Dict, productivity_gains: Dict) -> Dict:
        """Calculate wage effects from automation"""
        wage_effects = {
            'average_wage_change': 0,
            'wage_changes_by_category': {},
            'wage_inequality_change': 0
        }
        
        # Calculate wage changes by job category
        for category in JobCategory:
            category_jobs = [j for j in jobs.values() if j.category == category]
            if not category_jobs:
                continue
            
            # High-skill jobs benefit from productivity gains
            if category in [JobCategory.COGNITIVE_NON_ROUTINE, JobCategory.CREATIVE, 
                          JobCategory.MANAGEMENT, JobCategory.INTERPERSONAL]:
                wage_multiplier = 1 + productivity_gains['average_gain_per_worker'] * 0.3
            # Routine jobs face wage pressure
            elif category in [JobCategory.MANUAL_ROUTINE, JobCategory.COGNITIVE_ROUTINE]:
                wage_multiplier = 1 - 0.1  # 10% wage pressure
            else:
                wage_multiplier = 1 + productivity_gains['average_gain_per_worker'] * 0.1
            
            avg_wage_change = (wage_multiplier - 1) * 100  # Convert to percentage
            wage_effects['wage_changes_by_category'][category.value] = avg_wage_change
        
        # Calculate overall average wage change
        all_changes = list(wage_effects['wage_changes_by_category'].values())
        wage_effects['average_wage_change'] = np.mean(all_changes) if all_changes else 0
        
        # Calculate wage inequality change (high-skill jobs benefit more)
        high_skill_change = np.mean([
            wage_effects['wage_changes_by_category'].get(cat.value, 0)
            for cat in [JobCategory.COGNITIVE_NON_ROUTINE, JobCategory.CREATIVE, JobCategory.MANAGEMENT]
        ])
        
        low_skill_change = np.mean([
            wage_effects['wage_changes_by_category'].get(cat.value, 0)
            for cat in [JobCategory.MANUAL_ROUTINE, JobCategory.COGNITIVE_ROUTINE]
        ])
        
        wage_effects['wage_inequality_change'] = high_skill_change - low_skill_change
        
        return wage_effects
    
    def _analyze_wealth_distribution_changes(self, workers: Dict[str, Worker],
                                           displaced_workers: List[str],
                                           wage_effects: Dict,
                                           productivity_gains: Dict) -> Dict:
        """Analyze changes in wealth distribution"""
        
        # Calculate wealth changes for different worker groups
        wealth_changes = {
            'displaced_workers_wealth_loss': 0,
            'remaining_workers_wealth_gain': 0,
            'capital_owners_wealth_gain': 0,
            'overall_inequality_change': 0
        }
        
        # Displaced workers lose income and potentially wealth
        displaced_wealth_loss = 0
        for worker_id in displaced_workers:
            if worker_id in workers:
                worker = workers[worker_id]
                # Assume 6 months of income loss on average
                displaced_wealth_loss += worker.income * 0.5
        
        wealth_changes['displaced_workers_wealth_loss'] = displaced_wealth_loss
        
        # Remaining workers benefit from wage increases
        remaining_workers_gain = 0
        for worker_id, worker in workers.items():
            if worker_id not in displaced_workers and worker.current_job_id:
                job = self.jobs.get(worker.current_job_id)
                if job:
                    category_wage_change = wage_effects['wage_changes_by_category'].get(
                        job.category.value, 0
                    )
                    remaining_workers_gain += worker.income * (category_wage_change / 100)
        
        wealth_changes['remaining_workers_wealth_gain'] = remaining_workers_gain
        
        # Capital owners (those who own automation technology) benefit significantly
        # Assume productivity gains flow primarily to capital
        capital_owners_gain = productivity_gains['total_gain'] * 0.7  # 70% to capital
        wealth_changes['capital_owners_wealth_gain'] = capital_owners_gain
        
        # Calculate overall inequality change
        # Positive value means increasing inequality
        total_gains = remaining_workers_gain + capital_owners_gain
        total_losses = displaced_wealth_loss
        
        if total_gains > 0:
            capital_share_of_gains = capital_owners_gain / total_gains
            wealth_changes['overall_inequality_change'] = capital_share_of_gains - 0.3  # Baseline capital share
        
        return wealth_changes
    
    def _update_jobs_and_workers(self, jobs: Dict[str, Job], workers: Dict[str, Worker],
                               scenario: AutomationScenario, year: int) -> None:
        """Update jobs and workers based on automation effects"""
        
        # Update job worker counts based on automation
        for job_id, job in jobs.items():
            automation_impact = 0
            
            # Calculate reduction in workers needed
            for tech_id, tech in self.technologies.items():
                if job.category in tech.job_categories_affected:
                    # Assume some workers are replaced by technology
                    automation_impact += tech.productivity_multiplier * 0.1
            
            # Reduce worker count, but not below some minimum
            reduction_factor = min(0.5, automation_impact * scenario.technology_adoption_rate)
            job.current_workers = int(job.current_workers * (1 - reduction_factor))
        
        # Update worker skills through retraining
        retraining_probability = scenario.worker_retraining_investment * 0.1
        
        for worker_id, worker in workers.items():
            if np.random.random() < retraining_probability:
                # Improve technical and cognitive skills
                worker.skills[SkillType.TECHNICAL] = min(1.0, worker.skills[SkillType.TECHNICAL] + 0.1)
                worker.skills[SkillType.COGNITIVE] = min(1.0, worker.skills[SkillType.COGNITIVE] + 0.05)
    
    def _calculate_unemployment_rate(self, workers: Dict[str, Worker]) -> float:
        """Calculate current unemployment rate"""
        unemployed = sum(1 for w in workers.values() if w.current_job_id is None)
        return unemployed / len(workers) if workers else 0
    
    def _calculate_wealth_gini(self, workers: Dict[str, Worker]) -> float:
        """Calculate Gini coefficient for wealth distribution"""
        if not workers:
            return 0
        
        wealth_values = [w.wealth for w in workers.values()]
        return self._calculate_gini_coefficient(wealth_values)
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient"""
        if not values or len(values) == 0:
            return 0
        
        values = np.array(values)
        values = values[values >= 0]
        
        if len(values) == 0 or np.sum(values) == 0:
            return 0
        
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        index_sum = np.sum((np.arange(1, n + 1) * sorted_values))
        total_sum = np.sum(sorted_values)
        
        gini = (2 * index_sum) / (n * total_sum) - (n + 1) / n
        return max(0, min(1, gini))
    
    def _calculate_automation_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics for the automation simulation"""
        yearly_results = results['yearly_results']
        
        if not yearly_results:
            return {}
        
        # Extract time series data
        years = [r['year'] for r in yearly_results]
        unemployment_rates = [r['unemployment_rate'] for r in yearly_results]
        gini_coefficients = [r['gini_coefficient'] for r in yearly_results]
        displaced_workers = [r['displaced_workers'] for r in yearly_results]
        productivity_gains = [r['productivity_gains']['total_gain'] for r in yearly_results]
        
        summary = {
            'total_years_simulated': len(yearly_results),
            'final_unemployment_rate': unemployment_rates[-1],
            'unemployment_rate_change': unemployment_rates[-1] - unemployment_rates[0],
            'final_gini_coefficient': gini_coefficients[-1],
            'inequality_change': gini_coefficients[-1] - gini_coefficients[0],
            'total_workers_displaced': sum(displaced_workers),
            'total_productivity_gains': sum(productivity_gains),
            'average_annual_displacement': np.mean(displaced_workers),
            'peak_displacement_year': years[np.argmax(displaced_workers)] if displaced_workers else 0,
            'automation_trend': self._calculate_trend(productivity_gains)
        }
        
        # Calculate correlation between automation and inequality
        if len(productivity_gains) > 1 and len(gini_coefficients) > 1:
            correlation = np.corrcoef(productivity_gains, gini_coefficients)[0, 1]
            summary['automation_inequality_correlation'] = correlation
        
        return summary
    
    def analyze_automation_scenarios(self, scenarios: List[AutomationScenario],
                                   time_horizon: int = 20) -> Dict:
        """
        Analyze multiple automation scenarios
        
        Args:
            scenarios: List of automation scenarios to analyze
            time_horizon: Number of years to simulate for each scenario
            
        Returns:
            Dictionary with comparative analysis results
        """
        scenario_results = {}
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.name if scenario.name else f'scenario_{i+1}'
            
            # Run simulation for this scenario
            simulation_results = self.simulate_automation_impact(
                scenario, time_horizon, random_seed=42 + i
            )
            
            scenario_results[scenario_name] = {
                'scenario_parameters': scenario,
                'simulation_results': simulation_results
            }
        
        # Comparative analysis
        comparative_analysis = self._compare_scenarios(scenario_results)
        
        return {
            'individual_scenarios': scenario_results,
            'comparative_analysis': comparative_analysis
        }
    
    def _compare_scenarios(self, scenario_results: Dict) -> Dict:
        """Compare results across different scenarios"""
        comparison = {
            'unemployment_comparison': {},
            'inequality_comparison': {},
            'productivity_comparison': {},
            'displacement_comparison': {}
        }
        
        for scenario_name, results in scenario_results.items():
            summary = results['simulation_results']['summary']
            
            comparison['unemployment_comparison'][scenario_name] = {
                'final_rate': summary.get('final_unemployment_rate', 0),
                'change': summary.get('unemployment_rate_change', 0)
            }
            
            comparison['inequality_comparison'][scenario_name] = {
                'final_gini': summary.get('final_gini_coefficient', 0),
                'change': summary.get('inequality_change', 0)
            }
            
            comparison['productivity_comparison'][scenario_name] = {
                'total_gains': summary.get('total_productivity_gains', 0),
                'trend': summary.get('automation_trend', 'stable')
            }
            
            comparison['displacement_comparison'][scenario_name] = {
                'total_displaced': summary.get('total_workers_displaced', 0),
                'average_annual': summary.get('average_annual_displacement', 0)
            }
        
        # Find best and worst scenarios
        comparison['best_scenarios'] = self._identify_best_scenarios(comparison)
        comparison['worst_scenarios'] = self._identify_worst_scenarios(comparison)
        
        return comparison
    
    def _identify_best_scenarios(self, comparison: Dict) -> Dict:
        """Identify scenarios with best outcomes"""
        best = {}
        
        # Best for unemployment (lowest final rate)
        unemployment_data = comparison['unemployment_comparison']
        if unemployment_data:
            best_unemployment = min(unemployment_data.items(), key=lambda x: x[1]['final_rate'])
            best['lowest_unemployment'] = best_unemployment[0]
        
        # Best for inequality (lowest increase or highest decrease)
        inequality_data = comparison['inequality_comparison']
        if inequality_data:
            best_inequality = min(inequality_data.items(), key=lambda x: x[1]['change'])
            best['lowest_inequality_increase'] = best_inequality[0]
        
        # Best for productivity (highest gains)
        productivity_data = comparison['productivity_comparison']
        if productivity_data:
            best_productivity = max(productivity_data.items(), key=lambda x: x[1]['total_gains'])
            best['highest_productivity'] = best_productivity[0]
        
        return best
    
    def _identify_worst_scenarios(self, comparison: Dict) -> Dict:
        """Identify scenarios with worst outcomes"""
        worst = {}
        
        # Worst for unemployment (highest final rate)
        unemployment_data = comparison['unemployment_comparison']
        if unemployment_data:
            worst_unemployment = max(unemployment_data.items(), key=lambda x: x[1]['final_rate'])
            worst['highest_unemployment'] = worst_unemployment[0]
        
        # Worst for inequality (highest increase)
        inequality_data = comparison['inequality_comparison']
        if inequality_data:
            worst_inequality = max(inequality_data.items(), key=lambda x: x[1]['change'])
            worst['highest_inequality_increase'] = worst_inequality[0]
        
        # Worst for displacement (highest total displaced)
        displacement_data = comparison['displacement_comparison']
        if displacement_data:
            worst_displacement = max(displacement_data.items(), key=lambda x: x[1]['total_displaced'])
            worst['highest_displacement'] = worst_displacement[0]
        
        return worst
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def create_automation_jobs_dataset(self) -> List[Job]:
        """Create a realistic dataset of jobs with automation characteristics"""
        jobs_data = [
            # High automation risk jobs
            Job("cashier", "Cashier", JobCategory.COGNITIVE_ROUTINE, AutomationRisk.HIGH, 0.85,
                3500000, 25000, {SkillType.COGNITIVE: 0.3, SkillType.SOCIAL: 0.4}, 1, 0, 0.2),
            
            Job("data_entry", "Data Entry Clerk", JobCategory.COGNITIVE_ROUTINE, AutomationRisk.HIGH, 0.90,
                500000, 32000, {SkillType.COGNITIVE: 0.4, SkillType.TECHNICAL: 0.3}, 2, 1, 0.8),
            
            Job("assembly_worker", "Assembly Line Worker", JobCategory.MANUAL_ROUTINE, AutomationRisk.HIGH, 0.80,
                2000000, 35000, {SkillType.PHYSICAL: 0.7, SkillType.TECHNICAL: 0.3}, 1, 0, 0.1),
            
            # Medium automation risk jobs
            Job("accountant", "Accountant", JobCategory.COGNITIVE_ROUTINE, AutomationRisk.MEDIUM, 0.60,
                1200000, 55000, {SkillType.COGNITIVE: 0.8, SkillType.TECHNICAL: 0.6}, 4, 3, 0.7),
            
            Job("truck_driver", "Truck Driver", JobCategory.MANUAL_NON_ROUTINE, AutomationRisk.MEDIUM, 0.65,
                1800000, 45000, {SkillType.PHYSICAL: 0.6, SkillType.COGNITIVE: 0.4}, 2, 2, 0.3),
            
            Job("paralegal", "Paralegal", JobCategory.COGNITIVE_ROUTINE, AutomationRisk.MEDIUM, 0.55,
                300000, 48000, {SkillType.COGNITIVE: 0.7, SkillType.TECHNICAL: 0.5}, 3, 2, 0.6),
            
            # Low automation risk jobs
            Job("nurse", "Registered Nurse", JobCategory.INTERPERSONAL, AutomationRisk.LOW, 0.25,
                3000000, 65000, {SkillType.SOCIAL: 0.8, SkillType.COGNITIVE: 0.7, SkillType.PHYSICAL: 0.5}, 4, 2, 0.4),
            
            Job("teacher", "Elementary School Teacher", JobCategory.INTERPERSONAL, AutomationRisk.LOW, 0.20,
                1500000, 50000, {SkillType.SOCIAL: 0.9, SkillType.CREATIVE: 0.6, SkillType.COGNITIVE: 0.8}, 4, 1, 0.3),
            
            Job("software_engineer", "Software Engineer", JobCategory.COGNITIVE_NON_ROUTINE, AutomationRisk.LOW, 0.15,
                1000000, 85000, {SkillType.TECHNICAL: 0.9, SkillType.COGNITIVE: 0.9, SkillType.CREATIVE: 0.7}, 4, 3, 0.9),
            
            Job("therapist", "Physical Therapist", JobCategory.INTERPERSONAL, AutomationRisk.LOW, 0.10,
                250000, 75000, {SkillType.SOCIAL: 0.8, SkillType.PHYSICAL: 0.6, SkillType.COGNITIVE: 0.7}, 5, 3, 0.2),
            
            Job("manager", "General Manager", JobCategory.MANAGEMENT, AutomationRisk.LOW, 0.30,
                800000, 95000, {SkillType.SOCIAL: 0.8, SkillType.COGNITIVE: 0.8, SkillType.CREATIVE: 0.6}, 4, 8, 0.5),
            
            Job("artist", "Graphic Designer", JobCategory.CREATIVE, AutomationRisk.MEDIUM, 0.45,
                200000, 45000, {SkillType.CREATIVE: 0.9, SkillType.TECHNICAL: 0.7, SkillType.COGNITIVE: 0.6}, 3, 2, 0.8)
        ]
        
        return jobs_data
    
    def create_automation_technologies_dataset(self) -> List[AutomationTechnology]:
        """Create a realistic dataset of automation technologies"""
        technologies = [
            AutomationTechnology(
                "industrial_robots", "Industrial Robots", 0.8, 50000, 2.5,
                [JobCategory.MANUAL_ROUTINE, JobCategory.MANUAL_NON_ROUTINE], 3, 0.15
            ),
            
            AutomationTechnology(
                "ai_software", "AI Software Systems", 0.6, 100000, 3.0,
                [JobCategory.COGNITIVE_ROUTINE, JobCategory.COGNITIVE_NON_ROUTINE], 2, 0.25
            ),
            
            AutomationTechnology(
                "autonomous_vehicles", "Autonomous Vehicles", 0.4, 80000, 1.8,
                [JobCategory.MANUAL_NON_ROUTINE], 8, 0.20
            ),
            
            AutomationTechnology(
                "chatbots", "AI Chatbots and Virtual Assistants", 0.7, 20000, 2.0,
                [JobCategory.COGNITIVE_ROUTINE, JobCategory.INTERPERSONAL], 1, 0.30
            ),
            
            AutomationTechnology(
                "machine_learning", "Machine Learning Platforms", 0.5, 150000, 4.0,
                [JobCategory.COGNITIVE_ROUTINE, JobCategory.COGNITIVE_NON_ROUTINE], 3, 0.35
            )
        ]
        
        return technologies