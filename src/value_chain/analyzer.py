"""
Value Chain Analyzer

Main class for analyzing wealth value chains across production,
distribution, and consumption phases.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import networkx as nx
from scipy.optimize import minimize
import matplotlib.pyplot as plt

@dataclass
class ValueChainNode:
    """Represents a node in the value chain"""
    node_id: str
    node_type: str  # 'producer', 'distributor', 'consumer', 'intermediary'
    value_added: float
    costs: Dict[str, float]
    revenues: Dict[str, float]
    efficiency: float
    market_power: float

@dataclass
class ValueChainLink:
    """Represents a connection between value chain nodes"""
    from_node: str
    to_node: str
    flow_volume: float
    value_transfer: float
    transaction_costs: float
    markup: float

class ValueChainAnalyzer:
    """Comprehensive value chain analysis for wealth creation and distribution"""
    
    def __init__(self):
        self.nodes = {}
        self.links = []
        self.chain_graph = nx.DiGraph()
        
    def add_node(self, node: ValueChainNode):
        """Add a node to the value chain"""
        self.nodes[node.node_id] = node
        self.chain_graph.add_node(node.node_id, **node.__dict__)
    
    def add_link(self, link: ValueChainLink):
        """Add a link between nodes in the value chain"""
        self.links.append(link)
        self.chain_graph.add_edge(
            link.from_node, 
            link.to_node,
            **link.__dict__
        )
    
    def build_supply_chain(self, chain_config: Dict) -> Dict:
        """
        Build a complete supply chain from configuration
        
        Args:
            chain_config: Configuration dictionary defining the chain structure
            
        Returns:
            Dictionary with chain analysis results
        """
        # Clear existing chain
        self.nodes.clear()
        self.links.clear()
        self.chain_graph.clear()
        
        # Build nodes
        for node_config in chain_config.get('nodes', []):
            node = ValueChainNode(
                node_id=node_config['id'],
                node_type=node_config['type'],
                value_added=node_config.get('value_added', 0),
                costs=node_config.get('costs', {}),
                revenues=node_config.get('revenues', {}),
                efficiency=node_config.get('efficiency', 1.0),
                market_power=node_config.get('market_power', 0.5)
            )
            self.add_node(node)
        
        # Build links
        for link_config in chain_config.get('links', []):
            link = ValueChainLink(
                from_node=link_config['from'],
                to_node=link_config['to'],
                flow_volume=link_config.get('volume', 1.0),
                value_transfer=link_config.get('value', 0),
                transaction_costs=link_config.get('transaction_costs', 0),
                markup=link_config.get('markup', 0)
            )
            self.add_link(link)
        
        return self.analyze_complete_chain()
    
    def analyze_complete_chain(self) -> Dict:
        """Perform comprehensive analysis of the value chain"""
        analysis = {}
        
        # Basic chain metrics
        analysis['chain_metrics'] = self._calculate_chain_metrics()
        
        # Value flow analysis
        analysis['value_flows'] = self._analyze_value_flows()
        
        # Efficiency analysis
        analysis['efficiency_analysis'] = self._analyze_efficiency()
        
        # Market power analysis
        analysis['market_power'] = self._analyze_market_power()
        
        # Bottleneck identification
        analysis['bottlenecks'] = self._identify_bottlenecks()
        
        # Wealth distribution along chain
        analysis['wealth_distribution'] = self._analyze_wealth_distribution()
        
        # Risk analysis
        analysis['risk_analysis'] = self._analyze_chain_risks()
        
        return analysis
    
    def _calculate_chain_metrics(self) -> Dict:
        """Calculate basic metrics for the value chain"""
        metrics = {
            'total_nodes': len(self.nodes),
            'total_links': len(self.links),
            'chain_length': self._calculate_chain_length(),
            'total_value_added': sum(node.value_added for node in self.nodes.values()),
            'average_efficiency': np.mean([node.efficiency for node in self.nodes.values()]),
            'network_density': nx.density(self.chain_graph)
        }
        
        # Calculate node type distribution
        node_types = {}
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        metrics['node_type_distribution'] = node_types
        
        return metrics
    
    def _calculate_chain_length(self) -> Dict:
        """Calculate various measures of chain length"""
        if not self.chain_graph.nodes():
            return {'average_path_length': 0, 'diameter': 0, 'longest_path': 0}
        
        # Find source and sink nodes
        sources = [n for n in self.chain_graph.nodes() if self.chain_graph.in_degree(n) == 0]
        sinks = [n for n in self.chain_graph.nodes() if self.chain_graph.out_degree(n) == 0]
        
        if not sources or not sinks:
            return {'average_path_length': 0, 'diameter': 0, 'longest_path': 0}
        
        # Calculate path lengths
        path_lengths = []
        longest_path = 0
        
        for source in sources:
            for sink in sinks:
                try:
                    path_length = nx.shortest_path_length(self.chain_graph, source, sink)
                    path_lengths.append(path_length)
                    longest_path = max(longest_path, path_length)
                except nx.NetworkXNoPath:
                    continue
        
        return {
            'average_path_length': np.mean(path_lengths) if path_lengths else 0,
            'diameter': longest_path,
            'longest_path': longest_path
        }
    
    def _analyze_value_flows(self) -> Dict:
        """Analyze how value flows through the chain"""
        flows = {
            'total_flow_volume': sum(link.flow_volume for link in self.links),
            'total_value_transfer': sum(link.value_transfer for link in self.links),
            'average_markup': np.mean([link.markup for link in self.links]),
            'total_transaction_costs': sum(link.transaction_costs for link in self.links)
        }
        
        # Calculate flow concentration
        flow_volumes = [link.flow_volume for link in self.links]
        if flow_volumes:
            flows['flow_concentration_gini'] = self._calculate_gini(flow_volumes)
        
        # Node-level flow analysis
        node_flows = {}
        for node_id in self.nodes:
            inflow = sum(link.flow_volume for link in self.links if link.to_node == node_id)
            outflow = sum(link.flow_volume for link in self.links if link.from_node == node_id)
            
            node_flows[node_id] = {
                'inflow': inflow,
                'outflow': outflow,
                'net_flow': outflow - inflow,
                'flow_efficiency': outflow / inflow if inflow > 0 else 0
            }
        
        flows['node_flows'] = node_flows
        
        return flows
    
    def _analyze_efficiency(self) -> Dict:
        """Analyze efficiency across the value chain"""
        efficiency_analysis = {}
        
        # Overall efficiency metrics
        efficiencies = [node.efficiency for node in self.nodes.values()]
        efficiency_analysis['overall'] = {
            'mean_efficiency': np.mean(efficiencies),
            'median_efficiency': np.median(efficiencies),
            'min_efficiency': np.min(efficiencies),
            'max_efficiency': np.max(efficiencies),
            'efficiency_variance': np.var(efficiencies)
        }
        
        # Efficiency by node type
        efficiency_by_type = {}
        for node in self.nodes.values():
            if node.node_type not in efficiency_by_type:
                efficiency_by_type[node.node_type] = []
            efficiency_by_type[node.node_type].append(node.efficiency)
        
        for node_type, effs in efficiency_by_type.items():
            efficiency_analysis[f'{node_type}_efficiency'] = {
                'mean': np.mean(effs),
                'std': np.std(effs),
                'count': len(effs)
            }
        
        # Chain efficiency (multiplicative)
        chain_efficiency = 1.0
        for node in self.nodes.values():
            chain_efficiency *= node.efficiency
        efficiency_analysis['chain_efficiency'] = chain_efficiency
        
        return efficiency_analysis
    
    def _analyze_market_power(self) -> Dict:
        """Analyze market power distribution in the value chain"""
        market_power_analysis = {}
        
        # Overall market power distribution
        market_powers = [node.market_power for node in self.nodes.values()]
        market_power_analysis['distribution'] = {
            'mean': np.mean(market_powers),
            'median': np.median(market_powers),
            'gini_coefficient': self._calculate_gini(market_powers),
            'concentration_ratio': self._calculate_concentration_ratio(market_powers)
        }
        
        # Market power by position in chain
        # Calculate centrality measures
        centrality_measures = {
            'degree_centrality': nx.degree_centrality(self.chain_graph),
            'betweenness_centrality': nx.betweenness_centrality(self.chain_graph),
            'closeness_centrality': nx.closeness_centrality(self.chain_graph),
            'eigenvector_centrality': nx.eigenvector_centrality(self.chain_graph, max_iter=1000)
        }
        
        # Correlate market power with centrality
        correlations = {}
        for measure_name, centrality in centrality_measures.items():
            if len(centrality) > 1:
                powers = [self.nodes[node_id].market_power for node_id in centrality.keys()]
                centrality_values = list(centrality.values())
                correlation = np.corrcoef(powers, centrality_values)[0, 1]
                correlations[measure_name] = correlation
        
        market_power_analysis['centrality_correlations'] = correlations
        market_power_analysis['centrality_measures'] = centrality_measures
        
        return market_power_analysis
    
    def _identify_bottlenecks(self) -> Dict:
        """Identify bottlenecks in the value chain"""
        bottlenecks = {}
        
        # Capacity bottlenecks (lowest efficiency nodes)
        efficiencies = [(node_id, node.efficiency) for node_id, node in self.nodes.items()]
        efficiencies.sort(key=lambda x: x[1])
        bottlenecks['efficiency_bottlenecks'] = efficiencies[:3]  # Top 3 bottlenecks
        
        # Flow bottlenecks (nodes with highest betweenness centrality)
        betweenness = nx.betweenness_centrality(self.chain_graph)
        betweenness_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        bottlenecks['flow_bottlenecks'] = betweenness_sorted[:3]
        
        # Critical path analysis
        try:
            # Find longest path (critical path)
            sources = [n for n in self.chain_graph.nodes() if self.chain_graph.in_degree(n) == 0]
            sinks = [n for n in self.chain_graph.nodes() if self.chain_graph.out_degree(n) == 0]
            
            critical_paths = []
            for source in sources:
                for sink in sinks:
                    try:
                        path = nx.shortest_path(self.chain_graph, source, sink)
                        path_length = len(path)
                        critical_paths.append((path, path_length))
                    except nx.NetworkXNoPath:
                        continue
            
            if critical_paths:
                longest_path = max(critical_paths, key=lambda x: x[1])
                bottlenecks['critical_path'] = longest_path[0]
        except:
            bottlenecks['critical_path'] = []
        
        return bottlenecks
    
    def _analyze_wealth_distribution(self) -> Dict:
        """Analyze how wealth is distributed along the value chain"""
        wealth_analysis = {}
        
        # Calculate wealth metrics for each node
        node_wealth = {}
        for node_id, node in self.nodes.items():
            total_revenue = sum(node.revenues.values()) if node.revenues else 0
            total_costs = sum(node.costs.values()) if node.costs else 0
            profit = total_revenue - total_costs + node.value_added
            
            node_wealth[node_id] = {
                'revenue': total_revenue,
                'costs': total_costs,
                'value_added': node.value_added,
                'profit': profit,
                'profit_margin': profit / total_revenue if total_revenue > 0 else 0
            }
        
        # Aggregate wealth statistics
        profits = [w['profit'] for w in node_wealth.values()]
        revenues = [w['revenue'] for w in node_wealth.values()]
        
        wealth_analysis['aggregate'] = {
            'total_profit': sum(profits),
            'total_revenue': sum(revenues),
            'profit_gini': self._calculate_gini(profits) if profits else 0,
            'revenue_gini': self._calculate_gini(revenues) if revenues else 0
        }
        
        # Wealth by node type
        wealth_by_type = {}
        for node_id, wealth in node_wealth.items():
            node_type = self.nodes[node_id].node_type
            if node_type not in wealth_by_type:
                wealth_by_type[node_type] = {'profits': [], 'revenues': []}
            
            wealth_by_type[node_type]['profits'].append(wealth['profit'])
            wealth_by_type[node_type]['revenues'].append(wealth['revenue'])
        
        for node_type, data in wealth_by_type.items():
            wealth_analysis[f'{node_type}_wealth'] = {
                'total_profit': sum(data['profits']),
                'average_profit': np.mean(data['profits']),
                'total_revenue': sum(data['revenues']),
                'average_revenue': np.mean(data['revenues'])
            }
        
        wealth_analysis['node_wealth'] = node_wealth
        
        return wealth_analysis
    
    def _analyze_chain_risks(self) -> Dict:
        """Analyze risks in the value chain"""
        risk_analysis = {}
        
        # Single point of failure risk
        articulation_points = list(nx.articulation_points(self.chain_graph.to_undirected()))
        risk_analysis['single_points_of_failure'] = articulation_points
        
        # Concentration risk
        # Calculate how concentrated flows are
        flow_volumes = [link.flow_volume for link in self.links]
        if flow_volumes:
            risk_analysis['flow_concentration_risk'] = self._calculate_gini(flow_volumes)
        
        # Efficiency risk (variance in efficiency)
        efficiencies = [node.efficiency for node in self.nodes.values()]
        risk_analysis['efficiency_risk'] = np.var(efficiencies)
        
        # Market power concentration risk
        market_powers = [node.market_power for node in self.nodes.values()]
        risk_analysis['market_power_concentration'] = self._calculate_gini(market_powers)
        
        # Network resilience
        risk_analysis['network_resilience'] = self._calculate_network_resilience()
        
        return risk_analysis
    
    def _calculate_network_resilience(self) -> Dict:
        """Calculate network resilience metrics"""
        resilience = {}
        
        # Node connectivity
        if len(self.chain_graph.nodes()) > 1:
            resilience['node_connectivity'] = nx.node_connectivity(self.chain_graph)
            resilience['edge_connectivity'] = nx.edge_connectivity(self.chain_graph)
        else:
            resilience['node_connectivity'] = 0
            resilience['edge_connectivity'] = 0
        
        # Robustness to random failures
        original_efficiency = np.mean([node.efficiency for node in self.nodes.values()])
        
        # Simulate random node failures
        failure_impacts = []
        for _ in range(min(10, len(self.nodes))):  # Test up to 10 random failures
            if len(self.nodes) <= 1:
                break
                
            # Randomly select a node to fail
            failed_node = np.random.choice(list(self.nodes.keys()))
            
            # Calculate impact (simplified as loss of that node's efficiency contribution)
            remaining_efficiencies = [
                node.efficiency for node_id, node in self.nodes.items() 
                if node_id != failed_node
            ]
            
            if remaining_efficiencies:
                new_efficiency = np.mean(remaining_efficiencies)
                impact = (original_efficiency - new_efficiency) / original_efficiency
                failure_impacts.append(impact)
        
        if failure_impacts:
            resilience['average_failure_impact'] = np.mean(failure_impacts)
            resilience['max_failure_impact'] = np.max(failure_impacts)
        else:
            resilience['average_failure_impact'] = 0
            resilience['max_failure_impact'] = 0
        
        return resilience
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for a list of values"""
        if not values or len(values) == 0:
            return 0
        
        values = np.array(values)
        values = values[values >= 0]  # Remove negative values
        
        if len(values) == 0 or np.sum(values) == 0:
            return 0
        
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        index_sum = np.sum((np.arange(1, n + 1) * sorted_values))
        total_sum = np.sum(sorted_values)
        
        gini = (2 * index_sum) / (n * total_sum) - (n + 1) / n
        return max(0, min(1, gini))
    
    def _calculate_concentration_ratio(self, values: List[float], top_n: int = 4) -> float:
        """Calculate concentration ratio (share of top N entities)"""
        if not values:
            return 0
        
        sorted_values = sorted(values, reverse=True)
        total = sum(values)
        
        if total == 0:
            return 0
        
        top_sum = sum(sorted_values[:min(top_n, len(sorted_values))])
        return top_sum / total
    
    def optimize_chain(self, objective: str = 'efficiency') -> Dict:
        """
        Optimize the value chain for a given objective
        
        Args:
            objective: 'efficiency', 'profit', 'resilience', or 'equality'
            
        Returns:
            Dictionary with optimization results
        """
        if objective == 'efficiency':
            return self._optimize_for_efficiency()
        elif objective == 'profit':
            return self._optimize_for_profit()
        elif objective == 'resilience':
            return self._optimize_for_resilience()
        elif objective == 'equality':
            return self._optimize_for_equality()
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def _optimize_for_efficiency(self) -> Dict:
        """Optimize chain for maximum efficiency"""
        # This is a simplified optimization - in practice, this would be more complex
        current_efficiency = np.mean([node.efficiency for node in self.nodes.values()])
        
        # Identify improvement opportunities
        improvements = []
        for node_id, node in self.nodes.items():
            if node.efficiency < 0.8:  # Threshold for improvement
                potential_improvement = 0.8 - node.efficiency
                improvements.append({
                    'node_id': node_id,
                    'current_efficiency': node.efficiency,
                    'potential_improvement': potential_improvement,
                    'improvement_cost': potential_improvement * 1000  # Simplified cost model
                })
        
        improvements.sort(key=lambda x: x['potential_improvement'] / x['improvement_cost'], reverse=True)
        
        return {
            'current_efficiency': current_efficiency,
            'improvement_opportunities': improvements,
            'max_potential_efficiency': min(1.0, current_efficiency + sum(imp['potential_improvement'] for imp in improvements))
        }
    
    def _optimize_for_profit(self) -> Dict:
        """Optimize chain for maximum profit"""
        current_analysis = self.analyze_complete_chain()
        current_profit = current_analysis['wealth_distribution']['aggregate']['total_profit']
        
        # Identify profit optimization opportunities
        opportunities = []
        
        # Look for high-markup opportunities
        for link in self.links:
            if link.markup < 0.2:  # Low markup threshold
                potential_increase = 0.2 - link.markup
                opportunities.append({
                    'type': 'markup_increase',
                    'link': f"{link.from_node} -> {link.to_node}",
                    'current_markup': link.markup,
                    'potential_increase': potential_increase,
                    'estimated_profit_increase': potential_increase * link.flow_volume
                })
        
        # Look for cost reduction opportunities
        for node_id, node in self.nodes.items():
            total_costs = sum(node.costs.values()) if node.costs else 0
            if total_costs > 0:
                potential_reduction = total_costs * 0.1  # 10% cost reduction
                opportunities.append({
                    'type': 'cost_reduction',
                    'node_id': node_id,
                    'current_costs': total_costs,
                    'potential_reduction': potential_reduction,
                    'estimated_profit_increase': potential_reduction
                })
        
        opportunities.sort(key=lambda x: x['estimated_profit_increase'], reverse=True)
        
        return {
            'current_profit': current_profit,
            'optimization_opportunities': opportunities,
            'max_potential_profit': current_profit + sum(opp['estimated_profit_increase'] for opp in opportunities)
        }
    
    def _optimize_for_resilience(self) -> Dict:
        """Optimize chain for maximum resilience"""
        current_analysis = self.analyze_complete_chain()
        current_resilience = current_analysis['risk_analysis']['network_resilience']
        
        # Identify resilience improvements
        improvements = []
        
        # Add redundant connections
        single_points = current_analysis['risk_analysis']['single_points_of_failure']
        for point in single_points:
            improvements.append({
                'type': 'add_redundancy',
                'critical_node': point,
                'description': f"Add backup connections for {point}",
                'estimated_cost': 5000,
                'resilience_improvement': 0.1
            })
        
        # Diversify suppliers/customers
        for node_id, node in self.nodes.items():
            in_degree = self.chain_graph.in_degree(node_id)
            out_degree = self.chain_graph.out_degree(node_id)
            
            if in_degree == 1:  # Single supplier
                improvements.append({
                    'type': 'diversify_suppliers',
                    'node_id': node_id,
                    'description': f"Add alternative suppliers for {node_id}",
                    'estimated_cost': 3000,
                    'resilience_improvement': 0.05
                })
            
            if out_degree == 1:  # Single customer
                improvements.append({
                    'type': 'diversify_customers',
                    'node_id': node_id,
                    'description': f"Add alternative customers for {node_id}",
                    'estimated_cost': 3000,
                    'resilience_improvement': 0.05
                })
        
        improvements.sort(key=lambda x: x['resilience_improvement'] / x['estimated_cost'], reverse=True)
        
        return {
            'current_resilience': current_resilience,
            'resilience_improvements': improvements,
            'max_potential_resilience': min(1.0, sum(imp['resilience_improvement'] for imp in improvements))
        }
    
    def _optimize_for_equality(self) -> Dict:
        """Optimize chain for more equal wealth distribution"""
        current_analysis = self.analyze_complete_chain()
        current_gini = current_analysis['wealth_distribution']['aggregate']['profit_gini']
        
        # Identify equality improvements
        improvements = []
        
        # Redistribute high markups
        high_markup_links = [link for link in self.links if link.markup > 0.3]
        for link in high_markup_links:
            reduction = link.markup - 0.2  # Reduce to 20%
            improvements.append({
                'type': 'reduce_markup',
                'link': f"{link.from_node} -> {link.to_node}",
                'current_markup': link.markup,
                'proposed_reduction': reduction,
                'equality_improvement': reduction * 0.1  # Simplified impact
            })
        
        # Support low-efficiency nodes
        low_efficiency_nodes = [
            (node_id, node) for node_id, node in self.nodes.items() 
            if node.efficiency < 0.6
        ]
        
        for node_id, node in low_efficiency_nodes:
            improvement = 0.7 - node.efficiency
            improvements.append({
                'type': 'efficiency_support',
                'node_id': node_id,
                'current_efficiency': node.efficiency,
                'proposed_improvement': improvement,
                'equality_improvement': improvement * 0.05,
                'estimated_cost': improvement * 2000
            })
        
        improvements.sort(key=lambda x: x['equality_improvement'], reverse=True)
        
        return {
            'current_gini': current_gini,
            'equality_improvements': improvements,
            'target_gini': max(0, current_gini - sum(imp['equality_improvement'] for imp in improvements))
        }