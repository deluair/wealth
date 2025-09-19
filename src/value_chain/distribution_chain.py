"""
Distribution Chain Analysis

Classes for analyzing wealth distribution through supply chains,
including logistics, intermediaries, and distribution networks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

@dataclass
class DistributionNode:
    """Represents a node in the distribution chain"""
    id: str
    name: str
    node_type: str  # 'producer', 'wholesaler', 'retailer', 'consumer', 'logistics'
    location: Tuple[float, float]  # (latitude, longitude)
    capacity: float
    operating_cost: float
    markup_rate: float
    efficiency: float
    inventory_level: float = 0.0
    processing_time: float = 1.0
    quality_retention: float = 1.0

@dataclass
class DistributionLink:
    """Represents a connection between distribution nodes"""
    from_node: str
    to_node: str
    transport_cost: float
    transport_time: float
    capacity: float
    reliability: float
    distance: float
    transport_mode: str  # 'road', 'rail', 'air', 'sea', 'digital'

@dataclass
class Product:
    """Represents a product flowing through the distribution chain"""
    id: str
    name: str
    base_value: float
    perishability: float  # 0-1, higher means more perishable
    weight: float
    volume: float
    quality_sensitivity: float  # 0-1, sensitivity to handling

class DistributionChainAnalyzer:
    """Analyze wealth distribution through supply chains"""
    
    def __init__(self):
        self.network = nx.DiGraph()
        self.nodes = {}
        self.links = {}
        self.products = {}
    
    def build_distribution_network(self, nodes: List[DistributionNode],
                                 links: List[DistributionLink]) -> None:
        """Build the distribution network graph"""
        # Clear existing network
        self.network.clear()
        self.nodes.clear()
        self.links.clear()
        
        # Add nodes
        for node in nodes:
            self.nodes[node.id] = node
            self.network.add_node(
                node.id,
                name=node.name,
                type=node.node_type,
                location=node.location,
                capacity=node.capacity,
                cost=node.operating_cost,
                markup=node.markup_rate,
                efficiency=node.efficiency
            )
        
        # Add links
        for link in links:
            link_id = f"{link.from_node}->{link.to_node}"
            self.links[link_id] = link
            self.network.add_edge(
                link.from_node,
                link.to_node,
                cost=link.transport_cost,
                time=link.transport_time,
                capacity=link.capacity,
                reliability=link.reliability,
                distance=link.distance,
                mode=link.transport_mode
            )
    
    def analyze_distribution_network(self) -> Dict:
        """Comprehensive analysis of the distribution network"""
        analysis = {}
        
        # Network structure analysis
        analysis['network_structure'] = self._analyze_network_structure()
        
        # Flow analysis
        analysis['flow_analysis'] = self._analyze_flows()
        
        # Cost analysis
        analysis['cost_analysis'] = self._analyze_distribution_costs()
        
        # Efficiency analysis
        analysis['efficiency_analysis'] = self._analyze_distribution_efficiency()
        
        # Risk analysis
        analysis['risk_analysis'] = self._analyze_distribution_risks()
        
        # Bottleneck analysis
        analysis['bottleneck_analysis'] = self._identify_bottlenecks()
        
        # Market power analysis
        analysis['market_power'] = self._analyze_market_power()
        
        return analysis
    
    def _analyze_network_structure(self) -> Dict:
        """Analyze the structure of the distribution network"""
        structure = {}
        
        # Basic network metrics
        structure['basic_metrics'] = {
            'total_nodes': self.network.number_of_nodes(),
            'total_edges': self.network.number_of_edges(),
            'density': nx.density(self.network),
            'is_connected': nx.is_weakly_connected(self.network),
            'number_of_components': nx.number_weakly_connected_components(self.network)
        }
        
        # Node type distribution
        node_types = defaultdict(int)
        for node_id in self.network.nodes():
            node_type = self.nodes[node_id].node_type
            node_types[node_type] += 1
        
        structure['node_distribution'] = dict(node_types)
        
        # Centrality measures
        centrality_measures = {}
        
        # Degree centrality
        in_degree_centrality = nx.in_degree_centrality(self.network)
        out_degree_centrality = nx.out_degree_centrality(self.network)
        
        centrality_measures['degree_centrality'] = {
            'in_degree': in_degree_centrality,
            'out_degree': out_degree_centrality
        }
        
        # Betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(self.network)
        centrality_measures['betweenness_centrality'] = betweenness_centrality
        
        # Closeness centrality
        try:
            closeness_centrality = nx.closeness_centrality(self.network)
            centrality_measures['closeness_centrality'] = closeness_centrality
        except:
            centrality_measures['closeness_centrality'] = {}
        
        structure['centrality_measures'] = centrality_measures
        
        # Path analysis
        path_analysis = {}
        
        # Average shortest path length
        if nx.is_weakly_connected(self.network):
            try:
                avg_path_length = nx.average_shortest_path_length(self.network)
                path_analysis['average_path_length'] = avg_path_length
            except:
                path_analysis['average_path_length'] = None
        else:
            path_analysis['average_path_length'] = None
        
        # Diameter
        if nx.is_weakly_connected(self.network):
            try:
                diameter = nx.diameter(self.network)
                path_analysis['diameter'] = diameter
            except:
                path_analysis['diameter'] = None
        else:
            path_analysis['diameter'] = None
        
        structure['path_analysis'] = path_analysis
        
        return structure
    
    def _analyze_flows(self) -> Dict:
        """Analyze flows through the distribution network"""
        flow_analysis = {}
        
        # Capacity analysis
        total_node_capacity = sum(node.capacity for node in self.nodes.values())
        total_link_capacity = sum(link.capacity for link in self.links.values())
        
        flow_analysis['capacity_metrics'] = {
            'total_node_capacity': total_node_capacity,
            'total_link_capacity': total_link_capacity,
            'average_node_capacity': total_node_capacity / len(self.nodes) if self.nodes else 0,
            'average_link_capacity': total_link_capacity / len(self.links) if self.links else 0
        }
        
        # Flow distribution by node type
        capacity_by_type = defaultdict(float)
        for node in self.nodes.values():
            capacity_by_type[node.node_type] += node.capacity
        
        flow_analysis['capacity_by_type'] = dict(capacity_by_type)
        
        # Theoretical maximum flow
        # Find all source nodes (producers) and sink nodes (consumers)
        source_nodes = [node_id for node_id in self.network.nodes() 
                       if self.nodes[node_id].node_type == 'producer']
        sink_nodes = [node_id for node_id in self.network.nodes() 
                     if self.nodes[node_id].node_type == 'consumer']
        
        max_flows = []
        if source_nodes and sink_nodes:
            for source in source_nodes:
                for sink in sink_nodes:
                    try:
                        max_flow = nx.maximum_flow_value(
                            self.network, source, sink, capacity='capacity'
                        )
                        max_flows.append({
                            'source': source,
                            'sink': sink,
                            'max_flow': max_flow
                        })
                    except:
                        continue
        
        flow_analysis['maximum_flows'] = max_flows
        
        # Flow efficiency metrics
        if max_flows:
            total_max_flow = sum(flow['max_flow'] for flow in max_flows)
            avg_max_flow = total_max_flow / len(max_flows)
            
            flow_analysis['flow_efficiency'] = {
                'total_theoretical_flow': total_max_flow,
                'average_flow_per_path': avg_max_flow,
                'flow_utilization': total_max_flow / total_link_capacity if total_link_capacity > 0 else 0
            }
        
        return flow_analysis
    
    def _analyze_distribution_costs(self) -> Dict:
        """Analyze costs in the distribution network"""
        cost_analysis = {}
        
        # Node costs
        total_node_costs = sum(node.operating_cost for node in self.nodes.values())
        node_costs_by_type = defaultdict(float)
        
        for node in self.nodes.values():
            node_costs_by_type[node.node_type] += node.operating_cost
        
        cost_analysis['node_costs'] = {
            'total_operating_costs': total_node_costs,
            'average_node_cost': total_node_costs / len(self.nodes) if self.nodes else 0,
            'costs_by_type': dict(node_costs_by_type)
        }
        
        # Link costs
        total_transport_costs = sum(link.transport_cost for link in self.links.values())
        transport_costs_by_mode = defaultdict(float)
        
        for link in self.links.values():
            transport_costs_by_mode[link.transport_mode] += link.transport_cost
        
        cost_analysis['transport_costs'] = {
            'total_transport_costs': total_transport_costs,
            'average_link_cost': total_transport_costs / len(self.links) if self.links else 0,
            'costs_by_mode': dict(transport_costs_by_mode)
        }
        
        # Total system costs
        total_system_cost = total_node_costs + total_transport_costs
        
        cost_analysis['system_costs'] = {
            'total_system_cost': total_system_cost,
            'node_cost_share': total_node_costs / total_system_cost if total_system_cost > 0 else 0,
            'transport_cost_share': total_transport_costs / total_system_cost if total_system_cost > 0 else 0
        }
        
        # Cost per unit capacity
        total_capacity = sum(node.capacity for node in self.nodes.values())
        cost_analysis['cost_efficiency'] = {
            'cost_per_unit_capacity': total_system_cost / total_capacity if total_capacity > 0 else 0,
            'node_cost_per_capacity': total_node_costs / total_capacity if total_capacity > 0 else 0,
            'transport_cost_per_capacity': total_transport_costs / total_capacity if total_capacity > 0 else 0
        }
        
        return cost_analysis
    
    def _analyze_distribution_efficiency(self) -> Dict:
        """Analyze efficiency of the distribution network"""
        efficiency_analysis = {}
        
        # Node efficiency analysis
        node_efficiencies = [node.efficiency for node in self.nodes.values()]
        
        efficiency_analysis['node_efficiency'] = {
            'average_efficiency': np.mean(node_efficiencies),
            'median_efficiency': np.median(node_efficiencies),
            'min_efficiency': np.min(node_efficiencies),
            'max_efficiency': np.max(node_efficiencies),
            'efficiency_std': np.std(node_efficiencies)
        }
        
        # Efficiency by node type
        efficiency_by_type = defaultdict(list)
        for node in self.nodes.values():
            efficiency_by_type[node.node_type].append(node.efficiency)
        
        type_efficiency_stats = {}
        for node_type, efficiencies in efficiency_by_type.items():
            type_efficiency_stats[node_type] = {
                'average': np.mean(efficiencies),
                'count': len(efficiencies),
                'std': np.std(efficiencies)
            }
        
        efficiency_analysis['efficiency_by_type'] = type_efficiency_stats
        
        # Link efficiency (based on reliability and cost)
        link_efficiencies = []
        for link in self.links.values():
            # Efficiency as reliability per unit cost
            link_efficiency = link.reliability / link.transport_cost if link.transport_cost > 0 else 0
            link_efficiencies.append(link_efficiency)
        
        if link_efficiencies:
            efficiency_analysis['link_efficiency'] = {
                'average_efficiency': np.mean(link_efficiencies),
                'median_efficiency': np.median(link_efficiencies),
                'min_efficiency': np.min(link_efficiencies),
                'max_efficiency': np.max(link_efficiencies)
            }
        
        # Overall system efficiency
        # Weighted average of node efficiencies by capacity
        total_capacity = sum(node.capacity for node in self.nodes.values())
        weighted_efficiency = sum(
            node.efficiency * node.capacity for node in self.nodes.values()
        ) / total_capacity if total_capacity > 0 else 0
        
        efficiency_analysis['system_efficiency'] = {
            'weighted_average_efficiency': weighted_efficiency,
            'capacity_weighted': True
        }
        
        return efficiency_analysis
    
    def _analyze_distribution_risks(self) -> Dict:
        """Analyze risks in the distribution network"""
        risk_analysis = {}
        
        # Link reliability analysis
        link_reliabilities = [link.reliability for link in self.links.values()]
        
        risk_analysis['reliability_metrics'] = {
            'average_link_reliability': np.mean(link_reliabilities),
            'min_link_reliability': np.min(link_reliabilities),
            'reliability_std': np.std(link_reliabilities),
            'unreliable_links_count': sum(1 for r in link_reliabilities if r < 0.8)
        }
        
        # Single point of failure analysis
        critical_nodes = []
        for node_id in self.network.nodes():
            # Check if removing this node would disconnect the network
            temp_network = self.network.copy()
            temp_network.remove_node(node_id)
            
            if not nx.is_weakly_connected(temp_network) and nx.is_weakly_connected(self.network):
                critical_nodes.append(node_id)
        
        risk_analysis['critical_nodes'] = {
            'count': len(critical_nodes),
            'nodes': critical_nodes,
            'risk_level': 'High' if len(critical_nodes) > 0 else 'Low'
        }
        
        # Capacity risk analysis
        node_capacities = [node.capacity for node in self.nodes.values()]
        capacity_concentration = self._calculate_concentration_ratio(node_capacities, top_n=3)
        
        risk_analysis['capacity_concentration'] = {
            'top_3_concentration': capacity_concentration,
            'risk_level': 'High' if capacity_concentration > 0.7 else 'Medium' if capacity_concentration > 0.5 else 'Low'
        }
        
        # Geographic risk analysis
        locations = [node.location for node in self.nodes.values()]
        if locations:
            # Calculate geographic dispersion
            lats, lons = zip(*locations)
            lat_range = max(lats) - min(lats)
            lon_range = max(lons) - min(lons)
            
            risk_analysis['geographic_risk'] = {
                'latitude_range': lat_range,
                'longitude_range': lon_range,
                'geographic_dispersion': lat_range * lon_range,
                'risk_level': 'Low' if lat_range * lon_range > 100 else 'Medium' if lat_range * lon_range > 10 else 'High'
            }
        
        return risk_analysis
    
    def _identify_bottlenecks(self) -> Dict:
        """Identify bottlenecks in the distribution network"""
        bottleneck_analysis = {}
        
        # Capacity bottlenecks
        node_capacities = [(node.id, node.capacity) for node in self.nodes.values()]
        node_capacities.sort(key=lambda x: x[1])
        
        # Bottom 20% of nodes by capacity
        bottleneck_count = max(1, len(node_capacities) // 5)
        capacity_bottlenecks = node_capacities[:bottleneck_count]
        
        bottleneck_analysis['capacity_bottlenecks'] = {
            'nodes': capacity_bottlenecks,
            'count': len(capacity_bottlenecks),
            'total_bottleneck_capacity': sum(cap for _, cap in capacity_bottlenecks)
        }
        
        # Flow bottlenecks (nodes with high betweenness centrality but low capacity)
        try:
            betweenness = nx.betweenness_centrality(self.network)
            flow_bottlenecks = []
            
            for node_id, centrality in betweenness.items():
                node_capacity = self.nodes[node_id].capacity
                # High centrality but low capacity indicates potential bottleneck
                if centrality > 0.1 and node_capacity < np.median([n.capacity for n in self.nodes.values()]):
                    flow_bottlenecks.append({
                        'node_id': node_id,
                        'centrality': centrality,
                        'capacity': node_capacity,
                        'bottleneck_score': centrality / node_capacity if node_capacity > 0 else float('inf')
                    })
            
            # Sort by bottleneck score
            flow_bottlenecks.sort(key=lambda x: x['bottleneck_score'], reverse=True)
            
            bottleneck_analysis['flow_bottlenecks'] = {
                'nodes': flow_bottlenecks[:5],  # Top 5 flow bottlenecks
                'count': len(flow_bottlenecks)
            }
        except:
            bottleneck_analysis['flow_bottlenecks'] = {'nodes': [], 'count': 0}
        
        # Link bottlenecks (low capacity, high importance)
        link_bottlenecks = []
        for link_id, link in self.links.items():
            # Simple bottleneck score: inverse of capacity
            bottleneck_score = 1 / link.capacity if link.capacity > 0 else float('inf')
            link_bottlenecks.append({
                'link_id': link_id,
                'capacity': link.capacity,
                'bottleneck_score': bottleneck_score,
                'from_node': link.from_node,
                'to_node': link.to_node
            })
        
        link_bottlenecks.sort(key=lambda x: x['bottleneck_score'], reverse=True)
        
        bottleneck_analysis['link_bottlenecks'] = {
            'links': link_bottlenecks[:5],  # Top 5 link bottlenecks
            'count': len(link_bottlenecks)
        }
        
        return bottleneck_analysis
    
    def _analyze_market_power(self) -> Dict:
        """Analyze market power distribution in the network"""
        market_power = {}
        
        # Market concentration by node type
        capacity_by_type = defaultdict(float)
        for node in self.nodes.values():
            capacity_by_type[node.node_type] += node.capacity
        
        total_capacity = sum(capacity_by_type.values())
        
        market_shares = {}
        for node_type, capacity in capacity_by_type.items():
            market_shares[node_type] = capacity / total_capacity if total_capacity > 0 else 0
        
        market_power['market_shares_by_type'] = market_shares
        
        # Herfindahl-Hirschman Index (HHI) for each node type
        hhi_by_type = {}
        for node_type in capacity_by_type.keys():
            type_nodes = [node for node in self.nodes.values() if node.node_type == node_type]
            type_capacities = [node.capacity for node in type_nodes]
            type_total = sum(type_capacities)
            
            if type_total > 0:
                market_shares_squared = [(cap / type_total) ** 2 for cap in type_capacities]
                hhi = sum(market_shares_squared)
                hhi_by_type[node_type] = hhi
        
        market_power['hhi_by_type'] = hhi_by_type
        
        # Overall market concentration
        all_capacities = [node.capacity for node in self.nodes.values()]
        overall_hhi = self._calculate_hhi(all_capacities)
        
        market_power['overall_concentration'] = {
            'hhi': overall_hhi,
            'concentration_level': self._classify_concentration(overall_hhi)
        }
        
        # Top player analysis
        sorted_nodes = sorted(self.nodes.values(), key=lambda x: x.capacity, reverse=True)
        top_players = sorted_nodes[:5]
        
        top_player_analysis = []
        for i, node in enumerate(top_players):
            market_share = node.capacity / total_capacity if total_capacity > 0 else 0
            top_player_analysis.append({
                'rank': i + 1,
                'node_id': node.id,
                'node_type': node.node_type,
                'capacity': node.capacity,
                'market_share': market_share
            })
        
        market_power['top_players'] = top_player_analysis
        
        return market_power
    
    def simulate_product_flow(self, product: Product, source_node: str,
                            target_nodes: List[str], quantity: float) -> Dict:
        """
        Simulate product flow through the distribution network
        
        Args:
            product: Product to simulate
            source_node: Starting node
            target_nodes: Destination nodes
            quantity: Quantity to distribute
            
        Returns:
            Dictionary with flow simulation results
        """
        simulation_results = {}
        
        # Find optimal paths to each target
        path_results = []
        
        for target in target_nodes:
            try:
                # Find shortest path by cost
                path = nx.shortest_path(
                    self.network, source_node, target, weight='cost'
                )
                
                # Calculate path metrics
                path_cost = nx.shortest_path_length(
                    self.network, source_node, target, weight='cost'
                )
                
                path_time = sum(
                    self.network[path[i]][path[i+1]]['time']
                    for i in range(len(path)-1)
                )
                
                # Calculate value degradation
                value_degradation = self._calculate_value_degradation(
                    product, path, path_time
                )
                
                # Calculate final value
                final_value = product.base_value * (1 - value_degradation)
                
                path_results.append({
                    'target': target,
                    'path': path,
                    'path_length': len(path) - 1,
                    'total_cost': path_cost,
                    'total_time': path_time,
                    'value_degradation': value_degradation,
                    'final_value': final_value,
                    'profit_margin': (final_value - path_cost) / final_value if final_value > 0 else 0
                })
                
            except nx.NetworkXNoPath:
                path_results.append({
                    'target': target,
                    'path': None,
                    'error': 'No path available'
                })
        
        simulation_results['path_analysis'] = path_results
        
        # Aggregate results
        successful_paths = [p for p in path_results if 'error' not in p]
        
        if successful_paths:
            simulation_results['summary'] = {
                'successful_deliveries': len(successful_paths),
                'total_targets': len(target_nodes),
                'success_rate': len(successful_paths) / len(target_nodes),
                'average_cost': np.mean([p['total_cost'] for p in successful_paths]),
                'average_time': np.mean([p['total_time'] for p in successful_paths]),
                'average_value_degradation': np.mean([p['value_degradation'] for p in successful_paths]),
                'total_profit': sum((p['final_value'] - p['total_cost']) * quantity for p in successful_paths),
                'average_profit_margin': np.mean([p['profit_margin'] for p in successful_paths])
            }
        else:
            simulation_results['summary'] = {
                'successful_deliveries': 0,
                'total_targets': len(target_nodes),
                'success_rate': 0
            }
        
        return simulation_results
    
    def _calculate_value_degradation(self, product: Product, path: List[str],
                                   total_time: float) -> float:
        """Calculate value degradation along a path"""
        degradation = 0.0
        
        # Time-based degradation (perishability)
        time_degradation = product.perishability * (total_time / 24.0)  # Assuming time in hours
        degradation += time_degradation
        
        # Handling degradation (quality sensitivity)
        handling_steps = len(path) - 1
        handling_degradation = product.quality_sensitivity * handling_steps * 0.01
        degradation += handling_degradation
        
        # Node-specific degradation
        for node_id in path:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Quality retention affects degradation
                node_degradation = (1 - node.quality_retention) * 0.05
                degradation += node_degradation
        
        return min(1.0, degradation)  # Cap at 100% degradation
    
    def optimize_distribution_network(self, objective: str = 'cost') -> Dict:
        """
        Optimize the distribution network for different objectives
        
        Args:
            objective: 'cost', 'time', 'reliability', or 'efficiency'
            
        Returns:
            Dictionary with optimization results
        """
        optimization_results = {}
        
        if objective == 'cost':
            optimization_results = self._optimize_for_cost()
        elif objective == 'time':
            optimization_results = self._optimize_for_time()
        elif objective == 'reliability':
            optimization_results = self._optimize_for_reliability()
        elif objective == 'efficiency':
            optimization_results = self._optimize_for_efficiency()
        else:
            optimization_results = {'error': f'Unknown objective: {objective}'}
        
        return optimization_results
    
    def _optimize_for_cost(self) -> Dict:
        """Optimize network for minimum cost"""
        # This is a simplified optimization
        # In practice, this would involve complex optimization algorithms
        
        current_total_cost = sum(node.operating_cost for node in self.nodes.values())
        current_total_cost += sum(link.transport_cost for link in self.links.values())
        
        # Identify high-cost nodes and links
        high_cost_nodes = [
            node for node in self.nodes.values()
            if node.operating_cost > np.mean([n.operating_cost for n in self.nodes.values()])
        ]
        
        high_cost_links = [
            link for link in self.links.values()
            if link.transport_cost > np.mean([l.transport_cost for l in self.links.values()])
        ]
        
        # Suggest optimizations
        suggestions = []
        
        for node in high_cost_nodes:
            potential_savings = node.operating_cost * 0.1  # Assume 10% reduction possible
            suggestions.append({
                'type': 'node_optimization',
                'node_id': node.id,
                'current_cost': node.operating_cost,
                'potential_savings': potential_savings,
                'suggestion': 'Optimize operations or consider alternative location'
            })
        
        for link in high_cost_links:
            potential_savings = link.transport_cost * 0.15  # Assume 15% reduction possible
            suggestions.append({
                'type': 'link_optimization',
                'link_id': f"{link.from_node}->{link.to_node}",
                'current_cost': link.transport_cost,
                'potential_savings': potential_savings,
                'suggestion': 'Consider alternative transport mode or route'
            })
        
        total_potential_savings = sum(s['potential_savings'] for s in suggestions)
        
        return {
            'objective': 'cost_minimization',
            'current_total_cost': current_total_cost,
            'potential_savings': total_potential_savings,
            'savings_percentage': total_potential_savings / current_total_cost if current_total_cost > 0 else 0,
            'optimization_suggestions': suggestions
        }
    
    def _optimize_for_time(self) -> Dict:
        """Optimize network for minimum time"""
        # Calculate current average delivery time
        all_pairs_times = []
        
        for source in self.network.nodes():
            for target in self.network.nodes():
                if source != target:
                    try:
                        path_time = nx.shortest_path_length(
                            self.network, source, target, weight='time'
                        )
                        all_pairs_times.append(path_time)
                    except nx.NetworkXNoPath:
                        continue
        
        current_avg_time = np.mean(all_pairs_times) if all_pairs_times else 0
        
        # Identify slow links
        slow_links = [
            link for link in self.links.values()
            if link.transport_time > np.mean([l.transport_time for l in self.links.values()])
        ]
        
        suggestions = []
        for link in slow_links:
            time_reduction = link.transport_time * 0.2  # Assume 20% reduction possible
            suggestions.append({
                'type': 'speed_improvement',
                'link_id': f"{link.from_node}->{link.to_node}",
                'current_time': link.transport_time,
                'potential_reduction': time_reduction,
                'suggestion': 'Upgrade transport mode or improve logistics'
            })
        
        return {
            'objective': 'time_minimization',
            'current_average_time': current_avg_time,
            'optimization_suggestions': suggestions
        }
    
    def _optimize_for_reliability(self) -> Dict:
        """Optimize network for maximum reliability"""
        # Calculate current system reliability
        link_reliabilities = [link.reliability for link in self.links.values()]
        current_avg_reliability = np.mean(link_reliabilities)
        
        # Identify unreliable links
        unreliable_links = [
            link for link in self.links.values()
            if link.reliability < 0.8
        ]
        
        suggestions = []
        for link in unreliable_links:
            reliability_improvement = min(0.95 - link.reliability, 0.2)
            suggestions.append({
                'type': 'reliability_improvement',
                'link_id': f"{link.from_node}->{link.to_node}",
                'current_reliability': link.reliability,
                'potential_improvement': reliability_improvement,
                'suggestion': 'Improve infrastructure or add redundancy'
            })
        
        return {
            'objective': 'reliability_maximization',
            'current_average_reliability': current_avg_reliability,
            'optimization_suggestions': suggestions
        }
    
    def _optimize_for_efficiency(self) -> Dict:
        """Optimize network for maximum efficiency"""
        # Calculate current efficiency metrics
        node_efficiencies = [node.efficiency for node in self.nodes.values()]
        current_avg_efficiency = np.mean(node_efficiencies)
        
        # Identify inefficient nodes
        inefficient_nodes = [
            node for node in self.nodes.values()
            if node.efficiency < 0.7
        ]
        
        suggestions = []
        for node in inefficient_nodes:
            efficiency_improvement = min(0.9 - node.efficiency, 0.3)
            suggestions.append({
                'type': 'efficiency_improvement',
                'node_id': node.id,
                'current_efficiency': node.efficiency,
                'potential_improvement': efficiency_improvement,
                'suggestion': 'Optimize processes or upgrade technology'
            })
        
        return {
            'objective': 'efficiency_maximization',
            'current_average_efficiency': current_avg_efficiency,
            'optimization_suggestions': suggestions
        }
    
    def _calculate_concentration_ratio(self, values: List[float], top_n: int = 4) -> float:
        """Calculate concentration ratio for top N entities"""
        if not values:
            return 0
        
        sorted_values = sorted(values, reverse=True)
        top_values = sorted_values[:min(top_n, len(sorted_values))]
        total_value = sum(values)
        
        return sum(top_values) / total_value if total_value > 0 else 0
    
    def _calculate_hhi(self, values: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index"""
        if not values:
            return 0
        
        total = sum(values)
        if total == 0:
            return 0
        
        market_shares = [value / total for value in values]
        hhi = sum(share ** 2 for share in market_shares)
        
        return hhi
    
    def _classify_concentration(self, hhi: float) -> str:
        """Classify market concentration based on HHI"""
        if hhi < 0.15:
            return "Low concentration"
        elif hhi < 0.25:
            return "Moderate concentration"
        else:
            return "High concentration"