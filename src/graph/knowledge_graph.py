"""Knowledge graph construction utilities."""

import numpy as np
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr


class KnowledgeGraphBuilder:
    """Build knowledge graphs from student interaction data."""
    
    def __init__(self, method: str = 'hybrid'):
        """
        Initialize graph builder.
        
        Args:
            method: Graph construction method ('cooccurrence', 'prerequisite', 'hybrid')
        """
        self.method = method
        self.graph = None
        self.adjacency_matrix = None
        self.skill_nodes = None
    
    def build_from_cooccurrence(
        self,
        df: pd.DataFrame,
        threshold: float = 0.15,
        max_neighbors: int = 15
    ) -> nx.Graph:
        """Build graph from skill co-occurrence in sessions."""
        graph = nx.Graph()
        
        # Get unique skills
        skills = sorted(df['skill_id'].unique())
        graph.add_nodes_from(skills)
        self.skill_nodes = skills
        
        # Compute co-occurrence
        grouped = df.groupby('student_id')
        
        cooccurrence = {}
        for student_id, group in grouped:
            student_skills = group['skill_id'].unique()
            for i, skill1 in enumerate(student_skills):
                for skill2 in student_skills[i+1:]:
                    key = tuple(sorted([skill1, skill2]))
                    cooccurrence[key] = cooccurrence.get(key, 0) + 1
        
        # Add edges above threshold
        max_cooc = max(cooccurrence.values()) if cooccurrence else 1
        for (skill1, skill2), count in cooccurrence.items():
            weight = count / max_cooc
            if weight >= threshold:
                graph.add_edge(skill1, skill2, weight=weight)
        
        # Enforce max neighbors
        for node in graph.nodes():
            neighbors = sorted(
                graph.neighbors(node),
                key=lambda x: graph[node][x].get('weight', 0),
                reverse=True
            )[:max_neighbors]
            
            # Keep only top neighbors
            to_remove = [n for n in graph.neighbors(node) if n not in neighbors]
            for n in to_remove:
                graph.remove_edge(node, n)
        
        self.graph = graph
        self._compute_adjacency_matrix()
        return graph
    
    def build_from_prerequisite(
        self,
        df: pd.DataFrame,
        threshold: float = 0.1,
        max_neighbors: int = 15
    ) -> nx.DiGraph:
        """Build directed graph from prerequisite relationships."""
        graph = nx.DiGraph()
        
        # Get unique skills
        skills = sorted(df['skill_id'].unique())
        graph.add_nodes_from(skills)
        self.skill_nodes = skills
        
        # Compute opportunity counts
        skill_opportunities = df.groupby('skill_id').size()
        
        # For each skill pair, estimate prerequisite relationship
        for skill1 in skills:
            data1 = df[df['skill_id'] == skill1].sort_values('timestamp')
            
            for skill2 in skills:
                if skill1 == skill2:
                    continue
                
                data2 = df[df['skill_id'] == skill2].sort_values('timestamp')
                
                # Simple heuristic: if skill1 appears before skill2 frequently
                first_appear_1 = data1['student_id'].unique()
                first_appear_2 = data2['student_id'].unique()
                
                overlap = len(set(first_appear_1) & set(first_appear_2))
                if overlap > 0:
                    # Students who did skill1 before skill2
                    students_with_order = 0
                    for student_id in set(first_appear_1) & set(first_appear_2):
                        time1 = data1[data1['student_id'] == student_id]['timestamp'].min()
                        time2 = data2[data2['student_id'] == student_id]['timestamp'].min()
                        if time1 < time2:
                            students_with_order += 1
                    
                    if overlap > 0:
                        weight = students_with_order / overlap
                        if weight >= threshold:
                            graph.add_edge(skill1, skill2, weight=weight)
        
        # Enforce max neighbors per node
        for node in list(graph.nodes()):
            out_neighbors = sorted(
                graph.successors(node),
                key=lambda x: graph[node][x].get('weight', 0),
                reverse=True
            )[:max_neighbors]
            
            to_remove = [n for n in list(graph.successors(node)) if n not in out_neighbors]
            for n in to_remove:
                graph.remove_edge(node, n)
        
        self.graph = graph
        self._compute_adjacency_matrix()
        return graph
    
    def build_hybrid(
        self,
        df: pd.DataFrame,
        threshold: float = 0.15,
        max_neighbors: int = 15
    ) -> nx.Graph:
        """Build hybrid graph combining co-occurrence and prerequisite signals."""
        # Start with co-occurrence graph
        graph = self.build_from_cooccurrence(df, threshold, max_neighbors)
        
        # Add prerequisite edges
        pre_graph = self.build_from_prerequisite(df, threshold * 0.7, max_neighbors)
        
        # Merge graphs
        for u, v, data in pre_graph.edges(data=True):
            if graph.has_edge(u, v):
                # Average weights
                old_weight = graph[u][v].get('weight', 0.5)
                new_weight = (old_weight + data.get('weight', 0)) / 2
                graph[u][v]['weight'] = new_weight
            else:
                graph.add_edge(u, v, weight=data.get('weight', 0))
        
        self._compute_adjacency_matrix()
        return graph
    
    def _compute_adjacency_matrix(self):
        """Compute adjacency matrix from graph."""
        if self.graph is None or self.skill_nodes is None:
            return
        
        n = len(self.skill_nodes)
        adj = np.zeros((n, n))
        
        skill_to_idx = {skill: idx for idx, skill in enumerate(self.skill_nodes)}
        
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            i, j = skill_to_idx[u], skill_to_idx[v]
            adj[i][j] = weight
            adj[j][i] = weight  # Symmetric
        
        self.adjacency_matrix = adj
    
    def get_adjacency_matrix(self, skill_list: Optional[list] = None) -> np.ndarray:
        """Get adjacency matrix."""
        if skill_list is not None and self.skill_nodes != skill_list:
            self.skill_nodes = skill_list
            self._compute_adjacency_matrix()
        
        if self.adjacency_matrix is not None:
            return self.adjacency_matrix
        elif self.skill_nodes is not None:
            return np.zeros((len(self.skill_nodes), len(self.skill_nodes)))
        else:
            return np.zeros((0, 0))
    
    def save_graph(self, path: str):
        """Save graph to file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'graph': self.graph,
            'adjacency_matrix': self.adjacency_matrix,
            'skill_nodes': self.skill_nodes,
            'method': self.method
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_graph(path: str):
        """Load graph from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        builder = KnowledgeGraphBuilder(method=data.get('method', 'hybrid'))
        builder.graph = data['graph']
        builder.adjacency_matrix = data['adjacency_matrix']
        builder.skill_nodes = data['skill_nodes']
        
        return builder
