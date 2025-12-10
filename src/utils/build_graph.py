"""
Build knowledge graph from student interaction data.
"""

import argparse
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data import ASSISTmentsProcessor
from src.graph import KnowledgeGraphBuilder


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Build knowledge graph')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--method', type=str, default=None, 
                        choices=['cooccurrence', 'prerequisite', 'hybrid'],
                        help='Graph construction method (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override method if specified
    if args.method:
        config['graph']['construction_method'] = args.method
    
    method = config['graph']['construction_method']
    print(f"\nBuilding knowledge graph using '{method}' method...")
    
    # Load data
    processor = ASSISTmentsProcessor(
        config['data']['data_path'],
        min_interactions=config['data']['min_interactions']
    )
    
    df = processor.preprocess()
    
    # Build graph
    graph_builder = KnowledgeGraphBuilder(method=method)
    
    threshold = config['graph']['edge_threshold']
    max_neighbors = config['graph']['max_neighbors']
    
    if method == 'prerequisite':
        graph = graph_builder.build_from_prerequisite(
            df,
            threshold=threshold,
            max_neighbors=max_neighbors
        )
    elif method == 'cooccurrence':
        graph = graph_builder.build_from_cooccurrence(
            df,
            threshold=threshold,
            max_neighbors=max_neighbors
        )
    elif method == 'hybrid':
        graph = graph_builder.build_hybrid(
            df,
            threshold=threshold,
            max_neighbors=max_neighbors
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get adjacency matrix
    skill_list = sorted(list(processor.skill_map.keys()))
    adjacency = graph_builder.get_adjacency_matrix(skill_list)
    
    # Save graph
    output_path = config['graph']['graph_path']
    graph_builder.save_graph(output_path)
    
    print(f"\nKnowledge graph saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
