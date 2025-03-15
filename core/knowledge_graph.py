#<!-- filepath: /root/IRS/core/knowledge_graph.py -->
#!/usr/bin/env python3
# Knowledge Graph integration for IRS Tax Analysis System

import os
import re
import logging
import json
from typing import Dict, List, Set, Optional, Tuple, Union, Any
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import unittest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("knowledge_graph")

class TaxEntity:
    """Class representing a tax entity in the knowledge graph."""
    
    def __init__(self, name: str, entity_type: str, attributes: Dict[str, Any] = None):
        """Initialize a tax entity.
        
        Args:
            name: The entity name or identifier
            entity_type: Type of entity (e.g., 'deduction', 'form', 'taxpayer')
            attributes: Dictionary of entity attributes
        """
        self.name = name
        self.entity_type = entity_type
        self.attributes = attributes or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "name": self.name,
            "type": self.entity_type,
            "attributes": self.attributes
        }
    
    def __eq__(self, other):
        if not isinstance(other, TaxEntity):
            return False
        return self.name == other.name and self.entity_type == other.entity_type
    
    def __hash__(self):
        return hash((self.name, self.entity_type))
    
    def __str__(self):
        return f"{self.name} ({self.entity_type})"

class TaxKnowledgeGraph:
    """Knowledge Graph for tax domain knowledge."""
    
    def __init__(self, save_path: str = "./data/knowledge_graph.json"):
        """Initialize the knowledge graph.
        
        Args:
            save_path: Path to save/load the knowledge graph
        """
        self.graph = nx.DiGraph()
        self.save_path = save_path
        self.entity_types = set()
        self.relation_types = set()
        
        # Load existing graph if available
        self.load()
    
    def add_entity(self, entity: TaxEntity) -> bool:
        """Add an entity to the graph.
        
        Args:
            entity: The entity to add
            
        Returns:
            True if entity was added, False if it already existed
        """
        if entity.name not in [n for n in self.graph.nodes if self.graph.nodes[n].get("type") == entity.entity_type]:
            self.graph.add_node(entity.name, type=entity.entity_type, attributes=entity.attributes)
            self.entity_types.add(entity.entity_type)
            logger.debug(f"Added entity: {entity}")
            return True
        return False
    
    def add_relation(self, source: Union[str, TaxEntity], relation: str, target: Union[str, TaxEntity], 
                    attributes: Dict[str, Any] = None) -> bool:
        """Add a relation between two entities.
        
        Args:
            source: Source entity or name
            relation: Type of relation
            target: Target entity or name
            attributes: Additional attributes for the relation
            
        Returns:
            True if relation was added, False otherwise
        """
        # Get entity names
        source_name = source.name if isinstance(source, TaxEntity) else source
        target_name = target.name if isinstance(target, TaxEntity) else target
        
        # Add relation
        if not self.graph.has_edge(source_name, target_name) or self.graph[source_name][target_name].get("relation") != relation:
            self.graph.add_edge(source_name, target_name, relation=relation, attributes=(attributes or {}))
            self.relation_types.add(relation)
            logger.debug(f"Added relation: {source_name} --[{relation}]--> {target_name}")
            return True
        return False
    
    def get_entity(self, name: str, entity_type: Optional[str] = None) -> Optional[TaxEntity]:
        """Get an entity by name and optionally type.
        
        Args:
            name: Entity name
            entity_type: Optional entity type for disambiguation
            
        Returns:
            TaxEntity if found, None otherwise
        """
        if name not in self.graph.nodes:
            return None
        
        node_attrs = self.graph.nodes[name]
        
        if entity_type and node_attrs.get("type") != entity_type:
            return None
        
        return TaxEntity(
            name=name,
            entity_type=node_attrs.get("type", "unknown"),
            attributes=node_attrs.get("attributes", {})
        )
    
    def get_relations(self, entity: Union[str, TaxEntity], relation_type: Optional[str] = None, 
                     outgoing: bool = True) -> List[Tuple[str, str, str]]:
        """Get relations for an entity.
        
        Args:
            entity: Entity or entity name
            relation_type: Optional filter by relation type
            outgoing: If True, get outgoing relations; if False, get incoming relations
            
        Returns:
            List of (source, relation, target) tuples
        """
        entity_name = entity.name if isinstance(entity, TaxEntity) else entity
        
        if entity_name not in self.graph.nodes:
            return []
        
        results = []
        
        if outgoing:
            for _, target, data in self.graph.out_edges(entity_name, data=True):
                rel = data.get("relation")
                if relation_type is None or rel == relation_type:
                    results.append((entity_name, rel, target))
        else:
            for source, _, data in self.graph.in_edges(entity_name, data=True):
                rel = data.get("relation")
                if relation_type is None or rel == relation_type:
                    results.append((source, rel, entity_name))
        
        return results
    
    def query(self, query_entity: Union[str, TaxEntity], query_relation: str, 
             max_depth: int = 2) -> List[Dict[str, Any]]:
        """Query the knowledge graph.
        
        Args:
            query_entity: Starting entity or name for query
            query_relation: Relation type to follow
            max_depth: Maximum path depth to traverse
            
        Returns:
            List of result paths with entities and relations
        """
        entity_name = query_entity.name if isinstance(query_entity, TaxEntity) else query_entity
        
        if entity_name not in self.graph.nodes:
            return []
        
        # Find all paths from entity following specified relation type up to max_depth
        results = []
        visited = set([entity_name])
        self._dfs_query(entity_name, query_relation, [], results, visited, 0, max_depth)
        
        return results
    
    def _dfs_query(self, current: str, target_relation: str, current_path: List[Dict[str, Any]], 
                  results: List[Dict[str, Any]], visited: Set[str], depth: int, max_depth: int) -> None:
        """Recursive depth-first search for query.
        
        Args:
            current: Current entity name
            target_relation: Relation type to follow
            current_path: Current path being explored
            results: List to collect result paths
            visited: Set of visited entities
            depth: Current depth
            max_depth: Maximum depth to explore
        """
        if depth > max_depth:
            return
        
        # Check outgoing edges
        for _, target, data in self.graph.out_edges(current, data=True):
            relation = data.get("relation")
            
            # If this is the relation we're looking for
            if relation == target_relation:
                # Create path entry
                path_entry = {
                    "source": current,
                    "relation": relation,
                    "target": target,
                    "source_type": self.graph.nodes[current].get("type"),
                    "target_type": self.graph.nodes[target].get("type"),
                    "attributes": data.get("attributes", {})
                }
                
                # Add to results
                new_path = current_path + [path_entry]
                results.append(new_path)
                
                # Continue searching if not visited and under max depth
                if target not in visited and depth + 1 < max_depth:
                    visited_new = visited.copy()
                    visited_new.add(target)
                    self._dfs_query(target, target_relation, new_path, results, visited_new, depth + 1, max_depth)
    
    def visualize(self, output_file: Optional[str] = None, 
                 highlight_entities: List[str] = None,
                 highlight_relations: List[Tuple[str, str]] = None) -> None:
        """Visualize the knowledge graph.
        
        Args:
            output_file: Optional file to save the visualization
            highlight_entities: List of entity names to highlight
            highlight_relations: List of (source, target) pairs to highlight
        """
        plt.figure(figsize=(12, 10))
        
        # Create position layout
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        
        # Prepare node colors based on entity type
        entity_types = list(self.entity_types)
        color_map = plt.cm.tab20(range(len(entity_types)))
        type_to_color = {t: color_map[i % len(color_map)] for i, t in enumerate(entity_types)}
        
        # Get node colors
        node_colors = [type_to_color.get(self.graph.nodes[n].get("type", "unknown"), (0.7, 0.7, 0.7, 1.0)) for n in self.graph.nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=800,
            alpha=0.9
        )
        
        # Highlight specific entities if provided
        if highlight_entities:
            highlight_nodes = [n for n in self.graph.nodes if n in highlight_entities]
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=highlight_nodes,
                node_color='red',
                node_size=1000,
                alpha=0.8
            )
        
        # Draw edges
        edge_colors = []
        for u, v, data in self.graph.edges(data=True):
            if highlight_relations and (u, v) in highlight_relations:
                edge_colors.append('red')
            else:
                edge_colors.append('black')
        
        nx.draw_networkx_edges(
            self.graph, pos,
            width=1.0,
            alpha=0.7,
            edge_color=edge_colors,
            arrowsize=15
        )
        
        # Add edge labels
        edge_labels = {(u, v): data.get('relation', '') for u, v, data in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        # Add node labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=10,
            font_weight='bold'
        )
        
        # Add legend for entity types
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     label=t,
                                     markerfacecolor=type_to_color[t], 
                                     markersize=10) for t in entity_types]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title('Tax Knowledge Graph')
        plt.axis('off')
        
        # Save or display
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {output_file}")
        else:
            plt.show()
    
    def save(self) -> bool:
        """Save the knowledge graph to file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # Convert graph to JSON-serializable format
            data = {
                "nodes": [],
                "edges": [],
                "entity_types": list(self.entity_types),
                "relation_types": list(self.relation_types)
            }
            
            # Add nodes
            for node, attrs in self.graph.nodes(data=True):
                node_data = {
                    "name": node,
                    "type": attrs.get("type", "unknown"),
                    "attributes": attrs.get("attributes", {})
                }
                data["nodes"].append(node_data)
            
            # Add edges
            for u, v, attrs in self.graph.edges(data=True):
                edge_data = {
                    "source": u,
                    "target": v,
                    "relation": attrs.get("relation", "unknown"),
                    "attributes": attrs.get("attributes", {})
                }
                data["edges"].append(edge_data)
            
            # Save to file
            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Knowledge graph saved to {self.save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
            return False
    
    def load(self) -> bool:
        """Load the knowledge graph from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.save_path):
            logger.info(f"Knowledge graph file {self.save_path} not found, starting with empty graph")
            return False
        
        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)
            
            # Clear existing graph
            self.graph.clear()
            self.entity_types.clear()
            self.relation_types.clear()
            
            # Load entity types and relation types
            self.entity_types.update(data.get("entity_types", []))
            self.relation_types.update(data.get("relation_types", []))
            
            # Add nodes
            for node_data in data.get("nodes", []):
                self.graph.add_node(
                    node_data["name"],
                    type=node_data.get("type", "unknown"),
                    attributes=node_data.get("attributes", {})
                )
            
            # Add edges
            for edge_data in data.get("edges", []):
                self.graph.add_edge(
                    edge_data["source"],
                    edge_data["target"],
                    relation=edge_data.get("relation", "unknown"),
                    attributes=edge_data.get("attributes", {})
                )
            
            logger.info(f"Knowledge graph loaded from {self.save_path} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            return False
    
    def extract_from_text(self, text: str, entity_patterns: Dict[str, List[str]] = None) -> int:
        """Extract entities and relations from text.
        
        Args:
            text: Text to extract from
            entity_patterns: Dictionary mapping entity types to regex patterns
            
        Returns:
            Number of entities and relations extracted
        """
        if entity_patterns is None:
            # Default patterns for tax entities
            entity_patterns = {
                "form": [r"Form\s+([0-9A-Z\-]+)", r"([0-9]{3,4}[A-Z]?)(?:\s+form)"],
                "deduction": [r"([A-Za-z\s]+)\s+deduction"],
                "credit": [r"([A-Za-z\s]+)\s+credit"],
                "taxpayer": [r"taxpayer\s+([A-Za-z\s]+)", r"([A-Za-z\s]+)(?:'s tax)"]
            }
        
        count = 0
        
        # Extract entities
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group(1).strip()
                    entity = TaxEntity(entity_name, entity_type)
                    if self.add_entity(entity):
                        count += 1
        
        # TODO: Implement relation extraction (would require NLP parsing)
        
        return count

# Unit tests
class TestTaxKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.graph = TaxKnowledgeGraph(save_path="./test_knowledge_graph.json")
        
        # Add some test entities and relations
        self.entity1 = TaxEntity("1040", "form", {"description": "Individual Income Tax Return"})
        self.entity2 = TaxEntity("Standard Deduction", "deduction", {"description": "Fixed amount deduction"})
        self.entity3 = TaxEntity("John Doe", "taxpayer", {"income": 75000})
        
        self.graph.add_entity(self.entity1)
        self.graph.add_entity(self.entity2)
        self.graph.add_entity(self.entity3)
        
        self.graph.add_relation(self.entity3, "files", self.entity1)
        self.graph.add_relation(self.entity3, "claims", self.entity2)
    
    def tearDown(self):
        # Clean up test file
        if os.path.exists("./test_knowledge_graph.json"):
            os.remove("./test_knowledge_graph.json")
    
    def test_add_entity(self):
        entity = TaxEntity("1099", "form", {"description": "Miscellaneous Income"})
        result = self.graph.add_entity(entity)
        self.assertTrue(result)
        self.assertIn("1099", self.graph.graph.nodes)
    
    def test_add_relation(self):
        entity4 = TaxEntity("Child Tax Credit", "credit")
        self.graph.add_entity(entity4)
        
        result = self.graph.add_relation(self.entity3, "claims", entity4)
        self.assertTrue(result)
        self.assertTrue(self.graph.graph.has_edge("John Doe", "Child Tax Credit"))
    
    def test_get_entity(self):
        entity = self.graph.get_entity("1040", "form")
        self.assertIsNotNone(entity)
        self.assertEqual(entity.name, "1040")
        self.assertEqual(entity.entity_type, "form")
    
    def test_get_relations(self):
        relations = self.graph.get_relations("John Doe")
        self.assertEqual(len(relations), 2)
        
        # Filter by relation type
        relations = self.graph.get_relations("John Doe", "claims")
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0], ("John Doe", "claims", "Standard Deduction"))
    
    def test_save_and_load(self):
        # Save the graph
        self.graph.save()
        
        # Create a new graph and load
        new_graph = TaxKnowledgeGraph(save_path="./test_knowledge_graph.json")
        new_graph.load()
        
        # Check that entities and relations were preserved
        self.assertEqual(new_graph.graph.number_of_nodes(), 3)
        self.assertEqual(new_graph.graph.number_of_edges(), 2)
        self.assertIn("1040", new_graph.graph.nodes)
        self.assertTrue(new_graph.graph.has_edge("John Doe", "Standard Deduction"))
    
    def test_extract_from_text(self):
        text = """
        The taxpayer John Smith filed his Form 1040 last year.
        He claimed the Standard Deduction and Child Tax Credit.
        """
        
        count = self.graph.extract_from_text(text)
        self.assertGreater(count, 0)
        self.assertIn("John Smith", self.graph.graph.nodes)
        self.assertIn("Child Tax Credit", self.graph.graph.nodes)

if __name__ == "__main__":
    # Simple demonstration
    kg = TaxKnowledgeGraph()
    
    # Add entities
    form_1040 = TaxEntity("1040", "form", {"description": "Individual Income Tax Return"})
    form_w2 = TaxEntity("W-2", "form", {"description": "Wage and Tax Statement"})
    std_deduction = TaxEntity("Standard Deduction", "deduction", {"amount": 12950})
    taxpayer = TaxEntity("Jane Smith", "taxpayer", {"income": 85000})
    
    kg.add_entity(form_1040)
    kg.add_entity(form_w2)
    kg.add_entity(std_deduction)
    kg.add_entity(taxpayer)
    
    # Add relations
    kg.add_relation(taxpayer, "files", form_1040)
    kg.add_relation(taxpayer, "receives", form_w2)
    kg.add_relation(taxpayer, "claims", std_deduction)
    
    # Visualize
    kg.visualize()
    
    # Save graph
    kg.save()