"""
Git-backed Cortex Module
Implements Cortex operations using Git as the underlying storage system.
"""

import os
import git
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional
import hashlib
import json

# Assume HLLSet is available in the same folder
from HLLSets import HLLSet

@dataclass
class Context:
    """Represents a context in Cortex"""
    name: str
    basis_elements: List[str]  # List of HLLSet IDs
    
    def to_dict(self):
        return {
            'name': self.name,
            'basis_elements': self.basis_elements
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data['name'],
            basis_elements=data['basis_elements']
        )

@dataclass 
class Edge:
    """Represents an edge in Cortex"""
    edge_id: str
    tau: float
    rho: float
    phi: Optional[float] = None
    parent_edges: List[str] = None
    
    def __post_init__(self):
        if self.parent_edges is None:
            self.parent_edges = []
    
    def to_dict(self):
        return {
            'edge_id': self.edge_id,
            'tau': self.tau,
            'rho': self.rho,
            'phi': self.phi,
            'parent_edges': self.parent_edges
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            edge_id=data['edge_id'],
            tau=data['tau'],
            rho=data['rho'],
            phi=data.get('phi'),
            parent_edges=data.get('parent_edges', [])
        )

class GitBackedCortex:
    """
    Implements Cortex operations using Git as storage backend
    """
    
    def __init__(self, repo_path: str, init: bool = False):
        self.repo_path = repo_path
        
        if init:
            if not os.path.exists(repo_path):
                os.makedirs(repo_path)
            self.repo = git.Repo.init(repo_path)
            self._init_repo_structure()
        else:
            self.repo = git.Repo(repo_path)
        
        # Directory structure
        self.basis_dir = "basis/"
        self.contexts_dir = "contexts/"
        self.edges_dir = "edges/"
        self.layers_dir = "layers/"
    
    def _init_repo_structure(self):
        """Initialize the directory structure in the Git repo"""
        dirs = [self.basis_dir, self.contexts_dir, self.edges_dir, self.layers_dir]
        for dir_path in dirs:
            full_path = os.path.join(self.repo_path, dir_path)
            os.makedirs(full_path, exist_ok=True)
        
        # Create initial commit
        self.repo.index.commit("Initial Cortex repository structure")
    
    def _compute_sha(self, content: str) -> str:
        """Compute SHA1 hash for content (mimicking Git's hashing)"""
        return hashlib.sha1(content.encode()).hexdigest()
    
    def store_hllset(self, hllset: HLLSet) -> str:
        """
        Store HLLSet as Git blob in basis directory
        Returns: SHA of the stored HLLSet
        """
        # Serialize HLLSet - you might want to customize this
        content = json.dumps({
            'id': getattr(hllset, 'id', 'unknown'),
            'data': hllset.serialize() if hasattr(hllset, 'serialize') else str(hllset)
        })
        
        # Create blob in Git
        blob_sha = self.repo.git.hash_object('-w', '--stdin', input=content)
        
        # Store in basis directory
        blob_path = os.path.join(self.basis_dir, blob_sha)
        with open(os.path.join(self.repo_path, blob_path), 'w') as f:
            f.write(content)
        
        self.repo.index.add([blob_path])
        return blob_sha
    
    def get_hllset(self, sha: str) -> HLLSet:
        """Retrieve HLLSet from Git blob"""
        blob_path = os.path.join(self.basis_dir, sha)
        full_path = os.path.join(self.repo_path, blob_path)
        
        with open(full_path, 'r') as f:
            content = json.loads(f.read())
        
        # Deserialize HLLSet - you might want to customize this
        hllset = HLLSet()
        if hasattr(hllset, 'deserialize'):
            hllset.deserialize(content['data'])
        return hllset
    
    def create_context(self, context: Context) -> str:
        """
        Create context as Git tree commit
        Returns: Commit SHA
        """
        # Add all basis elements to index
        for basis_sha in context.basis_elements:
            basis_path = os.path.join(self.basis_dir, basis_sha)
            self.repo.index.add([basis_path])
        
        # Create commit for context
        commit = self.repo.index.commit(f"Context: {context.name}")
        
        # Store context metadata
        context_path = os.path.join(self.contexts_dir, f"{context.name}.json")
        with open(os.path.join(self.repo_path, context_path), 'w') as f:
            json.dump(context.to_dict(), f)
        
        self.repo.index.add([context_path])
        self.repo.index.commit(f"Context metadata: {context.name}")
        
        return commit.hexsha
    
    def create_edge(self, edge: Edge, parent_commits: List[str] = None) -> str:
        """
        Create edge as Git commit
        Returns: Commit SHA
        """
        if parent_commits is None:
            parent_commits = []
        
        # Create edge data file
        edge_content = json.dumps(edge.to_dict())
        edge_path = os.path.join(self.edges_dir, f"{edge.edge_id}.json")
        
        with open(os.path.join(self.repo_path, edge_path), 'w') as f:
            f.write(edge_content)
        
        self.repo.index.add([edge_path])
        
        # Get parent commit objects
        parent_commit_objs = []
        for parent_sha in parent_commits:
            try:
                parent_commit = self.repo.commit(parent_sha)
                parent_commit_objs.append(parent_commit)
            except:
                print(f"Warning: Parent commit {parent_sha} not found")
        
        # Create commit for edge
        commit_msg = f"Edge: {edge.edge_id} (τ={edge.tau}, ρ={edge.rho})"
        commit = self.repo.index.commit(commit_msg, parent_commits=parent_commit_objs)
        
        return commit.hexsha
    
    def create_layer_branch(self, layer_name: str, edges: List[Edge]) -> str:
        """
        Create a new branch representing a Cortex layer
        Returns: Branch name
        """
        # Create new branch
        current_branch = self.repo.active_branch.name
        self.repo.git.checkout('-b', layer_name)
        
        # Add all edges as commits
        for edge in edges:
            # Find parent commits (simplified - you might want more complex logic)
            parent_commits = []
            for parent_edge_id in edge.parent_edges:
                # Look for parent edge commits
                try:
                    # This is simplified - you'd want a better way to find parent commits
                    parent_commit = self._find_edge_commit(parent_edge_id)
                    if parent_commit:
                        parent_commits.append(parent_commit)
                except:
                    continue
            
            self.create_edge(edge, parent_commits)
        
        # Store layer metadata
        layer_metadata = {
            'name': layer_name,
            'edges': [edge.edge_id for edge in edges]
        }
        layer_path = os.path.join(self.layers_dir, f"{layer_name}.json")
        with open(os.path.join(self.repo_path, layer_path), 'w') as f:
            json.dump(layer_metadata, f)
        
        self.repo.index.add([layer_path])
        self.repo.index.commit(f"Layer metadata: {layer_name}")
        
        # Switch back to original branch
        self.repo.git.checkout(current_branch)
        
        return layer_name
    
    def _find_edge_commit(self, edge_id: str) -> Optional[str]:
        """Find commit SHA for a given edge ID"""
        # This is a simplified implementation
        # You might want to maintain an index of edge->commit mappings
        edge_path = os.path.join(self.edges_dir, f"{edge_id}.json")
        try:
            # Use git log to find commits that added this file
            commits = self.repo.git.log('--oneline', '--follow', '--', edge_path)
            if commits:
                return commits.split('\n')[0].split(' ')[0]
        except:
            pass
        return None
    
    def get_cortex_history(self) -> List[Dict]:
        """Get the commit history of the Cortex"""
        history = []
        for commit in self.repo.iter_commits():
            history.append({
                'sha': commit.hexsha,
                'message': commit.message.strip(),
                'author': f"{commit.author.name} <{commit.author.email}>",
                'date': commit.committed_datetime.isoformat()
            })
        return history
    
    def merge_contexts(self, context1: str, context2: str, merge_message: str = None) -> str:
        """
        Merge two contexts (similar to git merge)
        Returns: Merge commit SHA
        """
        if merge_message is None:
            merge_message = f"Merge contexts: {context1} and {context2}"
        
        # This is a simplified merge operation
        # In practice, you'd want more sophisticated context merging logic
        try:
            self.repo.git.merge(context1, context2, m=merge_message)
            return self.repo.head.commit.hexsha
        except git.GitCommandError as e:
            print(f"Merge conflict: {e}")
            # Handle merge conflicts here
            return None

# Demo and usage example
def demo():
    """Demonstrate Git-backed Cortex functionality"""
    
    # Initialize repository
    repo_path = "/tmp/cortex_demo"
    cortex = GitBackedCortex(repo_path, init=True)
    
    print("Git-backed Cortex Demo")
    print("=" * 50)
    
    # Create some sample HLLSets
    sample_hllsets = [
        HLLSet(),  # You'll need to initialize these properly
        HLLSet(),
        HLLSet()
    ]
    
    # Store HLLSets
    basis_shas = []
    for i, hllset in enumerate(sample_hllsets):
        sha = cortex.store_hllset(hllset)
        basis_shas.append(sha)
        print(f"Stored HLLSet {i} with SHA: {sha}")
    
    # Create contexts
    context_a = Context("context_A", basis_shas[:2])
    context_b = Context("context_B", basis_shas[1:])
    
    context_a_sha = cortex.create_context(context_a)
    context_b_sha = cortex.create_context(context_b)
    
    print(f"Created context A: {context_a_sha}")
    print(f"Created context B: {context_b_sha}")
    
    # Create edges
    edge1 = Edge("edge_1", tau=0.8, rho=0.6, parent_edges=[])
    edge2 = Edge("edge_2", tau=0.9, rho=0.7, parent_edges=["edge_1"])
    
    edge1_sha = cortex.create_edge(edge1)
    edge2_sha = cortex.create_edge(edge2, [edge1_sha])
    
    print(f"Created edge 1: {edge1_sha}")
    print(f"Created edge 2: {edge2_sha}")
    
    # Create layer branch
    layer_name = cortex.create_layer_branch("abstraction_layer_1", [edge1, edge2])
    print(f"Created layer branch: {layer_name}")
    
    # Show history
    history = cortex.get_cortex_history()
    print(f"\nCortex history ({len(history)} commits):")
    for commit in history[:5]:  # Show first 5 commits
        print(f"  {commit['sha'][:8]} - {commit['message']}")

if __name__ == "__main__":
    demo()