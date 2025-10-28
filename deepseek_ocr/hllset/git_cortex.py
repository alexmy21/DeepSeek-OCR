"""
Git-backed Cortex Module
"""

import os
import sys
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import hashlib
import json

# Import HllSet - this will use either Julia or Python implementation
try:
    from HLLSets import HllSet
    print("✅ HllSet imported successfully")
except ImportError as e:
    print(f"❌ HllSet import failed: {e}")
    
    # Minimal fallback
    class HllSet:
        def __init__(self, p=10):
            self.p = p
            self.data = {}
            
        def add(self, element, seed=0):
            return self
            
        @property
        def count(self):
            return 0
            
        def serialize(self):
            return json.dumps({"p": self.p})
            
        def deserialize(self, data):
            return self

@dataclass
class Context:
    name: str
    basis_elements: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self):
        """Convert Context to dictionary for serialization"""
        return {
            'name': self.name,
            'basis_elements': self.basis_elements,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create Context from dictionary"""
        return cls(
            name=data['name'],
            basis_elements=data['basis_elements'],
            metadata=data.get('metadata', {})
        )

@dataclass 
class Edge:
    edge_id: str
    tau: float
    rho: float
    parent_edges: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parent_edges is None:
            self.parent_edges = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self):
        """Convert Edge to dictionary for serialization"""
        return {
            'edge_id': self.edge_id,
            'tau': self.tau,
            'rho': self.rho,
            'parent_edges': self.parent_edges,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create Edge from dictionary"""
        return cls(
            edge_id=data['edge_id'],
            tau=data['tau'],
            rho=data['rho'],
            parent_edges=data.get('parent_edges', []),
            metadata=data.get('metadata', {})
        )

class GitBackedCortex:
    def __init__(self, repo_path: str, init: bool = False):
        try:
            import git
            self.repo = git.Repo.init(repo_path) if init else git.Repo(repo_path)
        except ImportError:
            raise ImportError("Install GitPython: pip install GitPython")
            
        self.repo_path = repo_path
        
        if init:
            # Create directory structure
            for dir_name in ["basis", "contexts", "edges", "layers"]:
                os.makedirs(os.path.join(repo_path, dir_name), exist_ok=True)
            self.repo.index.commit("Initial Cortex repository")
    
    def store_hllset(self, hllset: HllSet) -> str:
        """Store HllSet as Git blob"""
        content = hllset.serialize()
        
        # Use temporary file for git hash-object
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Use the temporary file with git hash-object
            blob_sha = self.repo.git.hash_object('-w', temp_path)
            
            # Write to the basis directory in the repo
            blob_path = f"basis/{blob_sha}.json"
            full_blob_path = os.path.join(self.repo_path, blob_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(full_blob_path), exist_ok=True)
            
            # Copy from temp file to repo
            import shutil
            shutil.copy2(temp_path, full_blob_path)
            
            # Add to git index
            self.repo.index.add([blob_path])
            self.repo.index.commit(f"Add HllSet: {blob_sha[:8]}")
            
            return blob_sha
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def get_hllset(self, sha: str) -> HllSet:
        """Retrieve HllSet from Git blob"""
        blob_path = f"basis/{sha}.json"
        full_path = os.path.join(self.repo_path, blob_path)
        
        with open(full_path, 'r') as f:
            content = f.read()
        
        hllset = HllSet(p=10)
        hllset.deserialize(content)
        return hllset
    
    def create_context(self, context: Context) -> str:
        """Create a context as a Git commit"""
        # Create context metadata file
        context_path = f"contexts/{context.name}.json"
        full_path = os.path.join(self.repo_path, context_path)
        
        with open(full_path, 'w') as f:
            json.dump(context.to_dict(), f, indent=2)
        
        self.repo.index.add([context_path])
        commit = self.repo.index.commit(f"Context: {context.name}")
        
        return commit.hexsha
    
    def get_context(self, context_name: str) -> Optional[Context]:
        """Retrieve context by name"""
        context_path = f"contexts/{context_name}.json"
        full_path = os.path.join(self.repo_path, context_path)
        
        if not os.path.exists(full_path):
            return None
        
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        return Context.from_dict(data)
    
    def create_edge(self, edge: Edge, parent_commits: List[str] = None) -> str:
        """Create an edge as a Git commit"""
        if parent_commits is None:
            parent_commits = []
        
        # Create edge metadata file
        edge_path = f"edges/{edge.edge_id}.json"
        full_path = os.path.join(self.repo_path, edge_path)
        
        with open(full_path, 'w') as f:
            json.dump(edge.to_dict(), f, indent=2)
        
        self.repo.index.add([edge_path])
        
        # Get parent commit objects
        parent_commit_objs = []
        for parent_sha in parent_commits:
            try:
                parent_commit = self.repo.commit(parent_sha)
                parent_commit_objs.append(parent_commit)
            except:
                print(f"Warning: Parent commit {parent_sha} not found")
        
        # Create commit
        commit_msg = f"Edge: {edge.edge_id} (τ={edge.tau}, ρ={edge.rho})"
        commit = self.repo.index.commit(commit_msg, parent_commits=parent_commit_objs)
        
        return commit.hexsha
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Retrieve edge by ID"""
        edge_path = f"edges/{edge_id}.json"
        full_path = os.path.join(self.repo_path, edge_path)
        
        if not os.path.exists(full_path):
            return None
        
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        return Edge.from_dict(data)
    
    def list_contexts(self) -> List[str]:
        """List all stored contexts"""
        contexts_dir = os.path.join(self.repo_path, "contexts")
        if not os.path.exists(contexts_dir):
            return []
        
        contexts = []
        for file in os.listdir(contexts_dir):
            if file.endswith('.json'):
                contexts.append(file[:-5])  # Remove .json extension
        return contexts
    
    def list_edges(self) -> List[str]:
        """List all stored edges"""
        edges_dir = os.path.join(self.repo_path, "edges")
        if not os.path.exists(edges_dir):
            return []
        
        edges = []
        for file in os.listdir(edges_dir):
            if file.endswith('.json'):
                edges.append(file[:-5])  # Remove .json extension
        return edges
    
    def get_history(self, max_count: int = 10) -> List[Dict]:
        """Get commit history"""
        history = []
        for commit in self.repo.iter_commits(max_count=max_count):
            history.append({
                'sha': commit.hexsha,
                'message': commit.message.strip(),
                'author': f"{commit.author.name} <{commit.author.email}>",
                'date': commit.committed_datetime.isoformat()
            })
        return history

def test_hllset_git_integration():
    """Test the integration between HllSet and GitCortex"""
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        print("Testing HllSet Git integration...")
        
        # Initialize cortex
        cortex = GitBackedCortex(temp_dir, init=True)
        print("✅ GitBackedCortex initialized")
        
        # Create and test HllSet
        hllset = HllSet(p=10)
        hllset.add("test_element_1")
        hllset.add("test_element_2")
        
        print(f"✅ HllSet created with count: {hllset.count}")
        
        # Store in Git
        sha = cortex.store_hllset(hllset)
        print(f"✅ HllSet stored with SHA: {sha}")
        
        # Retrieve from Git
        retrieved_hllset = cortex.get_hllset(sha)
        print(f"✅ HllSet retrieved with count: {retrieved_hllset.count}")
        
        # Test context creation
        context = Context("test_context", [sha])
        context_sha = cortex.create_context(context)
        print(f"✅ Context created with SHA: {context_sha[:8]}...")
        
        # Test context retrieval
        retrieved_context = cortex.get_context("test_context")
        print(f"✅ Context retrieved: {retrieved_context.name}")
        
        # Test edge creation
        edge = Edge("test_edge", tau=0.8, rho=0.6, parent_edges=[])
        edge_sha = cortex.create_edge(edge)
        print(f"✅ Edge created with SHA: {edge_sha[:8]}...")
        
        # Test edge retrieval
        retrieved_edge = cortex.get_edge("test_edge")
        print(f"✅ Edge retrieved: {retrieved_edge.edge_id}")
        
        # Test listing
        contexts = cortex.list_contexts()
        edges = cortex.list_edges()
        print(f"✅ Contexts: {contexts}")
        print(f"✅ Edges: {edges}")
        
        # Test history
        history = cortex.get_history(max_count=5)
        print(f"✅ History ({len(history)} commits)")
        for commit in history:
            print(f"   {commit['sha'][:8]} - {commit['message']}")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_hllset_git_integration()