import os
import json
import git
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from ..core.HLLSets import HllSet

@dataclass
class GitCortexConfig:
    repo_path: str = "./hllset_cortex_repo"
    hllset_dir: str = "HLLSets"
    context_dir: str = "contexts"
    raw_data_dir: str = "raw_data"
    hi_idx_dir: str = "hi_idx_states"

class GitCortex:
    def __init__(self, config: GitCortexConfig = None):
        self.config = config or GitCortexConfig()
        self.repo = self._init_repo()
    
    def _init_repo(self) -> git.Repo:
        """Initialize or open Git repository"""
        if not os.path.exists(self.config.repo_path):
            os.makedirs(self.config.repo_path)
            repo = git.Repo.init(self.config.repo_path)
            
            # Create initial commit
            self._create_initial_commit(repo)
        else:
            repo = git.Repo(self.config.repo_path)
        
        return repo
    
    def _create_initial_commit(self, repo: git.Repo):
        """Create initial commit with directory structure"""
        # Create directories
        for dir_name in [self.config.hllset_dir, self.config.context_dir, 
                        self.config.raw_data_dir, self.config.hi_idx_dir]:
            os.makedirs(os.path.join(self.config.repo_path, dir_name), exist_ok=True)
        
        # Create README
        readme_path = os.path.join(self.config.repo_path, "README.md")
        with open(readme_path, 'w') as f:
            f.write("# HLLSet Cortex Repository\n\n")
            f.write("This repository stores HLLSets, contexts, and relationships.\n")
        
        repo.index.add([readme_path])
        repo.index.commit("Initial commit: HLLSet Cortex structure")
    
    def store_hllset(self, hllset: HllSet) -> str:
        """Store HLLSet and return SHA1"""
        serialized = hllset.serialize()
        return self._store_blob(serialized, self.config.hllset_dir)
    
    def get_hllset(self, sha1: str) -> Optional[HllSet]:
        """Retrieve HLLSet by SHA1"""
        blob_content = self._get_blob_content(sha1)
        if blob_content:
            return HllSet.deserialize(blob_content)
        return None
    
    def store_raw_data(self, data: bytes, data_type: str = "text") -> str:
        """Store raw data and return SHA1"""
        metadata = {
            'data_type': data_type,
            'timestamp': self._current_timestamp()
        }
        wrapped_data = json.dumps(metadata).encode() + b'\0\0\0' + data
        return self._store_blob(wrapped_data, self.config.raw_data_dir)
    
    def create_context_commit(self, context_name: str, hllset_shas: Set[str],
                            parent_commits: List[str] = None) -> str:
        """Create a context as a Git commit"""
        # Create tree with context metadata
        context_data = {
            'name': context_name,
            'hllset_shas': list(hllset_shas),
            'timestamp': self._current_timestamp()
        }
        
        tree_content = json.dumps(context_data, indent=2).encode()
        tree_sha = self._store_blob(tree_content, self.config.context_dir)
        
        # Create commit
        commit_message = f"Context: {context_name} with {len(hllset_shas)} HLLSets"
        commit_sha = self._create_commit(tree_sha, parent_commits or [], commit_message)
        
        return commit_sha
    
    def store_hi_idx_state(self, hi_idx_state: Dict) -> str:
        """Store hi_idx state as JSON in Git"""
        json_content = json.dumps(hi_idx_state, indent=2).encode()
        filename = f"hi_idx_state_{self._current_timestamp()}.json"
        return self._store_blob(json_content, self.config.hi_idx_dir, filename)
    
    def _store_blob(self, data: bytes, directory: str, filename: str = None) -> str:
        """Store data as Git blob and return SHA1"""
        from git import Blob
        
        if filename:
            filepath = os.path.join(directory, filename)
        else:
            filepath = directory  # Git will handle the blob storage
        
        # Create blob
        blob_sha = self.repo.git.hash_object('-w', '--stdin', input=data)
        
        # Add to index if we have a filename
        if filename:
            # We need to create a tree with this blob
            self.repo.index.add([filepath])
            self.repo.index.commit(f"Add {filename}")
        
        return blob_sha
    
    def _get_blob_content(self, sha1: str) -> Optional[bytes]:
        """Get blob content by SHA1"""
        try:
            return self.repo.git.cat_file('-p', sha1).encode('latin1')
        except Exception:
            return None
    
    def _create_commit(self, tree_sha: str, parent_commits: List[str], 
                      message: str) -> str:
        """Create a Git commit"""
        parent_args = []
        for parent in parent_commits:
            parent_args.extend(['-p', parent])
        
        commit_sha = self.repo.git.commit_tree(tree_sha, *parent_args, 
                                             '-m', message)
        return commit_sha
    
    def _current_timestamp(self) -> str:
        """Get current timestamp as ISO string"""
        from datetime import datetime
        return datetime.now().isoformat()