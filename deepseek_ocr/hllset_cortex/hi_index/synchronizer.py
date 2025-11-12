import json
from typing import Set, Dict, Any
from .redis_index import RedisHIIndex
from ..git_cortex.git_store import GitCortex

class HIIndexSynchronizer:
    def __init__(self, hi_index: RedisHIIndex, git_cortex: GitCortex):
        self.hi_index = hi_index
        self.git_cortex = git_cortex
    
    def dump_state_to_git(self, commit_message: str = "hi_idx sync") -> str:
        """Dump current hi_idx state to Git as JSON"""
        # Get all records from Redis
        all_records = self.hi_index.get_all_records()
        
        # Convert to serializable format
        serializable_state = {}
        for hllset_sha, record in all_records.items():
            serializable_state[hllset_sha] = {
                'hllset_sha': record.hllset_sha,
                'raw_data_sha': record.raw_data_sha,
                'context_sha': record.context_sha,
                'cardinality': record.cardinality,
                'timestamp': record.timestamp,
                'base_cover': record.base_cover or [],
                'metadata': record.metadata or {}
            }
        
        # Store in Git
        return self.git_cortex.store_hi_idx_state(serializable_state)
    
    def load_state_from_git(self, commit_sha: str = None) -> Dict[str, Any]:
        """Load hi_idx state from Git commit"""
        # This would implement loading from a specific Git commit
        # For now, return empty dict
        return {}
    
    def selective_sync(self, focus_shas: Set[str]):
        """Keep only focused records in memory, sync others to Git"""
        all_records = self.hi_index.get_all_records()
        to_remove = set(all_records.keys()) - focus_shas
        
        if to_remove:
            # Sync the state before removal
            self.dump_state_to_git("Selective sync before memory reduction")
            
            # Remove from Redis (they can be lazy-loaded from Git)
            for hllset_sha in to_remove:
                self.hi_index.redis.delete(f"hi_idx:record:{hllset_sha}")
                self.hi_index.redis.zrem('hi_idx:cardinality_index', hllset_sha)