import redis
import json
from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass, asdict
from ..core.HLLSets import HLLSet
from ..git_cortex.git_store import GitCortex

@dataclass
class HIIndexRecord:
    hllset_sha: str
    raw_data_sha: str
    context_sha: Optional[str] = None
    cardinality: float = 0.0
    timestamp: str = None
    base_cover: List[str] = None
    metadata: Dict[str, Any] = None

class RedisHIIndex:
    def __init__(self, redis_client: redis.Redis, git_cortex: GitCortex,
                 max_memory_records: int = 10000):
        self.redis = redis_client
        self.git_cortex = git_cortex
        self.max_memory_records = max_memory_records
        
        # Initialize Redis structures
        self._init_redis_indexes()
    
    def _init_redis_indexes(self):
        """Initialize Redis indexes if they don't exist"""
        # Cardinality sorted set for efficient range queries
        if not self.redis.exists('hi_idx:cardinality_index'):
            self.redis.zadd('hi_idx:cardinality_index', {'dummy': 0})
            self.redis.delete('hi_idx:cardinality_index')  # Remove dummy
    
    def add_record(self, record: HIIndexRecord):
        """Add record to hi_idx"""
        # Store record data
        record_key = f"hi_idx:record:{record.hllset_sha}"
        self.redis.hset(record_key, mapping=asdict(record))
        
        # Update cardinality index
        self.redis.zadd('hi_idx:cardinality_index', 
                       {record.hllset_sha: record.cardinality})
        
        # Update memory optimization
        self._manage_memory_usage()
    
    def get_record(self, hllset_sha: str) -> Optional[HIIndexRecord]:
        """Get record by HLLSet SHA1"""
        record_key = f"hi_idx:record:{hllset_sha}"
        record_data = self.redis.hgetall(record_key)
        
        if not record_data:
            return None
        
        # Convert bytes to string and appropriate types
        decoded_data = {}
        for key, value in record_data.items():
            k = key.decode('utf-8')
            v = value.decode('utf-8')
            
            # Type conversion
            if k in ['cardinality']:
                decoded_data[k] = float(v)
            elif k in ['base_cover']:
                decoded_data[k] = json.loads(v) if v else []
            elif k in ['metadata']:
                decoded_data[k] = json.loads(v) if v else {}
            else:
                decoded_data[k] = v
        
        return HIIndexRecord(**decoded_data)
    
    def find_similar(self, query_hllset: HLLSet, 
                    similarity_threshold: float = 0.6,
                    max_results: int = 100) -> List[Dict]:
        """Find similar HLLSets using cardinality filtering"""
        query_cardinality = query_hllset.cardinality()
        
        # Phase 1: Cardinality filtering
        candidate_shas = self.redis.zrangebyscore(
            'hi_idx:cardinality_index',
            query_cardinality * 0.5,  # Lower bound
            query_cardinality * 2.0,  # Upper bound
            start=0, num=max_results
        )
        
        # Phase 2: Similarity checking
        results = []
        for candidate_sha in candidate_shas:
            candidate_sha = candidate_sha.decode('utf-8')
            candidate_record = self.get_record(candidate_sha)
            
            if not candidate_record:
                continue
            
            # Get HLLSet from Git Cortex
            candidate_hllset = self.git_cortex.get_hllset(candidate_sha)
            if not candidate_hllset:
                continue
            
            similarity = query_hllset.jaccard_similarity(candidate_hllset)
            
            if similarity >= similarity_threshold:
                results.append({
                    'hllset_sha': candidate_sha,
                    'similarity': similarity,
                    'cardinality': candidate_record.cardinality,
                    'record': candidate_record
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    def get_tokens_above_cardinality(self, threshold: float) -> Set[str]:
        """Get all HLLSet SHA1s with cardinality >= threshold"""
        shas = self.redis.zrangebyscore('hi_idx:cardinality_index', 
                                       threshold, float('inf'))
        return {sha.decode('utf-8') for sha in shas}
    
    def _manage_memory_usage(self):
        """Manage memory usage by syncing to Git if needed"""
        record_count = self.redis.zcard('hi_idx:cardinality_index')
        
        if record_count > self.max_memory_records * 1.2:  # 20% over limit
            self._sync_excess_to_git()
    
    def _sync_excess_to_git(self):
        """Sync excess records to Git and remove from Redis"""
        # This would be implemented in the synchronizer
        pass
    
    def get_all_records(self) -> Dict[str, HIIndexRecord]:
        """Get all records (for synchronization)"""
        records = {}
        pattern = "hi_idx:record:*"
        
        for key in self.redis.scan_iter(match=pattern):
            hllset_sha = key.decode('utf-8').split(':')[-1]
            record = self.get_record(hllset_sha)
            if record:
                records[hllset_sha] = record
        
        return records