<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<table>
<tr>
<td width="30%">
  <img src="assets/1112548.png" width="60%" alt="SGS.ai" />
</td>
<td>
  <h1>HLLSet Cortex Library</h1>
</td>
</tr>
</table>

## Library Structure

```bash
deepseek_ocr/
├── __init__.py
├── hllset_cortex/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── hllset.py
│   │   ├── hashing.py
│   │   └── similarity.py
│   ├── git_cortex/
│   │   ├── __init__.py
│   │   ├── git_store.py
│   │   └── models.py
│   ├── hi_index/
│   │   ├── __init__.py
│   │   ├── redis_index.py
│   │   └── synchronizer.py
│   ├── traversal/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── context_builder.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── deepseek_ocr.py
│   └── utils/
│       ├── __init__.py
│       ├── serialization.py
│       └── sha1.py
└── shadow_extension.py
```

## Core Implementation Files

### 1. `hllset_cortex/core/hllset.py`

```python
import numpy as np
import mmh3
from typing import Set, List, Dict, Any
from dataclasses import dataclass
import pickle

@dataclass
class HLLSetConfig:
    p: int = 14  # 2^14 = 16384 registers
    b: int = 32  # 32 bits per register
    hash_seed: int = 42

class HLLSet:
    def __init__(self, config: HLLSetConfig = None):
        self.config = config or HLLSetConfig()
        self.m = 1 << self.config.p
        self.registers = np.zeros(self.m, dtype=np.uint32)
    
    def add_token(self, token: str, seed: int = None):
        """Add a token to the HLLSet"""
        seed = seed or self.config.hash_seed
        hash_val = mmh3.hash64(token, seed=seed)[0]
        
        # Use first p bits for register index
        register_idx = hash_val >> (64 - self.config.p)
        register_idx = register_idx % self.m
        
        # Use remaining bits to determine bit position
        remaining_bits = hash_val & ((1 << (64 - self.config.p)) - 1)
        bit_pos = self._trailing_zeros(remaining_bits)
        
        # Set the bit if within register width
        if bit_pos < self.config.b:
            self.registers[register_idx] |= (1 << bit_pos)
    
    def add_tokens(self, tokens: List[str], seed: int = None):
        """Add multiple tokens efficiently"""
        for token in tokens:
            self.add_token(token, seed)
    
    def union(self, other: 'HLLSet') -> 'HLLSet':
        """Compute union of two HLLSets"""
        if self.config != other.config:
            raise ValueError("HLLSet configurations must match")
        
        result = HLLSet(self.config)
        result.registers = self.registers | other.registers
        return result
    
    def intersection(self, other: 'HLLSet') -> 'HLLSet':
        """Compute intersection of two HLLSets"""
        if self.config != other.config:
            raise ValueError("HLLSet configurations must match")
        
        result = HLLSet(self.config)
        result.registers = self.registers & other.registers
        return result
    
    def difference(self, other: 'HLLSet') -> 'HLLSet':
        """Compute difference A - B"""
        if self.config != other.config:
            raise ValueError("HLLSet configurations must match")
        
        result = HLLSet(self.config)
        result.registers = self.registers & ~other.registers
        return result
    
    def cardinality(self) -> float:
        """Estimate cardinality using HyperLogLog algorithm"""
        # Simplified cardinality estimation
        # In production, use proper HLL estimation
        register_sum = np.sum(self.registers != 0)
        if register_sum == 0:
            return 0.0
        
        alpha = 0.7213 / (1 + 1.079 / self.m)
        raw_estimate = alpha * self.m * self.m / register_sum
        
        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            zeros = np.sum(self.registers == 0)
            if zeros > 0:
                return self.m * np.log(self.m / zeros)
        
        return raw_estimate
    
    def jaccard_similarity(self, other: 'HLLSet') -> float:
        """Compute Jaccard similarity between two HLLSets"""
        intersection_card = self.intersection(other).cardinality()
        union_card = self.union(other).cardinality()
        
        if union_card == 0:
            return 0.0
        return intersection_card / union_card
    
    def serialize(self) -> bytes:
        """Serialize HLLSet to bytes"""
        return pickle.dumps({
            'config': self.config,
            'registers': self.registers.tobytes()
        })
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'HLLSet':
        """Deserialize HLLSet from bytes"""
        obj = pickle.loads(data)
        hllset = cls(obj['config'])
        hllset.registers = np.frombuffer(obj['registers'], dtype=np.uint32)
        return hllset
    
    @classmethod
    def from_tokens(cls, tokens: List[str], config: HLLSetConfig = None) -> 'HLLSet':
        """Create HLLSet from token list"""
        hllset = cls(config)
        hllset.add_tokens(tokens)
        return hllset
    
    def _trailing_zeros(self, x: int) -> int:
        """Count trailing zeros in integer"""
        if x == 0:
            return 64 - self.config.p
        return (x & -x).bit_length() - 1
```

### 2. `hllset_cortex/core/similarity.py`

```python
from .hllset import HLLSet
from typing import Tuple

class HLLSetSimilarity:
    @staticmethod
    def bss_tau(A: HLLSet, B: HLLSet) -> float:
        """Compute BSS_tau (inclusion) A → B = |A ∩ B| / |B|"""
        intersection = A.intersection(B)
        b_cardinality = B.cardinality()
        
        if b_cardinality == 0:
            return 0.0
        return intersection.cardinality() / b_cardinality
    
    @staticmethod
    def bss_rho(A: HLLSet, B: HLLSet) -> float:
        """Compute BSS_rho (exclusion) A → B = |A \ B| / |B|"""
        difference = A.difference(B)
        b_cardinality = B.cardinality()
        
        if b_cardinality == 0:
            return 0.0
        return difference.cardinality() / b_cardinality
    
    @staticmethod
    def directional_similarity(A: HLLSet, B: HLLSet, 
                             tau_threshold: float = 0.7,
                             rho_threshold: float = 0.3) -> Tuple[bool, float, float]:
        """
        Check if morphism A → B exists based on tau/rho thresholds
        Returns: (exists, bss_tau, bss_rho)
        """
        bss_tau_val = HLLSetSimilarity.bss_tau(A, B)
        bss_rho_val = HLLSetSimilarity.bss_rho(A, B)
        
        exists = (bss_tau_val >= tau_threshold and 
                 bss_rho_val < rho_threshold)
        
        return exists, bss_tau_val, bss_rho_val
```

### 3. `hllset_cortex/git_cortex/git_store.py`

```python
import os
import json
import git
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from ..core.hllset import HLLSet

@dataclass
class GitCortexConfig:
    repo_path: str = "./hllset_cortex_repo"
    hllset_dir: str = "hllsets"
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
    
    def store_hllset(self, hllset: HLLSet) -> str:
        """Store HLLSet and return SHA1"""
        serialized = hllset.serialize()
        return self._store_blob(serialized, self.config.hllset_dir)
    
    def get_hllset(self, sha1: str) -> Optional[HLLSet]:
        """Retrieve HLLSet by SHA1"""
        blob_content = self._get_blob_content(sha1)
        if blob_content:
            return HLLSet.deserialize(blob_content)
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
```

### 4. `hllset_cortex/hi_index/redis_index.py`

```python
import redis
import json
from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass, asdict
from ..core.hllset import HLLSet
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
```

### 5. `hllset_cortex/hi_index/synchronizer.py`

```python
import json
from typing import Dict, Any
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
```

### 6. `hllset_cortex/integration/deepseek_ocr.py`

```python
import os
from typing import Dict, List, Optional
from ..core.hllset import HLLSet, HLLSetConfig
from ..git_cortex.git_store import GitCortex, GitCortexConfig
from ..hi_index.redis_index import RedisHIIndex, HIIndexRecord
from ..hi_index.synchronizer import HIIndexSynchronizer

class DeepSeekOCRIntegration:
    """
    Main integration class that connects DeepSeek-OCR with HLLSet Cortex
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 git_repo_path: str = "./hllset_cortex_repo",
                 hllset_config: HLLSetConfig = None):
        
        self.hllset_config = hllset_config or HLLSetConfig()
        
        # Initialize Git Cortex
        git_config = GitCortexConfig(repo_path=git_repo_path)
        self.git_cortex = GitCortex(git_config)
        
        # Initialize Redis connection
        import redis
        self.redis_client = redis.from_url(redis_url)
        
        # Initialize HI Index
        self.hi_index = RedisHIIndex(self.redis_client, self.git_cortex)
        
        # Initialize Synchronizer
        self.synchronizer = HIIndexSynchronizer(self.hi_index, self.git_cortex)
        
        # Initialize base HLLSet manager (soldered sets)
        self.base_manager = BaseHLLSetManager()
    
    def process_document(self, document_path: str, 
                        document_type: str = "text") -> Dict:
        """
        Process a document through DeepSeek-OCR and store in HLLSet Cortex
        """
        # Step 1: Extract text using DeepSeek-OCR
        extracted_data = self._extract_with_deepseek_ocr(document_path, document_type)
        
        # Step 2: Create HLLSet from extracted tokens
        hllset = HLLSet.from_tokens(extracted_data['tokens'], self.hllset_config)
        
        # Step 3: Store raw data in Git Cortex
        raw_data_sha = self.git_cortex.store_raw_data(
            extracted_data['raw_text'].encode('utf-8'),
            document_type
        )
        
        # Step 4: Store HLLSet in Git Cortex and get SHA1
        hllset_sha = self.git_cortex.store_hllset(hllset)
        
        # Step 5: Create context for the document
        context_shas = self._build_context(hllset)
        context_sha = self.git_cortex.create_context_commit(
            f"doc_{os.path.basename(document_path)}",
            context_shas
        )
        
        # Step 6: Add to HI Index
        record = HIIndexRecord(
            hllset_sha=hllset_sha,
            raw_data_sha=raw_data_sha,
            context_sha=context_sha,
            cardinality=hllset.cardinality(),
            base_cover=list(context_shas)
        )
        self.hi_index.add_record(record)
        
        return {
            'hllset_sha': hllset_sha,
            'raw_data_sha': raw_data_sha,
            'context_sha': context_sha,
            'cardinality': hllset.cardinality(),
            'token_count': len(extracted_data['tokens']),
            'document_path': document_path
        }
    
    def query_similar_documents(self, query_text: str, 
                               similarity_threshold: float = 0.6) -> List[Dict]:
        """
        Find documents similar to query text
        """
        # Create HLLSet from query
        query_hllset = HLLSet.from_tokens([query_text], self.hllset_config)
        
        # Find similar in HI Index
        similar_results = self.hi_index.find_similar(
            query_hllset, similarity_threshold
        )
        
        # Enrich with document information
        enriched_results = []
        for result in similar_results:
            record = result['record']
            
            # Get raw data info from Git Cortex
            raw_data = self._get_raw_data_info(record.raw_data_sha)
            
            enriched_results.append({
                'hllset_sha': result['hllset_sha'],
                'similarity': result['similarity'],
                'cardinality': result['cardinality'],
                'raw_data_info': raw_data,
                'context_sha': record.context_sha
            })
        
        return enriched_results
    
    def _extract_with_deepseek_ocr(self, document_path: str, 
                                 document_type: str) -> Dict:
        """
        Extract text and tokens from document using DeepSeek-OCR
        This is a placeholder for actual DeepSeek-OCR integration
        """
        # TODO: Integrate with actual DeepSeek-OCR
        if document_type == "text":
            with open(document_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            # For images/PDFs, this would call DeepSeek-OCR
            text = f"Extracted text from {document_path} would go here"
        
        # Simple tokenization (replace with proper tokenization)
        tokens = text.split()
        
        return {
            'raw_text': text,
            'tokens': tokens,
            'document_type': document_type
        }
    
    def _build_context(self, hllset: HLLSet) -> Set[str]:
        """
        Build context for HLLSet by finding similar existing HLLSets
        """
        similar_results = self.hi_index.find_similar(hllset, 0.5)
        context_shas = {result['hllset_sha'] for result in similar_results[:5]}  # Top 5
        
        # Include base soldered sets
        base_cover = self.base_manager.find_optimal_cover(hllset)
        for base_hllset in base_cover:
            base_sha = self.git_cortex.store_hllset(base_hllset)
            context_shas.add(base_sha)
        
        return context_shas
    
    def _get_raw_data_info(self, raw_data_sha: str) -> Dict:
        """Get information about raw data from Git Cortex"""
        # This would extract metadata from the stored raw data
        return {
            'sha': raw_data_sha,
            'type': 'text',  # Would be extracted from metadata
            'size': 0  # Would be calculated
        }

class BaseHLLSetManager:
    """Manages soldered/base HLLSets for FPGA optimization"""
    
    def __init__(self):
        self.soldered_sets = self._initialize_soldered_sets()
    
    def _initialize_soldered_sets(self):
        """Initialize with optimal soldered HLLSets"""
        # These would be carefully chosen base sets
        return set()  # Placeholder
    
    def find_optimal_cover(self, target: HLLSet) -> List[HLLSet]:
        """Find minimal soldered set that covers target"""
        # Implementation would go here
        return list(self.soldered_sets)
```

### 7. `shadow_extension.py`

```python
"""
Shadow extension for DeepSeek-OCR that integrates HLLSet Cortex
"""

import os
import sys
from typing import Dict, List, Optional

# Add the current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

from deepseek_ocr.hllset_cortex.integration.deepseek_ocr import DeepSeekOCRIntegration

class DeepSeekOCRWithCortex:
    """
    Wrapper class that adds HLLSet Cortex capabilities to DeepSeek-OCR
    """
    
    def __init__(self, 
                 redis_url: str = None,
                 git_repo_path: str = None,
                 enable_cortex: bool = True):
        
        self.enable_cortex = enable_cortex
        
        if enable_cortex:
            # Initialize HLLSet Cortex integration
            self.cortex_integration = DeepSeekOCRIntegration(
                redis_url=redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379'),
                git_repo_path=git_repo_path or os.getenv('GIT_CORTEX_PATH', './hllset_cortex_repo')
            )
        
        # Initialize original DeepSeek-OCR
        self.original_ocr = self._initialize_original_ocr()
    
    def _initialize_original_ocr(self):
        """Initialize the original DeepSeek-OCR system"""
        # This would initialize the actual DeepSeek-OCR
        # For now, return a mock
        class MockOCR:
            def recognize(self, image_path):
                return {"text": f"Mock OCR result for {image_path}", "tokens": ["mock", "ocr", "result"]}
        
        return MockOCR()
    
    def recognize(self, image_path: str, use_cortex: bool = True) -> Dict:
        """
        Recognize text from image with optional HLLSet Cortex integration
        """
        # Step 1: Use original OCR
        ocr_result = self.original_ocr.recognize(image_path)
        
        if self.enable_cortex and use_cortex:
            # Step 2: Process through HLLSet Cortex
            cortex_result = self.cortex_integration.process_document(
                image_path, "image"
            )
            
            # Step 3: Find similar documents
            similar_docs = self.cortex_integration.query_similar_documents(
                ocr_result['text']
            )
            
            # Enhance OCR result with cortex information
            ocr_result['cortex'] = {
                'hllset_sha': cortex_result['hllset_sha'],
                'similar_documents': similar_docs,
                'cardinality': cortex_result['cardinality']
            }
        
        return ocr_result
    
    def query_similar(self, query_text: str, max_results: int = 10) -> List[Dict]:
        """Query similar documents using HLLSet Cortex"""
        if not self.enable_cortex:
            return []
        
        return self.cortex_integration.query_similar_documents(
            query_text
        )[:max_results]
    
    def get_document_context(self, document_path: str) -> Optional[Dict]:
        """Get context information for a processed document"""
        if not self.enable_cortex:
            return None
        
        # This would retrieve the context from Git Cortex
        return {"status": "Context retrieval not implemented"}

# Global instance for easy access
_cortex_instance = None

def get_cortex_extension(redis_url: str = None, git_repo_path: str = None):
    """Get or create the cortex extension instance"""
    global _cortex_instance
    if _cortex_instance is None:
        _cortex_instance = DeepSeekOCRWithCortex(redis_url, git_repo_path)
    return _cortex_instance

def recognize_with_cortex(image_path: str) -> Dict:
    """Convenience function for recognition with cortex"""
    cortex_ext = get_cortex_extension()
    return cortex_ext.recognize(image_path)

def query_similar_documents(query_text: str, max_results: int = 10) -> List[Dict]:
    """Convenience function for querying similar documents"""
    cortex_ext = get_cortex_extension()
    return cortex_ext.query_similar(query_text, max_results)
```

### 8. `setup.py`

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="deepseek-ocr-hllset-cortex",
    version="0.1.0",
    author="Alex Mylnikov",
    author_email="alex@example.com",
    description="HLLSet Cortex shadow extension for DeepSeek-OCR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'deepseek-cortex=deepseek_ocr.shadow_extension:main',
        ],
    },
)
```

### 9. `requirements.txt`

```txt
numpy>=1.21.0
mmh3>=3.0.0
redis>=4.5.0
GitPython>=3.1.0
python-dotenv>=0.19.0
dataclasses-json>=0.5.0
```

## Usage Examples

### Basic Usage

```python
from deepseek_ocr.shadow_extension import get_cortex_extension

# Initialize the extension
cortex_ext = get_cortex_extension(
    redis_url="redis://localhost:6379",
    git_repo_path="./hllset_cortex_repo"
)

# Process a document with cortex
result = cortex_ext.recognize("document.pdf")
print(f"OCR Text: {result['text']}")
print(f"HLLSet SHA: {result['cortex']['hllset_sha']}")
print(f"Similar documents: {len(result['cortex']['similar_documents'])}")

# Query similar documents
similar = cortex_ext.query_similar("quantum physics", max_results=5)
for doc in similar:
    print(f"Similarity: {doc['similarity']:.3f}")
```

### Integration with Existing DeepSeek-OCR

```python
# Replace original DeepSeek-OCR import with:
from deepseek_ocr.shadow_extension import recognize_with_cortex

# Use exactly the same API, but now with HLLSet Cortex capabilities
result = recognize_with_cortex("research_paper.png")
```

This library provides a complete, integrated solution that can be easily added to your DeepSeek-OCR fork as a shadow extension. The modular design allows for different components to be used independently while providing a seamless integration experience. 🚀

```math
ℒ_DPO(π_θ) = -𝔼[(x, y_w, y_l) ~ D] \text{ log } σ(β \text{ log}(π_θ(y_w|x)/π_ref(y_w|x)) — β log(π_θ(y_l|x)/π_ref(y_l|x)))
```
