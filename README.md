<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="assets/1112548.png" width="30%" alt="SGS.ai" />
</div>

# HLLSet Cortex: Semantic Intelligence Layer for DeepSeek-OCR

## 🧠 Overview

HLLSet Cortex is a **shadow indexing system** that adds semantic intelligence to DeepSeek-OCR by intercepting and enhancing all data exchange between users and the OCR engine. It transforms document processing into a **self-evolving knowledge graph** using HyperLogLog-based semantic fingerprints.

## 🎯 Core Strategy

### **Interception & Enhancement Pipeline**

```
User Prompt/Files → HLLSet Cortex → Enhanced Query → DeepSeek-OCR → Response → HLLSet Integration → Cortex Evolution
```

### **Key Principles**

1. **Everything becomes HLLSets**: All prompts, files, and OCR responses are converted to semantic fingerprints
2. **Cortex grows organically**: Each interaction creates new layers and relationships
3. **Query enhancement**: Cortex extends user queries with semantic context before OCR processing
4. **Layered architecture**: New knowledge pushes previous layers down, creating temporal relationships

## 🚀 Installation & Setup

### **1.1 Install DeepSeek-OCR Locally**

```bash
# Clone the repository
git clone https://github.com/alexmy21/DeepSeek-OCR
cd DeepSeek-OCR

# Create and activate conda environment
conda create -n deepseek-ocr python=3.10
conda activate deepseek-ocr

# Install dependencies
pip install -r requirements.txt

# Install DeepSeek-OCR
pip install -e .
```

### **1.2 Install HLLSet Cortex Extension**

```bash
# Install Julia for HLLSet core (if not already installed)
curl -fsSL https://install.julialang.org | sh

# Install Python-Julia bridge
pip install julia
python -c "import julia; julia.install()"

# Install HLLSets.jl package
julia -e 'using Pkg; Pkg.add("HLLSets")'

# Install the shadow indexer extension
pip install ./hllset_extension
```

## 🏗️ System Architecture

### **Core Components**

```bash
DeepSeek-OCR/
├── deepseek_ocr/              # Main package
│   ├── ocr.py                 # Original OCR interface (currently stub)
│   ├── hllset/                # HLLSet Cortex implementation
│   │   ├── core.py            # HLLSet class with similarity algorithms
│   │   ├── HLLSets.jl         # HLLSet class with similarity algorithms in Julia from SGS.ai
│   │   ├── HLLSets.py         # Wrapper of HLLSets.jl in Python
│   │   ├── indexer.py         # Shadow indexer (document index + morphisms)
│   │   ├── integration.py     # OCRWithHLLIndexing wrapper
│   │   ├── cortex.py          # Cortex category implementation
│   │   ├── fpga_optimized.py  # Hardware-accelerated version
│   │   └── memory_efficient.py # Memory optimizations
│   └── utils/
│       ├── semantic_tokenizer.py
│       └── memory_monitor.py
├── DeepSeek-OCR-master/       # Original upstream code
│   └── DeepSeek-OCR-vllm/     # vLLM-based implementation
│       ├── deepseek_ocr.py    # Main model (DeepseekOCRForCausalLM)
│       ├── config.py          # OCR configuration
│       ├── run_dpsk_ocr_image.py  # Image processing script
│       ├── run_dpsk_ocr_pdf.py    # PDF processing script
│       └── deepencoder/       # Vision encoders (SAM, CLIP)
├── examples/                  # Demonstration scripts
│   ├── demo_mock_ocr.py       # Mock OCR with HLLSet demo
│   ├── semantic_retrieval.py
│   ├── hllset_ocr_integration.py
│   └── advanced_retrieval.py
└── tests/
    └── test_hllset_integration.py
```

### **Data Flow**

1. **Interception**: All user-OCR communication is intercepted
2. **Conversion**: Text/images → Tokens → HLLSets
3. **Enhancement**: Queries are extended with Cortex context
4. **Integration**: OCR responses become new Cortex elements
5. **Evolution**: New layers are created with relationships

## 🔄 Workflow Details

### **2.1 Initial State: Empty Cortex**

When starting fresh:

```python
cortex = HLLSetCortex()  # Completely empty
# No HLLSets, no layers, no relationships
```

### **2.2 First Interaction Processing**

#### **Input**: User prompt + optional files

##### **Step 1: Conversion to Images**

```python
# Convert all input to images for consistent processing
prompt_image = convert_text_to_image(user_prompt)
file_images = [convert_to_image(file) for file in uploaded_files]
all_images = [prompt_image] + file_images
```

##### **Step 2: Tokenization & HLLSet Creation**

```python
# Use DeepSeek-OCR's tokenizer for consistency
tokens = deepseek_tokenizer.tokenize(images)
hllsets = [HLLSet.from_tokens(token_batch) for token_batch in tokens]

# Create initial Cortex layer
initial_layer = CortexLayer(hllsets, layer_id="initial")
cortex.add_layer(initial_layer)
```

##### **Step 3: OCR Processing with Enhanced Context**

```python
# Send original + HLLSet-enhanced query to OCR
enhanced_query = cortex.enhance_query(user_prompt, hllsets)
ocr_response = deepseek_ocr.process(enhanced_query, file_images)
```

##### **Step 4: Response Integration**

```python
# Convert OCR response to HLLSets and integrate
response_hllsets = convert_ocr_response_to_hllsets(ocr_response)
cortex.integrate_response(response_hllsets, initial_layer)
```

### **2.3 Subsequent Interactions**

#### **2.3.1 Query Enhancement Phase**

Before sending to DeepSeek-OCR:

```python
def enhance_query_with_cortex(user_input, cortex):
    # Convert user input to HLLSet
    input_hllset = HLLSet.from_text(user_input)
    
    # Find related HLLSets in Cortex
    related_hllsets = cortex.find_similar(input_hllset, top_k=5)
    
    # Extract semantic context from related HLLSets
    context = cortex.extract_context(related_hllsets)
    
    # Enhance original query
    enhanced_query = f"{user_input}\n\nContext: {context}"
    
    return enhanced_query, related_hllsets
```

#### **2.3.2 Layer Evolution**

```python
def process_interaction(user_input, files, cortex):
    # 1. Enhance query with Cortex context
    enhanced_query, context_hllsets = enhance_query_with_cortex(user_input, cortex)
    
    # 2. Process with DeepSeek-OCR
    ocr_response = deepseek_ocr.process(enhanced_query, files)
    
    # 3. Convert response to HLLSets
    response_hllsets = convert_ocr_response_to_hllsets(ocr_response)
    
    # 4. Create new Cortex layer
    new_layer = CortexLayer(
        elements=context_hllsets + response_hllsets,
        parent_layer=cortex.current_layer,
        relationships=establish_relationships(context_hllsets, response_hllsets)
    )
    
    # 5. Push layers down and add new layer
    cortex.push_layer(new_layer)
    
    return ocr_response, new_layer
```

## 🧩 Core Implementation Details

### **HLLSet Creation Pipeline**

```python
def create_hllsets_from_data(data):
    """Convert any data type to HLLSets"""
    if isinstance(data, str):
        # Text data
        tokens = tokenizer.tokenize_text(data)
    elif isinstance(data, Image):
        # Image data - use OCR to extract text first
        text = ocr_engine.extract_text(data)
        tokens = tokenizer.tokenize_text(text)
    elif isinstance(data, list):
        # Multiple items
        tokens = [token for item in data for token in tokenizer.tokenize(item)]
    
    return HLLSet.from_tokens(tokens)
```

### **Cortex Layer Management**

```python
class CortexLayer:
    def __init__(self, hllsets, parent_layer=None, relationships=None):
        self.hllsets = hllsets
        self.parent_layer = parent_layer
        self.relationships = relationships or []
        self.timestamp = datetime.now()
        self.layer_depth = 0 if parent_layer is None else parent_layer.layer_depth + 1

class HLLSetCortex:
    def __init__(self):
        self.layers = []  # Stack of layers, newest first
        self.entanglement_graph = EntanglementGraph()
    
    def push_layer(self, new_layer):
        """Add new layer and push previous layers down"""
        self.layers.insert(0, new_layer)  # Newest at index 0
        
        # Maintain maximum layer depth
        if len(self.layers) > MAX_LAYERS:
            self.layers.pop()  # Remove oldest layer
    
    def find_similar(self, query_hllset, top_k=5):
        """Find similar HLLSets across all layers"""
        similarities = []
        for layer in self.layers:
            for hllset in layer.hllsets:
                similarity = query_hllset.similarity_to(hllset)
                if similarity > SIMILARITY_THRESHOLD:
                    similarities.append((hllset, similarity, layer))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in similarities[:top_k]]
```

### **Entanglement Graph**

```python
class EntanglementGraph:
    """Manages relationships between HLLSets across layers"""
    
    def add_relationship(self, source_hllset, target_hllset, relationship_type, strength):
        self.graph.add_edge(
            source_hllset.id, 
            target_hllset.id,
            type=relationship_type,
            strength=strength,
            timestamp=datetime.now()
        )
    
    def get_context_paths(self, hllsets, max_hops=3):
        """Find contextual paths between HLLSets"""
        context_paths = []
        for hllset in hllsets:
            paths = nx.single_source_shortest_path_length(
                self.graph, hllset.id, cutoff=max_hops
            )
            context_paths.extend(paths)
        return context_paths
```

## 🔧 Development Roadmap

### **Phase 1: Core Interceptor** ✅

- [x] HLLSet implementation (Julia core + Python wrapper)
- [x] Basic Cortex layer structure
- [x] Data interception framework

### **Phase 2: Query Enhancement** 🚧

- [ ] Context extraction from Cortex
- [ ] Query enhancement algorithms
- [ ] Layer relationship establishment

### **Phase 3: Advanced Evolution**

- [ ] Multi-layer entanglement graphs
- [ ] Temporal relationship modeling
- [ ] Semantic context preservation

### **Phase 4: Production Ready**

- [ ] Performance optimization
- [ ] Memory management for large Cortices
- [ ] Integration with DeepSeek-OCR API

## 🎯 Usage Example

```python
from deepseek_ocr import DeepSeekOCR
from deepseek_ocr.hllset import HLLSetCortex

# Initialize systems
ocr = DeepSeekOCR()
cortex = HLLSetCortex()

# First interaction
user_prompt = "Explain this research paper about transformers"
files = ["research_paper.pdf"]

# Process through interceptor
response, new_layer = cortex.process_interaction(user_prompt, files)

print(f"Created Cortex layer: {new_layer.layer_depth}")
print(f"Added {len(new_layer.hllsets)} HLLSets to Cortex")

# Subsequent interaction - automatically enhanced
next_prompt = "What about attention mechanisms?"
next_response, next_layer = cortex.process_interaction(next_prompt, [])

# The second query was enhanced with context from the first interaction
```

## 🔬 Key Technical Insights

### **Why HLLSets?**

- **Constant-time similarity**: O(1) regardless of document size
- **Memory efficiency**: ~1KB per million documents
- **Mathematical foundation**: Based on HyperLogLog cardinality estimation
- **Hardware acceleration**: FPGA-ready architecture

### **Cortex Evolution Benefits**

1. **Contextual awareness**: Each query understands previous interactions
2. **Knowledge accumulation**: Cortex grows smarter with each use
3. **Semantic relationships**: Automatic discovery of conceptual connections
4. **Temporal understanding**: Layer structure captures evolution over time

## 🤝 Contributing

We welcome contributions to advance semantic document intelligence! Key areas:

- **HLLSet algorithm improvements**
- **Cortex layer optimization**
- **Query enhancement strategies**
- **Integration patterns with various OCR systems**

## 📚 Research Foundation

This work bridges:

- **HyperLogLog algorithms** for efficient similarity measurement
- **Category theory** for semantic relationship modeling
- **Entanglement graphs** for knowledge representation
- **Shadow indexing** for non-intrusive system enhancement

## References

1. <https://github.com/vllm-project/vllm>
2. <https://blocks.diy/website/home>
3. FPGA in Quantum Chips <https://www.msn.com/en-us/money/companies/amd-quietly-cracks-open-quantum-opportunity/ar-AA1PdFjn?ocid=hpmsn&cvid=65e4421733094cd7e26cbada26f4f90a&ei=17>
4. BI Explorer, BIE-1 <https://ysln.ycwb.com/content/2025-10/25/content_53753735.html>
5. <https://medium.com/ai-ai-oh/using-metadata-as-a-weapon-against-ai-fakes-635d0b6b01ec>
6. <https://matinmavaddat.substack.com/p/testing-systemic-qualities-understanding> <https://www.linkedin.com/pulse/why-good-metrics-dont-guarantee-systems-matin-mavaddat-8hmle/?trackingId=hzXlooAfTHGc8zUVOkxxlQ%3D%3D>

---
