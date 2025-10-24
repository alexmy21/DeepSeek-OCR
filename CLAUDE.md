# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of DeepSeek-OCR that adds **HLLSet Cortex** - a lightweight shadow indexing system for semantic document intelligence. The project combines DeepSeek's OCR capabilities with HyperLogLog-based semantic similarity for conceptual document search and relationship discovery.

**Key Innovation**: Shadow indexing runs alongside OCR processing to enable semantic search without modifying the original OCR workflow.

## Repository Structure

```bash
DeepSeek-OCR/
├── deepseek_ocr/              # Main package
│   ├── ocr.py                 # Original OCR interface (currently stub)
│   ├── hllset/                # HLLSet Cortex implementation
│   │   ├── core.py            # HLLSet class with similarity algorithms
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

## Development Commands

### Environment Setup

```bash
# Install dependencies
pip install -e .

# Or using requirements.txt
pip install -r requirements.txt

# Verify GPU availability
python detect_gpus.py
```

### Running OCR

**With vLLM (original implementation):**

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Process single image
python run_dpsk_ocr_image.py

# Process PDF
python run_dpsk_ocr_pdf.py

# Batch processing
python run_dpsk_ocr_eval_batch.py
```

**Configuration**: Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

- `MODEL_PATH`: Path to DeepSeek-OCR model
- `INPUT_PATH` / `OUTPUT_PATH`: Input/output locations
- `BASE_SIZE` / `IMAGE_SIZE`: Resolution settings
- `CROP_MODE`: Enable/disable dynamic cropping
- `MAX_CROPS`: Max crop tiles (reduce if GPU memory limited)
- `PROMPT`: OCR prompt template

### HLLSet Cortex Development

```bash
# Test HLLSet core functionality
python deepseek_ocr/hllset/core.py

# Test shadow indexer
python deepseek_ocr/hllset/indexer.py

# Run mock OCR demo with semantic search
python examples/demo_mock_ocr.py

# Run tests
python -m pytest tests/
```

## Architecture Notes

### HLLSet Cortex Design

**HLLSet (deepseek_ocr/hllset/core.py)**:

- HyperLogLog-based probabilistic data structure for efficient similarity
- `from_text()`: Creates HLLSet from OCR text using semantic tokenization
- `similarity_to()`: Computes BSS (Binary Similarity Score) between HLLSets
- Tokenization includes: words, bigrams, and sentence-level hashes
- Parameters: `tau` (inclusion threshold), `rho` (exclusion threshold)

**Shadow Indexer (deepseek_ocr/hllset/indexer.py)**:

- Non-intrusive: Runs alongside OCR without modifying original workflow
- `add_document()`: Indexes OCR output and discovers relationships
- `query_similar()`: Semantic search across indexed documents
- Persists index to disk (`hllset_index/index.json`)
- Tracks morphisms (document relationships) as a category structure

**Integration Pattern**:

- `OCRWithHLLIndexing` wraps existing OCR models
- Shadow indexing happens automatically during `process()` calls
- Optional: `enable_shadow_indexing=True` parameter
- Currently uses mock OCR; designed for easy integration with actual DeepSeek-OCR

### DeepSeek-OCR Architecture

**Model**: `DeepseekOCRForCausalLM` (vLLM-based)

- Vision encoders: SAM (Segment Anything) + CLIP
- Projector: MLP connecting vision features to language model
- Supports dynamic image cropping for high-resolution documents
- Multi-modal processing with image tokens

**Image Processing**:

- Global view: Base resolution (e.g., 1024x1024)
- Local views: Cropped tiles for detail (e.g., 640x640)
- Dynamic tiling based on aspect ratio
- Special tokens: `<image>`, `<|grounding|>`, `<|ref|>`, `<|det|>`

### GPU Configuration

**Target Hardware**: NVIDIA RTX 3060 12GB

- Set `CUDA_VISIBLE_DEVICES=0` to use specific GPU
- Adjust `MAX_CROPS` in config.py based on GPU memory
- Use `detect_gpus.py` to verify GPU detection

**Memory Management**:

- OCR models: ~3-5GB VRAM
- Shadow indexing: CPU-based (minimal memory)
- Reduce batch size or `MAX_CROPS` if OOM occurs

## Key Implementation Details

### HLLSet Tokenization Strategy

The tokenization in `core.py` captures semantic meaning at multiple levels:

1. **Word tokens**: Basic vocabulary (3-20 character words)
2. **Bigrams**: Captures phrases and word relationships
3. **Sentence hashes**: Captures document-level semantic chunks

This multi-level approach enables semantic similarity even when exact word matches are absent.

### Shadow Indexing Workflow

1. OCR processes document → extracts text
2. Indexer creates HLLSet from text (parallel operation)
3. Compares with existing documents using BSS similarity
4. Creates morphisms for relationships exceeding threshold `tau`
5. Persists index and morphisms to disk

### Morphisms and Category Theory

The system treats documents as objects in a category:

- **Objects**: Documents (represented as HLLSets)
- **Morphisms**: Similarity relationships (when similarity > tau)
- Enables compositional reasoning about document relationships

## Integration Notes

### Current State

The package has:

- ✅ Complete HLLSet implementation (CPU-based)
- ✅ Shadow indexer with persistence
- ✅ Integration wrapper (`OCRWithHLLIndexing`)
- ✅ Demo with mock OCR
- ⚠️ Placeholder for actual DeepSeek-OCR integration

### Integrating Real OCR

To connect actual DeepSeek-OCR models:

1. Update `deepseek_ocr/ocr.py` with real model loading
2. Modify `integration.py` to use actual OCR:

```python
from deepseek_ocr import DeepSeekOCR  # Real implementation
ocr_model = DeepSeekOCR.from_pretrained("deepseek-ai/DeepSeek-OCR")
```

3. Replace mock processing in `examples/demo_mock_ocr.py`

### Model Paths

- Upstream model: `deepseek-ai/DeepSeek-OCR` (Hugging Face)
- Set `MODEL_PATH` in `config.py` to local path or HF repo
- Requires: vision encoders, projector, language model weights

## Common Development Tasks

### Adding New Similarity Metrics

Edit `deepseek_ocr/hllset/core.py`:

- Add method to `HLLSet` class
- Consider register-based comparisons
- Test with sample documents in `if __name__ == "__main__"` block

### Modifying Tokenization Strategy

Edit `HLLSet.tokenize_text()` in `core.py`:

- Adjust regex patterns for word extraction
- Add/remove n-gram levels (unigrams, bigrams, trigrams)
- Test impact on semantic similarity

### Optimizing Index Performance

Check `deepseek_ocr/hllset/memory_efficient.py` for:

- Sparse register storage
- Incremental index updates
- Batch processing optimizations

### GPU Optimization

Edit `config.py` for memory constraints:

- Lower `MAX_CROPS` (default: 6, max: 9)
- Reduce `MAX_CONCURRENCY` (default: 100)
- Set `BASE_SIZE` and `IMAGE_SIZE` to smaller values
- Disable cropping: `CROP_MODE = False`

## Troubleshooting

**GPU not detected**:

```bash
python detect_gpus.py
# Check CUDA installation: nvidia-smi
# Verify PyTorch CUDA: python -c "import torch; print(torch.cuda.is_available())"
```

**OOM errors**:

- Reduce `MAX_CROPS` in `config.py`
- Lower batch size
- Use smaller resolution (BASE_SIZE=640, IMAGE_SIZE=512)

**Index corruption**:

```bash
# Delete and rebuild index
rm -rf hllset_index/
python examples/demo_mock_ocr.py
```

**Import errors**:

```bash
# Reinstall package
pip install -e .
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific component
python deepseek_ocr/hllset/core.py
python deepseek_ocr/hllset/indexer.py

# Test GPU functionality
python detect_gpus.py
```

## Contributing to HLLSet Cortex

When adding features:

1. HLLSet core changes go in `deepseek_ocr/hllset/core.py`
2. Indexing logic goes in `indexer.py`
3. Add demos to `examples/` showing value
4. Update integration in `integration.py`
5. Test with mock OCR before real integration
6. Document mathematical foundations in docstrings

The architecture is designed to be non-intrusive: shadow indexing adds semantic capabilities without breaking existing OCR workflows.
