# check_structure.py (place at DeepSeek-OCR/ root)
import os

def verify_structure():
    root = os.getcwd()
    print(f"Current directory: {root}")
    
    required_dirs = [
        'deepseek_ocr',
        'examples', 
        'tests',
        'docs'
    ]
    
    required_files = [
        'deepseek_ocr/__init__.py',
        'deepseek_ocr/hllset/__init__.py',
        'deepseek_ocr/hllset/core.py',
        'examples/semantic_retrieval.py',
        'tests/test_hllset_integration.py',
        'docs/semantic_indexing.md'
    ]
    
    print("\n📁 Checking directory structure...")
    for dir_path in required_dirs:
        exists = os.path.isdir(dir_path)
        status = "✅" if exists else "❌"
        print(f"{status} {dir_path}/")
    
    print("\n📄 Checking required files...")
    for file_path in required_files:
        exists = os.path.isfile(file_path)
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
    
    # Check if we're in the right place
    if os.path.basename(root) == 'deepseek_ocr':
        print("\n⚠️  WARNING: You're inside 'deepseek_ocr/' folder!")
        print("   Move up one level to the repository root.")
    else:
        print("\n🎉 Structure looks good!")

if __name__ == "__main__":
    verify_structure()