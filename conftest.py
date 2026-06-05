"""
Root conftest.py — adds all service src directories to sys.path.

This is the correct monorepo pattern for pytest: one root conftest,
no __init__.py in test directories, no per-service conftest path hacks.
"""
import sys
from pathlib import Path

root = Path(__file__).parent

# Internal packages
sys.path.insert(0, str(root / "packages" / "raglab-common" / "src"))
sys.path.insert(0, str(root / "packages" / "raglab-chunkers" / "src"))
sys.path.insert(0, str(root / "packages" / "raglab-retrievers" / "src"))

# All service src directories
_service_src_dirs = [
    str(root / 'services' / 'api-gateway' / 'src'),
    str(root / 'services' / 'auth' / 'src'),
    str(root / 'services' / 'config' / 'src'),
    str(root / 'services' / 'embedding' / 'src'),
    str(root / 'services' / 'graph' / 'src'),
    str(root / 'services' / 'indexing' / 'src'),
    str(root / 'services' / 'ingestion' / 'src'),
    str(root / 'services' / 'llm' / 'src'),
    str(root / 'services' / 'observability' / 'src'),
    str(root / 'services' / 'pipeline' / 'src'),
    str(root / 'services' / 'retrieval' / 'src'),
    str(root / 'services' / 'storage' / 'src'),
    str(root / 'services' / 'ui' / 'src'),
]

for d in _service_src_dirs:
    if d not in sys.path:
        sys.path.insert(0, d)
