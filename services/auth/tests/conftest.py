"""Configure sys.path for auth-service tests (R7)."""
import sys, os

# Auth service source
auth_src = os.path.join(os.path.dirname(__file__), "..", "src")
api_gw_src = os.path.join(os.path.dirname(__file__), 
    "..", "..", "api-gateway", "src")

for p in [auth_src, api_gw_src]:
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)
