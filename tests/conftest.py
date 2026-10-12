import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Clients are only constructed in these tests, never called
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")
