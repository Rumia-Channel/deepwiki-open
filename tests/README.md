# DeepWiki Tests

Test suite for the DeepWiki-Open project, organized by type and scope.

## Directory Structure

```
tests/
├── unit/                 # Unit tests — test individual components in isolation
│   ├── test_google_embedder.py          # Google AI embedder client tests
│   └── test_all_embedders.py            # All embedding backends (OpenAI, Google, Ollama, Bedrock)
├── integration/          # Integration tests — test component interactions
│   └── test_full_integration.py         # Full pipeline integration test
├── api/                  # API tests — test HTTP endpoints
│   └── test_api.py                      # API endpoint tests
├── run_tests.py          # Test runner script
└── __init__.py

test/                     # Additional standalone tests
└── test_extract_repo_name.py            # Repository name extraction tests
```

## Running Tests

### All Tests
```bash
python tests/run_tests.py
```

### Unit Tests Only
```bash
python tests/run_tests.py --unit
```

### Integration Tests Only
```bash
python tests/run_tests.py --integration
```

### API Tests Only
```bash
python tests/run_tests.py --api
```

### Individual Test Files
```bash
# Unit tests
python tests/unit/test_google_embedder.py
python tests/unit/test_all_embedders.py

# Integration tests
python tests/integration/test_full_integration.py

# API tests
python tests/api/test_api.py

# Standalone tests
python test/test_extract_repo_name.py
```

## Test Requirements

### Environment Variables
- `GOOGLE_API_KEY`: Required for Google AI embedder and Gemini model tests
- `OPENAI_API_KEY`: Required for OpenAI embedder and model tests
- `DEEPWIKI_EMBEDDER_TYPE`: Set to `openai`, `google`, `ollama`, or `bedrock` for embedder-specific tests

### Dependencies
All test dependencies are included in the main project requirements:
- `python-dotenv`: Loading environment variables
- `adalflow`: Core framework for embeddings and RAG
- `google-generativeai`: Google AI API client
- `openai`: OpenAI API client
- `requests`: HTTP API testing
- `pytest`: Test framework

## Test Categories

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Speed**: Fast (< 1 second per test)
- **Dependencies**: Minimal external dependencies
- **Examples**: Embedder response parsing, configuration loading, repo name extraction

### Integration Tests
- **Purpose**: Test how components work together
- **Speed**: Medium (1-10 seconds per test)
- **Dependencies**: May require API keys and external services
- **Examples**: End-to-end embedding pipeline, RAG workflow

### API Tests
- **Purpose**: Test HTTP endpoints and WebSocket connections
- **Speed**: Medium-slow (5-30 seconds per test)
- **Dependencies**: Requires running API server on port 8001
- **Examples**: Chat completion endpoints, streaming responses, health checks, wiki cache CRUD, auth validation

## Adding New Tests

1. **Choose the right category**: Determine if your test is unit, integration, or API
2. **Create the test file**: Place it in the appropriate subdirectory under `tests/`
3. **Follow naming convention**: `test_<component_name>.py`
4. **Add proper imports**: Use the project root path setup pattern
5. **Document the test**: Add docstrings explaining what the test does

## Troubleshooting

### Import Errors
If you get import errors, ensure the test file includes the project root path setup:

```python
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

### API Key Issues
Make sure you have a `.env` file in the project root with the required API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
DEEPWIKI_EMBEDDER_TYPE=openai
```

### Server Dependencies
For API tests, ensure the FastAPI server is running on the expected port:

```bash
python -m api.main
```
