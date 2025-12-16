# Deployment Guide: CQE Production System

**Version**: 1.0.0

This guide provides instructions for deploying and running the CQE Production System.

## 1. System Requirements

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows (WSL recommended)
- **Dependencies**: None. The system is 100% stdlib-only.

## 2. Deployment Options

The system is designed for three primary deployment targets:

1.  **Local Execution**: Run directly from the command line for analysis and research.
2.  **API Server**: Expose the CQE framework as a REST API.
3.  **Web Application**: Integrate into a web application for interactive use.

## 3. Local Deployment (Quick Start)

This is the simplest way to run the system.

1.  **Unpack the Archive**: Unzip the `CQE_PRODUCTION.zip` archive to your desired location.

2.  **Navigate to Directory**:
    ```bash
    cd /path/to/CQE_PRODUCTION
    ```

3.  **Run the Main Entry Point**:
    ```bash
    python3.11 main.py
    ```

    This will initialize the system and confirm that all components are correctly loaded.

## 4. API Server Deployment

To deploy as an API, you can use a simple web server framework like Flask or FastAPI (which would need to be installed).

**Example `api_server.py` (requires `flask`)**:

```python
from flask import Flask, jsonify, request
# Import CQE components
from core import e8_lattice
from tools import speedlight

app = Flask(__name__)

@app.route('/api/v1/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    # ... use CQE tools to process data ...
    result = speedlight.process(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 5. Initializing MonsterMoonshineDB

The included `monster_moonshine_db` can be initialized in a local SQLite database.

1.  **Install SQLite** (if not already present).

2.  **Create the Database**:
    ```bash
    sqlite3 data/monster_moonshine.db < data/monster_moonshine_db/schema.sql
    ```

3.  **Import Embeddings and Relationships**:
    ```bash
    sqlite3 data/monster_moonshine.db < data/monster_moonshine_db/import_embeddings.sql
    # Use .import for CSV
    sqlite3 data/monster_moonshine.db ".mode csv" ".import data/relationships.csv relationships"
    ```

This provides a fully searchable knowledge base for the Aletheia AI engine.
