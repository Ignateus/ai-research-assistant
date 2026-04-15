### AI Research Assistant
A CLI + API tool that takes a topic or a set of documents and produces structured research reports. It demonstrates the skills needed for AI/ML platforms engineer.

**Prerequisites:**

Make sure you have Python 3.10+ installed:
```
python3 --version
```

**Step 1:**
Navigate to the project (ai-research-assistant).

**Step 2: Create and activate the virtual environment.**
```
python3 -m venv .venv
source .venv/bin/activate
```

**Step 3: Install dependencies**
```
pip install -e ".[dev]"
```

**Step 4: Set up your API key.**
```
cp .env.example .env
```

> NOTE: ANTHROPIC_API_KEY is needed, contact the owner of this repo for this key.

**Step 5: Run the tests (optional but good to verify).**
```
pytest tests/ -v
```

**Step 6: Start the assistant.**
```
research
```
