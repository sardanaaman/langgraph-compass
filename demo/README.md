---
title: Compass Demo
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.3.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Intelligent follow-up question generation for agents
---

# Compass Demo

Interactive demo for [langgraph-compass](https://github.com/sardanaaman/langgraph-compass).

## Automated Deployment (Recommended)

The demo auto-deploys to Hugging Face Spaces when you push changes to the `demo/` folder.

### One-Time Setup

1. **Create a Hugging Face Space**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose **Gradio** as the SDK
   - Name it `compass-demo` (or whatever you prefer)
   - Set visibility (public recommended for discoverability)

2. **Create a Hugging Face Token**
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token with **Write** access
   - Copy the token

3. **Add Secrets/Variables to GitHub**

   Go to your GitHub repo → Settings → Secrets and variables → Actions

   **Secrets** (click "New repository secret"):
   - `HF_TOKEN`: Your Hugging Face write token
   - `OPENAI_API_KEY`: Your OpenAI API key (also add this to HF Space secrets)

   **Variables** (click "Variables" tab → "New repository variable"):
   - `HF_USERNAME`: Your Hugging Face username
   - `HF_SPACE_NAME`: `compass-demo` (or your Space name)

4. **Add OpenAI Key to HF Space**
   - Go to your Space → Settings → Variables and secrets
   - Add `OPENAI_API_KEY` as a secret

5. **Initial Push**
   - Run the workflow manually (Actions → "Sync Demo to Hugging Face Space" → "Run workflow")
   - Or push any change to `demo/`

Now any push to `demo/` on main will auto-sync to your Space!

## Manual Deployment

If you prefer to manage the Space separately:

1. **Clone your Space**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/compass-demo
   cd compass-demo
   ```

2. **Copy demo files**
   ```bash
   cp /path/to/langgraph-compass/demo/* .
   ```

3. **Push to deploy**
   ```bash
   git add .
   git commit -m "Update demo"
   git push
   ```

## Local Development

```bash
cd demo
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
python app.py
```

Then open http://localhost:7860

## Files

- `app.py` - Gradio application with two tabs (Try It, Compare Strategies)
- `requirements.txt` - Python dependencies
- `README.md` - This file (not used by Gradio)
