# Compass Demo

Interactive demo for [langgraph-compass](https://github.com/sardanaaman/langgraph-compass).

## Deploy to Hugging Face Spaces

1. **Create a new Space** at [huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose **Gradio** as the SDK
   - Name it something like `compass-demo`

2. **Clone your new Space**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/compass-demo
   cd compass-demo
   ```

3. **Copy the demo files**
   ```bash
   cp /path/to/langgraph-compass/demo/* .
   ```

4. **Add your OpenAI API key as a Space secret**
   - Go to Space Settings → Variables and secrets
   - Add `OPENAI_API_KEY` as a secret

5. **Push to deploy**
   ```bash
   git add .
   git commit -m "Initial demo"
   git push
   ```

6. Your demo will be live at `https://huggingface.co/spaces/YOUR_USERNAME/compass-demo`

## Local Development

```bash
cd demo
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
python app.py
```

Then open http://localhost:7860
