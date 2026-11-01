# ANN Model Explorer — Dark Streamlit UI

An interactive, dark-themed, animated Streamlit dashboard for your Keras/TensorFlow ANN.

Your model's architecture (auto-detected from `ANN_model.h5`):
`Input(4) → Dense(16, relu) → Dense(8, relu) → Dense(3, softmax)`

This shape (4 inputs, 3-class softmax output) matches the classic Iris dataset, so the
UI ships with Iris-style feature sliders and class labels by default (Setosa / Versicolor /
Virginica). **If your model was trained on different data**, just edit these four lines near
the top of `app.py`:

```python
FEATURE_NAMES = [...]     # your input feature names
FEATURE_RANGES = [...]    # (min, max, default) per feature
CLASS_NAMES = [...]       # your output class names
CLASS_EMOJIS = [...]      # optional emoji per class
```

## Setup

1. Put `ANN_model.h5` in the same folder as `app.py` (already done in this bundle).
2. Create a virtual environment (recommended) and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Your browser will open at `http://localhost:8501` with the dark UI.

## Features

- 🎨 Animated dark theme: gradient background, floating particles, glowing title
- 🎛️ Interactive sliders for each input feature
- 🚀 Animated "thinking" progress bar on prediction
- 📊 Animated probability bars + interactive Plotly bar chart
- 🎈 Confetti/balloons animation on high-confidence predictions
- 🕓 Session prediction history table
- ⚙️ Sidebar showing model architecture & parameter count

## Notes

- The dark theme also comes from `.streamlit/config.toml` — keep that folder alongside `app.py`.
- If `tensorflow` is heavy for your environment, you can swap the loader to `tf.keras.models.load_model` with `compile=False` for faster startup (already effectively used since we don't retrain).
