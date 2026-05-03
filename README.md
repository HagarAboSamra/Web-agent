# Web-agent
A modular pipeline that takes a URL and a natural language query, then predicts
which element on the page to interact with, and whether to use a cursor click or keyboard input.

# How it works

1. **screenshot.py** — opens the URL in a headless Chrome browser and captures a `210 × 160 × 3` RGB array in memory
2. **image_processing.py** — feeds the array into `Global_CNN`, a 6-layer network (5 conv + avg pool + 2 FC) that produces a 384-dim visual embedding
3. **DOM_elements.py** — extracts every DOM node from the page with its bounding box, XPath, text, and interaction flags
4. **feature_map.py** — scores each node against the query using an 11-feature text similarity matrix (exact match, token overlap, tag weight, etc.)
5. **local_cnn.py** — `Local_CNN` fuses the visual embedding with per-node DOM features to predict pixel location `(cx, cy)` and interaction type
6. **env.py** — orchestrates all five modules with a single browser session and parallel scoring

# Installation
pip install torch numpy selenium pillow chromedriver-autoinstaller

## Notes

- Models are randomly initialised — predictions improve once trained on labelled interaction data
- Set `wait_seconds=2.5` for heavy JS/SPA pages if the DOM comes back incomplete
- Set `visible_only=False` to include hidden elements in scoring
