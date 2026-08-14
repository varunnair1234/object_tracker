# Object Tracker Project

A real-time visual object tracker built in Python using **OpenCV** and **PyTorch**, using incremental **Principal Component Analysis (PCA)** to learn an object's variation patterns from history and locate it in each new frame.

The user draws a box around an object on the first frame, and the tracker follows it across every subsequent frame — adapting online to changes in lighting, rotation, and other appearance variations.

---

## Table of Contents

- [Overview](#overview)
- [The Pipeline](#the-pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [Code Walkthrough](#code-walkthrough)
  - [Setup and Imports](#setup-and-imports)
  - [Frame 1 Setup](#frame-1-setup)
  - [The Appearance Model](#the-appearance-model)
  - [Preprocessing](#preprocessing)
  - [Generating Candidates](#generating-candidates)
  - [Scoring and Winner Selection](#scoring-and-winner-selection)
  - [The Update Method](#the-update-method)
  - [PCA Computation](#pca-computation)
  - [Reconstruction Error in Detail](#reconstruction-error-in-detail)
- [Design Decisions](#design-decisions)
- [Common Q&A](#common-qa)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)


## Overview

This project is a **classical visual tracker** — no neural networks, no training data, no backpropagation. It uses PCA on a rolling buffer of past winning patches to learn the object's *subspace of appearances*, then locates the object in each new frame by finding the candidate patch that best fits inside that subspace.

**Why PCA?** Because the object doesn't have one fixed appearance — it changes over time due to lighting, rotation, expression, etc. PCA captures the *patterns of variation* the object has shown, letting the tracker recognize the object even as it changes.


## The Pipeline

1. Import all libraries and setup NumPy, PyTorch, and OpenCV.
2. The tracker is a `while` loop that updates in increments to see where the object went.
3. Within the first frame, a rectangle is drawn around the object using OpenCV.
4. The patch of the object is extracted and used to initialize the appearance model.
   - Compare the appearance model with candidate patches.
   - Establish a scoring system to decide which candidate patch has the lowest reconstruction error.
   - Out of all the candidates, pick the patch with the lowest score (best match).
   - Update the appearance model and repeat for the length of the video.


## Project Structure

```
object_tracker_project/
├── loop.py          # runtime: opens webcam, runs main loop, draws box
├── model.py         # AppearanceModel class (initialize, score, update)
├── patches.py       # preprocess() and generate_candidates()
├── requirements.txt
└── README.md
```


## Installation

```bash
# Clone the repo
git clone <repo-url>
cd object_tracker_project

# Create a virtual environment
python -m venv virtualenv
source virtualenv/bin/activate     # macOS/Linux
# virtualenv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

Dependencies:
- `opencv-python`
- `torch`
- `numpy`


## Usage

```bash
python loop.py
```

1. A window opens showing your webcam feed.
2. Drag a rectangle around the object you want to track.
3. Press **Space** or **Enter** to confirm.
4. The tracker follows the object with a green box.
5. Press **q** to quit.


## How It Works

### Principal Component Analysis (PCA)

PCA finds the directions in your data along which the data varies the most. It turns a large set of correlated variables into a smaller set of uncorrelated new variables called **principal components**.

**Example — a 2D ellipse:**
- The long axis is the first principal component — the direction of maximum variance.
- The short axis is the second principal component — the direction of second-highest variance.

**In this tracker, we use *Incremental PCA*** — a variant that updates the principal components as new patches arrive, without needing to keep old ones.

1. **PCA finds the direction of maximum variance.**
   - Imagine a big cloud of points floating in space. Each point is a flattened image patch. In 1024-dim space (too many dimensions to picture — shrink it to 2D mentally).
   - PCA picks the arrow pointing along the direction the cloud is most spread out — that's PC1. Then it picks the next arrow perpendicular to PC1 along which the cloud is most spread out — that becomes PC2.

2. **Dimensionality reduction** compresses correlated pixels into fewer meaningful numbers.
   - Pixels in an image aren't independent.
   - Instead of describing each patch by its 1024 pixel values, you describe it by its coordinates along the top few principal components.
   - If you keep 10 components, each patch becomes just 10 numbers — a compressed description that captures the meaningful structure.

3. **Reconstruction Error.**
   - Imagine you're standing in a room and someone asks: "How close to the floor is this fly?" You'd look straight down from the fly to the floor, mark the shadow, and measure the distance from the fly to its shadow.
   - That's what reconstruction does:
     - **Project** the patch onto the subspace (find the closest point on the subspace to the patch).
     - **Measure** the distance between the original patch and its shadow.
   - If the distance is near zero → tiny reconstruction error → great match.
   - The shadow is the model's attempt to rebuild the patch using only the top 10 principal components.


## Code Walkthrough

### Setup and Imports

```python
import cv2
import numpy as np
from model import AppearanceModel
import torch
from patches import preprocess, generate_candidates

webcam = cv2.VideoCapture(0)     # starts up the webcam

if webcam.isOpened():
    print('yes')
else:
    print('error')
    exit()

for _ in range(10):
    webcam.read()                # macOS webcam warmup
```

This block handles imports and checks whether the webcam is working. If it is, prints `'yes'`. If not, exits. We use **OpenCV, NumPy, and PyTorch** as the modules for this project.

### Frame 1 Setup

```python
x, y, w, h = cv2.selectROI('Select Object', frame)
patch = frame[y:y+h, x:x+w]
patch_arr = preprocess(patch)
patch_tensor = torch.tensor(patch_arr, dtype=torch.float32)

model = AppearanceModel(10)
model.initialize(patch_tensor)
```

This is one of the most important parts of the project. The webcam opens and we use it to track an object (my face) throughout the screen.

We use `cv2.selectROI` which lets us draw a rectangle around the object. We then calculate the image patch using the top-left coordinates from `selectROI` (`x, y`) and the width and height of the rectangle. In the frame:
- `y` = top-left coordinate (vertical)
- `y + h` = bottom coordinate
- `x` = top-left coordinate (horizontal)
- `x + w` = top-right coordinate
- Origin is at the top-left of the screen in computer graphics.

### Preprocessing — what does `preprocess` do?

When we compute the raw patch, we get three dimensions: `h, w, 3` — the number of rows, columns, and color channels (BGR for OpenCV). The shape is 3-dimensional because we have height, width, and color — but **PCA operates only on 1D vectors, not 3D**.

PCA has operations like mean subtraction, matrix multiplication, and SVD, which all require 1D or 2D inputs. The raw patch is also large, so we reduce it to 1024 numbers, which is a medium ground. We also convert it to a tensor because we use PyTorch.

### The Appearance Model

```python
class AppearanceModel():
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.singular_values = None
        self.n_samples = 0
        self.buffer = []
        self.buffer_size = 30

    def initialize(self, patch):
        self.mean = patch
        self.n_samples = 1

    def score(self, patch):
        mean_center = patch - self.mean       # 1024-dim vector
        return torch.norm(mean_center)         # magnitude of mean_center
```

We initialize the model with **10 components** (10 principal components). Many values are set to `None` until we update them later.

**Buffer** is a list that contains winning patches. With each new frame, there's a new winning patch, so we use a **FIFO (first-in, first-out)** structure to pop the first value and make space for the newest one.

**When initializing**, we set the mean as the patch itself since there's only one patch to work with. `initialize` puts real data into the model and runs when the tracker has an initial patch.

**Score method:** takes a patch tensor as input, calculates how far each pixel is from the mean across all 1024 dimensions, and stores it in `mean_center`. If `mean_center` is near zero, the patch is very close to the mean. If it's higher, the candidate is very different. We then compute the length of the vector using `torch.norm` — the Euclidean distance through PyTorch.

*Note:* Euclidean distance is a weak signal for scoring, but works fine as a fallback if the object hasn't changed much (used before PCA has enough data).

### Preprocessing

```python
def preprocess(patch):
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)   # convert to grayscale
    patch = cv2.resize(patch, (32, 32))               # PCA requires fixed shape
    patch = patch.flatten()                            # 1D vectors for PCA
    return patch
```

We convert the patches to **grayscale** (easier for the model due to reduced lighting-shift sensitivity). We resize to **32×32** because PCA requires equal dimensions across patches. We then flatten into a 1D vector for PCA and return the patch. 32×32 is the perfect middle ground: small enough to be fast, large enough to preserve appearance.

### Generating Candidates

```python
def generate_candidates(frame, x, y, w, h, search_radius=15, stride=4):
    candidates = []
    for dx in range(-search_radius, search_radius + 1, stride):
        for dy in range(-search_radius, search_radius + 1, stride):
            cx = x + dx
            cy = y + dy
            patch = frame[cy : cy+h, cx : cx+w]
            if patch.shape != (h, w, 3):
                continue
            candidates.append([cx, cy, patch])
    return candidates
```

**Visualization:**

```
┌────────────────────────────────────────────┐  ← frame
│                                            │
│       ┌────────────────────────────┐       │
│       │ • • • • • • • •            │       │
│       │ • • • • • • • •            │       │
│       │ • • ┌──────────┐ • •       │       │  ← candidates
│       │ • • │  object  │ • •       │          in green
│       │ • • │  (face)  │ • •       │
│       │ • • └──────────┘ • •       │
│       │ • • • • • • • •            │       │
│       │ • • • • • • • •            │       │
│       └────────────────────────────┘       │
│                                            │
└────────────────────────────────────────────┘
```

Each green dot represents the top-left corner of a candidate box. Candidates cover roughly the same area, differing by small position shifts.

We use a **nested for loop** to search within each candidate offset — for example, from -15 to 15 with a stride of 4 pixels (which represents how many pixels separate each candidate point). The nested loops iterate through vertical and horizontal offsets, `dy` and `dx`.

- `cx, cy` = absolute positions of the candidates, computed by summing the initial positions `x, y` with the offsets `dx, dy`.
- The candidate patch is cropped using the same method as the initial patch: adding the absolute position to the width/height of the image patch.
  - `cy : cy + h` = top edge → bottom edge
  - `cx : cx + w` = left edge → right edge

We also add a check to see if the candidate falls partially off the frame edge. `continue` jumps to the next loop iteration, skipping partial candidates — so sometimes we get fewer than 64 candidates. We append each valid candidate as a triple `[cx, cy, patch]` to the candidates list.

### Scoring and Winner Selection

```python
# Scoring Section
candidates = generate_candidates(frame, x, y, w, h)

scored = []

for cx, cy, patch in candidates:
    patch = preprocess(patch)
    patch_tensor = torch.tensor(patch, dtype=torch.float32)
    score = model.score(patch_tensor)
    scored.append((cx, cy, score, patch_tensor))

winner = min(scored, key=lambda item: item[2])
x, y, winner_tensor = winner[0], winner[1], winner[3]

cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

We create the `patch_tensor` and score the tensor based on the Euclidean length of the candidate patches from the mean. **Winner is calculated using a lambda function**, and then we derive `x, y`, and `winner_tensor` from `winner`, which returns 4 items.

**When we reassign `x, y`, we're updating the tracker's position for the next frame.** Because `x, y` were declared outside the loop, each iteration overwrites them with the next position. `generate_candidates` looks at the new position based on the `x, y` from `winner` — **that's how the tracker follows the object.**

We also use `cv2.rectangle` to draw a rectangle on the frame at the new position, with `(x, y)` as the top-left corner and `(x+w, y+h)` as the bottom-right corner.

### The Update Method

```python
def update(self, patch):
    buffer = self.buffer
    buffer.append(patch)
    if len(buffer) > self.buffer_size:
        buffer.pop(0)
    
    # PCA computation
    if len(buffer) >= self.n_components + 1:
        stacked = torch.stack(buffer)
        self.mean = torch.mean(stacked, dim=0)
        centered = stacked - self.mean
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        self.components = Vt[:self.n_components]
        self.singular_values = S[:self.n_components]
```

`update` is probably the most important part of this tracker. It creates a **buffer of the last 30 winning patches**, and appends each winning patch to the buffer. We keep the buffer at a fixed length of 30 elements so it doesn't take up too much memory. We use FIFO to pop the first element to make space for the newest.

### PCA Computation

I wanted to dive deeper into PCA computation because it's probably the most important part of the project.

**We start running PCA only if there are 11 or more patches in the buffer.** To find `n` principal directions, we need at least `n + 1` data points. With `n_components = 10`, we need 11 patches. PCA would produce nonsense with fewer than 11 patches.

**We convert the buffer into a single 2D matrix** of shape `(N, 1024)` with N tensors, each of shape `1024` values (remember 32 × 32 = 1024). We name this variable `stacked` — we need it as a matrix so we can perform matrix operations. Each row of `stacked` is one patch, and each column is one of the 1024 pixel positions across all patches.

**Then we compute the mean** using PyTorch's built-in `torch.mean`. We get a 1D tensor of 1024 values, where each value is the mean of all N winning patches at that pixel position.

**We mean-center the data** — subtracting the mean vector from every row of the stacked matrix. `centered` is an `(N, 1024)` matrix where each row is a patch minus the mean. Centering means moving the data cloud to the center of the graph, with the mean positioned at the origin. **Then PCA can worry about the direction of *variance* rather than the direction of *location*.**

**After centering, we run Singular Value Decomposition (SVD)** on the centered matrix. The operation returns three tensors:
- **U**: shape `(N, N)`, describes how each patch is expressed as coefficients along the principal directions.
- **S**: singular values in decreasing order. `S[i]` tells us how much variance direction `i` captures.
- **Vt**: shape `(N, 1024)`, each row is a principal direction in 1024-dim space. `Vt[0]` is the direction of maximum variance — the first principal component.

**We get the first 10 rows of Vt** — the 10 most important directions of variance. These directions are used through the `score` method to compute reconstruction error, asking how close we are to the candidate in the subspace.

**We also slice the top 10 singular values** — the importance of our 10 directions. While Vt gives us the directions themselves, singular values tell us the importance of each direction.

### Reconstruction Error in Detail

```python
def score(self, patch):
    mean_center = patch - self.mean
    if self.components is None:
        return torch.norm(mean_center)
    
    coefficients = self.components @ mean_center
    reconstruction = coefficients @ self.components
    residual = mean_center - reconstruction
    return torch.norm(residual)
```

This is the reconstruction error part.

- `self.components` is a matrix of shape `(10, 1024)` — 10 of the top principal directions, each 1024 pixels.
- `mean_center` has shape `(1024,)`, representing how far each candidate patch is from the mean.
- **Multiplying components with mean_center gives us a matrix of shape `(10,)`** — the **projection coefficients**, or how much of the patch's variation lies along each principal direction.
- The 10 numbers are a **compressed description** of the candidate — 10 coordinates instead of 1024.

**Shape reference:**

| Variable | Shape | Meaning |
|---|---|---|
| `self.mean` | `(1024,)` | Average value of each pixel position across recent patches |
| `stacked` | `(N, 1024)` | N rows (one per patch) × 1024 columns (one per pixel position) |
| `mean_center` | `(1024,)` | How far each candidate patch is from the mean |
| `reconstruction` | `(1024,)` | The 1024-dim rebuilt version of the candidate patch |
| `residual` | `(1024,)` | The 1024-dim difference between the original and the reconstruction |

**Reconstruction** = `coefficients @ components`. Multiplying coefficients (shape `(10,)`) with components (shape `(10, 1024)`) gives us a matrix of 1024 values (1D). This operation takes the 10 coefficients and, for each principal direction, multiplies the direction with its respective coefficient and sums them. **That is the reconstruction of the winning patch.**

**Residual** is a `(1024,)` vector representing the information that was lost in the compression. `torch.norm` calculates the length of the vector from the origin to the residual point.

**In simpler words:**
1. **Matrix multiplication 1:** Describes the candidate using 10 numbers along the principal directions.
2. **Matrix multiplication 2:** Recreates the candidate using the 10 numbers.
3. **Norm:** Measures how much the recreation missed.



## Design Decisions

**Grayscale + 32×32:** small enough for fast PCA, large enough to preserve the object's identity. Grayscale is more robust to lighting changes than color.

**64 candidates per frame:** search radius 15, stride 4. Search radius 15 covers realistic frame-to-frame motion; stride 4 keeps the count small enough for real-time scoring.

**10 principal components:** enough to capture real variation, restrictive enough to reject background clutter. Too few → subspace too rigid; too many → subspace too flexible, no discrimination.

**Buffer size 30:** enough samples for stable PCA (3× more than components), recent enough to stay adaptive.

**Store only winners in the buffer:** clean data for PCA. Storing all candidates would poison the subspace with background patterns.

**PCA reruns every frame:** ~1 ms cost per frame, keeps the subspace fresh as the object changes.


## Common Q&A

- **What's this project?** *"A real-time visual tracker built in Python with OpenCV and PyTorch, using incremental PCA to learn an object's variation patterns from history."*
- **What does PCA do?** *"Finds the directions along which the data varies most. In this tracker, those directions describe how the object's appearance changes over time — like lighting or rotation."*
- **Why center the data?** *"PCA measures from the origin, but variance is really about spread around the mean. Centering shifts the mean to the origin so PCA finds shape, not location."*
- **What's reconstruction error?** *"The residual after projecting a candidate onto the PCA subspace and rebuilding it — measures how well the candidate fits within the object's known variations."*
- **Why only 10 components?** *"Enough to capture real variation, restrictive enough to reject background clutter. Too many components make the subspace too flexible."*
- **Why a 30-patch buffer?** *"Enough samples for stable PCA, recent enough to stay adaptive."*
- **Why store only winners in the buffer?** *"Storing all candidates would poison PCA with background patterns; the subspace would learn to represent clutter instead of the object."*
- **Why not just Euclidean distance?** *"Distance measures against a single reference point (the mean). Reconstruction error measures against the whole subspace of allowed variations — much more powerful for objects that change."*

---

## Limitations

1. **Fast motion** — the object can't move more than 15 pixels per frame; otherwise it falls outside the search radius.
2. **Drift** — if the tracker briefly slips, a background patch enters the buffer and can corrupt PCA over time.
3. **Occlusion** — briefly hiding the object causes the tracker to lock onto background.
4. **Similar-looking distractors** — a second face could match the same subspace.
5. **Scale changes** — the box size is fixed, so objects moving closer/farther can be lost.
6. **Cold start** — frames 1–10 use the weaker Euclidean fallback before PCA kicks in.

---

## Future Improvements

1. **Confidence-based updates** — skip adding a patch to the buffer if the winner's score is unusually high, preventing drift.
2. **True incremental PCA** — the Sequential Karhunen-Loève (SKL) algorithm from Ross et al. (2008), which updates the subspace without full SVD recomputation.
3. **Multi-scale candidates** — generate candidates at 0.9× and 1.1× the box size to handle scale changes.
4. **Motion prediction** — use a Kalman filter to predict where the object will be, letting the search radius shrink while catching faster movement.
5. **Occlusion recovery** — detect when reconstruction error stays high across many frames and pause updates until re-acquisition.

---

## Credits

Based on the ideas from **Ross et al. (2008), *"Incremental Learning for Robust Visual Tracking"*** — a foundational paper in classical visual tracking with subspace-based appearance models.
