# Pipeline for Object Tracker 

### Stage 1: Imports

Import: NumPy, PyTorch, OpenCV, Requests

```python
import numpy as np
import torch
import opencv
import requests
```

### Stage 2


The tracker become a while loop that keeps udpating in increments to see where the objects went.
1. Within the first frame, a rectange is drawn using OpenCV.
2. The image patch of the object is extracted and used to initialize the appearance model.
    1. Within this, a while loop is created which compares the appearance model with candidate patches.
    2. We establish a scoring system to decide which candidate patch has the highest score.
    3. Out of all the patches closest to the candidate patch, we pick the patch with the highest score.
    4. Update the appearance model and repeat for the length of the entire video.
This is the pipeline of the tracker. Very simple.


### Explanations


