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


For this project, we use **Principal Component Analysis** or PCA, which finds the direction in our data along which the data varies the most. Basically, it turns a large set of correlated variables into a smaller set of uncorrelated new variables called principle components. For example:
    - Take a 2D Ellipse:
        - The long axis would ideally be the first principal component, being the direction of maximum variance.
        - The short axis would be the second principal component, being perpendicular the first and the second highest variance.
We will be utilizing Incremental Principal Component Analysis, which is a variant that updates the principal components as new patrches arrive without needing to keep the old ones.

1. PCA finds direction of maximum variance:
    1. Imagine if we have a big cloud of points floating in space and each point is a flattened image patch. You can't picture 4,096 dimensions so shrink it into 2D.
    2. PCA picks the arrow pointing along the direction the cloud is most spread out, which is PC1. Then it picks the next arrow perpendicular to PC1 along which the cloud is most spread out, becoming PC2.
2. Dimensionality reduction compresses correlated pixels into fewer meaningful numbers because pixels in an image aren't independent.
    1. Instead of describing each path by its 4,096 pixel values, you describe it by its coordinates along the top few principal components.
    2. If you keep 10 components, each patch becomes just 10 numbers. These 10 numbers are more meaningful because it gives you information about the characterestics of the patch.
3. Reconstruction Error
    1. Imagine you are standing in a room and someone asks you 'How close to the floor is this fly?'. To answer, you'd look straight down from the fly to the floor, mark the shadow and measure the distance from the fly to its shadow.
    2. That's what reconstruction does. 
        1. Projects the patch onto the subspace (finding the point on the surface that's closest to the patch)
        2. Measure the distance between the original patch and its shadow
    3. If the distance is newar 0, there is a tiny reconstruction error and a great match.
    4. The shadow is the model's attempt to rebuild the patch using only the top 10 principal components.