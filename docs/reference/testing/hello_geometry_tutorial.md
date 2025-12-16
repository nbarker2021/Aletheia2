
This Jupyter notebook guides users through basic CQE concepts via interactive code.

## 1. Generating a Simple Lattice
```python
import numpy as np
# Create a 2D square lattice
points = [(i,j) for i in range(-2,3) for j in range(-2,3)]
print(points)
```

## 2. Visualizing the Lattice
```python
import matplotlib.pyplot as plt
xs, ys = zip(*points)
plt.scatter(xs, ys)
plt.title('2D Square Lattice')
plt.show()
```

## 3. Simple Embedding Function
```python
# Toroidal projection for 2D
def toroidal_embed(pt, size=5):
    return (pt[0] % size, pt[1] % size)
embedded = [toroidal_embed(p) for p in points]
print(embedded)
```

## 4. CQE Concept: Root Systems
```python
# Define simple roots for A2 (hexagonal lattice)
roots = [(1,0), (-0.5,np.sqrt(3)/2),(-0.5,-np.sqrt(3)/2)]
for r in roots:
    plt.arrow(0,0,r[0],r[1], head_width=0.1)
plt.axis('equal'); plt.show()
```
