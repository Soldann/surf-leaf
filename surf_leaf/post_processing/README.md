# Surf-LEAF/post-processing

This subfolder contains the post-processing algorithm used in the Surf-LEAF pipeline for cleaning up the mesh after running Advancing Front Reconstruction.
It performs automatic mesh cleaning, hole filling, smoothing, and simplification using [PyMeshLab](https://pymeshlab.readthedocs.io/en/latest/index.html).

## 1. Environment Setup
```bash
conda create -n mesh-fix python=3.10
conda activate mesh-fix
pip install pymeshlab
```

## 2. Usage Example
Quick start
```bash
python mesh_fix.py --infile mesh.ply
```
Or with full parameters
```bash
python mesh_fix.py --infile mesh.ply --outfile mesh_fixed.ply --alpha_fraction 0.002 --stepsmoothnum 3 --targetperc 0.5
```
**Parameter Description**

| Parameter         | Type   | Default           | Description                                                        |
|-------------------|--------|-------------------|--------------------------------------------------------------------|
| --infile          | str    | required          | Path to the input mesh file (e.g. mesh.ply)                        |
| --outfile         | str    | infile + `_fixed` | Path to save the fixed mesh                                        |
| --alpha_fraction  | float  | 0.001             | Ball size (fraction) for alpha wrapping                            |
| --stepsmoothnum   | int    | 1                 | Number of HC Laplacian smoothing iterations                        |
| --targetperc      | float  | 0.6               | Target percentage reduction for mesh simplification (0~1)          |

## 3. Mesh Fixing Pipeline
### 1. Remove isolated pieces
    Delete isolated connected components whose diameter is smaller than the specified constant.
### 2. Mesh smoothing ([Two Steps Smoothing](https://www.researchgate.net/publication/47861030_A_comparison_of_mesh_smoothing_methods))
    A feature preserving/enhancing fairing filter based on two stages:
    1. Normal Smoothing, where similar normals are averaged together.
    2. Vertex reposition, where vertices are moved to fit on the new normals.
### 3. Create a new mesh object using [Alpha Wrapping](https://doc.cgal.org/latest/Alpha_wrap_3/index.html#Chapter_3D_Alpha_wrapping)
### 4. [HC Laplacian Smoothing](https://onlinelibrary.wiley.com/doi/10.1111/1467-8659.00334)
    A mesh smoothing technique that reduces noise while minimizing shrinkage and deformation by pushing vertices back toward their original positions after smoothing.
### 5. Mesh simplification ([Quadric Edge Collapse Decimation](https://mgarland.org/files/papers/quadrics.pdf))
    Simplify a mesh using a quadric based edge-collapse strategy. A variant of the well known Garland and Heckbert simplification algorithm with different weighting schemes to better cope with aspect ration and planar/degenerate quadrics areas.

## References
* [PyMeshLab filter list](https://pymeshlab.readthedocs.io/en/latest/filter_list.html#filter-list)
