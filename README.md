Implementation of the Dowker-Rips complex introduced in [<em>Flagifying the Dowker Complex</em>](https://arxiv.org/abs/2508.08025).
The complex is implemented as a class named `DowkerRipsComplex` that largely follows the API conventions from `scikit-learn`.

---

__Example of running DowkerRipsComplex__

The following is an example of computing persistent homology of the filtered complex $`\left\{\mathrm{DR}_{\varepsilon}(X,Y)\right\}_{\varepsilon\in\mathbb{R}^{+}}`$, that is, of the Dowker-Rips complex with relations $`R_{\varepsilon}\subseteq X\times Y`$ defined by $`(x,y)\in R_{\varepsilon}`$ iff $`d(x,y)\leq\varepsilon`$ for $`\varepsilon\geq 0`$, and where $`X`$ and $`Y`$ are subsets of $`\mathbb{R}^{n}`$ equipped with the Euclidean norm.
In the following example, we refer to $`X`$ and $`Y`$ as vertices and witnesses, respectively.

```
>>> from drc import DowkerRipsComplex
>>> from sklearn.datasets import make_blobs
>>> X, y = make_blobs(
        n_samples=200,
        centers=[[-1, 0], [1, 0]],
        cluster_std=0.75,
        random_state=42,
    )
>>> vertices, witnesses = X[y == 0], X[y == 1]
>>> drc = DowkerRipsComplex()  # use default parameters
>>> persistence = drc.fit_transform([vertices, witnesses])
>>> persistence
[array([[0.07438909, 0.1733489 ],
        [0.08154549, 0.24042536],
        [0.17218398, 0.24239226],
        [0.13146845, 0.25247845],
        [0.16269606, 0.2926637 ],
        [0.10576964, 0.32222554],
        [0.1382231 , 0.358332  ],
        [0.07358199, 0.3740825 ],
        [0.39632082, 0.4189592 ],
        [0.24082384, 0.577262  ],
        [0.02419385,        inf]], dtype=float32),
 array([[0.5035793 , 0.55263996]], dtype=float32)]
```

The output above is a list of arrays, where the $`i`$-th array contains (birth, death)-times of homological generators in dimension $`i-1`$.
Validity of Dowker-Rips duality can be verified by swapping the roles of vertices and witnesses as follows.

```
>>> import numpy as np
>>> persistence_swapped = DowkerRipsComplex().fit_transform([witnesses, vertices])
>>> all(
        np.allclose(homology, homology_swapped)
        for homology, homology_swapped
        in zip(persistence, persistence_swapped)
    )
True
```

Any `DowkerRipsComplex` object accepts further parameters during instantiation.
A full description of these can be displayed by calling `help(DowkerRipsComplex)`.
These parameters, among other things, allow the user to specify persistence-related parameters such as the maximal homological dimension to compute or which metric to use.

Behind the scenes, `DowkerRipsComplex` computes the input matrix required for Ripser from a matrix of pairwise distances between vertices and witnesses.
This process is implemented in two ways, one relying on NumPy and one relying on Numba.
Since Numba creates some overhead stemming from the compilation of Python code into machine code, the NumPy-implementation is usually faster than the Numba-implementation on smaller data sets.
The NumPy-implementation, however, creates a large temporary array, which can lead to OOM errors on larger datasets, in which case the Numba-implementation must be used.
Setting `use_numpy=True` during instantiation of an instance of `DowkerRipsComplex` forces the use of the NumPy-implementation; in case of an OOM error, the computation will fall back on the Numba-implementation and a corresponding warning is raised.

---

__Installation and requirements__

The package can be installed via `pip` by running `pip install -U dowker-rips-complex`.

Required Python dependencies are specified in `pyproject.toml`.
Provided that `uv` is installed, these dependencies can be installed by running `uv pip install -r pyproject.toml`.
The environment specified in `uv.lock` can be recreated by running `uv sync`.

---

__Installing from PyPI for `uv` users__

```
$ uv init
$ uv add dowker-rips-complex
$ uv run python
>>> from drc import DowkerRipsComplex
>>> ...
```
