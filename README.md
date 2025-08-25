# LinearRegression

Implementation of a **scikit-learn-like linear regression model** with CPU and GPU support.

This model has been built **from scratch**, inspired by `sklearn.linear_model`.  
It is trained using **batch gradient descent**, either on **CPU** or **GPU** (depending on the user’s choice).  

It includes the following methods:
- `fit`
- `transform`
- `fit_transform`

In the `main.py` you can see a small implementation of the class

This can be consider as a side project in my AI learning path.

---

## Requirements

- Python 3.9+
- numpy
- matplotlib
- numba (with CUDA support for GPU execution)
For GPU support, make sure you have CUDA installed and a compatible NVIDIA driver

You can install the dependencies manually:

```bash
pip install numpy matplotlib numba
```

or by running
```bash
pip install -r requirements.txt
