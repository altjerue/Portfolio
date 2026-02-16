# Jesús M. Rueda-Becerril | Data Science Portfolio

[![LinkedIn](https://img.shields.io/badge/LinkedIn-jeruebe-blue)](https://www.linkedin.com/in/jeruebe)
[![Website](https://img.shields.io/badge/Website-altjerue.github.io-green)](https://altjerue.github.io)
[![DataCamp](https://img.shields.io/badge/DataCamp-Portfolio-orange)](https://www.datacamp.com/portfolio/jmruebe)

---

## About This Portfolio

I'm a data scientist with a PhD in Physics and a decade of experience building end-to-end computational systems, from HPC astrophysics simulations to production ML pipelines. This portfolio showcases projects where I implement ML algorithms from first principles, apply data science to real-world problems, and demonstrate the full cycle from problem framing to working code.

**Core competencies:** Python · Machine Learning · HPC · Geospatial Analysis · Statistical Modeling · Scientific Computing

---

## Projects

### Machine Learning from Scratch

> **Why build from scratch?** Understanding what libraries like scikit-learn and XGBoost do under the hood is what separates engineers who use ML tools from engineers who can debug, adapt, and extend them. These implementations prove mathematical depth, not just API familiarity.

| Project | Algorithm | Key Concepts | Dataset |
|---------|-----------|--------------|---------|
| [Neural Network](./AIML/neuralnet_from_scratch.py) | 2-Layer NN | Backpropagation, tanh activation, gradient descent | XOR problem |
| [Decision Tree](./AIML/dectree_from_scratch.py) | CART | Gini impurity, information gain, recursive splitting | Iris |
| [Random Forest](./AIML/ranforest_from_scratch.py) | Ensemble | Bootstrap sampling, feature randomness, majority voting | Iris |
| [Gradient Boosting](./AIML/gradboost_from_scratch.py) | Boosting | Residual fitting, softmax, shrinkage, additive models | Iris |

See the [AIML README](./AIML/README.md) for full implementation details, benchmark results, and visualizations.

---

### Data Science Projects (DataCamp)

Applied data science projects covering the full analytical workflow: data cleaning, EDA, feature engineering, modeling, and communication.

| Project | Techniques | Description |
|---------|-----------|-------------|
| [Sleep Disorders Predictor](./DataCamp/sleep/sleep_data.ipynb) | Logistic Regression, EDA | Identifies factors contributing to sleep disorders and builds a binary classifier |
| [Avocado Toast Supply Chain](./DataCamp/avocado_toast/avocado_toast_analysis.ipynb) | Data cleaning, EDA | Traces global supply chain origins for avocado, olive oil, and sourdough |
| [Nobel Prize Analysis](./DataCamp/nobel_prizes/nobel_prizes.ipynb) | EDA, Linear Regression | Explores trends in Nobel Prize data across gender, age, and category over decades |

---

### Competitive Programming

Consistent coding practice through algorithm challenges. Solutions to:

- [Advent of Code](https://github.com/altjerue/advent-code)
- [Project Euler](https://projecteuler.net) (in progress)

---

## Open Source Scientific Software

In addition to this portfolio, I maintain open-source scientific computing tools developed during my research career:

| Project | Language | Description | Publications |
|---------|----------|-------------|--------------|
| [Paramo](https://github.com/altjerue/paramo) | Fortran 95 + Python | HPC radiative transfer simulation code with OpenMP optimization (60x speedup) | 5 papers |
| [Tleco](https://github.com/altjerue/tleco) | Rust + Python | Toolkit for modeling radiative signatures from relativistic outflows | ApJ 2024 |

---

## Technical Stack

```
Languages:    Python, C/C++, Fortran, R, SQL, Java, Rust, Shell
ML/DS:        NumPy, Pandas, Scikit-learn, PyTorch, Matplotlib, Seaborn
Geospatial:   QGIS, Sentinel-1/2 imagery, Rasterio
HPC:          OpenMP, MPI, SLURM, PBS, Frontera (TACC)
DevOps:       Git, Docker, Jenkins, OpenShift, Kafka
Cloud:        Azure, Databricks
```

---

*This portfolio is actively maintained. New projects added regularly.*
