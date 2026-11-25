# Notebooks

Use this folder for exploratory data analysis, such as inspecting price distributions and missing values. Keep benchmark logic in the Python package so the same experiment can run from the command line.

Follow the [dataset download instructions](../data/README.md), then open [CarDekho EDA](cardekho_eda.ipynb) and run its cells in order. Use the `ml-benchmark` kernel. If you already have a notebook frontend and need to register the kernel, run these commands:

```bash
conda run -n ml-benchmark python -m pip install ipykernel
conda run -n ml-benchmark python -m ipykernel install --user --name ml-benchmark --display-name "Python (ml-benchmark)"
```

The notebook writes selected charts to `results/published/`.
