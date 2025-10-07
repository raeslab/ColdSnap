[![Run Pytest](https://github.com/raeslab/ColdSnap/actions/workflows/autopytest.yml/badge.svg)](https://github.com/raeslab/ColdSnap/actions/workflows/autopytest.yml) [![Coverage](https://raw.githubusercontent.com/raeslab/ColdSnap/main/docs/coverage-badge.svg)](https://raw.githubusercontent.com/raeslab/ColdSnap/main/docs/coverage-badge.svg) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)  [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# ColdSnap: Freeze ML models and their training/testing data

The ColdSnap framework allows for training/testing data as well as machine learning models to be "frozen" aka serialized
to disk.

Machine learning projects often require careful tracking and storage of not only model architectures and parameters but 
also the datasets they were trained on. Having a robust mechanism for storing both models and their associated data 
snapshots is essential for reproducibility, version control, and long-term evaluation of model performance. **ColdSnap** 
was created to address these needs by providing a unified framework where machine learning models and their 
corresponding datasets can be seamlessly stored, serialized, and evaluated. By preserving both the model and 
data as a single unit, ColdSnap enables consistent evaluation across iterations, aids in model comparisons, 
and ensures that all aspects of a model’s creation—data transformations, training splits, and performance 
metrics—are easily retrievable, facilitating high-quality machine learning workflows.

## Installation

[//]: # (```python)

[//]: # (pip install coldsnap)

[//]: # (```)

## How to use ColdSnap

The code below can be found in `./docs/example`, in a nutshell you create a Data object, which contains your training and testing data, that data is added to a model along with the classifier to use and that can be serialized to disk. This model, along with the data, can be loaded again from another script/notebook. The `create_overview` function can summarize a list of models.

### Creating Snapshot of Data and Models

The code below shows how to create and store Data and Models.

```python
from coldsnap import Data, Model

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

iris = datasets.load_iris(as_frame=True)
iris_df = pd.merge(
    iris.data, iris.target, how="inner", left_index=True, right_index=True
)

if __name__ == "__main__":
    try:
        os.mkdir("./tmp/")
    except FileExistsError:
        pass

    cs_data = Data.from_df(
        iris_df, "target", random_state=1910, description="Iris Dataset"
    )

    cs_data.to_pickle("./tmp/iris_data.pkl.gz")

    # Create random forest classifier
    clf = RandomForestClassifier(random_state=1910)

    cs_model = Model(
        data=cs_data,
        clf=clf,
        description="RandomForestClassifier, default params on Iris dataset",
    )
    cs_model.fit()

    cs_model.to_pickle("./tmp/iris_model.pkl.gz")
```

### Using Transformers with ColdSnap

ColdSnap also supports sklearn transformers like `StandardScaler`, `PCA`, etc. This is useful for saving fitted
preprocessing pipelines along with your data. The example below shows how to fit a `StandardScaler` and save it as a snapshot.

```python
from coldsnap import Data, Model

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

iris = datasets.load_iris(as_frame=True)
iris_df = pd.merge(
    iris.data, iris.target, how="inner", left_index=True, right_index=True
)

if __name__ == "__main__":
    try:
        os.mkdir("./tmp/")
    except FileExistsError:
        pass

    # Create data object
    cs_data = Data.from_df(
        iris_df, "target", random_state=1910, description="Iris Dataset"
    )

    # Create a StandardScaler transformer
    scaler = StandardScaler()

    # Create a Model with the transformer using the 'estimator' parameter
    cs_scaler_model = Model(
        data=cs_data,
        estimator=scaler,
        description="StandardScaler for Iris dataset",
    )

    # Fit the scaler
    cs_scaler_model.fit()

    # Transform the training data
    X_train_scaled = cs_scaler_model.transform(cs_data.X_train)

    print("Original data (first 3 samples):")
    print(cs_data.X_train.head(3))
    print("\nScaled data (first 3 samples):")
    print(pd.DataFrame(X_train_scaled, columns=cs_data.features).head(3))

    # Save the fitted transformer
    cs_scaler_model.to_pickle("./tmp/iris_scaler.pkl.gz")

    # Later, you can load and use it on new data
    loaded_scaler = Model.from_pickle("./tmp/iris_scaler.pkl.gz")
    X_test_scaled = loaded_scaler.transform(cs_data.X_test)

    print("\nTransformer successfully saved and loaded!")
```

### Loading a Model

Once a model has been stored, it can easily be loaded using `Model.from_pickle(path)`. Once loaded,
details on the model and its performance can be retrieved using `.summary`. 

````python
from coldsnap import Model

if __name__ == "__main__":
    try:
        cs_model = Model.from_pickle("./tmp/iris_model.pkl.gz")
    except OSError:
        print("Model not found, run the script to create models first !")
        quit()

    print(cs_model.summary())
````

### Creating an Overview of Your Models

To quickly compare a number of models the function `create_overview` can be used as shown below.

```python
from coldsnap.utils import create_overview

if __name__ == "__main__":
    paths = [
        "./tmp/iris_model.pkl.gz",
        "./tmp/iris_model_svc.pkl.gz",
        "./tmp/iris_model_dt.pkl.gz",
    ]

    overview_df = create_overview(paths)

    print(overview_df.to_markdown())
```

The table below shows the output, you get for each model in the input list the summary and evaluation criteria.

|    | path                        | model_code   | model_description                                      | model_hash                                                       | data_code   | data_description   | data_hash                                                        |   num_features | features                                                                 |   num_classes | classes   |   accuracy |   precision |   recall |       f1 |   roc_auc |
|---:|:----------------------------|:-------------|:-------------------------------------------------------|:-----------------------------------------------------------------|:------------|:-------------------|:-----------------------------------------------------------------|---------------:|:-------------------------------------------------------------------------|--------------:|:----------|-----------:|------------:|---------:|---------:|----------:|
|  0 | ./tmp/iris_model.pkl.gz     | RF01         | RandomForestClassifier, default params on Iris dataset | b3f8665bce0ee979b51c9729019ae76d7ed3b83522024b9fb3375e1b96a3dc11 | IrD         | Iris Dataset       | 975cdbb5f836a810ad019751a998b18683437093f372f4545fd00be5335d5e4b |              4 | sepal length (cm), sepal width (cm), petal length (cm), petal width (cm) |             3 | 0, 1, 2   |   0.973684 |    0.975564 | 0.973684 | 0.973545 |  0.997973 |
|  1 | ./tmp/iris_model_svc.pkl.gz | SVC01        | SVC (with probabilities) on Iris dataset               | 280f5c4ca76b77144bbe7e9768bfc663b45fdafe61be3bbdc793458597f75e07 | IrD         | Iris Dataset       | 975cdbb5f836a810ad019751a998b18683437093f372f4545fd00be5335d5e4b |              4 | sepal length (cm), sepal width (cm), petal length (cm), petal width (cm) |             3 | 0, 1, 2   |   0.973684 |    0.975564 | 0.973684 | 0.973545 |  0.997973 |
|  2 | ./tmp/iris_model_dt.pkl.gz  | DT01         | DecisionTreeClassifier (max_depth=2) on Iris dataset   | 3814de3d290288de03f1b2388897c964b967c7f8ffa44303c61c41575da5d856 | IrD         | Iris Dataset       | 975cdbb5f836a810ad019751a998b18683437093f372f4545fd00be5335d5e4b |              4 | sepal length (cm), sepal width (cm), petal length (cm), petal width (cm) |             3 | 0, 1, 2   |   0.947368 |    0.947368 | 0.947368 | 0.947368 |  0.975673 |


### Evaluating Model performance

There are a few common metrics built into ColdSnap. See the example below (which assumes a model is loaded in cs_model).

```python
print(cs_model.evaluate())

# Confusion matrix
print(cs_model.confusion_matrix())

fig, ax = plt.subplots()
disp = cs_model.display_confusion_matrix(ax=ax, cmap="Blues")
plt.show()

# ROC curve
fig, ax = plt.subplots()

roc_disp = cs_model.display_roc_curve(ax=ax)

plt.show()

# SHAP beeswarm
cs_model.display_shap_beeswarm()
```

## Contributing

Any contributions you make are **greatly appreciated**.

  * Found a bug or have some suggestions? Open an [issue](https://github.com/raeslab/coldsnap/issues).
  * Pull requests are welcome! Though open an [issue](https://github.com/raeslab/coldsnap/issues) first to discuss which features/changes you wish to implement.

## Contact

ColdSnap was developed by [Sebastian Proost](https://sebastian.proost.science/) at the 
[RaesLab](https://raeslab.sites.vib.be/en). ColdSnap is available under the 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) 
license. 

For commercial access inquiries, please contact [Jeroen Raes](mailto:jeroen.raes@kuleuven.vib.be).
