from coldsnap import Data, Model

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
        iris_df,
        "target",
        random_state=1910,
        description="Iris Dataset",
        short_description="IrD",
    )

    cs_data.to_pickle("./tmp/iris_data.pkl.gz")

    # Create random forest classifier
    clf = RandomForestClassifier(random_state=1910)

    cs_model = Model(
        data=cs_data,
        clf=clf,
        description="RandomForestClassifier, default params on Iris dataset",
        short_description="RF01",
    )
    cs_model.fit()

    cs_model.to_pickle("./tmp/iris_model.pkl.gz")

    # Create support vector classifier
    clf = SVC(random_state=1910, probability=True)

    cs_model = Model(
        data=cs_data,
        clf=clf,
        description="SVC (with probabilities) on Iris dataset",
        short_description="SVC01",
    )
    cs_model.fit()

    cs_model.to_pickle("./tmp/iris_model_svc.pkl.gz")

    # Create DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=2, random_state=1910)

    cs_model = Model(
        data=cs_data,
        clf=clf,
        description="DecisionTreeClassifier (max_depth=2) on Iris dataset",
        short_description="DT01",
    )
    cs_model.fit()

    cs_model.to_pickle("./tmp/iris_model_dt.pkl.gz")
