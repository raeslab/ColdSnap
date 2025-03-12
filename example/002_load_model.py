from coldsnap import Model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    try:
        cs_model = Model.from_pickle("./tmp/iris_model.pkl.gz")
    except OSError:
        print("Model not found, run the script to create models first !")
        quit()

    print(cs_model.summary())
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
