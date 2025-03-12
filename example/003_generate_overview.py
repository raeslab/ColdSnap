from coldsnap.utils import create_overview

if __name__ == "__main__":
    paths = [
        "./tmp/iris_model.pkl.gz",
        "./tmp/iris_model_svc.pkl.gz",
        "./tmp/iris_model_dt.pkl.gz",
    ]

    overview_df = create_overview(paths)

    print(overview_df.to_markdown())

    overview_df.to_excel("./tmp/overview.xlsx")
