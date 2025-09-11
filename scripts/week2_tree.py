from ai4h.helpers import get_iris_data_split
from ai4h.models.tree import build_tree, gini


def main():
    X_train, X_test, y_train, y_test = get_iris_data_split()

    # Build own tree
    tree = build_tree(X_train, y_train, impurity_fn=gini, min_samples_leaf=5)

    print(" Grown tree ".center(40, "-"))
    print(tree.pretty())

    print(" Accuracy test set ".center(40, "-"))
    yhat = tree.predict(X_test)
    print(f"Accuracy: {(yhat == y_test).mean():.2f}")


if __name__ == "__main__":
    main()
