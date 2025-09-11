import numpy as np
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifierSK
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifierSkLearn

from ai4h.helpers import get_iris_data_split
from ai4h.models.ensemble import RandomForestClassifier
from ai4h.models.tree import DecisionTreeClassifier


def main():
    """
    Script to:
    - fit decision tree and assert it matches with sklearn
    - fit random forest and assert it matches with sklearn
    """
    X_train, X_test, y_train, y_test = get_iris_data_split()

    clf = DecisionTreeClassifierSkLearn(min_samples_leaf=5)
    clf.fit(X_train, y_train)
    clf_o = DecisionTreeClassifier(min_samples_leaf=5)
    clf_o.fit(X_train, y_train)

    yhatsk = clf_o.predict(X_test)
    yhato = clf.predict(X_test)
    assert np.allclose(yhato, yhatsk)

    rfo = RandomForestClassifier(
        n_estimators=(n_estimators := 500),
        min_samples_leaf=5,
        random_state=2025,
    )
    rfo.fit(X_train, y_train)

    rf = RandomForestClassifierSK(
        n_estimators=n_estimators, min_samples_leaf=5, random_state=2025
    )
    rf.fit(X_train, y_train)

    yhato = rfo.predict(X_test)
    yhatsk = rf.predict(X_test)

    assert np.allclose(yhato, yhatsk)


if __name__ == "__main__":
    main()
