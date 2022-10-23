import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def logisticRegression(x_train, y_train, x_test, y_test):

    try:
        clf = LogisticRegression().fit(x_train, y_train)
        y_pred = clf.predict(x_test)
    except:
        print('error')
        
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    # confusion = confusion_matrix(y_pred=y_pred, y_true= y_test)

    print(accuracy)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig('confusion.png')
