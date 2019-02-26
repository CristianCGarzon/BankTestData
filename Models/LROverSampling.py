import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from Models.DataSet import DataSet


class LearningRateOverSampling:

    def processLROverSampling():
        os_data_X, os_data_y, X_test, y_test = DataSet.getParticionedDataSetOverSampled()

        # Creating a logisitc regression model
        log_model = LogisticRegression()
        log_model.fit(os_data_X, os_data_y.values.ravel())
        y_pred = log_model.predict(X_test)
        print("The classification report of LrOverSampling is: \n", classification_report(y_test, y_pred))

        # Draw the ROC curve to check how much the model is capable of distinguishing between classes
        logit_roc_auc = roc_auc_score(y_test, log_model.predict(X_test))
        fpr, tpr, thresholds = roc_curve(y_test, log_model.predict_proba(X_test)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('../Log_ROC_Over_Sampling')
        plt.show()
