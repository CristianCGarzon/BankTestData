from datetime import datetime

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from Models.DataSet import DataSet
from sklearn.externals import joblib

class SupportVectorClassifier:


    def processSVCOverSampling():

        print("\nStart time: " + str(datetime.now()))

        os_data_X, os_data_y, X_test, y_test = DataSet.getParticionedDataSetOverSampled()

        svc_classifier = SVC(kernel='linear', C=1)
        svc_classifier.fit(os_data_X, os_data_y.values.ravel())
        y_pred = svc_classifier.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        print("\nFinish time: " + str(datetime.now()))

        # Save the model to re use later
        filename = '../LROverSampling.sav'
        joblib.dump(svc_classifier, filename)

        """
        # Draw the ROC curve to check how much the model is capable of distinguishing between classes
        logit_roc_auc = roc_auc_score(y_test, logModel.predict(X_test))
        fpr, tpr, thresholds = roc_curve(y_test, logModel.predict_proba(X_test)[:,1])
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC_Over_SVC')
        plt.show()
        """
