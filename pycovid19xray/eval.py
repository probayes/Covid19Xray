from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def report_metrics(y_test, pred,
                   target_names=['COVID-19', "Normal", "Pneumonia"]):
    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    classif_report = classification_report(y_test, pred,
                                           target_names=target_names)
    #cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    print("\n===========")
    print(matrix)
    #class_acc = np.array(cm_norm.diagonal())
    class_acc = [matrix[i,i]/np.sum(matrix[i,:])
                 if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens {0}: {1:.3f}, {2}: {3:.3f}, {4}: {5:.3f}'
          .format(target_names[0],
                  class_acc[0],
                  target_names[1],
                  class_acc[1],
                  target_names[2],
                  class_acc[2]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i])
            if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV {0}: {1:.3f}, {2}: {3:.3f}, {4}: {5:.3f}'
          .format(target_names[0],
                  ppvs[0],
                  target_names[1],
                  ppvs[1],
                  target_names[2],
                  ppvs[2]))
    print("===========")
    print(classif_report)
