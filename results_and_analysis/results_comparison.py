import math
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


def compare_unet_pfcnn_results(unet_prediction_accuracies, pred_accuracy_dict, t_out):
    higher_count = 0
    higher_val = []
    high_conf = []
    avg_higher=0
    pfcnn_sum = 0

    lower_count = 0
    lower_val = []
    low_conf = []
    avg_lower=0
    unet_sum = 0

    equal_count = 0
    equal_val = []

    unet_conf_arr = []
    pfcnn_conf_arr = []

    for i in range(0, len(t_out)):
        unet_dict = unet_prediction_accuracies[i]
        native_dict = pred_accuracy_dict[i]
        
        conf_unet = float(unet_dict.get('a'))
        conf_pfcnn = float(native_dict.get('a'))
        
        unet_conf_arr.append(conf_unet)
        pfcnn_conf_arr.append(conf_pfcnn)
        
        unet_sum += conf_unet
        pfcnn_sum += conf_pfcnn
        
        if conf_pfcnn > conf_unet:
            higher_count += 1
            hv = conf_pfcnn-conf_unet
            higher_val.append(hv)
            high_conf.append(conf_pfcnn)
            
        elif conf_pfcnn < conf_unet:
            lower_count += 1
            lv = conf_pfcnn-conf_unet
            lower_val.append(lv)
            low_conf.append(conf_pfcnn)
            
        elif conf_pfcnn == conf_unet:
            equal_count += 1
            lower_val.append(conf_pfcnn)
            
    avg_higher_diff=sum(higher_val) / len(higher_val)
    avg_lower_diff=sum(lower_val) / len(lower_val)
    avg_high_conf = sum(high_conf) / len(high_conf)
    avg_low_conf = sum(low_conf) / len(low_conf)
    avg_pfcnn = pfcnn_sum/len(t_out)
    avg_unet = unet_sum/len(t_out)
            
    ## Reporting ##
    print('REPORTING FOR E LIST:\n')
    print('Higher confidence in PFCNN than unet: ' + str(higher_count) + ' roi out of '+str(len(t_out)))
    print('Lower confidence in Probing FCNN than unet: ' + str(lower_count) + ' roi out of '+str(len(t_out))+'\n')

    print('Average higher difference of confidence (D1): '+str(avg_higher_diff))
    print('Average lower difference of confidence (D2): '+str(avg_lower_diff))
    print('Difference (D1 - D2): '+str(avg_higher_diff-(-avg_lower_diff))+'\n')

    print('Average higher confidence: '+str(avg_high_conf))
    print('Average lower confidence: '+str(avg_low_conf)+'\n')

    print('Average confidence in unet: '+str(avg_unet))
    print('Average confidence in pfcnn: '+str(avg_pfcnn))
    #print('Their average ((pfcnn+unet)/2): '+str((avg_pfcnn+avg_unet)/2))
    print('Their difference (pfcnn-unet): '+str(avg_unet-avg_pfcnn)+'\n')

    print('Equal confidence in Probing FCNN and unet: ' + str(equal_count) + ' out of '+str(len(t_out)))

    print('###################\n\n')


def pfcnn_confidence_vs_unet_confidence_plotting(pred_accuracy_dict, unet_prediction_accuracies):
    x_mypred = []
    y_unetpred = []

    x_ = []
    y_ = []

    index_list = random.sample(range(0,len(pred_accuracy_dict)), 400)

    for indx in index_list:
        x_mypred.append(pred_accuracy_dict[indx].get('a'))
        
    for indx in index_list:
        y_unetpred.append(unet_prediction_accuracies[indx].get('a'))
        
    x_mypred = np.array(x_mypred)
    y_unetpred = np.array(y_unetpred)

    for itm in x_mypred:
        x_.append(itm.item())
        
    for itm in y_unetpred:
        y_.append(itm.item())

    x_.append(0.0)
    y_.append(0.0)
    x_.append(1.0)
    y_.append(1.0)
        
    x_ = np.array(x_)
    y_ = np.array(y_)

    x_ = x_.astype(np.float)
    y_ = y_.astype(np.float)

    plt.rcParams["figure.figsize"] = (20,10)

    plt.xlabel('Confidence values of Probing FCNN Classifier')
    plt.ylabel('Confidence values of U-net Classifier')
    #plt.title('scatter plot: probing fcnn confidence score vs. unet confidence score (E list 100 points)')

    plt.plot(x_, y_, linestyle='none', marker='o')
    plt.show()


def get_confusion_matrix(nd_predict, test_classes):
    # Confusion matrix
    # confusion_matrix(y_true, y_pred)
    results = confusion_matrix(nd_predict, test_classes.numpy(), labels=[0, 1])
    print('Confusion Matrix :')
    print(results) 
    #print('Accuracy Score :',accuracy_score(test_classes, predicted))
    #print('Report : ')
    #print(classification_report(test_classes, predicted))
    return results
