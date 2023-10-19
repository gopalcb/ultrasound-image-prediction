from matplotlib import pyplot as plt
import cv2


#  write output in PNG file and store in HDD
def bulk_write(test_files_title, nd_predict):
    # Write NN output
    image_slice = 0
    xys = []
    r_path = 'D:/SCA/256_256/nn_output/bounding_boxes/'

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size

    #len(nd_predict)
    for i in range(0, len(nd_predict)):
        
        print(test_files_title[i])
        f_arr = test_files_title[i].split('_')
        
        p_sl = f_arr[0]
        i_s = f_arr[1]
        i_sub = f_arr[2]
        o_type = f_arr[3]
        x_val = f_arr[4]
        y_val = f_arr[5].split('.')[0]
        
        img_path = r_path + p_sl + '/'+ i_s + '_' + i_sub +'.png'
        imgcv = cv2.imread(img_path)
        
        x1 = int(x_val)-16
        y1 = int(y_val)-16
        x2 = int(x_val)+16
        y2 = int(y_val)+16
        
        #conf = 0.99
        
        #label = 'C' if nd_predict[i] == 1 else 'E'
        label = 'C' if o_type == 'c' else 'E'
        
        cv2.rectangle(imgcv,(x1,y1),(x2,y2),(0,255,0),1)
        
        labelSize=cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.4,2)
        
        _x1 = x1
        _y1 = y1
        _x2 = _x1+labelSize[0][0]
        _y2 = y1-int(labelSize[0][1])
        
        cv2.rectangle(imgcv,(_x1,_y1),(_x2,_y2),(0,255,0),cv2.FILLED)
        cv2.putText(imgcv,label,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,0,0),1)
        
        cv2.imwrite(img_path,imgcv)

    print('Image write complete')
    return True