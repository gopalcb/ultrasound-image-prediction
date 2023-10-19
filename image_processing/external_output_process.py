"""
this script file will process another prediction data to compare
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import os
from os import listdir
from os.path import isfile, join
import imageio
from scipy.io import loadmat


def get_gaussian_results():
    stdv = 1.0
    x = 0.0
    y = 0.0
    g = []

    for i in range(0, 33):
        
        la = []
        for j in range (0,33):
            x = (2.0 * 4 * j) / ((float)(33 - 1)) - 4 * stdv;
            y = (2.0 * 4 * i) / ((float)(33 - 1)) - 4 * stdv;
            
            v = np.exp((-1 * (x * x + y * y)) / (2.0 * math.pi * stdv * stdv));
            la.append(v)
            
        g.append(la)

    return g


def process_and_store_image_data():
    dataset=pd.read_csv('D:/SCA/u_net_output/cath_echo_confidence_single.txt', header=None, delimiter="\t")   
    dataset = dataset.values

    for ds in dataset:
        #1/45/1/c/44,129-100,76-81,141-112,158-143,89-169,153-194,95-206,154-221,205
        d = ds[0].split('/')
        
        if len(d) == 6:
        
            p = int(d[0])
            s = int(d[1])
            t = d[3]
            cls = d[5].split('-')
            confidences = d[4].split('#')

            f_ul = loadmat('D:/dataset/Labelled_Data/' + str(p).zfill(3) + 'm/UltrasounLabel.mat')
            ul = f_ul["USImage"]
            img = ul[:, :, s]
            nd_arr = np.array(img)

            for indx in range(0, len(cls)):
                x = int(cls[indx].split(',')[0])
                y = int(cls[indx].split(',')[1])

                conf = confidences[indx]

                #print(x)

                xl = 0
                xr = 0
                yt = 0
                yb = 0

                if x >= 128:
                    xl = x-128
                    xr = x + 128
                else:
                    xl = 0
                    xr = 256

                if y >= 128:
                    yt = y-128
                    yb = y + 128
                else:
                    yt = 0
                    yb = 256

                if xr-xl < 256:
                    print('xxx')

                if yb-yt < 256:
                    print('yyy')

                crop_arr = []
                gaus = []
                gi = 0
                flg = 0
                for r in range(yt, yb):
                    temp = []
                    gt = []
                    gj = 0
                    for c in range(xl, xr):
                        if r >= 354 or c >= 692:
                            temp.append(0)
                            gt.append(0)
                        else:
                            if r >= y - 15 and r <= y + 15 and c >= x - 15 and c <= x + 15:
                                temp.append(nd_arr[r][c])
                                gt.append(g[gi][gj])
                                gj = gj + 1
                                flg = 1
                            else:
                                temp.append(0)
                                gt.append(0)

                    if flg == 1:
                        gi = gi + 1        
                    crop_arr.append(temp)
                    gaus.append(gt)

                crop_arr = np.array(crop_arr)
                gaus = np.array(gaus)

                fn = str(p)+'_'+str(s)+'_1_'+t+'_h_'+str(x)+'_'+str(y)+'_'+str(conf)+'.png'
                fnr = str(p)+'_'+str(s)+'_1_'+t+'_h_'+str(x)+'_'+str(y)+'_'+str(conf)

                imageio.imwrite('D:/SCA/u_net_output/png_h/'+fn, crop_arr)
                img_1 = imageio.imread('D:/SCA/u_net_output/png_h/'+fn)
                img_1.tofile('D:/SCA/u_net_output/h/'+fnr)


                fn = str(p)+'_'+str(s)+'_1_'+t+'_g_'+str(x)+'_'+str(y)+'_'+str(conf)+'.png'
                fnr = str(p)+'_'+str(s)+'_1_'+t+'_g_'+str(x)+'_'+str(y)+'_'+str(conf)

                imageio.imwrite('D:/SCA/u_net_output/png_g/'+fn, gaus)
                img_2 = imageio.imread('D:/SCA/u_net_output/png_g/'+fn)
                img_2.tofile('D:/SCA/u_net_output/g/'+fnr)
