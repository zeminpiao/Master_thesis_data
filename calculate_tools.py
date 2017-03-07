from PIL import Image, ImageTk
import numpy
numpy.set_printoptions(threshold=numpy.nan)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import scipy
import scipy.io as sio
from sklearn.metrics import mean_squared_error
import operator
import os
import dxchange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from xraydb import XrayDB

#%matplotlib notebook


def data_initialization(data_se, data_in):
    
    #Read raw open beam, separate and interlacing data
    path = "/ufs/piao/Desktop/Data_2.24/"
    files_ob = [path+str(x)+'_OB.tif' for x in range(42, 242, 10)]
    img_ob = dxchange.reader.read_tiff(files_ob[0])
    data_ob = numpy.zeros([img_ob.shape[0],img_ob.shape[1], 20])
    ii = 0
    for file in files_ob:
        data_ob[:,:,ii] = dxchange.reader.read_tiff(file)
        ii += 1
    data_subtr_ob = data_ob.copy()
    for ii in range(0, 19):
        data_subtr_ob[:,:,ii] = data_subtr_ob[:,:,ii] - data_ob[:,:,ii+1]
    
    files_separate = [path+str(x)+'_3_separate.tif' for x in range(42, 242, 10)]
    img_spr = dxchange.reader.read_tiff(files_separate[0])
    data_separate = numpy.zeros([img_spr.shape[0],img_spr.shape[1], 20])
    ii = 0
    for file in files_separate:
        data_separate[:,:,ii] = dxchange.reader.read_tiff(file)
        ii += 1
    data_subtr_separate = data_separate.copy()
    for ii in range(0, 19):
        data_subtr_separate[:,:,ii] = data_subtr_separate[:,:,ii] - data_separate[:,:,ii+1]
    #global data_se
    #data_se = numpy.zeros([516, 516, 20])
    for ii in range(0, 20):
        data_se[:,:,ii] = numpy.subtract(numpy.log(data_subtr_ob[:,:,ii]), numpy.log(data_subtr_separate[:,:,ii]))

        
    files_interlacing = [path+str(x)+'_3_interlacing.tif' for x in range(42, 242, 10)]
    img_itlc = dxchange.reader.read_tiff(files_interlacing[0])
    data_interlacing = numpy.zeros([img_itlc.shape[0],img_itlc.shape[1], 20])
    ii = 0
    for file in files_interlacing:
        data_interlacing[:,:,ii] = dxchange.reader.read_tiff(file)
        ii += 1
    data_subtr_interlacing = data_interlacing.copy()
    for ii in range(0, 19):
        data_subtr_interlacing[:,:,ii] = data_subtr_interlacing[:,:,ii] - data_interlacing[:,:,ii+1]
    #global data_in
    #data_in = numpy.zeros([516, 516, 20])
    for ii in range(0, 20):
        data_in[:,:,ii] = numpy.subtract(numpy.log(data_subtr_ob[:,:,ii]), numpy.log(data_subtr_interlacing[:,:,ii]))
    data_se_mf = ndimage.filters.median_filter(data_se, (4, 4, 1))
    data_se_in = ndimage.filters.median_filter(data_in, (4, 4, 1))
    
    return data_se, data_in, data_se_mf, data_se_in



#Showing the spectrum of one specific pixel
def show_spectrum(x, y, formating, data_se, data_in, Energy_Value):
    spectrum_test = []
    if formating == 'separate':
        for i in range(0,len(Energy_Value)):
            spectrum_test.append(data_se[y, x, i])
    elif formating == 'interlacing':
        for i in range(0,len(Energy_Value)):
            spectrum_test.append(data_in[y, x, i])
    else:
        raise NameError('Wrong dataset name')
    #print spectrum_test
    mc_coefficient = [ i for i in spectrum_test ]
    #plt.figure()
    #plt.title('spectrum of point ('+str(x)+', '+str(y)+')')
    plt.plot(Energy_Value, mc_coefficient, linewidth=0.1)
    #plt.show()
    return spectrum_test


def show_spectrum_region((x1, y1), (x2, y2), formating_input, data_se, data_in, Energy_Value):
    spectrum_region = []
    plt.figure()
    plt.title('spectrum figure for region from '+str((x1, y1))+' to '+str((x2, y2)))
    if x1 > x2:
        for i in range(x2, x1):
            if y1 > y2:
                for j in range(y2, y1):
                    spectrum_region.append(show_spectrum(i, j, formating_input, data_se, data_in, Energy_Value))
            elif y1 == y2:
                spectrum_region.append(show_spectrum(i, y1, formating_input, data_se, data_in, Energy_Value))
            elif y1 < y2:
                for j in range(y1, y2):
                    spectrum_region.append(show_spectrum(i, j, formating_input, data_se, data_in, Energy_Value))
    elif x1 == x2:
        for j in range(y1, y2):
            if y1 > y2:
                for j in range(y2, y1):
                    spectrum_region.append(show_spectrum(x1, j, formating_input, data_se, data_in, Energy_Value))
            elif y1 == y2:
                spectrum_region.append(show_spectrum(x1, y1, formating_input, data_se, data_in, Energy_Value))
            elif y1 < y2:
                for j in range(y1, y2):
                    spectrum_region.append(show_spectrum(x1, j, formating_input, data_se, data_in, Energy_Value))
    elif x1 < x2:
        for i in range(x1, x2):
            if y1 > y2:
                for j in range(y2, y1):
                    spectrum_region.append(show_spectrum(i, j, formating_input, data_se, data_in, Energy_Value))
            elif y1 == y2:
                spectrum_region.append(show_spectrum(i, y1, formating_input, data_se, data_in, Energy_Value))
            elif y1 < y2:
                for j in range(y1, y2):
                    spectrum_region.append(show_spectrum(i, j, formating_input, data_se, data_in, Energy_Value))
    #plt.show(1)
    
    #plt.figure(2)
    if formating_input == 'separate':
        im = data_se[:,:,5]
        fig, ax = plt.subplots()
        ax.imshow(im)
        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='red',facecolor='none')
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
        ax.add_patch(rect)
    #fig.show(2)
    elif formating_input == 'interlacing':
        im = data_in[:,:,5]
        ig, ax = plt.subplots()
        ax.imshow(im)
        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='red',facecolor='none')
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
        ax.add_patch(rect)
    else:
        raise NameError('Wrong dataset name')
    coords = []
    
    return spectrum_region

def choose_region(formating_input):
    global coords
    coords = []
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata

        # print 'x = %d, y = %d'%(
        #     ix, iy)

        # assign global variable to access outside of function
        global coords
        coords.append((int(ix), int(iy)))
        print coords
        # Disconnect after 2 clicks
        if len(coords) == 2:
            fig.canvas.mpl_disconnect(cid)
            plt.close(1024242)
        return

    if formating_input == 'separate':
        im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/after_ob_correction/92_3_separate_after_ob.tif')
    elif formating_input == 'interlacing':
        im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/after_ob_correction/92_3_interlacing_after_ob.tif')
    else:
        raise NameError('Wrong dataset name')
    origin_array = numpy.array(im_origin)
    origin_float = origin_array.astype(float)

    
    fig = plt.figure(1024242)
    plt.imshow(origin_float)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(1024242)

    return coords

#Average of all the energy bin
def show_average_and_variance((x1, y1), (x2, y2), formating_input, data_se, data_in, Energy_Value):
    result_spectrum_mean = []
    result_variance = []
    if formating_input == 'separate':
        for energy_bin in range(0, len(Energy_Value)):
            mean_one_bin = numpy.nanmean(data_se[y1:y2, x1:x2, energy_bin])
            #print mean_one_bin
            variance_one_bin = numpy.nanvar(data_se[y1:y2, x1:x2, energy_bin])
            #print variance_one_bin
            result_spectrum_mean.append(mean_one_bin)
            result_variance.append(variance_one_bin)
    elif formating_input == 'interlacing' :
        for energy_bin in range(0, len(Energy_Value)):
            mean_one_bin = numpy.nanmean(data_in[y1:y2, x1:x2, energy_bin])
            #print mean_one_bin
            variance_one_bin = numpy.nanvar(data_in[y1:y2, x1:x2, energy_bin])
            #print variance_one_bin
            result_spectrum_mean.append(mean_one_bin)
            result_variance.append(variance_one_bin)
    #print result_spectrum_mean
    plt.figure()
    plt.title('Average spectrum for the region from' +str((x1, y1))+' to '+str((x2, y2)))
    plt.plot(Energy_Value, result_spectrum_mean)
    #plt.close()
    #print result_variance
    plt.figure()
    plt.title('Spectrum variance between for the region from' +str((x1, y1))+' to '+str((x2, y2)))
    plt.plot(Energy_Value, result_variance)
    return result_spectrum_mean, result_variance

def theoretical_spectrum(element, Energy_Value):
    return XrayDB().mu_elam(element, Energy_Value)


#find the optimal k for kmeans clustering:
def find_optimal_k_by_PCA(input_data):
    
    test_PCA_data = numpy.array(input_data)
    test_PCA_data = numpy.nan_to_num(test_PCA_data)
    #print test_PCA_data.shape
    PCA_data = PCA()

    #plt.figure()

    a = PCA()
    e = a.fit(test_PCA_data)
    ratio_sum = 0
    for i in range(0, 100):
        ratio_sum = ratio_sum + e.explained_variance_ratio_[i]
        if ratio_sum >0.99:
            break
    return i+1
    

