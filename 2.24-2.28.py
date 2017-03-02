
# coding: utf-8

# From the 3d plot of images after open beam correction and images after median filtering for each threshold value, we could see that:
# 1. median filtering cuts down noises inside the metal regions, because we can see the variance inside the metal regions has been reduced. 
# 2. The lower the threshold, the more variance inside the metal regions, as well as the variance between regions becomes higher. So to detect region of different metals, we could use the data of lower threshold. The higher the threshold, the less noise inside the metal regions but the variance between different metal region will become unclear with the increment of the threshold.
# 3. Intuitively seeing in the 3d plot, the  noise is variant with the energy level so we can use median filter on that.

# In[727]:

from PIL import Image, ImageTk
import numpy
numpy.set_printoptions(threshold=numpy.nan)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import scipy
import scipy.io as sio

import operator
import os
import dxchange
get_ipython().magic(u'matplotlib notebook')


# In[ ]:




# In[2]:

'''#Doing the open beam correction 52 THL separate dataset
#for i in range(42,242,10):
#    im = Image.open('/ufs/piao/Desktop/Data_2.24/'+str(i)+'_3_separate.tif')
#    ob = Image.open('/ufs/piao/Desktop/Data_2.24/'+str(i)+'_OB.tif')
im = Image.open('/ufs/piao/Desktop/Data_2.24/52_3_separate.tif')
ob = Image.open('/ufs/piao/Desktop/Data_2.24/52_OB.tif')
original = numpy.array(im)
original_log = numpy.log(original)
open_beam = numpy.array(ob)
open_beam_log = numpy.log(open_beam)
    #division = numpy.divide(open_beam_float, original_float)
    #print division.shape
result = numpy.subtract(open_beam_log, original_log)
    #result[result<0.09] = 0
result_image = Image.fromarray(result)
    #result_image.save('/ufs/piao/Desktop/Data_2.24/after_ob_correction/'+str(i)+'_3_separate_after_ob.tif')
result_image.save('/ufs/piao/Desktop/Data_2.24/after_ob_correction/manual_52_3_separate_after_ob.tif')'''


# In[178]:

'''#thresholding for 42 pic to denoise
im = Image.open('/ufs/piao/Desktop/Data_2.24/after_ob_correction/42_3_separate_after_ob.tif')
im_array = numpy.array(im)
#im_array[(im_array<3.2) & (im_array>2.8)] = 0
im_image = Image.fromarray(im_array)
im_image.save('/ufs/piao/Desktop/Data_2.24/after_ob_correction/42thresholding.tif')
'''


# In[180]:

'''#Doing the open beam correction for separate dataset
for i in range(42,242,10):
    im = Image.open('/ufs/piao/Desktop/Data_2.24/'+str(i)+'_3_separate.tif')
    ob = Image.open('/ufs/piao/Desktop/Data_2.24/'+str(i)+'_OB.tif')
    #im = Image.open('/ufs/piao/Desktop/Data_2.24/52_3_separate.tif')
    #ob = Image.open('/ufs/piao/Desktop/Data_2.24/52_OB.tif')
    original = numpy.array(im)
    original_log = numpy.log(original)
    open_beam = numpy.array(ob)
    open_beam_log = numpy.log(open_beam)
    #division = numpy.divide(open_beam_float, original_float)
    #print division.shape
    result = numpy.subtract(open_beam_log, original_log)
    #result[result<0.09] = 0
    result_image = Image.fromarray(result)
    result_image.save('/ufs/piao/Desktop/Data_2.24/after_ob_correction/'+str(i)+'_3_separate_after_ob.tif')
    #result_image.save('/ufs/piao/Desktop/Data_2.24/after_ob_correction/manual_52_3_separate_after_ob.tif')
'''


# In[181]:

'''Doing the open beam correction for interlacing dataset
for i in range(42,242,10):
    im = Image.open('/ufs/piao/Desktop/Data_2.24/'+str(i)+'_3_interlacing.tif')
    ob = Image.open('/ufs/piao/Desktop/Data_2.24/'+str(i)+'_OB.tif')
    original = numpy.array(im)
    original_log = numpy.log(original)
    open_beam = numpy.array(ob)
    open_beam_log = numpy.log(open_beam)
    #eliminate the noise
    #result[result<0.09] = 0
    result = numpy.subtract(open_beam_log, original_log)
    result_image = Image.fromarray(result)
    
    result_image.save('/ufs/piao/Desktop/Data_2.24/after_ob_correction/'+str(i)+'_3_interlacing_after_ob.tif')'''


# In[3]:


# Get the files only:
path = "/ufs/piao/Desktop/Data_2.24/"

files = [path+str(x)+'_OB.tif' for x in range(42, 242, 10)]

img = dxchange.reader.read_tiff(files[0])

data_ob = numpy.zeros([img.shape[0],img.shape[1], 20])

ii = 0
for file in files:
    data_ob[:,:,ii] = dxchange.reader.read_tiff(file)
    ii += 1


# In[4]:

path = "/ufs/piao/Desktop/Data_2.24/"

files = [path+str(x)+'_3_separate.tif' for x in range(42, 242, 10)]

img = dxchange.reader.read_tiff(files[0])

data_separate = numpy.zeros([img.shape[0],img.shape[1], 20])

ii = 0
for file in files:
    data_separate[:,:,ii] = dxchange.reader.read_tiff(file)
    ii += 1


# In[244]:

data_subtr_separate = data_separate.copy()
data_subtr_ob = data_ob.copy()
for ii in range(0, 19):
    data_subtr_separate[:,:,ii] = data_subtr_separate[:,:,ii] - data_separate[:,:,ii+1]
    data_subtr_ob[:,:,ii] = data_subtr_ob[:,:,ii] - data_ob[:,:,ii+1]


# In[245]:

data = numpy.zeros([516, 516, 20])
for ii in range(0, 20):
    data[:,:,ii] = numpy.subtract(numpy.log(data_subtr_ob[:,:,ii]), numpy.log(data_subtr_separate[:,:,ii]))


# In[246]:

plt.imshow(data[:,:,5])
plt.show()


# In[98]:




# In[554]:

#Create the Energy axis corresponding to the Threshold value
Energy_Value = [0.0008706*i*i + 0.1*i+3.392 for i in range(47, 147, 10)]
print Energy_Value


# In[24]:




# In[25]:




# In[461]:

#Showing the spectrum of one specific pixel
def show_spectrum(x, y, formating):
    spectrum_test = []
    for i in range(0,10):
        spectrum_test.append(data[y, x, i])
    #print spectrum_test
    mc_coefficient = [ i for i in spectrum_test ]
    #plt.figure()
    #plt.title('spectrum of point ('+str(x)+', '+str(y)+')')
    plt.plot(Energy_Value, mc_coefficient, linewidth=0.1)
    #plt.show()


# In[756]:

'''plt.figure()
plt.title('spectrum for specific region')
show_spectrum(112, 112, 'separate')
show_spectrum(113, 112, 'separate')
show_spectrum(112, 113, 'separate')
show_spectrum(113, 113, 'separate')
plt.show()'''


# In[457]:




# In[250]:

def show_spectrum_region((x1, y1), (x2, y2)):
    plt.figure(10)
    plt.title('spectrum figure for region from '+str((x1, y1))+' to '+str((x2, y2)))
    if x1 > x2:
        for i in range(x2, x1):
            if y1 > y2:
                for j in range(y2, y1):
                    show_spectrum(i, j, 'separate')
            elif y1 == y2:
                show_spectrum(i, y1, 'separate')
            elif y1 < y2:
                for j in range(y1, y2):
                    show_spectrum(i, j, 'separate')
    elif x1 == x2:
        for j in range(y1, y2):
            if y1 > y2:
                for j in range(y2, y1):
                    show_spectrum(x1, j, 'separate')
            elif y1 == y2:
                show_spectrum(x1, y1, 'separate')
            elif y1 < y2:
                for j in range(y1, y2):
                    show_spectrum(x1, j, 'separate')
    elif x1 < x2:
        for i in range(x1, x2):
            if y1 > y2:
                for j in range(y2, y1):
                    show_spectrum(i, j, 'separate')
            elif y1 == y2:
                show_spectrum(i, y1, 'separate')
            elif y1 < y2:
                for j in range(y1, y2):
                    show_spectrum(i, j, 'separate')
    #plt.show(1)
    
    #plt.figure(2)
    
    im = data[:,:,5]
    fig, ax = plt.subplots()
    ax.imshow(im)
    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='red',facecolor='none')
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ax.add_patch(rect)
    #fig.show(2)
    
    coords = []


# In[241]:




# In[737]:

show_spectrum_region((150, 240), (160, 245))


# In[371]:

plt.close(1)


# In[738]:

img = ndimage.filters.median_filter(data, (4, 4, 1))


# In[540]:

#Showing the spectrum of one specific pixel
def show_spectrum_after(x, y, formating):
    spectrum_test = []
    for i in range(0,10):
        spectrum_test.append(img[y, x, i])
    #print spectrum_test
    mc_coefficient = [ i for i in spectrum_test ]
    #plt.figure()
    #plt.title('spectrum of point ('+str(x)+', '+str(y)+')')
    #plt.plot(Energy_Value, mc_coefficient, 'ro')
    plt.plot(Energy_Value, mc_coefficient, linewidth=0.1)
    #plt.show()


# In[363]:

def show_spectrum_region_after_mf((x1, y1), (x2, y2)):
    plt.figure(4)
    plt.title('spectrum figure for region from '+str((x1, y1))+' to '+str((x2, y2)))
    if x1 > x2:
        for i in range(x2, x1):
            if y1 > y2:
                for j in range(y2, y1):
                    show_spectrum_after(i, j, 'separate')
            elif y1 == y2:
                show_spectrum_after(i, y1, 'separate')
            elif y1 < y2:
                for j in range(y1, y2):
                    show_spectrum_after(i, j, 'separate')
    elif x1 == x2:
        for j in range(y1, y2):
            if y1 > y2:
                for j in range(y2, y1):
                    show_spectrum_after(x1, j, 'separate')
            elif y1 == y2:
                show_spectrum_after(x1, y1, 'separate')
            elif y1 < y2:
                for j in range(y1, y2):
                    show_spectrum_after(x1, j, 'separate')
    elif x1 < x2:
        for i in range(x1, x2):
            if y1 > y2:
                for j in range(y2, y1):
                    show_spectrum_after(i, j, 'separate')
            elif y1 == y2:
                show_spectrum_after(i, y1, 'separate')
            elif y1 < y2:
                for j in range(y1, y2):
                    show_spectrum_after(i, j, 'separate')
        #plt.figure(2)
     
    #plt.show(1)
    
    #plt.title('image sample')
    im = data[:,:,5]
    fig, ax = plt.subplots()
    ax.imshow(im)
    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='red',facecolor='none')
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ax.add_patch(rect)
    #fig.show(2)
    coords = []


# In[483]:

print coords


# In[522]:

#Average of all the energy bin
def show_average_and_variance((x1, y1), (x2, y2)):
    result_spectrum_mean = []
    result_variance = []
    for energy_bin in range(0, 10):
        mean_one_bin = numpy.nanmean(data[y1:y2, x1:x2, energy_bin])
        #print mean_one_bin
        variance_one_bin = numpy.nanvar(data[y1:y2, x1:x2, energy_bin])
        #print variance_one_bin
        result_spectrum_mean.append(mean_one_bin)
        result_variance.append(variance_one_bin)
    print result_spectrum_mean
    plt.figure(101)
    plt.plot(Energy_Value, result_spectrum_mean)
    #plt.close()
    print result_variance
    plt.figure(103)
    plt.plot(Energy_Value, result_variance)


# In[539]:

#Average of all the energy bin
def show_average_and_variance_after_mf((x1, y1), (x2, y2)):
    result_spectrum_mean = []
    result_variance = []
    for energy_bin in range(0, 10):
        mean_one_bin = numpy.nanmean(img[y1:y2, x1:x2, energy_bin])
        #print mean_one_bin
        variance_one_bin = numpy.nanvar(img[y1:y2, x1:x2, energy_bin])
        #print variance_one_bin
        result_spectrum_mean.append(mean_one_bin)
        result_variance.append(variance_one_bin)
    print result_spectrum_mean
    plt.figure(101)
    plt.plot(Energy_Value, result_spectrum_mean)
    #plt.close()
    print result_variance
    plt.figure(103)
    plt.plot(Energy_Value, result_variance)


# In[548]:

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
        plt.close(3)
    return


im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/after_ob_correction/92_3_separate_after_ob.tif')
origin_array = numpy.array(im_origin)
origin_float = origin_array.astype(float)

coords = []
fig = plt.figure(3)
plt.imshow(origin_float)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show(3)


# In[356]:

#coords2 = [(coords[0][1],coords[0][0]), (coords[1][1],coords[1][0])]


# In[556]:

show_spectrum_region(coords[0], coords[1])


# In[550]:

show_spectrum_region_after_mf(coords[0], coords[1])


# In[709]:

plt.clf()
plt.cla()
plt.close()
plt.clf()
plt.cla()
plt.close()
plt.clf()
plt.cla()
plt.close()
plt.clf()
plt.cla()
plt.close()
plt.clf()
plt.cla()
plt.close()
plt.clf()
plt.cla()
plt.close()
plt.clf()
plt.cla()
plt.close()
plt.clf()
plt.cla()
plt.close()
plt.clf()
plt.cla()
plt.close()


# In[512]:




# In[739]:

show_average_and_variance(coords[0], coords[1])


# In[740]:

show_average_and_variance_after_mf(coords[0], coords[1])


# In[750]:

#simple calibration regarding the first value is the correct value
a = [5.411926703072643, 1.8805652338921721, 1.1713508250840026, 0.8518970916166495, 0.70442292344329061, 0.52283946541211868, 0.41611437629723858, 0.32311383005315319, 0.27294877675840923, 0.20954124467732876]
real_spectrum = [i*4.8252 for i in a]
print real_spectrum

print Energy_Value
plt.figure(1022222)
plt.plot(Energy_Value, real_spectrum)
plt.show()


# In[748]:

#simple calibration by average value
a = [5.411926703072643, 1.8805652338921721, 1.1713508250840026, 0.8518970916166495, 0.70442292344329061, 0.52283946541211868, 0.41611437629723858, 0.32311383005315319, 0.27294877675840923, 0.20954124467732876]
a_mean = sum(a) / float(len(a))

#print a_mean

theoretical_mean = 7.0362

real_spectrum = [i*(theoretical_mean/a_mean) for i in a]
#print real_spectrum

#print Energy_Value
plt.figure(1020)
real_figure = plt.plot(Energy_Value, real_spectrum,'r')
theoretical_figure = plt.plot(Energy_Value, [26.115, 15.6443, 9.7395, 6.2889, 4.1844, 2.8856, 2.0430, 1.4852, 1.1154, 0.8603], 'b')
red_patch = patches.Patch(color='red', label='real figure')
blue_patch = patches.Patch(color='blue', label='theoretical figure')
yellow_patch = patches.Patch(color='yellow', label='difference')
subtraction = map(operator.sub, real_spectrum, [26.115, 15.6443, 9.7395, 6.2889, 4.1844, 2.8856, 2.0430, 1.4852, 1.1154, 0.8603])
difference = map(operator.abs, subtraction)
different_figure = plt.plot(Energy_Value, difference, 'y')

plt.legend(handles=[red_patch, blue_patch, yellow_patch])
plt.show()
print difference


# Our dataset could not perform well in low range of energy level, due to the bin size is relatively big.
# 
# 
# In this case, we entrust the y - axis value to represent the photon count at the average of lower bound and upper bound energy level. As a result, the result 
# From the simple calibration, for our dataset, the higher the energy level, the more accurate the spectrum is. This is because in high energy level, the variation between bins is low. 

# In[ ]:




# In[751]:

#numpy.nanmean(data[124:151,147:179,0])

#print data[124:151,147:179,0]


# In[752]:

'''#Showing the spectrum of one specific pixel
def show_spectrum_openbeam(x, y):
    
    spectrum_test = []
    for i in range(52,182,10):
        im = Image.open('/ufs/piao/Desktop/Data_2.24/'+str(i)+'_OB.tif')
        original = numpy.array(im)
        original_float = original.astype(float)
        spectrum_test.append(original_float.item((x,y)))
    #print spectrum_test
    mc_coefficient = [ i for i in spectrum_test ]
    print mc_coefficient
    plt.figure()
    plt.title('spectrum of point ('+str(x)+', '+str(y)+')')
    plt.plot(Energy_Value, mc_coefficient, 'ro')
    plt.show()
'''


# In[ ]:




# In[ ]:




# In[ ]:




# In[753]:

print Energy_Value


# In[652]:

theoretical_energy_level = [1.000E-3, 1.500E-3, 2.000E-3, 3.000E-3, 4.000E-3, 5.000E-3, 6.000E-3, 8.000E-3, 1.000E-2, 1.500E-2, 2.000E-2, 3.000E-2, 4.000E-2, 5.000E-2, 6.000E-2, 8.000E-2, 1.000E-1, 1.500E-1, 2.000E-1, 3.000E-1, 4.000E-1, 5.000E-1, 6.000E-1, 8.000E-1, 1.000   , 1.250   , 1.500   , 2.000   , 3.000   , 4.000   , 5.000   , 6.000   , 8.000   , 1.000E+1, 1.500E+1, 2.000E+1];
#mac1 = sio.loadmat('mac.mat')


# In[706]:

print theoretical_energy_level[8]

print theoretical_energy_level[13]


# In[651]:

#Mass attenuation coefficient for all elements in theoretical_energy_level

mac1 = [[  7.21700000e+00,   2.14800000e+00,   1.05900000e+00,
          5.61200000e-01,   4.54600000e-01,   4.19300000e-01,
          4.04200000e-01,   3.91400000e-01,   3.85400000e-01,
          3.76400000e-01,   3.69500000e-01,   3.57000000e-01,
          3.45800000e-01,   3.35500000e-01,   3.26000000e-01,
          3.09100000e-01,   2.94400000e-01,   2.65100000e-01,
          2.42900000e-01,   2.11200000e-01,   1.89300000e-01,
          1.72900000e-01,   1.59900000e-01,   1.40500000e-01,
          1.26300000e-01,   1.12900000e-01,   1.02700000e-01,
          8.76900000e-02,   6.92100000e-02,   5.80600000e-02,
          5.04900000e-02,   4.49800000e-02,   3.74600000e-02,
          3.25400000e-02,   2.53900000e-02,   2.15300000e-02],
       [  6.08400000e+01,   1.67600000e+01,   6.86300000e+00,
          2.00700000e+00,   9.32900000e-01,   5.76600000e-01,
          4.19500000e-01,   2.93300000e-01,   2.47600000e-01,
          2.09200000e-01,   1.96000000e-01,   1.83800000e-01,
          1.76300000e-01,   1.70300000e-01,   1.65100000e-01,
          1.56200000e-01,   1.48600000e-01,   1.33600000e-01,
          1.22400000e-01,   1.06400000e-01,   9.53500000e-02,
          8.70700000e-02,   8.05400000e-02,   7.07600000e-02,
          6.36200000e-02,   5.68800000e-02,   5.17300000e-02,
          4.42200000e-02,   3.50300000e-02,   2.94900000e-02,
          2.57700000e-02,   2.30700000e-02,   1.94000000e-02,
          1.70300000e-02,   1.36300000e-02,   1.18300000e-02],
       [  2.33900000e+02,   6.66800000e+01,   2.70700000e+01,
          7.54900000e+00,   3.11400000e+00,   1.61900000e+00,
          9.87500000e-01,   5.05400000e-01,   3.39500000e-01,
          2.17600000e-01,   1.85600000e-01,   1.64400000e-01,
          1.55100000e-01,   1.48800000e-01,   1.43800000e-01,
          1.35600000e-01,   1.28900000e-01,   1.15800000e-01,
          1.06000000e-01,   9.21000000e-02,   8.24900000e-02,
          7.53200000e-02,   6.96800000e-02,   6.12100000e-02,
          5.50300000e-02,   4.92100000e-02,   4.47600000e-02,
          3.83000000e-02,   3.04300000e-02,   2.57200000e-02,
          2.25700000e-02,   2.03000000e-02,   1.72500000e-02,
          1.52900000e-02,   1.25200000e-02,   1.10900000e-02],
       [  6.04100000e+02,   1.79700000e+02,   7.46900000e+01,
          2.12700000e+01,   8.68500000e+00,   4.36900000e+00,
          2.52700000e+00,   1.12400000e+00,   6.46600000e-01,
          3.07000000e-01,   2.25100000e-01,   1.79200000e-01,
          1.64000000e-01,   1.55400000e-01,   1.49300000e-01,
          1.40100000e-01,   1.32800000e-01,   1.19000000e-01,
          1.08900000e-01,   9.46300000e-02,   8.47100000e-02,
          7.73900000e-02,   7.15500000e-02,   6.28600000e-02,
          5.65200000e-02,   5.05400000e-02,   4.59700000e-02,
          3.93800000e-02,   3.13800000e-02,   2.66400000e-02,
          2.34700000e-02,   2.12100000e-02,   1.81900000e-02,
          1.62700000e-02,   1.36100000e-02,   1.22700000e-02],
       [  1.22900000e+03,   3.76600000e+02,   1.59700000e+02,
          4.66700000e+01,   1.92700000e+01,   9.68300000e+00,
          5.53800000e+00,   2.34600000e+00,   1.25500000e+00,
          4.82700000e-01,   3.01400000e-01,   2.06300000e-01,
          1.79300000e-01,   1.66500000e-01,   1.58300000e-01,
          1.47200000e-01,   1.39100000e-01,   1.24300000e-01,
          1.13600000e-01,   9.86200000e-02,   8.83400000e-02,
          8.06500000e-02,   7.46000000e-02,   6.54900000e-02,
          5.89000000e-02,   5.26600000e-02,   4.79100000e-02,
          4.10800000e-02,   3.28400000e-02,   2.79800000e-02,
          2.47600000e-02,   2.24800000e-02,   1.94500000e-02,
          1.75500000e-02,   1.49500000e-02,   1.36800000e-02],
       [  2.21100000e+03,   7.00200000e+02,   3.02600000e+02,
          9.03300000e+01,   3.77800000e+01,   1.91200000e+01,
          1.09500000e+01,   4.57600000e+00,   2.37300000e+00,
          8.07100000e-01,   4.42000000e-01,   2.56200000e-01,
          2.07600000e-01,   1.87100000e-01,   1.75300000e-01,
          1.61000000e-01,   1.51400000e-01,   1.34700000e-01,
          1.22900000e-01,   1.06600000e-01,   9.54600000e-02,
          8.71500000e-02,   8.05800000e-02,   7.07600000e-02,
          6.36100000e-02,   5.69000000e-02,   5.17900000e-02,
          4.44200000e-02,   3.56200000e-02,   3.04700000e-02,
          2.70800000e-02,   2.46900000e-02,   2.15400000e-02,
          1.95900000e-02,   1.69800000e-02,   1.57500000e-02],
       [  3.31100000e+03,   1.08300000e+03,   4.76900000e+02,
          1.45600000e+02,   6.16600000e+01,   3.14400000e+01,
          1.80900000e+01,   7.56200000e+00,   3.87900000e+00,
          1.23600000e+00,   6.17800000e-01,   3.06600000e-01,
          2.28800000e-01,   1.98000000e-01,   1.81700000e-01,
          1.63900000e-01,   1.52900000e-01,   1.35300000e-01,
          1.23300000e-01,   1.06800000e-01,   9.55700000e-02,
          8.71900000e-02,   8.06300000e-02,   7.08100000e-02,
          6.36400000e-02,   5.69300000e-02,   5.18000000e-02,
          4.45000000e-02,   3.57900000e-02,   3.07300000e-02,
          2.74200000e-02,   2.51100000e-02,   2.20900000e-02,
          2.02400000e-02,   1.78200000e-02,   1.67300000e-02],
       [  4.59000000e+03,   1.54900000e+03,   6.94900000e+02,
          2.17100000e+02,   9.31500000e+01,   4.79000000e+01,
          2.77000000e+01,   1.16300000e+01,   5.95200000e+00,
          1.83600000e+00,   8.65100000e-01,   3.77900000e-01,
          2.58500000e-01,   2.13200000e-01,   1.90700000e-01,
          1.67800000e-01,   1.55100000e-01,   1.36100000e-01,
          1.23700000e-01,   1.07000000e-01,   9.56600000e-02,
          8.72900000e-02,   8.07000000e-02,   7.08700000e-02,
          6.37200000e-02,   5.69700000e-02,   5.18500000e-02,
          4.45900000e-02,   3.59700000e-02,   3.10000000e-02,
          2.77700000e-02,   2.55200000e-02,   2.26300000e-02,
          2.08900000e-02,   1.86600000e-02,   1.77000000e-02],
       [  5.64900000e+03,   1.97900000e+03,   9.04700000e+02,
          2.88800000e+02,   1.25600000e+02,   6.51400000e+01,
          3.78900000e+01,   1.60200000e+01,   8.20500000e+00,
          2.49200000e+00,   1.13300000e+00,   4.48700000e-01,
          2.82800000e-01,   2.21400000e-01,   1.92000000e-01,
          1.63900000e-01,   1.49600000e-01,   1.29800000e-01,
          1.17600000e-01,   1.01500000e-01,   9.07300000e-02,
          8.27400000e-02,   7.64900000e-02,   6.71700000e-02,
          6.03700000e-02,   5.39900000e-02,   4.91500000e-02,
          4.22800000e-02,   3.42200000e-02,   2.96000000e-02,
          2.66300000e-02,   2.45700000e-02,   2.19500000e-02,
          2.03900000e-02,   1.84600000e-02,   1.76900000e-02],
       [  7.40900000e+03,   2.66600000e+03,   1.24300000e+03,
          4.05100000e+02,   1.78500000e+02,   9.33900000e+01,
          5.46700000e+01,   2.32800000e+01,   1.19700000e+01,
          3.61300000e+00,   1.60600000e+00,   5.92300000e-01,
          3.47300000e-01,   2.57900000e-01,   2.16100000e-01,
          1.78100000e-01,   1.60000000e-01,   1.37000000e-01,
          1.23600000e-01,   1.06400000e-01,   9.50200000e-02,
          8.66400000e-02,   8.00600000e-02,   7.02900000e-02,
          6.31600000e-02,   5.64600000e-02,   5.14500000e-02,
          4.43000000e-02,   3.59400000e-02,   3.12200000e-02,
          2.81800000e-02,   2.61000000e-02,   2.34800000e-02,
          2.19700000e-02,   2.01300000e-02,   1.94600000e-02],
       [  6.54200000e+02,   3.19400000e+03,   1.52100000e+03,
          5.07000000e+02,   2.26100000e+02,   1.19400000e+02,
          7.03000000e+01,   3.01800000e+01,   1.55700000e+01,
          4.69400000e+00,   2.05700000e+00,   7.19700000e-01,
          3.96900000e-01,   2.80400000e-01,   2.26800000e-01,
          1.79600000e-01,   1.58500000e-01,   1.33500000e-01,
          1.19900000e-01,   1.02900000e-01,   9.18500000e-02,
          8.37200000e-02,   7.73600000e-02,   6.78800000e-02,
          6.10000000e-02,   5.45400000e-02,   4.96800000e-02,
          4.28200000e-02,   3.48700000e-02,   3.03700000e-02,
          2.75300000e-02,   2.55900000e-02,   2.31900000e-02,
          2.18100000e-02,   2.02300000e-02,   1.97000000e-02],
       [  9.22500000e+02,   4.00400000e+03,   1.93200000e+03,
          6.58500000e+02,   2.97400000e+02,   1.58300000e+02,
          9.38100000e+01,   4.06100000e+01,   2.10500000e+01,
          6.35800000e+00,   2.76300000e+00,   9.30600000e-01,
          4.88100000e-01,   3.29200000e-01,   2.57000000e-01,
          1.95100000e-01,   1.68600000e-01,   1.39400000e-01,
          1.24500000e-01,   1.06500000e-01,   9.49200000e-02,
          8.64700000e-02,   7.98800000e-02,   7.00800000e-02,
          6.29600000e-02,   5.62900000e-02,   5.12900000e-02,
          4.42600000e-02,   3.61300000e-02,   3.15900000e-02,
          2.87300000e-02,   2.68100000e-02,   2.44500000e-02,
          2.31300000e-02,   2.16800000e-02,   2.12700000e-02],
       [  1.18500000e+03,   4.02200000e+02,   2.26300000e+03,
          7.88000000e+02,   3.60500000e+02,   1.93400000e+02,
          1.15300000e+02,   5.03300000e+01,   2.62300000e+01,
          7.95500000e+00,   3.44100000e+00,   1.12800000e+00,
          5.68500000e-01,   3.68100000e-01,   2.77800000e-01,
          2.01800000e-01,   1.70400000e-01,   1.37800000e-01,
          1.22300000e-01,   1.04200000e-01,   9.27600000e-02,
          8.44500000e-02,   7.80200000e-02,   6.84100000e-02,
          6.14600000e-02,   5.49600000e-02,   5.00600000e-02,
          4.32400000e-02,   3.54100000e-02,   3.10600000e-02,
          2.83600000e-02,   2.65500000e-02,   2.43700000e-02,
          2.31800000e-02,   2.19500000e-02,   2.16800000e-02],
       [  1.57000000e+03,   5.35500000e+02,   2.77700000e+03,
          9.78400000e+02,   4.52900000e+02,   2.45000000e+02,
          1.47000000e+02,   6.46800000e+01,   3.38900000e+01,
          1.03400000e+01,   4.46400000e+00,   1.43600000e+00,
          7.01200000e-01,   4.38500000e-01,   3.20700000e-01,
          2.22800000e-01,   1.83500000e-01,   1.44800000e-01,
          1.27500000e-01,   1.08200000e-01,   9.61400000e-02,
          8.74800000e-02,   8.07700000e-02,   7.08200000e-02,
          6.36100000e-02,   5.68800000e-02,   5.18300000e-02,
          4.48000000e-02,   3.67800000e-02,   3.24000000e-02,
          2.96700000e-02,   2.78800000e-02,   2.57400000e-02,
          2.46200000e-02,   2.35200000e-02,   2.33800000e-02],
       [  1.91300000e+03,   6.54700000e+02,   3.01800000e+02,
          1.11800000e+03,   5.24200000e+02,   2.86000000e+02,
          1.72600000e+02,   7.66000000e+01,   4.03500000e+01,
          1.23900000e+01,   5.35200000e+00,   1.70000000e+00,
          8.09600000e-01,   4.91600000e-01,   3.49400000e-01,
          2.32400000e-01,   1.86500000e-01,   1.43200000e-01,
          1.25000000e-01,   1.05500000e-01,   9.35900000e-02,
          8.51100000e-02,   7.85400000e-02,   6.88400000e-02,
          6.18200000e-02,   5.52600000e-02,   5.03900000e-02,
          4.35800000e-02,   3.59000000e-02,   3.17200000e-02,
          2.91500000e-02,   2.74700000e-02,   2.55200000e-02,
          2.45200000e-02,   2.36400000e-02,   2.36300000e-02],
       [  2.42900000e+03,   8.34200000e+02,   3.85300000e+02,
          1.33900000e+03,   6.33800000e+02,   3.48700000e+02,
          2.11600000e+02,   9.46500000e+01,   5.01200000e+01,
          1.55000000e+01,   6.70800000e+00,   2.11300000e+00,
          9.87200000e-01,   5.84900000e-01,   4.05300000e-01,
          2.58500000e-01,   2.02000000e-01,   1.50600000e-01,
          1.30200000e-01,   1.09100000e-01,   9.66500000e-02,
          8.78100000e-02,   8.10200000e-02,   7.09800000e-02,
          6.37300000e-02,   5.69700000e-02,   5.19300000e-02,
          4.49800000e-02,   3.71500000e-02,   3.29300000e-02,
          3.03600000e-02,   2.87200000e-02,   2.68200000e-02,
          2.58900000e-02,   2.51700000e-02,   2.52900000e-02],
       [  2.83200000e+03,   9.77100000e+02,   4.52000000e+02,
          1.47300000e+03,   7.03700000e+02,   3.90100000e+02,
          2.38400000e+02,   1.07500000e+02,   5.72500000e+01,
          1.78400000e+01,   7.73900000e+00,   2.42500000e+00,
          1.11700000e+00,   6.48300000e-01,   4.39500000e-01,
          2.69600000e-01,   2.05000000e-01,   1.48000000e-01,
          1.26600000e-01,   1.05400000e-01,   9.31100000e-02,
          8.45300000e-02,   7.79500000e-02,   6.82600000e-02,
          6.12800000e-02,   5.47800000e-02,   4.99400000e-02,
          4.32800000e-02,   3.58500000e-02,   3.18800000e-02,
          2.95000000e-02,   2.79800000e-02,   2.62800000e-02,
          2.54900000e-02,   2.49600000e-02,   2.52000000e-02],
       [  3.18400000e+03,   1.10500000e+03,   5.12000000e+02,
          1.70300000e+02,   7.57200000e+02,   4.22500000e+02,
          2.59300000e+02,   1.18000000e+02,   6.31600000e+01,
          1.98300000e+01,   8.62900000e+00,   2.69700000e+00,
          1.22800000e+00,   7.01200000e-01,   4.66400000e-01,
          2.76000000e-01,   2.04300000e-01,   1.42700000e-01,
          1.20500000e-01,   9.95300000e-02,   8.77600000e-02,
          7.95800000e-02,   7.33500000e-02,   6.41900000e-02,
          5.76200000e-02,   5.15000000e-02,   4.69500000e-02,
          4.07400000e-02,   3.38400000e-02,   3.01900000e-02,
          2.80200000e-02,   2.66700000e-02,   2.51700000e-02,
          2.45100000e-02,   2.41800000e-02,   2.45300000e-02],
       [  4.05800000e+03,   1.41800000e+03,   6.59200000e+02,
          2.19800000e+02,   9.25600000e+02,   5.18900000e+02,
          3.20500000e+02,   1.46900000e+02,   7.90700000e+01,
          2.50300000e+01,   1.09300000e+01,   3.41300000e+00,
          1.54100000e+00,   8.67900000e-01,   5.67800000e-01,
          3.25100000e-01,   2.34500000e-01,   1.58200000e-01,
          1.31900000e-01,   1.08000000e-01,   9.49500000e-02,
          8.60000000e-02,   7.92200000e-02,   6.92900000e-02,
          6.21600000e-02,   5.55600000e-02,   5.06800000e-02,
          4.39900000e-02,   3.66600000e-02,   3.28200000e-02,
          3.05400000e-02,   2.91500000e-02,   2.76600000e-02,
          2.70400000e-02,   2.68700000e-02,   2.73700000e-02],
       [  4.86700000e+03,   1.71400000e+03,   7.99900000e+02,
          2.67600000e+02,   1.21800000e+02,   6.02600000e+02,
          3.73100000e+02,   1.72600000e+02,   9.34100000e+01,
          2.97900000e+01,   1.30600000e+01,   4.08000000e+00,
          1.83000000e+00,   1.01900000e+00,   6.57800000e-01,
          3.65600000e-01,   2.57100000e-01,   1.67400000e-01,
          1.37600000e-01,   1.11600000e-01,   9.78300000e-02,
          8.85100000e-02,   8.14800000e-02,   7.12200000e-02,
          6.38800000e-02,   5.70900000e-02,   5.20700000e-02,
          4.52400000e-02,   3.78000000e-02,   3.39500000e-02,
          3.17000000e-02,   3.03500000e-02,   2.89200000e-02,
          2.83900000e-02,   2.83800000e-02,   2.90300000e-02],
       [  5.23800000e+03,   1.85800000e+03,   8.70600000e+02,
          2.92200000e+02,   1.33200000e+02,   6.30500000e+02,
          3.93300000e+02,   1.82800000e+02,   9.95200000e+01,
          3.20200000e+01,   1.40900000e+01,   4.40900000e+00,
          1.96900000e+00,   1.08700000e+00,   6.93200000e-01,
          3.75300000e-01,   2.57700000e-01,   1.61900000e-01,
          1.31000000e-01,   1.05200000e-01,   9.19300000e-02,
          8.30500000e-02,   7.63900000e-02,   6.67500000e-02,
          5.98500000e-02,   5.34700000e-02,   4.87800000e-02,
          4.24300000e-02,   3.55400000e-02,   3.20200000e-02,
          2.99900000e-02,   2.87800000e-02,   2.75600000e-02,
          2.71500000e-02,   2.73200000e-02,   2.80400000e-02],
       [  5.86900000e+03,   2.09600000e+03,   9.86000000e+02,
          3.32300000e+02,   1.51700000e+02,   6.83800000e+02,
          4.32300000e+02,   2.02300000e+02,   1.10700000e+02,
          3.58700000e+01,   1.58500000e+01,   4.97200000e+00,
          2.21400000e+00,   1.21300000e+00,   7.66100000e-01,
          4.05200000e-01,   2.72100000e-01,   1.64900000e-01,
          1.31400000e-01,   1.04300000e-01,   9.08100000e-02,
          8.19100000e-02,   7.52900000e-02,   6.57200000e-02,
          5.89100000e-02,   5.26300000e-02,   4.80100000e-02,
          4.18000000e-02,   3.51200000e-02,   3.17300000e-02,
          2.98200000e-02,   2.86800000e-02,   2.75900000e-02,
          2.72700000e-02,   2.76200000e-02,   2.84400000e-02],
       [  6.49500000e+03,   2.34200000e+03,   1.10600000e+03,
          3.74300000e+02,   1.71200000e+02,   9.29100000e+01,
          4.68700000e+02,   2.21700000e+02,   1.21800000e+02,
          3.98300000e+01,   1.76800000e+01,   5.56400000e+00,
          2.47200000e+00,   1.34700000e+00,   8.43800000e-01,
          4.37100000e-01,   2.87700000e-01,   1.68200000e-01,
          1.31800000e-01,   1.03400000e-01,   8.96500000e-02,
          8.07400000e-02,   7.41400000e-02,   6.46600000e-02,
          5.79400000e-02,   5.17500000e-02,   4.72200000e-02,
          4.11500000e-02,   3.46600000e-02,   3.14100000e-02,
          2.96000000e-02,   2.85500000e-02,   2.75900000e-02,
          2.73800000e-02,   2.78600000e-02,   2.87700000e-02],
       [  7.40500000e+03,   2.69400000e+03,   1.27700000e+03,
          4.33900000e+02,   1.98800000e+02,   1.08000000e+02,
          5.16000000e+02,   2.51300000e+02,   1.38600000e+02,
          4.57100000e+01,   2.03800000e+01,   6.43400000e+00,
          2.85600000e+00,   1.55000000e+00,   9.63900000e-01,
          4.90500000e-01,   3.16600000e-01,   1.78800000e-01,
          1.37800000e-01,   1.06700000e-01,   9.21300000e-02,
          8.28100000e-02,   7.59800000e-02,   6.62000000e-02,
          5.93000000e-02,   5.29500000e-02,   4.83200000e-02,
          4.21300000e-02,   3.55900000e-02,   3.23500000e-02,
          3.05700000e-02,   2.95600000e-02,   2.86900000e-02,
          2.85500000e-02,   2.92000000e-02,   3.02600000e-02],
       [  8.09300000e+03,   2.98400000e+03,   1.42100000e+03,
          4.85100000e+02,   2.22900000e+02,   1.21200000e+02,
          7.35000000e+01,   2.73400000e+02,   1.51400000e+02,
          5.02700000e+01,   2.25300000e+01,   7.14100000e+00,
          3.16900000e+00,   1.71400000e+00,   1.06000000e+00,
          5.30600000e-01,   3.36700000e-01,   1.83800000e-01,
          1.39100000e-01,   1.06200000e-01,   9.13300000e-02,
          8.19200000e-02,   7.50900000e-02,   6.53700000e-02,
          5.85200000e-02,   5.22400000e-02,   4.76900000e-02,
          4.16200000e-02,   3.52400000e-02,   3.21300000e-02,
          3.04500000e-02,   2.95200000e-02,   2.87500000e-02,
          2.87100000e-02,   2.95100000e-02,   3.06800000e-02],
       [  9.08500000e+03,   3.39900000e+03,   1.62600000e+03,
          5.57600000e+02,   2.56700000e+02,   1.39800000e+02,
          8.48400000e+01,   3.05600000e+02,   1.70600000e+02,
          5.70800000e+01,   2.56800000e+01,   8.17600000e+00,
          3.62900000e+00,   1.95800000e+00,   1.20500000e+00,
          5.95200000e-01,   3.71700000e-01,   1.96400000e-01,
          1.46000000e-01,   1.09900000e-01,   9.40000000e-02,
          8.41400000e-02,   7.70400000e-02,   6.69900000e-02,
          5.99500000e-02,   5.35000000e-02,   4.88300000e-02,
          4.26500000e-02,   3.62100000e-02,   3.31200000e-02,
          3.14600000e-02,   3.05700000e-02,   2.99100000e-02,
          2.99400000e-02,   3.09200000e-02,   3.22400000e-02],
       [  9.79600000e+03,   3.69700000e+03,   1.77900000e+03,
          6.12900000e+02,   2.83000000e+02,   1.54300000e+02,
          9.37000000e+01,   3.24800000e+02,   1.84100000e+02,
          6.20100000e+01,   2.80300000e+01,   8.96200000e+00,
          3.98100000e+00,   2.14400000e+00,   1.31400000e+00,
          6.41400000e-01,   3.94900000e-01,   2.02300000e-01,
          1.47600000e-01,   1.09400000e-01,   9.31100000e-02,
          8.31500000e-02,   7.60400000e-02,   6.60400000e-02,
          5.90600000e-02,   5.27000000e-02,   4.81000000e-02,
          4.20400000e-02,   3.58000000e-02,   3.28300000e-02,
          3.12700000e-02,   3.04500000e-02,   2.99100000e-02,
          3.00200000e-02,   3.11500000e-02,   3.25600000e-02],
       [  9.85500000e+03,   4.23400000e+03,   2.04900000e+03,
          7.09400000e+02,   3.28200000e+02,   1.79300000e+02,
          1.09000000e+02,   4.95200000e+01,   2.09000000e+02,
          7.08100000e+01,   3.22000000e+01,   1.03400000e+01,
          4.60000000e+00,   2.47400000e+00,   1.51200000e+00,
          7.30600000e-01,   4.44000000e-01,   2.20800000e-01,
          1.58200000e-01,   1.15400000e-01,   9.76500000e-02,
          8.69800000e-02,   7.94400000e-02,   6.89100000e-02,
          6.16000000e-02,   5.49400000e-02,   5.01500000e-02,
          4.38700000e-02,   3.74500000e-02,   3.44400000e-02,
          3.28900000e-02,   3.21000000e-02,   3.16400000e-02,
          3.18500000e-02,   3.32000000e-02,   3.47600000e-02],
       [  1.05700000e+04,   4.41800000e+03,   2.15400000e+03,
          7.48800000e+02,   3.47300000e+02,   1.89900000e+02,
          1.15600000e+02,   5.25500000e+01,   2.15900000e+02,
          7.40500000e+01,   3.37900000e+01,   1.09200000e+01,
          4.86200000e+00,   2.61300000e+00,   1.59300000e+00,
          7.63000000e-01,   4.58400000e-01,   2.21700000e-01,
          1.55900000e-01,   1.11900000e-01,   9.41300000e-02,
          8.36200000e-02,   7.62500000e-02,   6.60500000e-02,
          5.90100000e-02,   5.26100000e-02,   4.80300000e-02,
          4.20500000e-02,   3.59900000e-02,   3.31800000e-02,
          3.17700000e-02,   3.10800000e-02,   3.07400000e-02,
          3.10300000e-02,   3.24700000e-02,   3.40800000e-02],
       [  1.55300000e+03,   4.82500000e+03,   2.37500000e+03,
          8.31100000e+02,   3.86500000e+02,   2.11800000e+02,
          1.29000000e+02,   5.87500000e+01,   2.33100000e+02,
          8.11700000e+01,   3.71900000e+01,   1.20700000e+01,
          5.38400000e+00,   2.89200000e+00,   1.76000000e+00,
          8.36400000e-01,   4.97300000e-01,   2.34100000e-01,
          1.61700000e-01,   1.14100000e-01,   9.53900000e-02,
          8.45000000e-02,   7.69500000e-02,   6.65600000e-02,
          5.94100000e-02,   5.29600000e-02,   4.83400000e-02,
          4.23500000e-02,   3.63400000e-02,   3.36000000e-02,
          3.22500000e-02,   3.16000000e-02,   3.13800000e-02,
          3.17500000e-02,   3.33500000e-02,   3.50900000e-02],
       [  1.69700000e+03,   5.08700000e+03,   2.51500000e+03,
          8.85700000e+02,   4.13000000e+02,   2.26600000e+02,
          1.38200000e+02,   6.30200000e+01,   3.42100000e+01,
          8.53700000e+01,   3.92800000e+01,   1.28100000e+01,
          5.72600000e+00,   3.07600000e+00,   1.86800000e+00,
          8.82300000e-01,   5.19700000e-01,   2.38700000e-01,
          1.61900000e-01,   1.12300000e-01,   9.32500000e-02,
          8.23600000e-02,   7.48700000e-02,   6.46600000e-02,
          5.76700000e-02,   5.13900000e-02,   4.69200000e-02,
          4.11300000e-02,   3.53800000e-02,   3.28000000e-02,
          3.15600000e-02,   3.09900000e-02,   3.08600000e-02,
          3.13000000e-02,   3.30000000e-02,   3.47900000e-02],
       [  1.89300000e+03,   5.47500000e+03,   2.71100000e+03,
          9.61300000e+02,   4.49700000e+02,   2.47200000e+02,
          1.50900000e+02,   6.89000000e+01,   3.74200000e+01,
          9.15200000e+01,   4.22200000e+01,   1.38500000e+01,
          6.20700000e+00,   3.33500000e+00,   2.02300000e+00,
          9.50100000e-01,   5.55000000e-01,   2.49100000e-01,
          1.66100000e-01,   1.13100000e-01,   9.32700000e-02,
          8.21200000e-02,   7.45200000e-02,   6.42600000e-02,
          5.72700000e-02,   5.10100000e-02,   4.65700000e-02,
          4.08600000e-02,   3.52400000e-02,   3.27500000e-02,
          3.15800000e-02,   3.10700000e-02,   3.10300000e-02,
          3.15600000e-02,   3.34000000e-02,   3.52800000e-02],
       [  2.12100000e+03,   5.22700000e+03,   2.93100000e+03,
          1.04900000e+03,   4.92000000e+02,   2.70900000e+02,
          1.65600000e+02,   7.57300000e+01,   4.11500000e+01,
          9.85600000e+01,   4.56400000e+01,   1.50600000e+01,
          6.76000000e+00,   3.63500000e+00,   2.20300000e+00,
          1.03000000e+00,   5.97100000e-01,   2.62200000e-01,
          1.71900000e-01,   1.15000000e-01,   9.41400000e-02,
          8.25900000e-02,   7.48300000e-02,   6.44000000e-02,
          5.73500000e-02,   5.10600000e-02,   4.66100000e-02,
          4.09300000e-02,   3.53900000e-02,   3.29600000e-02,
          3.18700000e-02,   3.14100000e-02,   3.14600000e-02,
          3.20700000e-02,   3.40500000e-02,   3.60300000e-02],
       [  2.31700000e+03,   5.33600000e+03,   3.09800000e+03,
          1.11600000e+03,   5.25200000e+02,   2.89600000e+02,
          1.77300000e+02,   8.11600000e+01,   4.41400000e+01,
          1.03300000e+02,   4.81800000e+01,   1.59600000e+01,
          7.18400000e+00,   3.86400000e+00,   2.34100000e+00,
          1.09000000e+00,   6.27800000e-01,   2.70300000e-01,
          1.74200000e-01,   1.14400000e-01,   9.29900000e-02,
          8.12900000e-02,   7.35000000e-02,   6.31400000e-02,
          5.61900000e-02,   4.99900000e-02,   4.56400000e-02,
          4.01000000e-02,   3.47600000e-02,   3.24700000e-02,
          3.14500000e-02,   3.10500000e-02,   3.11900000e-02,
          3.18600000e-02,   3.39500000e-02,   3.59900000e-02],
       [  2.62400000e+03,   1.00200000e+03,   3.40700000e+03,
          1.23100000e+03,   5.81500000e+02,   3.21300000e+02,
          1.96800000e+02,   9.02600000e+01,   4.91200000e+01,
          1.11900000e+02,   5.26600000e+01,   1.75300000e+01,
          7.90000000e+00,   4.26400000e+00,   2.58200000e+00,
          1.19800000e+00,   6.86100000e-01,   2.89900000e-01,
          1.83800000e-01,   1.18600000e-01,   9.56300000e-02,
          8.32800000e-02,   7.51500000e-02,   6.44300000e-02,
          5.72800000e-02,   5.09400000e-02,   4.65000000e-02,
          4.08900000e-02,   3.55200000e-02,   3.32700000e-02,
          3.22900000e-02,   3.19400000e-02,   3.21800000e-02,
          3.29300000e-02,   3.52100000e-02,   3.73800000e-02],
       [  2.85400000e+03,   1.09300000e+03,   3.59900000e+03,
          1.30500000e+03,   6.18600000e+02,   3.42500000e+02,
          2.10100000e+02,   9.65100000e+01,   5.25700000e+01,
          1.16800000e+02,   5.54800000e+01,   1.85400000e+01,
          8.38900000e+00,   4.52300000e+00,   2.73900000e+00,
          1.26700000e+00,   7.22100000e-01,   2.99800000e-01,
          1.87200000e-01,   1.18600000e-01,   9.48000000e-02,
          8.22600000e-02,   7.41000000e-02,   6.34000000e-02,
          5.63100000e-02,   5.00500000e-02,   4.56900000e-02,
          4.02000000e-02,   3.50100000e-02,   3.28600000e-02,
          3.19600000e-02,   3.16800000e-02,   3.19900000e-02,
          3.28000000e-02,   3.51800000e-02,   3.74100000e-02],
       [  3.17400000e+03,   1.21900000e+03,   3.41000000e+03,
          1.41800000e+03,   6.74800000e+02,   3.74400000e+02,
          2.30000000e+02,   1.05800000e+02,   5.76600000e+01,
          1.90900000e+01,   5.98000000e+01,   2.00900000e+01,
          9.11200000e+00,   4.91800000e+00,   2.97900000e+00,
          1.37500000e+00,   7.79900000e-01,   3.18700000e-01,
          1.96000000e-01,   1.21900000e-01,   9.67000000e-02,
          8.36000000e-02,   7.51000000e-02,   6.41200000e-02,
          5.68900000e-02,   5.05300000e-02,   4.61300000e-02,
          4.06100000e-02,   3.54500000e-02,   3.33500000e-02,
          3.25000000e-02,   3.22700000e-02,   3.26700000e-02,
          3.35700000e-02,   3.61000000e-02,   3.84500000e-02],
       [  3.49400000e+03,   1.34700000e+03,   2.58900000e+03,
          1.52500000e+03,   7.29700000e+02,   4.05800000e+02,
          2.49600000e+02,   1.15000000e+02,   6.27400000e+01,
          2.07900000e+01,   6.38600000e+01,   2.15700000e+01,
          9.81800000e+00,   5.30600000e+00,   3.21400000e+00,
          1.48100000e+00,   8.36500000e-01,   3.36900000e-01,
          2.04200000e-01,   1.24700000e-01,   9.81100000e-02,
          8.44300000e-02,   7.57000000e-02,   6.44700000e-02,
          5.71400000e-02,   5.07200000e-02,   4.63000000e-02,
          4.07900000e-02,   3.56900000e-02,   3.36500000e-02,
          3.28600000e-02,   3.26800000e-02,   3.31700000e-02,
          3.41400000e-02,   3.68300000e-02,   3.92700000e-02],
       [  3.86400000e+03,   1.49300000e+03,   7.42200000e+02,
          1.65400000e+03,   7.93600000e+02,   4.42400000e+02,
          2.72500000e+02,   1.25800000e+02,   6.87100000e+01,
          2.27900000e+01,   6.85500000e+01,   2.33000000e+01,
          1.06500000e+01,   5.76400000e+00,   3.49300000e+00,
          1.60700000e+00,   9.04700000e-01,   3.59500000e-01,
          2.14900000e-01,   1.28900000e-01,   1.00600000e-01,
          8.61300000e-02,   7.70300000e-02,   6.54600000e-02,
          5.79500000e-02,   5.14100000e-02,   4.69200000e-02,
          4.13700000e-02,   3.62800000e-02,   3.42800000e-02,
          3.35500000e-02,   3.34100000e-02,   3.39900000e-02,
          3.50400000e-02,   3.79000000e-02,   4.04800000e-02],
       [  4.21000000e+03,   1.63100000e+03,   8.11500000e+02,
          1.77200000e+03,   8.50700000e+02,   4.75500000e+02,
          2.93500000e+02,   1.35600000e+02,   7.41700000e+01,
          2.46300000e+01,   7.23700000e+01,   2.48500000e+01,
          1.13900000e+01,   6.17300000e+00,   3.74400000e+00,
          1.72100000e+00,   9.65800000e-01,   3.79000000e-01,
          2.23700000e-01,   1.31800000e-01,   1.01800000e-01,
          8.69300000e-02,   7.75600000e-02,   6.57100000e-02,
          5.81000000e-02,   5.15000000e-02,   4.70000000e-02,
          4.14600000e-02,   3.64400000e-02,   3.45100000e-02,
          3.38400000e-02,   3.37400000e-02,   3.44100000e-02,
          3.55400000e-02,   3.85500000e-02,   4.12200000e-02],
       [  4.60000000e+03,   1.78600000e+03,   8.89300000e+02,
          1.90600000e+03,   9.16400000e+02,   5.13400000e+02,
          3.17200000e+02,   1.46900000e+02,   8.03800000e+01,
          2.67200000e+01,   7.71200000e+01,   2.66600000e+01,
          1.22300000e+01,   6.64400000e+00,   4.03200000e+00,
          1.85200000e+00,   1.03700000e+00,   4.02300000e-01,
          2.34400000e-01,   1.35700000e-01,   1.04000000e-01,
          8.83100000e-02,   7.85800000e-02,   6.64200000e-02,
          5.86600000e-02,   5.19600000e-02,   4.74100000e-02,
          4.18500000e-02,   3.68600000e-02,   3.49800000e-02,
          3.43600000e-02,   3.43200000e-02,   3.50800000e-02,
          3.62800000e-02,   3.94400000e-02,   4.22400000e-02],
       [  4.94200000e+03,   1.92500000e+03,   9.59300000e+02,
          2.01100000e+03,   9.70300000e+02,   5.45000000e+02,
          3.37300000e+02,   1.56500000e+02,   8.57600000e+01,
          2.85400000e+01,   8.05400000e+01,   2.81000000e+01,
          1.29400000e+01,   7.03700000e+00,   4.27400000e+00,
          1.96200000e+00,   1.09600000e+00,   4.20800000e-01,
          2.42300000e-01,   1.37900000e-01,   1.04700000e-01,
          8.84800000e-02,   7.85100000e-02,   6.61900000e-02,
          5.83700000e-02,   5.16700000e-02,   4.71300000e-02,
          4.16300000e-02,   3.67500000e-02,   3.49600000e-02,
          3.43900000e-02,   3.44000000e-02,   3.52300000e-02,
          3.65000000e-02,   3.97800000e-02,   4.26400000e-02],
       [  5.35600000e+03,   2.09200000e+03,   1.04400000e+03,
          1.86200000e+03,   1.03600000e+03,   5.83600000e+02,
          3.61900000e+02,   1.68300000e+02,   9.23100000e+01,
          3.07600000e+01,   1.41000000e+01,   2.99300000e+01,
          1.38100000e+01,   7.52100000e+00,   4.57100000e+00,
          2.09900000e+00,   1.16900000e+00,   4.44900000e-01,
          2.53400000e-01,   1.41800000e-01,   1.06600000e-01,
          8.96800000e-02,   7.93500000e-02,   6.67400000e-02,
          5.87600000e-02,   5.19700000e-02,   4.74000000e-02,
          4.18900000e-02,   3.70500000e-02,   3.53200000e-02,
          3.48200000e-02,   3.48700000e-02,   3.57800000e-02,
          3.71200000e-02,   4.05500000e-02,   4.35300000e-02],
       [  5.71800000e+03,   2.24000000e+03,   1.12000000e+03,
          1.96300000e+03,   1.09500000e+03,   6.16500000e+02,
          3.83200000e+02,   1.78500000e+02,   9.80000000e+01,
          3.27000000e+01,   1.49900000e+01,   3.13900000e+01,
          1.45200000e+01,   7.92600000e+00,   4.82200000e+00,
          2.21500000e+00,   1.23200000e+00,   4.64700000e-01,
          2.61800000e-01,   1.44000000e-01,   1.07400000e-01,
          8.99200000e-02,   7.93300000e-02,   6.64700000e-02,
          5.84600000e-02,   5.16600000e-02,   4.71000000e-02,
          4.16400000e-02,   3.69200000e-02,   3.52700000e-02,
          3.48200000e-02,   3.49100000e-02,   3.59100000e-02,
          3.73000000e-02,   4.08400000e-02,   4.38700000e-02],
       [  6.16900000e+03,   2.42600000e+03,   1.21400000e+03,
          4.44100000e+02,   1.17000000e+03,   6.58900000e+02,
          4.10100000e+02,   1.91500000e+02,   1.05300000e+02,
          3.51800000e+01,   1.61300000e+01,   3.33000000e+01,
          1.54400000e+01,   8.44800000e+00,   5.14700000e+00,
          2.36500000e+00,   1.31400000e+00,   4.91600000e-01,
          2.74200000e-01,   1.48500000e-01,   1.09700000e-01,
          9.13400000e-02,   8.03500000e-02,   6.71100000e-02,
          5.89400000e-02,   5.20400000e-02,   4.74400000e-02,
          4.19700000e-02,   3.72800000e-02,   3.56800000e-02,
          3.52900000e-02,   3.54200000e-02,   3.65000000e-02,
          3.79700000e-02,   4.16600000e-02,   4.48100000e-02],
       [  6.53800000e+03,   2.57900000e+03,   1.29200000e+03,
          4.73000000e+02,   1.22700000e+03,   6.91200000e+02,
          4.30800000e+02,   2.01700000e+02,   1.11000000e+02,
          3.71500000e+01,   1.70400000e+01,   3.46500000e+01,
          1.61400000e+01,   8.85000000e+00,   5.39900000e+00,
          2.48100000e+00,   1.37700000e+00,   5.11500000e-01,
          2.82700000e-01,   1.50600000e-01,   1.10300000e-01,
          9.13400000e-02,   8.01000000e-02,   6.66900000e-02,
          5.84900000e-02,   5.15900000e-02,   4.70200000e-02,
          4.16200000e-02,   3.70400000e-02,   3.55200000e-02,
          3.51800000e-02,   3.53700000e-02,   3.65100000e-02,
          3.80200000e-02,   4.18100000e-02,   4.50100000e-02],
       [  7.03900000e+03,   2.79000000e+03,   1.40100000e+03,
          5.13600000e+02,   1.30500000e+03,   7.38500000e+02,
          4.61000000e+02,   2.16400000e+02,   1.19300000e+02,
          3.99800000e+01,   1.83600000e+01,   3.66800000e+01,
          1.72000000e+01,   9.44400000e+00,   5.76600000e+00,
          2.65100000e+00,   1.47000000e+00,   5.42600000e-01,
          2.97200000e-01,   1.56000000e-01,   1.13100000e-01,
          9.32100000e-02,   8.15300000e-02,   6.76600000e-02,
          5.92100000e-02,   5.21700000e-02,   4.75400000e-02,
          4.20900000e-02,   3.75400000e-02,   3.60600000e-02,
          3.57700000e-02,   3.60100000e-02,   3.72300000e-02,
          3.88200000e-02,   4.27600000e-02,   4.60900000e-02],
       [  7.35000000e+03,   2.93100000e+03,   1.47300000e+03,
          5.41400000e+02,   1.17000000e+03,   7.68500000e+02,
          4.79300000e+02,   2.25400000e+02,   1.24400000e+02,
          4.17800000e+01,   1.92000000e+01,   3.76500000e+01,
          1.77800000e+01,   9.77900000e+00,   5.97500000e+00,
          2.75100000e+00,   1.52400000e+00,   5.59300000e-01,
          3.03800000e-01,   1.57100000e-01,   1.12900000e-01,
          9.25000000e-02,   8.06400000e-02,   6.67000000e-02,
          5.82600000e-02,   5.12900000e-02,   4.67300000e-02,
          4.13900000e-02,   3.69800000e-02,   3.55900000e-02,
          3.53600000e-02,   3.56300000e-02,   3.69100000e-02,
          3.85300000e-02,   4.25300000e-02,   4.58700000e-02],
       [  7.80900000e+03,   3.13100000e+03,   1.57800000e+03,
          5.80800000e+02,   1.23100000e+03,   8.13400000e+02,
          5.07200000e+02,   2.39100000e+02,   1.32100000e+02,
          4.44500000e+01,   2.04400000e+01,   3.94900000e+01,
          1.87300000e+01,   1.03000000e+01,   6.30600000e+00,
          2.90700000e+00,   1.60900000e+00,   5.87600000e-01,
          3.16700000e-01,   1.61400000e-01,   1.14900000e-01,
          9.37100000e-02,   8.13800000e-02,   6.70700000e-02,
          5.84900000e-02,   5.14400000e-02,   4.68400000e-02,
          4.15100000e-02,   3.71500000e-02,   3.58200000e-02,
          3.56400000e-02,   3.59600000e-02,   3.73000000e-02,
          3.89800000e-02,   4.31100000e-02,   4.65400000e-02],
       [  8.15700000e+03,   3.29600000e+03,   1.66500000e+03,
          6.14300000e+02,   9.39300000e+02,   8.47100000e+02,
          5.29400000e+02,   2.50000000e+02,   1.38400000e+02,
          4.66400000e+01,   2.14600000e+01,   4.12100000e+01,
          1.94200000e+01,   1.07000000e+01,   6.56400000e+00,
          3.02900000e+00,   1.67600000e+00,   6.09100000e-01,
          3.26000000e-01,   1.63900000e-01,   1.15600000e-01,
          9.37400000e-02,   8.11300000e-02,   6.66200000e-02,
          5.80000000e-02,   5.09500000e-02,   4.63800000e-02,
          4.11200000e-02,   3.68600000e-02,   3.56100000e-02,
          3.54800000e-02,   3.58300000e-02,   3.72400000e-02,
          3.89500000e-02,   4.31500000e-02,   4.66200000e-02],
       [  8.58200000e+03,   3.49100000e+03,   1.76700000e+03,
          6.53600000e+02,   3.16900000e+02,   8.84600000e+02,
          5.56900000e+02,   2.63100000e+02,   1.45900000e+02,
          4.92300000e+01,   2.26800000e+01,   7.63100000e+00,
          2.02700000e+01,   1.12000000e+01,   6.87900000e+00,
          3.17600000e+00,   1.75800000e+00,   6.36100000e-01,
          3.38100000e-01,   1.67700000e-01,   1.17200000e-01,
          9.45300000e-02,   8.15300000e-02,   6.67000000e-02,
          5.79700000e-02,   5.08600000e-02,   4.62800000e-02,
          4.10500000e-02,   3.68600000e-02,   3.56700000e-02,
          3.55900000e-02,   3.59800000e-02,   3.74500000e-02,
          3.92100000e-02,   4.35100000e-02,   4.70400000e-02],
       [  8.43400000e+03,   3.60800000e+03,   1.83200000e+03,
          6.79200000e+02,   3.29700000e+02,   9.01400000e+02,
          5.72100000e+02,   2.70200000e+02,   1.50100000e+02,
          5.07800000e+01,   2.34100000e+01,   7.87800000e+00,
          2.06400000e+01,   1.14500000e+01,   7.04100000e+00,
          3.25500000e+00,   1.80100000e+00,   6.49200000e-01,
          3.42900000e-01,   1.67900000e-01,   1.16300000e-01,
          9.32800000e-02,   8.02200000e-02,   6.53800000e-02,
          5.66900000e-02,   4.96700000e-02,   4.51800000e-02,
          4.00900000e-02,   3.60600000e-02,   3.49400000e-02,
          3.49200000e-02,   3.53400000e-02,   3.68300000e-02,
          3.86000000e-02,   4.29000000e-02,   4.64200000e-02],
       [  9.09600000e+03,   3.91900000e+03,   1.99700000e+03,
          7.42000000e+02,   3.60700000e+02,   8.43000000e+02,
          6.17300000e+02,   2.92200000e+02,   1.62600000e+02,
          5.51200000e+01,   2.54300000e+01,   8.56100000e+00,
          2.21000000e+01,   1.23200000e+01,   7.57900000e+00,
          3.51000000e+00,   1.94200000e+00,   6.97800000e-01,
          3.66300000e-01,   1.77100000e-01,   1.21700000e-01,
          9.70100000e-02,   8.31300000e-02,   6.74900000e-02,
          5.84100000e-02,   5.11100000e-02,   4.64700000e-02,
          4.12400000e-02,   3.71600000e-02,   3.60700000e-02,
          3.60800000e-02,   3.65500000e-02,   3.81500000e-02,
          4.00200000e-02,   4.45500000e-02,   4.82300000e-02],
       [  9.41300000e+03,   4.08500000e+03,   2.08800000e+03,
          7.78000000e+02,   3.78700000e+02,   6.39200000e+02,
          6.37600000e+02,   3.03200000e+02,   1.69000000e+02,
          5.74300000e+01,   2.65200000e+01,   8.93000000e+00,
          2.27000000e+01,   1.27200000e+01,   7.82500000e+00,
          3.63300000e+00,   2.01100000e+00,   7.20200000e-01,
          3.76000000e-01,   1.79700000e-01,   1.22300000e-01,
          9.69900000e-02,   8.28100000e-02,   6.69600000e-02,
          5.78500000e-02,   5.05400000e-02,   4.59400000e-02,
          4.07800000e-02,   3.68100000e-02,   3.57700000e-02,
          3.58300000e-02,   3.63400000e-02,   3.79700000e-02,
          3.98700000e-02,   4.44500000e-02,   4.81500000e-02],
       [  9.36500000e+03,   4.33500000e+03,   2.22600000e+03,
          8.31900000e+02,   4.05500000e+02,   2.30300000e+02,
          6.71100000e+02,   3.21400000e+02,   1.79300000e+02,
          6.10400000e+01,   2.82200000e+01,   9.50700000e+00,
          2.38100000e+01,   1.34000000e+01,   8.24800000e+00,
          3.83600000e+00,   2.12400000e+00,   7.58900000e-01,
          3.94100000e-01,   1.86300000e-01,   1.25700000e-01,
          9.91200000e-02,   8.43100000e-02,   6.78900000e-02,
          5.85400000e-02,   5.10800000e-02,   4.64100000e-02,
          4.12100000e-02,   3.72500000e-02,   3.62500000e-02,
          3.63500000e-02,   3.68900000e-02,   3.86000000e-02,
          4.05700000e-02,   4.52900000e-02,   4.91000000e-02],
       [  8.54300000e+03,   4.49900000e+03,   2.31900000e+03,
          8.69600000e+02,   4.24600000e+02,   2.41400000e+02,
          6.89800000e+02,   3.33400000e+02,   1.86000000e+02,
          6.34700000e+01,   2.93800000e+01,   9.90400000e+00,
          2.45700000e+01,   1.37900000e+01,   8.51100000e+00,
          3.96300000e+00,   2.19600000e+00,   7.82800000e-01,
          4.04600000e-01,   1.89100000e-01,   1.26500000e-01,
          9.92300000e-02,   8.41000000e-02,   6.74400000e-02,
          5.80300000e-02,   5.05800000e-02,   4.59200000e-02,
          4.07800000e-02,   3.69200000e-02,   3.59800000e-02,
          3.61200000e-02,   3.66900000e-02,   3.84400000e-02,
          4.04200000e-02,   4.51800000e-02,   4.90200000e-02],
       [  9.08700000e+03,   4.77200000e+03,   2.46400000e+03,
          9.26700000e+02,   4.53100000e+02,   2.57800000e+02,
          6.31900000e+02,   3.52900000e+02,   1.96700000e+02,
          6.73100000e+01,   3.11900000e+01,   1.05200000e+01,
          2.57900000e+01,   1.44700000e+01,   8.96200000e+00,
          4.17700000e+00,   2.31500000e+00,   8.23900000e-01,
          4.23900000e-01,   1.96100000e-01,   1.30100000e-01,
          1.01500000e-01,   8.57000000e-02,   6.84300000e-02,
          5.87600000e-02,   5.11000000e-02,   4.64000000e-02,
          4.12200000e-02,   3.73700000e-02,   3.64600000e-02,
          3.66400000e-02,   3.72600000e-02,   3.90700000e-02,
          4.11300000e-02,   4.60300000e-02,   4.99600000e-02],
       [  9.71100000e+03,   5.03300000e+03,   2.60700000e+03,
          9.85700000e+02,   4.81100000e+02,   2.74000000e+02,
          4.90800000e+02,   3.73200000e+02,   2.08200000e+02,
          7.14300000e+01,   3.31200000e+01,   1.11900000e+01,
          5.21500000e+00,   1.52000000e+01,   9.44700000e+00,
          4.40900000e+00,   2.44500000e+00,   8.68700000e-01,
          4.45200000e-01,   2.03900000e-01,   1.34200000e-01,
          1.04100000e-01,   8.75700000e-02,   6.96200000e-02,
          5.96100000e-02,   5.18100000e-02,   4.70100000e-02,
          4.17700000e-02,   3.79200000e-02,   3.70500000e-02,
          3.72700000e-02,   3.79200000e-02,   3.98100000e-02,
          4.19400000e-02,   4.69900000e-02,   5.10300000e-02],
       [  1.05800000e+04,   5.09000000e+03,   2.76800000e+03,
          1.04700000e+03,   5.13100000e+02,   2.92400000e+02,
          5.14500000e+02,   3.95000000e+02,   2.20900000e+02,
          7.59700000e+01,   3.52600000e+01,   1.19200000e+01,
          5.55700000e+00,   1.59900000e+01,   9.97700000e+00,
          4.66400000e+00,   2.58800000e+00,   9.18000000e-01,
          4.68700000e-01,   2.12600000e-01,   1.38900000e-01,
          1.07100000e-01,   8.97600000e-02,   7.10500000e-02,
          6.07000000e-02,   5.26800000e-02,   4.77900000e-02,
          4.24400000e-02,   3.85800000e-02,   3.77400000e-02,
          3.80000000e-02,   3.87000000e-02,   4.06700000e-02,
          4.28800000e-02,   4.80900000e-02,   5.22600000e-02],
       [  6.62700000e+03,   5.27300000e+03,   2.87800000e+03,
          1.09300000e+03,   5.36600000e+02,   3.06100000e+02,
          1.92700000e+02,   4.09400000e+02,   2.30000000e+02,
          7.92500000e+01,   3.68400000e+01,   1.24700000e+01,
          5.80900000e+00,   1.65000000e+01,   1.03300000e+01,
          4.83900000e+00,   2.68700000e+00,   9.52200000e-01,
          4.84400000e-01,   2.17800000e-01,   1.41200000e-01,
          1.08300000e-01,   9.04000000e-02,   7.12700000e-02,
          6.07200000e-02,   5.26200000e-02,   4.76900000e-02,
          4.23600000e-02,   3.85600000e-02,   3.77600000e-02,
          3.80600000e-02,   3.87900000e-02,   4.08100000e-02,
          4.30400000e-02,   4.83200000e-02,   5.25700000e-02],
       [  2.05600000e+03,   5.55300000e+03,   3.04800000e+03,
          1.16200000e+03,   5.70900000e+02,   3.26000000e+02,
          2.05300000e+02,   4.31400000e+02,   2.44000000e+02,
          8.41100000e+01,   3.91600000e+01,   1.32700000e+01,
          6.18100000e+00,   1.73500000e+01,   1.08700000e+01,
          5.10700000e+00,   2.84000000e+00,   1.00500000e+00,
          5.09800000e-01,   2.27300000e-01,   1.46200000e-01,
          1.11500000e-01,   9.27600000e-02,   7.28000000e-02,
          6.18800000e-02,   5.35000000e-02,   4.84900000e-02,
          4.30800000e-02,   3.92500000e-02,   3.84900000e-02,
          3.88200000e-02,   3.95900000e-02,   4.16900000e-02,
          4.40100000e-02,   4.94600000e-02,   5.38000000e-02],
       [  2.10700000e+03,   5.35800000e+03,   3.12000000e+03,
          1.19300000e+03,   5.87300000e+02,   3.35600000e+02,
          2.11500000e+02,   4.40100000e+02,   2.49900000e+02,
          8.63300000e+01,   4.02500000e+01,   1.36500000e+01,
          6.36200000e+00,   1.77400000e+01,   1.10700000e+01,
          5.21200000e+00,   2.90100000e+00,   1.02700000e+00,
          5.19200000e-01,   2.29600000e-01,   1.46600000e-01,
          1.11200000e-01,   9.21800000e-02,   7.20100000e-02,
          6.10600000e-02,   5.27100000e-02,   4.77300000e-02,
          4.24000000e-02,   3.86800000e-02,   3.79700000e-02,
          3.83300000e-02,   3.91200000e-02,   4.12400000e-02,
          4.35500000e-02,   4.89900000e-02,   5.33500000e-02],
       [  2.21600000e+03,   5.62400000e+03,   3.27800000e+03,
          1.25600000e+03,   6.19300000e+02,   3.54200000e+02,
          2.23400000e+02,   3.98900000e+02,   2.62900000e+02,
          9.08700000e+01,   4.24200000e+01,   1.44100000e+01,
          6.71600000e+00,   1.85000000e+01,   1.15500000e+01,
          5.45500000e+00,   3.04000000e+00,   1.07600000e+00,
          5.42500000e-01,   2.38000000e-01,   1.50900000e-01,
          1.13900000e-01,   9.40400000e-02,   7.31200000e-02,
          6.18600000e-02,   5.33100000e-02,   4.82200000e-02,
          4.28500000e-02,   3.91300000e-02,   3.84500000e-02,
          3.88500000e-02,   3.96700000e-02,   4.18500000e-02,
          4.42200000e-02,   4.98000000e-02,   5.42300000e-02],
       [  2.29100000e+03,   5.04100000e+03,   3.36000000e+03,
          1.29200000e+03,   6.38000000e+02,   3.65300000e+02,
          2.30500000e+02,   4.06800000e+02,   2.69300000e+02,
          9.33500000e+01,   4.36300000e+01,   1.48400000e+01,
          6.92000000e+00,   3.85900000e+00,   1.17500000e+01,
          5.57300000e+00,   3.10900000e+00,   1.10000000e+00,
          5.53400000e-01,   2.41000000e-01,   1.51700000e-01,
          1.13900000e-01,   9.37100000e-02,   7.25200000e-02,
          6.12000000e-02,   5.26200000e-02,   4.75900000e-02,
          4.22800000e-02,   3.86500000e-02,   3.80200000e-02,
          3.84400000e-02,   3.92800000e-02,   4.14700000e-02,
          4.38400000e-02,   4.94300000e-02,   5.38500000e-02],
       [  2.39600000e+03,   5.31400000e+03,   3.50700000e+03,
          1.35400000e+03,   6.69700000e+02,   3.83800000e+02,
          2.42300000e+02,   3.13300000e+02,   2.81500000e+02,
          9.80200000e+01,   4.58800000e+01,   1.56300000e+01,
          7.28800000e+00,   4.06400000e+00,   1.22300000e+01,
          5.82600000e+00,   3.25300000e+00,   1.15100000e+00,
          5.77500000e-01,   2.49800000e-01,   1.56200000e-01,
          1.16700000e-01,   9.56200000e-02,   7.36500000e-02,
          6.19600000e-02,   5.32100000e-02,   4.80800000e-02,
          4.27200000e-02,   3.90800000e-02,   3.84800000e-02,
          3.89400000e-02,   3.98100000e-02,   4.20600000e-02,
          4.44900000e-02,   5.01700000e-02,   5.47000000e-02],
       [  2.49400000e+03,   5.55000000e+03,   3.46700000e+03,
          1.40500000e+03,   6.95300000e+02,   3.98800000e+02,
          2.52000000e+02,   3.26900000e+02,   2.90200000e+02,
          1.01600000e+02,   4.76500000e+01,   1.62500000e+01,
          7.58200000e+00,   4.22700000e+00,   1.25900000e+01,
          6.01200000e+00,   3.36000000e+00,   1.18900000e+00,
          5.95300000e-01,   2.55800000e-01,   1.59000000e-01,
          1.18100000e-01,   9.64400000e-02,   7.39300000e-02,
          6.20400000e-02,   5.31800000e-02,   4.80100000e-02,
          4.26500000e-02,   3.90500000e-02,   3.84800000e-02,
          3.89600000e-02,   3.98600000e-02,   4.21400000e-02,
          4.46000000e-02,   5.03300000e-02,   5.49200000e-02],
       [  2.61600000e+03,   5.84700000e+03,   3.59000000e+03,
          1.46500000e+03,   7.26400000e+02,   4.17000000e+02,
          2.63600000e+02,   1.27100000e+02,   3.01200000e+02,
          1.06000000e+02,   4.98000000e+01,   1.70100000e+01,
          7.94000000e+00,   4.42500000e+00,   1.30900000e+01,
          6.24400000e+00,   3.49200000e+00,   1.23600000e+00,
          6.17800000e-01,   2.63900000e-01,   1.62900000e-01,
          1.20500000e-01,   9.80000000e-02,   7.47700000e-02,
          6.25700000e-02,   5.35100000e-02,   4.82800000e-02,
          4.28900000e-02,   3.93000000e-02,   3.87700000e-02,
          3.92700000e-02,   4.02000000e-02,   4.25300000e-02,
          4.50400000e-02,   5.08800000e-02,   5.55100000e-02],
       [  2.74800000e+03,   6.06900000e+03,   3.52300000e+03,
          1.52600000e+03,   7.58700000e+02,   4.35900000e+02,
          2.75700000e+02,   1.33000000e+02,   3.12900000e+02,
          1.10600000e+02,   5.20400000e+01,   1.78000000e+01,
          8.31500000e+00,   4.63400000e+00,   1.36200000e+01,
          6.47800000e+00,   3.62800000e+00,   1.28500000e+00,
          6.41500000e-01,   2.72400000e-01,   1.67200000e-01,
          1.23000000e-01,   9.96800000e-02,   7.56900000e-02,
          6.31700000e-02,   5.39200000e-02,   4.86000000e-02,
          4.31700000e-02,   3.95800000e-02,   3.90700000e-02,
          3.96100000e-02,   4.05600000e-02,   4.29400000e-02,
          4.54900000e-02,   5.14100000e-02,   5.61300000e-02],
       [  2.89900000e+03,   3.93700000e+03,   3.68600000e+03,
          1.59400000e+03,   7.94600000e+02,   4.56900000e+02,
          2.89200000e+02,   1.39700000e+02,   2.83000000e+02,
          1.15700000e+02,   5.45300000e+01,   1.86800000e+01,
          8.73500000e+00,   4.86700000e+00,   1.40900000e+01,
          6.74100000e+00,   3.78000000e+00,   1.34000000e+00,
          6.68200000e-01,   2.82200000e-01,   1.72200000e-01,
          1.26100000e-01,   1.01800000e-01,   7.69300000e-02,
          6.40400000e-02,   5.45500000e-02,   4.91200000e-02,
          4.36200000e-02,   4.00200000e-02,   3.95400000e-02,
          4.01000000e-02,   4.10800000e-02,   4.35200000e-02,
          4.61200000e-02,   5.21600000e-02,   5.69800000e-02],
       [  3.01700000e+03,   1.35000000e+03,   3.79700000e+03,
          1.64000000e+03,   8.19300000e+02,   4.71700000e+02,
          2.98800000e+02,   1.44400000e+02,   2.89300000e+02,
          1.19300000e+02,   5.62800000e+01,   1.93200000e+01,
          9.04000000e+00,   5.03800000e+00,   3.14700000e+00,
          6.90900000e+00,   3.88100000e+00,   1.37800000e+00,
          6.86000000e-01,   2.88200000e-01,   1.74900000e-01,
          1.27400000e-01,   1.02500000e-01,   7.71000000e-02,
          6.39700000e-02,   5.43900000e-02,   4.89500000e-02,
          4.34700000e-02,   3.98900000e-02,   3.94400000e-02,
          4.00200000e-02,   4.10100000e-02,   4.34700000e-02,
          4.60900000e-02,   5.21800000e-02,   5.70000000e-02],
       [  3.18700000e+03,   1.42400000e+03,   3.45200000e+03,
          1.71000000e+03,   8.56000000e+02,   4.93400000e+02,
          3.12900000e+02,   1.51300000e+02,   2.21100000e+02,
          1.24700000e+02,   5.88100000e+01,   2.02300000e+01,
          9.47200000e+00,   5.27900000e+00,   3.29700000e+00,
          7.16100000e+00,   4.03300000e+00,   1.43300000e+00,
          7.13000000e-01,   2.98100000e-01,   1.79900000e-01,
          1.30500000e-01,   1.04600000e-01,   7.82900000e-02,
          6.47800000e-02,   5.49600000e-02,   4.94100000e-02,
          4.38500000e-02,   4.02800000e-02,   3.98500000e-02,
          4.04500000e-02,   4.14700000e-02,   4.39800000e-02,
          4.66400000e-02,   5.28200000e-02,   5.77300000e-02],
       [  3.33500000e+03,   1.48900000e+03,   3.59800000e+03,
          1.76800000e+03,   8.85900000e+02,   5.11300000e+02,
          3.24400000e+02,   1.57100000e+02,   2.30100000e+02,
          1.29000000e+02,   6.08700000e+01,   2.09800000e+01,
          9.82800000e+00,   5.47800000e+00,   3.42000000e+00,
          7.35200000e+00,   4.15400000e+00,   1.47700000e+00,
          7.33900000e-01,   3.05400000e-01,   1.83400000e-01,
          1.32400000e-01,   1.05800000e-01,   7.88000000e-02,
          6.50200000e-02,   5.50500000e-02,   4.94400000e-02,
          4.38700000e-02,   4.03000000e-02,   3.98900000e-02,
          4.05200000e-02,   4.15500000e-02,   4.40900000e-02,
          4.67700000e-02,   5.30100000e-02,   5.79600000e-02],
       [  3.51000000e+03,   1.56600000e+03,   3.77100000e+03,
          1.83800000e+03,   9.22200000e+02,   5.32800000e+02,
          3.38200000e+02,   1.63900000e+02,   2.37900000e+02,
          1.34000000e+02,   6.33400000e+01,   2.18700000e+01,
          1.02500000e+01,   5.71700000e+00,   3.56900000e+00,
          7.58700000e+00,   4.30200000e+00,   1.53100000e+00,
          7.59800000e-01,   3.14900000e-01,   1.88100000e-01,
          1.35200000e-01,   1.07600000e-01,   7.98100000e-02,
          6.56700000e-02,   5.54500000e-02,   4.97700000e-02,
          4.41300000e-02,   4.05700000e-02,   4.01800000e-02,
          4.08200000e-02,   4.18800000e-02,   4.44600000e-02,
          4.71700000e-02,   5.35000000e-02,   5.85200000e-02],
       [  3.68300000e+03,   1.64300000e+03,   3.92200000e+03,
          1.90200000e+03,   9.56400000e+02,   5.53400000e+02,
          3.51400000e+02,   1.70500000e+02,   9.69100000e+01,
          1.38900000e+02,   6.57300000e+01,   2.27300000e+01,
          1.06700000e+01,   5.94900000e+00,   3.71300000e+00,
          7.81000000e+00,   4.43800000e+00,   1.58100000e+00,
          7.84400000e-01,   3.23800000e-01,   1.92500000e-01,
          1.37800000e-01,   1.09300000e-01,   8.06600000e-02,
          6.61800000e-02,   5.57700000e-02,   5.00000000e-02,
          4.43300000e-02,   4.07500000e-02,   4.03800000e-02,
          4.10300000e-02,   4.21000000e-02,   4.47200000e-02,
          4.74700000e-02,   5.38400000e-02,   5.89300000e-02],
       [  3.87200000e+03,   1.72900000e+03,   3.77300000e+03,
          1.97200000e+03,   9.94300000e+02,   5.75900000e+02,
          3.66000000e+02,   1.77800000e+02,   1.01100000e+02,
          1.44000000e+02,   6.83500000e+01,   2.36700000e+01,
          1.11200000e+01,   6.20600000e+00,   3.87200000e+00,
          8.06900000e+00,   4.58700000e+00,   1.63700000e+00,
          8.11900000e-01,   3.33900000e-01,   1.97600000e-01,
          1.40900000e-01,   1.11400000e-01,   8.17900000e-02,
          6.68800000e-02,   5.62700000e-02,   5.03800000e-02,
          4.46300000e-02,   4.10500000e-02,   4.06800000e-02,
          4.13700000e-02,   4.24600000e-02,   4.51100000e-02,
          4.79000000e-02,   5.43700000e-02,   5.95300000e-02],
       [  4.03200000e+03,   1.80100000e+03,   2.21800000e+03,
          1.93800000e+03,   1.02300000e+03,   5.93600000e+02,
          3.77600000e+02,   1.83600000e+02,   1.04500000e+02,
          1.47800000e+02,   7.03900000e+01,   2.44300000e+01,
          1.14900000e+01,   6.41400000e+00,   4.00200000e+00,
          8.29000000e+00,   4.69600000e+00,   1.68000000e+00,
          8.32700000e-01,   3.41400000e-01,   2.01100000e-01,
          1.42800000e-01,   1.12500000e-01,   8.22400000e-02,
          6.70500000e-02,   5.62500000e-02,   5.03400000e-02,
          4.45800000e-02,   4.10000000e-02,   4.06500000e-02,
          4.13400000e-02,   4.24400000e-02,   4.51100000e-02,
          4.79100000e-02,   5.44200000e-02,   5.95600000e-02],
       [  4.24300000e+03,   1.89800000e+03,   1.03200000e+03,
          2.01100000e+03,   1.06300000e+03,   6.17800000e+02,
          3.93500000e+02,   1.91400000e+02,   1.09000000e+02,
          1.53000000e+02,   7.31700000e+01,   2.54600000e+01,
          1.19900000e+01,   6.69300000e+00,   4.17600000e+00,
          8.58500000e+00,   4.85500000e+00,   1.74000000e+00,
          8.62800000e-01,   3.52500000e-01,   2.06800000e-01,
          1.46300000e-01,   1.14900000e-01,   8.35700000e-02,
          6.79400000e-02,   5.68800000e-02,   5.08300000e-02,
          4.50100000e-02,   4.13900000e-02,   4.10400000e-02,
          4.17400000e-02,   4.28600000e-02,   4.55800000e-02,
          4.84400000e-02,   5.50200000e-02,   6.02400000e-02],
       [  4.43300000e+03,   1.98600000e+03,   1.08100000e+03,
          1.96500000e+03,   1.10000000e+03,   6.40200000e+02,
          4.08300000e+02,   1.98700000e+02,   1.13200000e+02,
          1.57800000e+02,   7.57400000e+01,   2.64100000e+01,
          1.24500000e+01,   6.95400000e+00,   4.33900000e+00,
          8.73100000e+00,   4.99300000e+00,   1.79500000e+00,
          8.89600000e-01,   3.62500000e-01,   2.11800000e-01,
          1.49200000e-01,   1.16800000e-01,   8.45600000e-02,
          6.85700000e-02,   5.72700000e-02,   5.11200000e-02,
          4.52200000e-02,   4.16000000e-02,   4.12400000e-02,
          4.19600000e-02,   4.31000000e-02,   4.58400000e-02,
          4.87200000e-02,   5.53700000e-02,   6.06400000e-02],
       [  4.65200000e+03,   2.08900000e+03,   1.13700000e+03,
          2.04900000e+03,   1.14400000e+03,   6.66100000e+02,
          4.25300000e+02,   2.07200000e+02,   1.18100000e+02,
          1.63700000e+02,   7.88300000e+01,   2.75200000e+01,
          1.29800000e+01,   7.25600000e+00,   4.52800000e+00,
          2.18500000e+00,   5.15800000e+00,   1.86000000e+00,
          9.21400000e-01,   3.74400000e-01,   2.18000000e-01,
          1.53000000e-01,   1.19400000e-01,   8.60300000e-02,
          6.95300000e-02,   5.79400000e-02,   5.16700000e-02,
          4.57000000e-02,   4.20100000e-02,   4.16600000e-02,
          4.23900000e-02,   4.35500000e-02,   4.63300000e-02,
          4.92600000e-02,   5.59800000e-02,   6.13600000e-02],
       [  4.83000000e+03,   2.17400000e+03,   1.18400000e+03,
          2.11700000e+03,   1.17900000e+03,   6.86900000e+02,
          4.38700000e+02,   2.14000000e+02,   1.22100000e+02,
          1.68100000e+02,   8.12300000e+01,   2.84100000e+01,
          1.34200000e+01,   7.50400000e+00,   4.68300000e+00,
          2.25900000e+00,   5.27900000e+00,   1.90900000e+00,
          9.45600000e-01,   3.83400000e-01,   2.22400000e-01,
          1.55500000e-01,   1.21000000e-01,   8.67900000e-02,
          6.99300000e-02,   5.81400000e-02,   5.17900000e-02,
          4.57500000e-02,   4.20700000e-02,   4.17200000e-02,
          4.24600000e-02,   4.36200000e-02,   4.64300000e-02,
          4.93700000e-02,   5.61300000e-02,   6.15400000e-02],
       [  5.00800000e+03,   2.25900000e+03,   1.23100000e+03,
          2.18800000e+03,   1.21200000e+03,   7.06800000e+02,
          4.51800000e+02,   2.20800000e+02,   1.26000000e+02,
          1.49700000e+02,   8.36100000e+01,   2.92900000e+01,
          1.38500000e+01,   7.75100000e+00,   4.83800000e+00,
          2.33200000e+00,   5.39800000e+00,   1.95700000e+00,
          9.69600000e-01,   3.92300000e-01,   2.26700000e-01,
          1.58000000e-01,   1.22600000e-01,   8.75100000e-02,
          7.03100000e-02,   5.83200000e-02,   5.18700000e-02,
          4.58100000e-02,   4.21000000e-02,   4.17500000e-02,
          4.24800000e-02,   4.36600000e-02,   4.64700000e-02,
          4.94300000e-02,   5.62200000e-02,   6.16200000e-02],
       [  5.21000000e+03,   2.35600000e+03,   1.28500000e+03,
          1.96500000e+03,   1.25100000e+03,   7.30400000e+02,
          4.67200000e+02,   2.28700000e+02,   1.30600000e+02,
          1.11600000e+02,   8.63600000e+01,   3.03200000e+01,
          1.43600000e+01,   8.04100000e+00,   5.02100000e+00,
          2.41900000e+00,   5.54900000e+00,   2.01400000e+00,
          9.98500000e-01,   4.03100000e-01,   2.32300000e-01,
          1.61400000e-01,   1.24800000e-01,   8.87000000e-02,
          7.10200000e-02,   5.87600000e-02,   5.22200000e-02,
          4.60600000e-02,   4.23400000e-02,   4.19700000e-02,
          4.27200000e-02,   4.39100000e-02,   4.67500000e-02,
          4.97200000e-02,   5.65800000e-02,   6.20600000e-02],
       [  5.44100000e+03,   2.46800000e+03,   1.34800000e+03,
          2.05300000e+03,   1.29600000e+03,   7.58000000e+02,
          4.85500000e+02,   2.37800000e+02,   1.36000000e+02,
          1.16000000e+02,   8.95200000e+01,   3.15200000e+01,
          1.49500000e+01,   8.37900000e+00,   5.23300000e+00,
          2.52200000e+00,   5.73900000e+00,   2.08200000e+00,
          1.03300000e+00,   4.16300000e-01,   2.39100000e-01,
          1.65600000e-01,   1.27700000e-01,   9.03600000e-02,
          7.21400000e-02,   5.95500000e-02,   5.28500000e-02,
          4.65900000e-02,   4.27900000e-02,   4.24200000e-02,
          4.31700000e-02,   4.43700000e-02,   4.72500000e-02,
          5.02500000e-02,   5.72100000e-02,   6.27600000e-02],
       [  5.72400000e+03,   2.60400000e+03,   1.42300000e+03,
          2.15500000e+03,   1.29900000e+03,   7.93100000e+02,
          5.08500000e+02,   2.49400000e+02,   1.42700000e+02,
          1.21900000e+02,   9.35200000e+01,   3.30300000e+01,
          1.56900000e+01,   8.80200000e+00,   5.49900000e+00,
          2.64900000e+00,   5.99100000e+00,   2.17000000e+00,
          1.07800000e+00,   4.33500000e-01,   2.48300000e-01,
          1.71400000e-01,   1.31800000e-01,   9.28600000e-02,
          7.39100000e-02,   6.08700000e-02,   5.39400000e-02,
          4.74900000e-02,   4.36200000e-02,   4.32300000e-02,
          4.39900000e-02,   4.52200000e-02,   4.81600000e-02,
          5.12400000e-02,   5.83500000e-02,   6.39800000e-02],
       [  5.86800000e+03,   2.73100000e+03,   1.49500000e+03,
          2.25000000e+03,   1.27500000e+03,   8.25100000e+02,
          5.29900000e+02,   2.60100000e+02,   1.49000000e+02,
          1.27900000e+02,   9.70400000e+01,   3.44200000e+01,
          1.63800000e+01,   9.19600000e+00,   5.74800000e+00,
          2.76900000e+00,   6.17400000e+00,   2.24900000e+00,
          1.11800000e+00,   4.49100000e-01,   2.56500000e-01,
          1.76500000e-01,   1.35400000e-01,   9.49500000e-02,
          7.53800000e-02,   6.18900000e-02,   5.47900000e-02,
          4.82100000e-02,   4.42400000e-02,   4.38300000e-02,
          4.46100000e-02,   4.58500000e-02,   4.88300000e-02,
          5.19500000e-02,   5.91800000e-02,   6.49300000e-02],
       [  5.82600000e+03,   2.71900000e+03,   1.49000000e+03,
          1.55400000e+03,   1.26600000e+03,   8.16300000e+02,
          5.24000000e+02,   2.57700000e+02,   1.47700000e+02,
          1.25900000e+02,   9.56300000e+01,   3.40800000e+01,
          1.62400000e+01,   9.12500000e+00,   5.70600000e+00,
          2.74900000e+00,   6.08600000e+00,   2.21500000e+00,
          1.10100000e+00,   4.42000000e-01,   2.51800000e-01,
          1.72800000e-01,   1.32200000e-01,   9.23000000e-02,
          7.30300000e-02,   5.98400000e-02,   5.29000000e-02,
          4.64900000e-02,   4.26400000e-02,   4.22400000e-02,
          4.29800000e-02,   4.41800000e-02,   4.70500000e-02,
          5.00700000e-02,   5.70400000e-02,   6.26000000e-02],
       [  6.23800000e+03,   2.84600000e+03,   1.56200000e+03,
          1.99400000e+03,   1.32200000e+03,   8.48900000e+02,
          5.45100000e+02,   2.68400000e+02,   1.53900000e+02,
          5.57300000e+01,   9.93300000e+01,   3.55000000e+01,
          1.69400000e+01,   9.52500000e+00,   5.95900000e+00,
          2.87100000e+00,   1.65500000e+00,   2.29500000e+00,
          1.14100000e+00,   4.57700000e-01,   2.60100000e-01,
          1.78000000e-01,   1.35800000e-01,   9.44400000e-02,
          7.45000000e-02,   6.08700000e-02,   5.37600000e-02,
          4.72100000e-02,   4.32400000e-02,   4.28300000e-02,
          4.35800000e-02,   4.47900000e-02,   4.76900000e-02,
          5.07700000e-02,   5.78600000e-02,   6.35000000e-02],
       [  6.20100000e+03,   2.95000000e+03,   1.62000000e+03,
          6.66400000e+02,   1.36700000e+03,   8.74100000e+02,
          5.61300000e+02,   2.76900000e+02,   1.58900000e+02,
          5.76000000e+01,   1.02300000e+02,   3.66400000e+01,
          1.75000000e+01,   9.85000000e+00,   6.16600000e+00,
          2.97100000e+00,   1.71200000e+00,   2.35500000e+00,
          1.17200000e+00,   4.69600000e-01,   2.66200000e-01,
          1.81700000e-01,   1.38300000e-01,   9.57900000e-02,
          7.53500000e-02,   6.14100000e-02,   5.41500000e-02,
          4.75100000e-02,   4.34800000e-02,   4.30400000e-02,
          4.37900000e-02,   4.50000000e-02,   4.79300000e-02,
          5.10200000e-02,   5.81500000e-02,   6.38300000e-02],
       [  6.46900000e+03,   3.08200000e+03,   1.69600000e+03,
          6.98100000e+02,   1.42600000e+03,   8.68700000e+02,
          5.82900000e+02,   2.87800000e+02,   1.65300000e+02,
          6.00200000e+01,   1.06200000e+02,   3.81100000e+01,
          1.82300000e+01,   1.02700000e+01,   6.43300000e+00,
          3.10000000e+00,   1.78600000e+00,   2.43400000e+00,
          1.21300000e+00,   4.85900000e-01,   2.74900000e-01,
          1.87200000e-01,   1.42100000e-01,   9.80400000e-02,
          7.68600000e-02,   6.25100000e-02,   5.50400000e-02,
          4.82200000e-02,   4.40900000e-02,   4.36300000e-02,
          4.43800000e-02,   4.56000000e-02,   4.85600000e-02,
          5.17100000e-02,   5.89500000e-02,   6.47300000e-02],
       [  6.61400000e+03,   3.16100000e+03,   1.74200000e+03,
          7.18000000e+02,   1.25300000e+03,   8.87800000e+02,
          5.94500000e+02,   2.93900000e+02,   1.68900000e+02,
          6.14100000e+01,   9.36800000e+01,   3.89200000e+01,
          1.86500000e+01,   1.05200000e+01,   6.59200000e+00,
          3.17800000e+00,   1.83000000e+00,   2.47200000e+00,
          1.23400000e+00,   4.93900000e-01,   2.78900000e-01,
          1.89500000e-01,   1.43500000e-01,   9.86100000e-02,
          7.70900000e-02,   6.25100000e-02,   5.49800000e-02,
          4.81200000e-02,   4.39600000e-02,   4.34700000e-02,
          4.42100000e-02,   4.54200000e-02,   4.83600000e-02,
          5.14900000e-02,   5.87100000e-02,   6.44700000e-02],
       [  6.53000000e+03,   3.32700000e+03,   1.83400000e+03,
          7.55800000e+02,   1.31500000e+03,   8.75900000e+02,
          6.21700000e+02,   3.07400000e+02,   1.76900000e+02,
          6.44200000e+01,   7.02500000e+01,   4.07700000e+01,
          1.95700000e+01,   1.10500000e+01,   6.92900000e+00,
          3.34200000e+00,   1.92400000e+00,   2.57500000e+00,
          1.28800000e+00,   5.15200000e-01,   2.90400000e-01,
          1.96900000e-01,   1.48700000e-01,   1.01800000e-01,
          7.93700000e-02,   6.42000000e-02,   5.63700000e-02,
          4.92700000e-02,   4.49600000e-02,   4.44500000e-02,
          4.51900000e-02,   4.64100000e-02,   4.94300000e-02,
          5.26100000e-02,   6.00000000e-02,   6.59200000e-02],
       [  6.62600000e+03,   3.38200000e+03,   1.86500000e+03,
          7.69200000e+02,   1.32900000e+03,   8.89100000e+02,
          6.28400000e+02,   3.10800000e+02,   1.79100000e+02,
          6.52800000e+01,   7.10600000e+01,   4.12800000e+01,
          1.98300000e+01,   1.12100000e+01,   7.03500000e+00,
          3.39500000e+00,   1.95400000e+00,   2.59100000e+00,
          1.29800000e+00,   5.19200000e-01,   2.92200000e-01,
          1.97600000e-01,   1.49000000e-01,   1.01600000e-01,
          7.89600000e-02,   6.37000000e-02,   5.58700000e-02,
          4.87800000e-02,   4.44700000e-02,   4.39200000e-02,
          4.46300000e-02,   4.58300000e-02,   4.87900000e-02,
          5.19500000e-02,   5.92700000e-02,   6.51200000e-02],
       [  6.95000000e+03,   3.49000000e+03,   1.96000000e+03,
          8.09000000e+02,   1.39000000e+03,   9.32000000e+02,
          6.57000000e+02,   3.25000000e+02,   1.87000000e+02,
          6.84000000e+01,   7.45000000e+01,   4.31000000e+01,
          2.08000000e+01,   1.18000000e+01,   7.39000000e+00,
          3.57000000e+00,   2.05000000e+00,   2.70000000e+00,
          1.35000000e+00,   5.41000000e-01,   3.04000000e-01,
          2.05000000e-01,   1.54000000e-01,   1.05000000e-01,
          8.13000000e-02,   6.54000000e-02,   5.73000000e-02,
          4.99000000e-02,   4.55000000e-02,   4.49000000e-02,
          4.56000000e-02,   4.68000000e-02,   4.98000000e-02,
          5.30000000e-02,   6.05000000e-02,   6.65000000e-02],
       [  7.19000000e+03,   3.62000000e+03,   2.04000000e+03,
          8.39000000e+02,   1.43000000e+03,   9.65000000e+02,
          6.76000000e+02,   3.35000000e+02,   1.94000000e+02,
          7.06000000e+01,   7.71000000e+01,   4.45000000e+01,
          2.15000000e+01,   1.22000000e+01,   7.65000000e+00,
          3.70000000e+00,   2.13000000e+00,   2.77000000e+00,
          1.39000000e+00,   5.57000000e-01,   3.12000000e-01,
          2.10000000e-01,   1.58000000e-01,   1.07000000e-01,
          8.26000000e-02,   6.63000000e-02,   5.80000000e-02,
          5.05000000e-02,   4.59000000e-02,   4.53000000e-02,
          4.60000000e-02,   4.72000000e-02,   5.02000000e-02,
          5.35000000e-02,   6.10000000e-02,   6.71000000e-02],
       [  7.37000000e+03,   3.73000000e+03,   2.10000000e+03,
          8.64000000e+02,   1.05000000e+03,   9.90000000e+02,
          6.64000000e+02,   3.43000000e+02,   1.98000000e+02,
          7.24000000e+01,   7.94000000e+01,   4.55000000e+01,
          2.20000000e+01,   1.25000000e+01,   7.86000000e+00,
          3.80000000e+00,   2.19000000e+00,   2.82000000e+00,
          1.42000000e+00,   5.68000000e-01,   3.18000000e-01,
          2.14000000e-01,   1.60000000e-01,   1.08000000e-01,
          8.33000000e-02,   6.67000000e-02,   5.82000000e-02,
          5.06000000e-02,   4.60000000e-02,   4.53000000e-02,
          4.60000000e-02,   4.71000000e-02,   5.02000000e-02,
          5.34000000e-02,   6.10000000e-02,   6.70000000e-02],
       [  7.54000000e+03,   3.83000000e+03,   2.15000000e+03,
          8.89000000e+02,   1.04000000e+03,   1.02000000e+03,
          6.79000000e+02,   3.51000000e+02,   2.03000000e+02,
          7.41000000e+01,   8.14000000e+01,   4.65000000e+01,
          2.26000000e+01,   1.28000000e+01,   8.07000000e+00,
          3.91000000e+00,   2.25000000e+00,   2.87000000e+00,
          1.44000000e+00,   5.79000000e-01,   3.24000000e-01,
          2.18000000e-01,   1.63000000e-01,   1.09000000e-01,
          8.41000000e-02,   6.71000000e-02,   5.85000000e-02,
          5.08000000e-02,   4.61000000e-02,   4.53000000e-02,
          4.60000000e-02,   4.71000000e-02,   5.01000000e-02,
          5.34000000e-02,   6.10000000e-02,   6.70000000e-02],
       [  7.84000000e+03,   3.95000000e+03,   2.25000000e+03,
          9.29000000e+02,   4.84000000e+02,   1.06000000e+03,
          6.68000000e+02,   3.64000000e+02,   2.10000000e+02,
          7.71000000e+01,   8.39000000e+01,   4.83000000e+01,
          2.35000000e+01,   1.34000000e+01,   8.42000000e+00,
          4.08000000e+00,   2.35000000e+00,   2.97000000e+00,
          1.50000000e+00,   6.00000000e-01,   3.36000000e-01,
          2.25000000e-01,   1.68000000e-01,   1.13000000e-01,
          8.62000000e-02,   6.87000000e-02,   5.97000000e-02,
          5.18000000e-02,   4.69000000e-02,   4.61000000e-02,
          4.68000000e-02,   4.79000000e-02,   5.09000000e-02,
          5.42000000e-02,   6.19000000e-02,   6.81000000e-02],
       [  7.89000000e+03,   4.06000000e+03,   2.31000000e+03,
          9.54000000e+02,   4.97000000e+02,   9.27000000e+02,
          6.85000000e+02,   3.72000000e+02,   2.15000000e+02,
          7.89000000e+01,   8.58000000e+01,   4.92000000e+01,
          2.40000000e+01,   1.37000000e+01,   8.64000000e+00,
          4.19000000e+00,   2.41000000e+00,   3.03000000e+00,
          1.52000000e+00,   6.11000000e-01,   3.42000000e-01,
          2.29000000e-01,   1.71000000e-01,   1.14000000e-01,
          8.70000000e-02,   6.91000000e-02,   6.00000000e-02,
          5.20000000e-02,   4.70000000e-02,   4.62000000e-02,
          4.68000000e-02,   4.79000000e-02,   5.09000000e-02,
          5.42000000e-02,   6.19000000e-02,   6.81000000e-02],
       [  7.79000000e+03,   4.22000000e+03,   2.40000000e+03,
          9.92000000e+02,   5.17000000e+02,   9.59000000e+02,
          7.11000000e+02,   3.83000000e+02,   2.22000000e+02,
          8.16000000e+01,   4.01000000e+01,   5.08000000e+01,
          2.49000000e+01,   1.42000000e+01,   8.97000000e+00,
          4.36000000e+00,   2.51000000e+00,   3.12000000e+00,
          1.57000000e+00,   6.31000000e-01,   3.52000000e-01,
          2.36000000e-01,   1.75000000e-01,   1.17000000e-01,
          8.89000000e-02,   7.04000000e-02,   6.11000000e-02,
          5.28000000e-02,   4.76000000e-02,   4.67000000e-02,
          4.74000000e-02,   4.84000000e-02,   5.15000000e-02,
          5.48000000e-02,   6.26000000e-02,   6.89000000e-02],
       [  7.13000000e+03,   4.32000000e+03,   2.46000000e+03,
          1.01000000e+03,   5.29000000e+02,   9.77000000e+02,
          7.25000000e+02,   3.89000000e+02,   2.26000000e+02,
          8.31000000e+01,   4.09000000e+01,   5.17000000e+01,
          2.54000000e+01,   1.45000000e+01,   9.16000000e+00,
          4.45000000e+00,   2.57000000e+00,   3.14000000e+00,
          1.59000000e+00,   6.40000000e-01,   3.57000000e-01,
          2.39000000e-01,   1.77000000e-01,   1.18000000e-01,
          8.94000000e-02,   7.07000000e-02,   6.12000000e-02,
          5.28000000e-02,   4.75000000e-02,   4.66000000e-02,
          4.72000000e-02,   4.82000000e-02,   5.12000000e-02,
          5.45000000e-02,   6.23000000e-02,   6.86000000e-02]]


# In[675]:

print mac1[12]
print theoretical_energy_level


# In[696]:

x = numpy.array(theoretical_energy_level)

y = numpy.array(mac1[12])

print x.shape
print y.shape
plt.figure(102335)
plt.xscale('log')
plt.plot(x, y)


# In[ ]:




# In[ ]:




# In[ ]:




# In[755]:

'''# test validation of data 42
im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/42_3_separate.tif')
im_openbeam = Image.open('/ufs/piao/Desktop/Data_2.24/42_OB.tif')
origin_array = numpy.array(im_origin)
origin_float = origin_array.astype(float)
openbeam = numpy.array(im_openbeam)
openbeam_float = openbeam.astype(float)
divide = numpy.divide(openbeam_float, origin_float)
log = numpy.log(divide)
print log
log_image = Image.fromarray(log)
log_image.save('/ufs/piao/Desktop/Data_2.24/sum/log_image_42.tif')'''


# In[131]:

#details on average image separate
im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/sum/average_separate_after_ob.tif')
origin_array = numpy.array(im_origin)
origin_float = origin_array.astype(float)

plt.imshow(origin_float)
plt.title('average image visualization')
plt.colorbar()
plt.show()


# In[133]:

#details on maximum image separate
im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/after_ob_correction/after_mf/52_3_separate_after_ob_mf.tif')
origin_array = numpy.array(im_origin)
origin_float = origin_array.astype(float)

plt.imshow(origin_float)
plt.title('average image visualization')
plt.colorbar()
plt.show()


# In[134]:

#details on original image separate 52
im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/52_3_separate.tif')
origin_array = numpy.array(im_origin)
origin_float = origin_array.astype(float)

plt.imshow(origin_float)
plt.title('original image visualization')
plt.colorbar()
plt.show()


# In[114]:

#details on openbeam image separate 52
im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/after_ob_correction/92_3_separate_after_ob.tif')
origin_array = numpy.array(im_origin)
origin_float = origin_array.astype(float)

plt.imshow(origin_float)
plt.title('openbeam image visualization')
plt.colorbar()
plt.show()


# In[141]:

#details on openbeam image separate 52
im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/imageJ_52_3_separate.tif')
origin_array = numpy.array(im_origin)

plt.imshow(origin_array)
plt.title('openbeam image visualization')
plt.colorbar()
plt.show()


# In[209]:

im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/after_ob_correction/after_mf/132_3_separate_after_ob_mf-1.tif')
origin_array = numpy.array(im_origin)
plt.imshow(origin_array)
plt.title('openbeam image visualization')
plt.colorbar()
plt.show()


# In[211]:

im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/after_ob_correction/after_mf/132_3_separate_after_ob_mf.tif')
origin_array = numpy.array(im_origin)
plt.imshow(origin_array)
plt.title('openbeam image visualization')
plt.colorbar()
plt.show()


# In[179]:

im_origin = Image.open('/ufs/piao/Desktop/Data_2.24/52_3_separate.tif')
origin_array = numpy.array(im_origin)
plt.imshow(origin_array)
plt.title('openbeam image visualization')
plt.colorbar()
plt.show()


# In[ ]:



