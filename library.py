#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:33:43 2017

@author: piao
"""


def counting_on_threshold_source_real(threshold_kev):
    i = 0

    for element in source_xray[0]:
        if element > threshold_kev:
            break
        i = i + 1
    
    return source_xray[1][i]


def simple_integral(list_target, N):
    value_to_return = 0.0
    for i in range(0, len(list_target)-1):
        a0 = numpy.array(list_target[i])
        a1 = numpy.array(list_target[i+1])
        value_to_return = (value_to_return + ((a0 + a1) * N ) / 2.0)
        #print list_target[i]
        #print list_target[i + 1]
        #print ((list_target[i] + list_target[i+1])*N )/2.0
        #print value_to_return
    return value_to_return

def Integral_after_Energy(E, i, thickness, mode, dn_count):
    list_to_return = []
    negative_input = numpy.negative(xray.density(i)\
                                    * xray.mu_elam(i, E)*thickness)
    if mode == 'x_ray_spectral':
        #4150 is calibration of photon count based on siemens data
        intensity = x_ray_spectral_array * numpy.exp(negative_input) / dn_count
        print intensity
    else:
        intensity = 10000 * numpy.exp(negative_input)
    #print intensity
    intensity = numpy.ndarray.tolist(intensity)
    for i in range(0, ((E[-1] - E[0]) / 10) + 1):
        list_to_return.append(simple_integral(intensity, 10))
        intensity[i] = 0
    return list_to_return

#Transfering the spectral into integral values for all elements

def Integral_OB(E, spectra_XRAY):
    list_to_return = []
    intensity = spectra_XRAY
    for i in range(0, ((E[-1] - E[0]) / 10) + 1):
        list_to_return.append(simple_integral(intensity, 10))
        intensity[i] = 0
    return list_to_return


#based on bin size choose the bin
def choose_correct_sample_energy(N, range_of_energy):
    a = range_of_energy[0]
    list_to_return = []
    slices =  (range_of_energy[1] - range_of_energy[0]) / N
    list_to_return.append(a)
    for i in range(1, N + 1):
        list_to_return.append(a + i*slices)
    return list_to_return


def subtract_after_binning(list_integral, list_bin_energy):
    list_to_return = []
    #list_to_return.append\(list_integral[(list_bin_energy[0] / 10)] - list_integral[(list_bin_energy[1] / 10)])
    for i in range(1, len(list_bin_energy)):
        #print (list_bin_energy[i] / 10)
        list_to_return.append((list_integral[((list_bin_energy[i-1]\
                                               - list_bin_energy[0]) / 10)] - \
                               list_integral[((list_bin_energy[i]\
                                               - list_bin_energy[0]) / 10)]))
    return list_to_return

def after_binning_for_element(integral_with_error_value, a):
    integral_value_list_noise = []
    integral_value_list_noise.append(0)

    list_8bin_after_sub = subtract_after_binning(\
                                                 integral_with_error_value\
                                                 , a)
    array_after_sub_20 = []
    for i in list_8bin_after_sub:
        array_after_sub_20.append((float(i) / float(a[1] - a[0])))
    return array_after_sub_20

def show_theoretical_spectral_after_binning(ele_num, spctr_num, N_N, printing):
    '''sample energy level'''
    a =  choose_correct_sample_energy(N_N, [12000, 26000])
    #print a
    '''middle point of each two sampling energy level'''
    b = []
    for ii in range(0, len(a) - 1):
        b.append((a[ii]+a[ii+1]) / 2)

    b = numpy.array(b)

    list_to_return = []
    #if printing == 'print':
        #plt.figure()
        #plt.title('Mass attenuation coefficient prediction after binning with bin number '+str(N_N) )

    for i in range(1, spctr_num + 1):
        
        list_error_I = []
        list_error_I0 = []
        for j in range(0, len(integral_value_list[ele_num])):
            test = numpy.random.normal\
            (integral_value_list[ele_num][j],\
             numpy.sqrt(integral_value_list[ele_num][j]), 1)
            list_error_I.append(test)
        for k in range(0, len(integral_value_xrayspec)):
            test = numpy.random.normal\
            (integral_value_xrayspec[k], \
             numpy.sqrt(integral_value_xrayspec[k]),1)
            list_error_I0.append(test)
        list_after_sub_I = after_binning_for_element(list_error_I, a)
        list_after_sub_I0 = after_binning_for_element(list_error_I0, a)
        
        

        #print list_array_element[ele_num]
        #print list_array_element[ele_num]
        #print array_to_be_div
        mac_after_binning_11 = -numpy.log(numpy.divide(list_after_sub_I,  \
                                                       list_after_sub_I0))\
                                                       .astype(float) / 5e-3

        if printing == 'print':
            plt.plot(b, mac_after_binning_11,  'p')
            #plt.plot(b, mac_after_binning_12, linewidth=0.1)
            #plt.plot(b, mac_after_binning_13, linewidth=0.1)

        
        list_to_return.append(mac_after_binning_11)
        #list_to_return.append(mac_after_binning_NOISE_12)
        #list_to_return.append(mac_after_binning_NOISE_13)
    '''
    test = xray.mu_elam(ele_num, b)
    plt.plot(b, numpy.array(test)*xray.density(ele_num))
    
    test1 = xray.mu_elam(ele_num - 1, b)
    plt.plot(b, numpy.array(test1)*xray.density(ele_num))
    
    test2 = xray.mu_elam(ele_num + 1, b)
    plt.plot(b, numpy.array(test2)*xray.density(ele_num))
    
    test2 = xray.mu_elam(ele_num + 1, b)
    plt.plot(b, numpy.array(test2)*xray.density(ele_num))
    test3 = xray.mu_elam(ele_num - 2, b)
    plt.plot(b, numpy.array(test3)*xray.density(ele_num))
    test4 = xray.mu_elam(ele_num + 2, b)
    plt.plot(b, numpy.array(test4)*xray.density(ele_num))'''
#plt.plot(b, after_binning_for_element(29))
#plt.plot(Energy_Value, integral_value_list_noise[31])
    if printing == 'print':
        plt.show()
    #array_to_return = numpy.array(list_to_return)
    return list_to_return
