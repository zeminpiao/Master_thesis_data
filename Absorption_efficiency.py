#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:56:29 2017

@author: piao
"""

from xraydb import XrayDB
import matplotlib.pyplot as plt
import operator
import numpy
import scipy.optimize as optimization
import scipy


xray = XrayDB()
Energy_Value = [i for i in range(12000, 26010, 10)]
dn_count = 1
#real Xray source data 

source_xray = numpy.loadtxt('/ufs/piao/Desktop/data_xray.txt')
source_xray = source_xray.transpose()
source_xray[0] = source_xray[0] * 0.01165211 - 0.147653
#print source_xray


#detector efficiency data

detector_efficiency = numpy.loadtxt('/ufs/piao/Downloads/Default Dataset.csv', delimiter=',')
detector_efficiency = detector_efficiency.transpose()
detector_efficiency[1] = detector_efficiency[1] / 100
#print detector_efficiency

xvals = numpy.linspace(12, 26, 1401)

x = detector_efficiency[0]
#print x
detector_efficiency_linear_interpolated = numpy.interp(xvals, x, detector_efficiency[1])

#print test


def counting_on_threshold_source_real(threshold_kev):
    i = 0

    for element in source_xray[0]:
        if element > threshold_kev:
            break
        i = i + 1
    
    return source_xray[1][i]

x_energy = numpy.linspace(12, 26, 1401)

x_ray_spectral_array = numpy.array([counting_on_threshold_source_real(i) for i in x_energy])



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


#Considering the absorption efficiency of silicon detector with thickness of 300um
def Integral_after_Energy(E, i, thickness, mode, dn_count):
    list_to_return = []
    negative_input = numpy.negative(xray.density(i)\
                                    * xray.mu_elam(i, E)*thickness)
    if mode == 'x_ray_spectral':
        #4150 is calibration of photon count based on siemens data
        intensity = x_ray_spectral_array *\
        detector_efficiency_linear_interpolated*\
        numpy.exp(negative_input) / dn_count
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

def Integral_OB_ideal(E, spectra_XRAY):
    list_to_return = []
    intensity = numpy.copy(spectra_XRAY)
    for i in range(0, ((E[-1] - E[0]) / 10) + 1):
        list_to_return.append(simple_integral(intensity, 10))
        intensity[i] = 0
    return list_to_return

def Integral_OB(E, spectra_XRAY):
    list_to_return = []
    intensity = spectra_XRAY * detector_efficiency_linear_interpolated
    for i in range(0, ((E[-1] - E[0]) / 10) + 1):
        list_to_return.append(simple_integral(intensity, 10))
        intensity[i] = 0
    return list_to_return


integral_value_xrayspec = Integral_OB(Energy_Value, \
                                      x_ray_spectral_array / dn_count)

plt.figure()
plt.title('X-ray Source spectra comparison with and without the effect of detector efficiency')
plt.plot(Energy_Value, x_ray_spectral_array, 'r')
plt.plot(Energy_Value, x_ray_spectral_array * detector_efficiency_linear_interpolated, 'g')

plt.figure()
plt.title('Integral spectra comparison with and without the effect of detector efficiency')
plt.plot(Energy_Value, integral_value_xrayspec, 'g')
plt.plot(Energy_Value, Integral_OB_ideal(Energy_Value, x_ray_spectral_array), 'r')
