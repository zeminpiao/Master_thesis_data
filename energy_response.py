#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:17:55 2017

@author: piao
"""

from xraydb import XrayDB
import matplotlib.pyplot as plt
import operator
import numpy

source_xray = numpy.loadtxt('/ufs/piao/Desktop/data_xray.txt')
source_xray = source_xray.transpose()
source_xray[0] = source_xray[0] * 0.01165211 - 0.147653
print source_xray

detector_efficiency = numpy.loadtxt('/ufs/piao/Downloads/Default Dataset.csv', delimiter=',')
detector_efficiency = detector_efficiency.transpose()
detector_efficiency[1] = detector_efficiency[1] / 100
print detector_efficiency


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
    intensity = numpy.copy(spectra_XRAY)
    for i in range(0, ((E[-1] - E[0]) / 10) + 1):
        list_to_return.append(simple_integral(intensity, 10))
        intensity[i] = 0
    return list_to_return


x_energy = numpy.linspace(12, 26, 1401)

x_ray_spectral_array = numpy.array([counting_on_threshold_source_real(i)  for i in x_energy])

xray = XrayDB()

integral_value_xrayspec = Integral_OB([i for i in range(12000, 26010, 10)], \
                                      x_ray_spectral_array)

import math
def Gaussian_response(FWHM,mu,x):
    sigma = FWHM / 2.355
    const_gaussian = 1 / numpy.sqrt((2*math.pi*sigma*sigma))
    return const_gaussian * numpy.exp(-((x-mu)**2 ) / (2*sigma*sigma))


x_ray_spectral_after_energy_response = numpy.copy(x_ray_spectral_array)
for i in range(0, x_ray_spectral_after_energy_response.shape[0]):
    ''' array_to_time = [Gaussian_response(100, j, i) * x_ray_spectral_array[j]\
                     for j in range(0, x_ray_spectral_after_energy_response.shape[0] / 5)]\
    +[Gaussian_response(200, j, i) * x_ray_spectral_array[j] for j in \
      range(x_ray_spectral_after_energy_response.shape[0] / 5, x_ray_spectral_after_energy_response.shape[0] *2 / 5)]\
    +[Gaussian_response(300, j, i) * x_ray_spectral_array[j] for j in \
      range(x_ray_spectral_after_energy_response.shape[0]*2 / 5, x_ray_spectral_after_energy_response.shape[0] *3 / 5)]\
    +[Gaussian_response(400, j, i) * x_ray_spectral_array[j] for j in \
      range(x_ray_spectral_after_energy_response.shape[0]*3 / 5, x_ray_spectral_after_energy_response.shape[0] *4 / 5)]\
    +[Gaussian_response(500, j, i) * x_ray_spectral_array[j] for j in \
     range(x_ray_spectral_after_energy_response.shape[0]*4 / 5, x_ray_spectral_after_energy_response.shape[0] * 5 / 5)]'''

    list_to_sum = [Gaussian_response((j / 10) + 200, j, i) * x_ray_spectral_array[j] for j \
                   in range(0, x_ray_spectral_after_energy_response.shape[0])]
    x_ray_spectral_after_energy_response[i] = sum(list_to_sum)
print x_ray_spectral_after_energy_response
print x_ray_spectral_array
plt.figure()
plt.title('Comparison of x-ray spectral with and without considering energy resolution')
plt.plot(x_energy, x_ray_spectral_after_energy_response, 'g')
plt.plot(x_energy, x_ray_spectral_array, 'r')

integral_value_energy_response = Integral_OB([i for i in range(12000, 26010, 10)], x_ray_spectral_after_energy_response)
plt.figure()
plt.title('Comparison of integral spectral with and without considering energy resolution')
plt.plot(x_energy,integral_value_xrayspec, 'r')
plt.plot(x_energy, integral_value_energy_response, 'g')


plt.figure()
plt.title('Subtraction of two integral x-ray spectral with and without considering energy resolution')
plt.plot(x_energy, numpy.array(integral_value_energy_response) - numpy.array(integral_value_xrayspec), 'r')
plt.plot(x_energy, numpy.array(x_ray_spectral_after_energy_response) - numpy.array(x_ray_spectral_array), 'g')