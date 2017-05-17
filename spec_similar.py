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
print source_xray


#detector efficiency data

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


#Transfering the spectral into integral values for all elements
integral_value_list = []
#just for dislocation
integral_value_list.append([0])
for i in range(1, 31):
    integral_value_list.append(Integral_after_Energy(Energy_Value,\
                                                     i, 5e-3,\
                                                     'x_ray_spectral', dn_count))
    
integral_value_xrayspec = Integral_OB(Energy_Value, \
                                      x_ray_spectral_array / dn_count)


integral_after_error_ob = []
for i in range(0, len(integral_value_xrayspec)):
    #print i
    test = numpy.random.normal(integral_value_xrayspec[i], numpy.sqrt(integral_value_xrayspec[i]),1)
    
    #test = integral_value_xrayspec[i]
    integral_after_error_ob.append(test)
    
integral_with_error_value = []
for i in range(1, 31):
    integral_after_error = []
    for j in range(0, len(integral_value_list[i])):
        test = numpy.random.normal(integral_value_list[i][j],numpy.sqrt(integral_value_list[i][j]), 1)
        #test = integral_value_list[i][j]
        integral_after_error.append(test)
    integral_with_error_value.append(integral_after_error)
    
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
              0.1, 1)
             #numpy.sqrt(integral_value_list[ele_num][j]), 1)
                                                
            list_error_I.append(test)
        for k in range(0, len(integral_value_xrayspec)):
            test = numpy.random.normal\
            (integral_value_xrayspec[k], \
             0.1, 1)
             #numpy.sqrt(integral_value_xrayspec[k]),1)
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

def mse_print_accuracy_result(element_number, number_of_bins):
    list_to_return = []
    a =  choose_correct_sample_energy(number_of_bins, [12000, 26000])
    #print a 
    '''middle point of each two sampling energy level'''
    b = []
    for ii in range(0, len(a) - 1):
        b.append((a[ii]+a[ii+1]) / 2)
    #print b
    linear_attenuation_coefficient_list = []
    for i in range(1, 78):
        list_mac = xray.mu_elam(i, b)
        linear_attenuation_coefficient_list.append(list_mac * xray.density(i))
    linear_attenuation_coefficient_array = numpy.array(linear_attenuation_coefficient_list)
    
    
    
    mac_after_binning = show_theoretical_spectral_after_binning\
    (element_number, 200,  number_of_bins, 'noprint')
        #print mac_after_binning[0]
    accuracy = 0
    for j in range(0, 200):
        test_lmsqr = numpy.array(mac_after_binning[j])
        #print test_lmsqr
        #print linear_attenuation_coefficient_array[element_number - 1]
        lmsqr = 1000000
        index = 0
        list_lmsqr = []
        sum_lmsqr = 0
        for k in range(1, 78):
            #print test_lmsqr
            #print mass_attenuation_coefficient_array[i - 1]
            mse = ((test_lmsqr - linear_attenuation_coefficient_array[k - 1]) ** 2).mean()
            #print 'the',k,'th mse', mse
            list_lmsqr.append(mse)
            sum_lmsqr = sum_lmsqr + mse
        sme_array = numpy.array(list_lmsqr)
        for kk in range(0, 77):
            if (sme_array[kk] / sum_lmsqr) < lmsqr:
                lmsqr = (sme_array[kk] / sum_lmsqr)
                index = kk + 1
        #print index
        if index == element_number:
        #global accuracy
            accuracy = accuracy + 1
    list_to_return.append(float(accuracy) / 200)
        
    return list_to_return

def manhattan_print_accuracy_result(element_number, number_of_bins):
    list_to_return = []
    a =  choose_correct_sample_energy(number_of_bins, [12000, 26000])
    #print a 
    '''middle point of each two sampling energy level'''
    b = []
    for ii in range(0, len(a) - 1):
        b.append((a[ii]+a[ii+1]) / 2)
    #print b
    linear_attenuation_coefficient_list = []
    for i in range(1, 78):
        list_mac = xray.mu_elam(i, b)
        linear_attenuation_coefficient_list.append(list_mac * xray.density(i))
    linear_attenuation_coefficient_array = numpy.array(linear_attenuation_coefficient_list)
    
    
    
    mac_after_binning = show_theoretical_spectral_after_binning\
    (element_number, 200,  number_of_bins, 'noprint')
        #print mac_after_binning[0]
    accuracy = 0
    for j in range(0, 200):
        test_lmsqr = numpy.array(mac_after_binning[j])
        #print test_lmsqr
        #print linear_attenuation_coefficient_array[element_number - 1]
        lmsqr = 1000000
        index = 0
        list_lmsqr = []
        sum_lmsqr = 0
        for k in range(1, 78):
            #print test_lmsqr
            #print mass_attenuation_coefficient_array[i - 1]
            mse = scipy.spatial.distance.cityblock(test_lmsqr, linear_attenuation_coefficient_array[k - 1])
            #print 'the',k,'th mse', mse
            list_lmsqr.append(mse)
            sum_lmsqr = sum_lmsqr + mse
        sme_array = numpy.array(list_lmsqr)
        for kk in range(0, 77):
            if (sme_array[kk] / sum_lmsqr) < lmsqr:
                lmsqr = (sme_array[kk] / sum_lmsqr)
                index = kk + 1
        #print index
        if index == element_number:
        #global accuracy
            accuracy = accuracy + 1
    list_to_return.append(float(accuracy) / 200)
        
    return list_to_return



#Spectral Gradient Angle
def sga_print_accuracy_result(element_number, number_of_bins):
    list_to_return = []
    a =  choose_correct_sample_energy(number_of_bins, [12000, 26000])
    #print a 
    '''middle point of each two sampling energy level'''
    b = []
    for ii in range(0, len(a) - 1):
        b.append((a[ii]+a[ii+1]) / 2)
    #print b
    linear_attenuation_coefficient_list = []
    for i in range(1, 78):
        list_mac = xray.mu_elam(i, b)
        linear_attenuation_coefficient_list.append(list_mac * xray.density(i))
    linear_attenuation_coefficient_array = numpy.array(linear_attenuation_coefficient_list)
    
    mac_after_binning = show_theoretical_spectral_after_binning(element_number, 200, number_of_bins, 'noprint')
    accuracy = 0
    for j in range(0, 200):

        lmsga = 1000000
        index = 0
        list_sga = []
        sum_sga = 0
        for k in range(1, 78):
            spectral_gradient_test = \
            numpy.array([mac_after_binning[j][ii]\
                         - mac_after_binning[j][ii - 1] \
                                            for ii in\
                                            range(1, len(mac_after_binning[j]))])
            spectral_gradient_theo = numpy.array\
            ([linear_attenuation_coefficient_array[k - 1][ii] - \
              linear_attenuation_coefficient_array[k - 1][ii - 1] \
                                                  for ii in range(1, len(linear_attenuation_coefficient_array[k - 1]))])
                #print spectral_gradient_test
                #print mac_after_binning[j]
                #print spectral_gradient_theo
                #print mass_attenuation_coefficient_array[k - 1]
            sga = ((spectral_gradient_test - spectral_gradient_theo) ** 2).mean()
            list_sga.append(sga)
            sum_sga = sum_sga + sga
        sga_array = numpy.array(list_sga)
        for kk in range(0, 77):
            if (sga_array[kk] / sum_sga) < lmsga:
                lmsga = (sga_array[kk] / sum_sga)
                index = kk + 1
            #print index
        if index == element_number:
            accuracy = accuracy + 1
    list_to_return.append(float(accuracy) / 200)
    return list_to_return

def sid_print_accuracy_result_angled(element_number, number_of_bins):
    list_to_return = []
    a =  choose_correct_sample_energy(number_of_bins, [12000, 26000])
    #print a 
    '''middle point of each two sampling energy level'''
    b = []
    for ii in range(0, len(a) - 1):
        b.append((a[ii]+a[ii+1]) / 2)
    #print b
    linear_attenuation_coefficient_list = []
    for i in range(1, 78):
        list_mac = xray.mu_elam(i, b)
        linear_attenuation_coefficient_list.append(list_mac * xray.density(i))
    linear_attenuation_coefficient_array = numpy.array(linear_attenuation_coefficient_list)
    
    mac_after_binning = show_theoretical_spectral_after_binning(element_number, 200, number_of_bins, 'noprint')
    accuracy = 0
    for j in range(0, 200):
        test_lmsqr = numpy.array(mac_after_binning[j])
        mean_test_lsmqr = numpy.mean(test_lmsqr)
        list_sid = []
        sum_sid = 0
        lmsid = 1000000
        index = 0
        for k in range(1, 70):
            mac_array_mean = numpy.mean(linear_attenuation_coefficient_array[k - 1])
            sid = numpy.dot((test_lmsqr / mean_test_lsmqr - linear_attenuation_coefficient_array[k - 1] / mac_array_mean), \
                            numpy.log(test_lmsqr / mean_test_lsmqr)\
                            - numpy.log(linear_attenuation_coefficient_array[k - 1] / mac_array_mean))
            #sid = numpy.dot((test_lmsqr - mass_attenuation_coefficient_array[k - 1]), numpy.log(test_lmsqr) - numpy.log(mass_attenuation_coefficient_array[k - 1]))
            list_sid.append(sid)
            sum_sid = sum_sid + sid
        sid_array = numpy.array(list_sid)
        for kk in range(0, 69):
            if (sid_array[kk] / sum_sid) < lmsid:
                lmsid = (sid_array[kk] / sum_sid)
                index = kk + 1
            #print index
        if index == element_number:
            accuracy = accuracy + 1
    list_to_return.append(float(accuracy) / 200)
    return list_to_return



def sid_print_accuracy_result(element_number, number_of_bins):
    list_to_return = []
    a =  choose_correct_sample_energy(number_of_bins, [12000, 26000])
    #print a 
    '''middle point of each two sampling energy level'''
    b = []
    for ii in range(0, len(a) - 1):
        b.append((a[ii]+a[ii+1]) / 2)
    #print b
    linear_attenuation_coefficient_list = []
    for i in range(1, 78):
        list_mac = xray.mu_elam(i, b)
        linear_attenuation_coefficient_list.append(list_mac * xray.density(i))
    linear_attenuation_coefficient_array = numpy.array(linear_attenuation_coefficient_list)
    
    mac_after_binning = show_theoretical_spectral_after_binning(element_number, 200, number_of_bins, 'noprint')
    accuracy = 0
    for j in range(0, 200):
        test_lmsqr = numpy.array(mac_after_binning[j])
        #mean_test_lsmqr = numpy.mean(test_lmsqr)
        list_sid = []
        sum_sid = 0
        lmsid = 1000000
        index = 0
        for k in range(1, 70):
            #mac_array_mean = numpy.mean(linear_attenuation_coefficient_array[k - 1])
            sid = numpy.dot((test_lmsqr - linear_attenuation_coefficient_array[k - 1]), numpy.log(test_lmsqr) - numpy.log(linear_attenuation_coefficient_array[k - 1]))
                #sid = numpy.dot((test_lmsqr - mass_attenuation_coefficient_array[k - 1]), numpy.log(test_lmsqr) - numpy.log(mass_attenuation_coefficient_array[k - 1]))
            list_sid.append(sid)
            sum_sid = sum_sid + sid
        sid_array = numpy.array(list_sid)
        for kk in range(0, 69):
            if (sid_array[kk] / sum_sid) < lmsid:
                lmsid = (sid_array[kk] / sum_sid)
                index = kk + 1
            #print index
        if index == element_number:
            accuracy = accuracy + 1
    list_to_return.append(float(accuracy) / 200)
    return list_to_return

def spcra_print_accuracy_result(element_number, number_of_bins):
    list_to_return = []
    a =  choose_correct_sample_energy(number_of_bins, [12000, 26000])
    #print a 
    '''middle point of each two sampling energy level'''
    b = []
    for ii in range(0, len(a) - 1):
        b.append((a[ii]+a[ii+1]) / 2)
    #print b
    linear_attenuation_coefficient_list = []
    for i in range(1, 78):
        list_mac = xray.mu_elam(i, b)
        linear_attenuation_coefficient_list.append(list_mac * xray.density(i))
    linear_attenuation_coefficient_array = numpy.array(linear_attenuation_coefficient_list)
    
    mac_after_binning = show_theoretical_spectral_after_binning(element_number, 200, number_of_bins, 'noprint')
    accuracy = 0
    for j in range(0, 200):
        test_spcra = numpy.array(mac_after_binning[j])
        mean_test_spcra = numpy.mean(test_spcra)
        list_spcra = []
        sum_spcra = 0
        lmspcra = 1000000
        index = 0
        for k in range(1, 70):
            #mac_array_mean = numpy.mean(mass_attenuation_coefficient_array[k - 1])
            #sid = numpy.dot((test_lmsqr / mean_test_lsmqr - mass_attenuation_coefficient_array[k - 1] / mac_array_mean), \
                             #numpy.log(test_lmsqr / mean_test_lsmqr)\
                            #- numpy.log(mass_attenuation_coefficient_array[k - 1] / mac_array_mean))
            spcra = numpy.dot((test_spcra - mean_test_spcra), \
                              (linear_attenuation_coefficient_array[k - 1] - \
                               numpy.mean(linear_attenuation_coefficient_array[k - 1]))\
                               ) / (numpy.linalg.norm(test_spcra - mean_test_spcra) * numpy.linalg.norm(\
                    linear_attenuation_coefficient_array[k - 1] - numpy.mean(linear_attenuation_coefficient_array[k - 1])))
            list_spcra.append(spcra)
            sum_spcra = sum_spcra + spcra
        spcra_array = numpy.array(list_spcra)
        for kk in range(0, 69):
            if (spcra_array[kk] / sum_spcra) < lmspcra:
                lmspcra = (spcra_array[kk] / sum_spcra)
                index = kk + 1
       # print index
        if index == element_number:
            accuracy = accuracy + 1
    list_to_return.append(float(accuracy) / 200)
    return list_to_return