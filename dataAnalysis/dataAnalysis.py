# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:23:12 2019

@author: dgonc
"""

#import pandas as pd 
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import math
import numpy as np
import statistics as st

#file_ab = pd.read_excel('dadoscolunaab.xlsx')
#file_no = pd.read_excel('dadoscolunano.xlsx')

#txt_file_ab = file_ab.to_csv('dadoscolunaab.txt', sep="\t",index=True)
#txt_file_no = file_no.to_csv('dadoscolunano.txt', sep="\t",index=True)
data_ab = pl.loadtxt('dadoscolunaab.txt',usecols=(0,1,2,3,4,5,6,7,8,9) ,skiprows=1)
data_no = pl.loadtxt('dadoscolunano.txt',usecols=(0,1,2,3,4,5,6,7,8))

arr = np.array([0, 0, 0, 1, 1, 1, 2, 3, 3, 4, 8]) # auxiliar array

#index GÉNERO	IDADE	OPER	PI	PT	LLA	SS	PR	GS	CLASS

index_ab = data_ab[:,0] #Abnormal people index number (means the amount of people)
index_no = data_no[:,0] #Normal people index number (means the amount of people)

gender_ab = data_ab[:,1]
gender_no = data_no[:,1]

pi_ab = data_ab[:,4]

ab_amount = len(index_ab)
no_amount = len(index_no)

total_amount = ab_amount+no_amount #Total amount of people NO+AB

age_ab = data_ab[:,2] #Abnormal class ages
age_no = data_no[:,2] #Normal class ages

cirur_ab = data_ab[:,3] #Abnormal Cirurgy amount
cirur_no = data_no[:,3] #Normal Cirurgy amount


ss_ab = data_ab[:,7] #SS abnormal class variable column
ss_no = data_no[:,6] #SS normal class variable column

gs_ab = data_ab[:,9] #GS abnormal class variable column
gs_no = data_no[:,8] #GS normal class variable column

#print('Number of abnormal column people: ', ab_amount) #Abnormal class variable
#print('Number of normal column people: ', no_amount) #Normal class variable

#sum of all abnormal SS and GS
ss_ab_sum = pl.sum(ss_ab)
gs_ab_sum = pl.sum(gs_ab)
print('Sum of all abnormal elements')
print('Abnormal SS: ',ss_ab_sum)
print('Abnormal GS: ',gs_ab_sum)
#sum of all normal SS and GS
print('Sum of all normal elements')
ss_no_sum = pl.sum(ss_no)
gs_no_sum = pl.sum(gs_no)
print('Normal SS: ',ss_no_sum)
print('Abnormal GS: ',gs_no_sum,'\n')

ss_ab_mean = pl.mean(ss_ab) #SS Abnormal Class mean
ss_no_mean = pl.mean(ss_no) #SS Normal Class mean
gs_ab_mean = pl.mean(gs_ab)
gs_no_mean = pl.mean(gs_no)
print('Mean')
print('Abnormal SS: ', ss_ab_mean)
print('Normal SS: ', ss_no_mean)
print('Abnormal GS: ', gs_ab_mean)
print('Normal GS: ', gs_no_mean,'\n')

#pattern deviation
ss_ab_pat_dev = st.stdev(ss_ab)
gs_ab_pat_dev = st.stdev(gs_ab)
ss_no_pat_dev = st.stdev(ss_no)
gs_no_pat_dev = st.stdev(gs_no)
print('Standard Deviation:')
print('Abnormal SS: ',ss_ab_pat_dev)
print('Abnormal GS: ',gs_ab_pat_dev)
print('Normal SS: ',ss_no_pat_dev)
print('Normal GS: ',gs_no_pat_dev,'\n')

#median
ss_ab_median = pl.median(ss_ab)
gs_ab_median = pl.median(gs_ab) 
ss_no_median = pl.median(ss_no)
gs_no_median = pl.median(gs_no)
print('Median:')
print('Abnormal SS: ',ss_ab_median)
print('Abnormal GS: ',gs_ab_median)
print('Normal SS: ',ss_no_median)
print('Normal GS: ',gs_no_median,'\n')

#variance
ss_ab_variance = st.variance(ss_ab)
gs_ab_variance = st.variance(gs_ab)
ss_no_variance = st.variance(ss_no)
gs_no_variance = st.variance(gs_no)
print('Variance')
print('Abnormal SS: ',ss_ab_variance)
print('Abnormal GS: ',gs_ab_variance)
print('Normal SS: ',ss_no_variance)
print('Normal GS: ',gs_no_variance,'\n')

#Sturges Law
def create_class(variable,amount):
    ln_n = math.log(amount) 
    k = 1+(ln_n/math.log(2))
    minimum = min(variable)
    maximum = max(variable)
    h = (maximum-minimum)/k
    class_quant = int(k//1)

    return class_quant,h;

#it can easily be done like this
#sturges_law = np.histogram(ss_ab, bins='sturges', density=True)

cl_ss_ab = create_class(ss_ab,ab_amount)
cl_ss_no = create_class(ss_no,no_amount)
cl_gs_ab = create_class(gs_ab,ab_amount)
cl_gs_no = create_class(gs_no,no_amount)
cl_age_ab = create_class(age_ab,ab_amount)
cl_age_no = create_class(age_no,no_amount)

#classes amplitude
cl_ss_ab_h = pl.array(cl_ss_ab[1])
cl_ss_no_h = pl.array(cl_ss_no[1])
cl_gs_ab_h = pl.array(cl_gs_ab[1])
cl_gs_no_h = pl.array(cl_gs_no[1])
cl_age_ab_h = pl.array(cl_age_ab[1])
cl_age_no_h = pl.array(cl_age_no[1])

#classes quantity

#creating an histogram model
st_ss_ab = np.histogram(ss_ab, bins=pl.arange(min(ss_ab),max(ss_ab)+cl_ss_ab_h,cl_ss_ab_h), density=True)
st_ss_no = np.histogram(ss_no, bins=pl.arange(min(ss_no),max(ss_no)+cl_ss_no_h,cl_ss_no_h), density=True)
st_gs_ab = np.histogram(gs_ab, bins=pl.arange(min(gs_ab),max(gs_ab)+cl_gs_ab_h,cl_gs_ab_h), density=True)
st_gs_no = np.histogram(gs_no, bins=pl.arange(min(gs_no),max(gs_no)+cl_gs_no_h,cl_gs_no_h), density=True)
st_age_ab = np.histogram(age_ab, bins=pl.arange(min(age_ab),max(age_ab)+cl_age_ab_h,cl_age_ab_h), density=True)
st_age_no = np.histogram(age_no, bins=pl.arange(min(age_no),max(age_no)+cl_age_no_h,cl_age_no_h), density=True)

classes_ss_ab = pl.array(st_ss_ab[1])
classes_ss_no = pl.array(st_ss_no[1])
classes_gs_ab = pl.array(st_gs_ab[1])
classes_gs_no = pl.array(st_gs_no[1])
classes_age_ab = pl.array(st_age_ab[1])
classes_age_no = pl.array(st_age_no[1])

#amount of classes and counter of classes
#abnormal
hist_ss_ab = np.histogram(ss_ab,classes_ss_ab)
hist_gs_ab = np.histogram(gs_ab,classes_gs_ab)
hist_age_ab = np.histogram(age_ab,st_age_ab[1])[0]
count_classes_ss_ab = pl.array(hist_ss_ab[0])
count_classes_gs_ab = pl.array(hist_gs_ab[0])
count_classes_age_ab = pl.array(hist_age_ab)

#normal
hist_ss_no = np.histogram(ss_no)
hist_gs_no = np.histogram(gs_no)
hist_age_no = np.histogram(age_no,st_age_no[1])
count_classes_ss_no = pl.array(hist_ss_no[0])
count_classes_gs_no = pl.array(hist_gs_no[0])
count_classes_age_no = pl.array(hist_age_no)[0]

#plotting histograms
#pl.hist(ss_ab,bins=pl.arange(min(ss_ab),max(ss_ab)+cl_ss_ab_h,cl_ss_ab_h))
#pl.hist(ss_ab,cumulative=True,bins=pl.arange(min(ss_ab),max(ss_ab)+cl_ss_ab_h,cl_ss_ab_h))
#pl.hist(gs_ab,bins=pl.arange(min(gs_ab),max(gs_ab)+cl_gs_ab_h,cl_gs_ab_h))
#pl.hist(ss_no,cumulative=True,bins=pl.arange(min(ss_no),max(ss_no)+cl_ss_no_h,cl_ss_no_h))
#pl.hist(gs_no,bins=pl.arange(min(gs_no),max(gs_no)+cl_gs_no_h,cl_gs_no_h))

#Counting elements function
def count_elem(elem):
    x = 0
    y = 0
    z = 0
    for i in range(len(elem)):
        if (elem[i] == 0):
            x= x+1
        elif (elem[i]==1):
            y= y+1
        elif (elem[i]==2):
            z=z+1
    return x, y, z;

ab_elem_gen = pl.array(count_elem(gender_ab))
no_elem_gen = pl.array(count_elem(gender_no))
ab_elem_cir = pl.array(count_elem(cirur_ab))
no_elem_cir = pl.array(count_elem(cirur_no))

ab_male = ab_elem_gen[0]
ab_female = ab_elem_gen[1]
no_male = no_elem_gen[0]
no_female = no_elem_gen[1]
ab_zero_cir = ab_elem_cir[0]
ab_one_cir = ab_elem_cir[1]
ab_two_cir = ab_elem_cir[2]
no_zero_cir = no_elem_cir[0]
no_one_cir = no_elem_cir[1]
no_two_cir = no_elem_cir[2]

#Relative Frequency function
def rel_freq(variable,amount):
    aux = []
    rel_freqVal = -1
    try:
        validation = variable[0]
        for i in range(len(variable)):
            aux += [(variable[i]/amount)]
    except:
        rel_freqVal = (variable/amount)
    if(rel_freqVal == -1):
        return aux;
    else:
        return rel_freqVal;

#Relative Frequency Values
rel_freq_ab = rel_freq(ab_amount,total_amount)
rel_freq_no = rel_freq(no_amount,total_amount)
rel_freq_ab_male = rel_freq(ab_male,ab_amount)
rel_freq_ab_female = rel_freq(ab_female,ab_amount)
rel_freq_no_male = rel_freq(no_male,no_amount)
rel_freq_no_female = rel_freq(no_female,no_amount)
rel_freq_classes_ss_ab = rel_freq(count_classes_ss_ab,ab_amount)
rel_freq_classes_ss_no = rel_freq(count_classes_ss_no,no_amount)
rel_freq_classes_gs_ab = rel_freq(count_classes_gs_ab,ab_amount)
rel_freq_classes_gs_no = rel_freq(count_classes_gs_no,no_amount)
rel_freq_age_ab = rel_freq(count_classes_age_ab,ab_amount)
rel_freq_age_no = rel_freq(count_classes_age_no,no_amount)
rel_freq_zerocir_ab = rel_freq(ab_zero_cir,ab_amount)
rel_freq_onecir_ab = rel_freq(ab_one_cir,ab_amount)
rel_freq_twocir_ab = rel_freq(ab_two_cir,ab_amount)


#Creating circular graph
"""labels = ['Male Normal','Female Normal']
sizes = [rel_freq_no_male, rel_freq_no_female]
colors = ['yellowgreen', 'gold']
pie_plot = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=90)
plt.legend(labels, loc="best")
plt.title('Gender Relative Frequency')
plt.axis('equal')
plt.tight_layout()
plt.show()"""

#Creating Bar graph
"""groups = len(to_plot_no)
fig, ax = plt.subplots()
index = pl.arange(groups)
bar_width = 0.25
opacity = 0.8
rects1 = plt.bar(index, to_plot_ab*100, bar_width, alpha=opacity, color='b', label='Abnormal')
rects2 = plt.bar(index + bar_width, to_plot_no*100, bar_width, alpha=opacity, color='g', label='Normal')
plt.xlabel('Variables')
plt.ylabel('Relative Frequency')
plt.title('Comparison')
plt.xticks(index+0.1, ('Classes', 'Male', 'Female'))
plt.legend()
plt.tight_layout()
plt.show()"""

#Absolute Frequency function
def abs_freq(variable,amount):
    aux = []
    for i in range(len(variable)):
        a = variable[i]/amount
        aux += [a]
    return aux;

#cumulative frequency
abs_ss_ab_model = stats.cumfreq(ss_ab, numbins=len(classes_ss_ab)-1, defaultreallimits=(min(classes_ss_ab),max(classes_ss_ab)))
abs_ss_ab = abs_ss_ab_model[0]
cum_freq_ss_ab = abs_freq(abs_ss_ab,ab_amount)

abs_gs_ab_model = stats.cumfreq(gs_ab, numbins=len(classes_gs_ab)-1, defaultreallimits=(min(classes_gs_ab),max(classes_gs_ab)))
abs_gs_ab = abs_gs_ab_model[0]
cum_freq_gs_ab = abs_freq(abs_gs_ab,ab_amount)

abs_ss_no_model = stats.cumfreq(ss_no, numbins=len(classes_ss_no)-1, defaultreallimits=(min(classes_ss_no),max(classes_ss_no)))
abs_ss_no = abs_ss_no_model[0]
cum_freq_ss_no = abs_freq(abs_ss_no,no_amount)

abs_gs_no_model = stats.cumfreq(gs_no, numbins=len(classes_gs_no)-1, defaultreallimits=(min(classes_gs_no),max(classes_gs_no)))
abs_gs_no = abs_gs_no_model[0]
cum_freq_gs_no = abs_freq(abs_gs_no,no_amount)

print('Abnormal SS classes')
print('Classes: \n',classes_ss_ab)
print('ni: ',count_classes_ss_ab)
print('Relative Frequency: \n',rel_freq_classes_ss_ab)
print('Cumulative Frequency: \n',cum_freq_ss_ab)
print('Amplitude: ',cl_ss_ab_h,'\n')

print('Normal SS classes')
print('Classes: \n',classes_ss_no)
print('ni: \n',count_classes_ss_no)
print('Relative Frequence: \n',rel_freq_classes_ss_no)
print('Cumulative Frequence: \n',cum_freq_ss_no)
print('Amplitude: \n',cl_ss_no_h,'\n')

print('Normal GS classes')
print('Classes: \n',classes_gs_no)
print('ni: \n',count_classes_gs_no)
print('Relative Frequence: \n',rel_freq_classes_gs_no)
print('Cumulative Frequence: \n',cum_freq_ss_no)
print('Amplitude: \n',cl_gs_no_h,'\n')

print('Abnormal GS classes')
print('Classes: \n',classes_gs_ab)
print('ni: \n',count_classes_gs_ab)
print('Relative Frequency: \n',rel_freq_classes_gs_ab)
print('Cumulative Frequency: \n',cum_freq_gs_no)
print('Amplitude: \n',cl_gs_ab_h,'\n')

print('Abnormal age classes')
print('Classes: \n',classes_age_ab)
print('ni: \n',count_classes_age_ab)
print('Relative Frequency: \n',rel_freq_age_ab)
print('Amplitude: \n',cl_age_ab_h,'\n')

print('Normal age classes')
print('Classes: \n',classes_age_no)
print('ni: \n',count_classes_age_no)
print('Relative Frequency: \n',rel_freq_age_no)
print('Amplitude: \n',cl_age_no_h,'\n')

#mode
def mode_making(variable):
    counting = st._counts(variable)
    conv_to_array = pl.array(counting[0])
    mode = conv_to_array[0]
    quantity = conv_to_array[1]
    
    return mode, quantity;

ss_ab_mode_func = mode_making(ss_ab)
ss_no_mode = mode_making(ss_no)[0]
ss_ab_mode = ss_ab_mode_func[0]
gs_ab_mode = mode_making(gs_ab)[0]
gs_no_mode = mode_making(gs_no)[0]

print('Abnormal SS Mode: ',ss_ab_mode)
print('Normal SS Mode: ',ss_no_mode)
print('Abnormal GS Mode: ',gs_ab_mode)
print('Normal GS Mode: ',gs_no_mode,'\n')


#ss_ab_mode_pos = int(ss_ab_mode_func[1]) #quantity of elements

age_ab_mode_func = mode_making(age_ab)
age_ab_mode = age_ab_mode_func[0]

def modal_class(variable1,variable2,amplitude):
    x = max(variable1)
    y = 0
    class_y = 0
    for i in range(len(variable1)):
        if (variable1[i]==x):
            y = variable2[i]
    class_y = [y,y+amplitude]
    return x, class_y;

ss_ab_modal_model = pl.array(modal_class(count_classes_ss_ab,classes_ss_ab,cl_ss_ab_h))
ss_ab_modal_mode = ss_ab_modal_model[0]
ss_ab_modal_class = ss_ab_modal_model[1]
gs_ab_modal_model = pl.array(modal_class(count_classes_gs_ab,classes_gs_ab,cl_gs_ab_h))
gs_ab_modal_class = gs_ab_modal_model[1]
age_ab_modal_class = pl.array(modal_class(count_classes_age_ab,classes_age_ab,cl_age_ab_h))
#print('Abnormal age modal class: ',age_ab_modal_mode)
print('Abnormal SS modal class: ',ss_ab_modal_class)
print('Abnormal GS modal class: ',gs_ab_modal_class)

ss_no_modal_model = pl.array(modal_class(count_classes_ss_no,classes_ss_no,cl_ss_no_h))
ss_no_modal_mode = ss_no_modal_model[0]
ss_no_modal_class = ss_no_modal_model[1]
gs_no_modal_model = pl.array(modal_class(count_classes_gs_no,classes_gs_no,cl_gs_no_h))
gs_no_modal_class = gs_no_modal_model[1]
print('Normal SS modal class: ',ss_no_modal_class)
print('Normal GS modal class: ',gs_no_modal_class,'\n')

#to_plot puts the values above to an array so you can plot the graphs in an easier way
to_plot_ab = pl.array([rel_freq_ab,rel_freq_ab_male,rel_freq_ab_female])
to_plot_no = pl.array([rel_freq_no,rel_freq_no_male,rel_freq_no_female])

#Linear Regression Models
x_no = pl.array(ss_no.reshape((-1,1))) #X - Independent Variable
y_no = pl.array(gs_no) #Y - Dependent Variable

x_ab = pl.array(ss_ab.reshape((-1,1))) #X - Independent Variable
#ss_x_ab = pl.array(ss_ab) #Y - Dependent Variable
y_ab = pl.array(gs_ab) #Y - Dependent Variable

no_model = LinearRegression().fit(x_no,y_no) #Normal Class linear regression model
no_y_pred = no_model.predict(x_no) #Prediction X based (line)
no_cor_coef = no_model.score(x_no,y_no) #R^2 Coeficient
r_no_coef = math.sqrt(no_cor_coef) #Square route R^2 = R
print('Normal linear regression coeficient: ',r_no_coef)

ab_model = LinearRegression().fit(x_ab,y_ab) #Normal Class linear regression model
ab_y_pred = ab_model.predict(x_ab) #Prediction X based (line)
ab_cor_coef = ab_model.score(x_ab,y_ab) #R^2 Coeficient
r_ab_coef = math.sqrt(ab_cor_coef) #Square route R^2 = R
print('Abnormal linear regression coeficient: ',r_ab_coef,'\n')

"""plt.subplot(221)
plt.title('Normal GS/SS')
plt.xlabel('SS')
plt.ylabel('GS')
plt.scatter(x_no,y_no)
plt.plot(x_no,no_y_pred,color='red')
plt.show()

plt.subplot(222)
plt.title('Abnormal GS/SS')
plt.xlabel('SS')
plt.ylabel('GS')
plt.scatter(x_ab,y_ab)
plt.plot(x_ab,ab_y_pred,color='red')
plt.show()"""