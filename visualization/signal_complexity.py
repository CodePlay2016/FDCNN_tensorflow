#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 16:09:32 2018

@author: codeplay2018
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import os, glob

# global settings
# data_dir = "/home/codeplay2018/E/code/lab/data/TestDataFromWen/arranged/steady_condition/"
data_dir = "E:\\code\\lab\\data\\TestDataFromWen\\arranged\\steady_condition\\"
fs = 12000
font_name='Time New Roman'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.default'] = 'regular' # set math text font the same as normal text
file_list = glob.glob(data_dir+'*12k*.txt')
list.sort(file_list)
show_file_list = ['pmt_'+str(fs//1000)+'k_10.txt',
                  'pmt_'+str(fs//1000)+'k_20.txt',
                  'pmt_'+str(fs//1000)+'k_30.txt',
                  'pmt_'+str(fs//1000)+'k_40.txt',
                  'pmt_'+str(fs//1000)+'k_50.txt']
dataset_list = ["A", "B", "C", "D", "E"]
bigtitle_font = {'family':'serif', 'fontsize':17}
subtitle_font = {'fontsize':16}

def dofft(signal, fs):
    N = len(signal)
    fsignal = np.abs(np.fft.fft(signal))*2/N
    fsignal = fsignal[:N//2]
    f = fs*np.arange(0,N//2)/N
    return fsignal, f

def plot_tNf_for_signal():
    plt.figure(figsize=(12,6))
    plt.subplots_adjust(top=0.75,bottom=0.05)
    label_size = 12
    for ii, filename in enumerate(file_list):
        if ii in range(0,5):
            signal = np.loadtxt(filename)[:,3]
            N = len(signal) # sample poingts
            t = np.arange(0,N/fs,1/fs)
            transformed, f = dofft(signal, fs)
            rf = int(filename.split('/')[-1].split('_')[-1].split('.')[0])//2

            ax = plt.subplot(5,2,ii%5*2+1)
            plt.plot(t[:fs],signal[:fs])
            ax.set_ylim(-1,1)
            ax.set_xlim(0,max(t[:fs]))
            # plt.title("time %d"%rf)
            plt.title("WC-%s: Time Series"%dataset_list[ii%5],fontsize=label_size)
            if ii%5 == 4:
                plt.xlabel("Time/s",fontsize=label_size)
            if ii%5 == 2:
                plt.ylabel("Amplitude",fontsize=label_size)

            ax = plt.subplot(5,2,ii%5*2+2)
            plt.plot(f,transformed)
            ax.set_ylim(0,max(transformed))
            ax.set_xlim(0,6000)
            # plt.title("FFT %d"%rf)
            print(filename)
            plt.title("WC-%s: Fourier Spectrum"%dataset_list[ii%5],fontsize=label_size)
            if ii%5 == 4:
                plt.xlabel("Frequency/Hz",fontsize=label_size)
    plt.tight_layout(pad=2, w_pad=2, h_pad=0.5)
    plt.suptitle("(a) Normal gearbox",fontsize=14)
    
    plt.show()


#%% 
## draw the spectrum between two ro-speed to show the differentce of frequency component
def plot_spectrum_difference():
    plt.figure(figsize=(10,10))
    for ii in range(len(show_file_list)):
        filename = show_file_list[ii]
        signal = np.loadtxt(os.path.join(data_dir, filename))[:,3]
        N = len(signal) # sample poingts
        transformed, f = dofft(signal, fs)
        rf = int(filename.split('_')[-1].split('.')[0])//2
        mf = rf/5*109.37 # meshing freq
        x_start, x_end = (int((mf-100)/fs*N),int((mf+100)/fs*N))
        mesh_range = list(range(x_start, x_end))
        f, fsignal = (f[mesh_range],transformed[mesh_range])
        y_max = np.max(fsignal)
        max_index = fsignal.argmax()
        
        plt.subplot(len(show_file_list),1,ii+1)
        plt.plot(f, fsignal, lw=1)
        ax = plt.gca()
        if ii < 3:
            plt.annotate('meshing frequency',xy=(mf,0.1*y_max),xytext=(-30,50), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', facecolor='r', lw=0.8,connectionstyle='arc3,rad=-0.15'),
                fontsize=13, ha='right', va='center',color='r')
        else:
            plt.annotate('meshing frequency',xy=(mf,0.9*y_max),xytext=(30,0), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', facecolor='r', lw=0.8), fontsize=13,
                ha='left', va='center',color='r')
        xticks = list(plt.xticks()[0]) + [mf]
        if ii in (4,2,1): 
            xticks.remove(xticks[5]) # remove occlusion

        plt.title('Working condition '+dataset_list[4-ii]+': input shaft speed of %dHz'%(rf), **subtitle_font)
        plt.xticks(xticks)
        tick = ax.xaxis.get_major_ticks()[-1]
        tick.label1.set_color('r')
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=min(f), right=max(f))
        ax.set_xlabel('Frequency/Hz',fontsize=14)
        ax.set_ylabel('Amplitude',fontsize=14)
        # plt.subplots_adjust(bottom=0.05)
        plt.tight_layout()
    # plt.suptitle('Side band near meshing frequency under two different rotational speed',
    #             **bigtitle_font)
    plt.show()

# f_list_25Hz = [508.1,513.54,518.5,,,,,,
#                 ,,,,,]
f_dict_25Hz = {r'$f_{mesh}-f_{carrier}-f_{planet pass}-f_{planet}$':508.1, r'$f_{mesh}-2f_{carrier}-f_{planet pass}$':513.54, 
               r'$f_{mesh}-f_{carrier}-f_{planet pass}$':518.5, r'$f_{mesh}-f_{planet pass}$':523.9,
               r'$f_{mesh}-f_{faulty planet}$':530.57, r'$f_{mesh}-f_{planet}$':534.77,
               r'$f_{mesh}-f_{carrier}$':540.2, r'$f_{mesh}$':545.8,r'$f_{mesh}+f_{carrier}$':551.2,
               r'$f_{mesh}+f_{planet}$':556.61, r'$f_{mesh}+f_{planet pass}$':567.59, r'$f_{mesh}+f_{planet pass}+f_{carrier}$':573.1,
               r'$f_{mesh}+f_{planet pass}+f_{planet}$':577.95,r'$f_{mesh}+f_{planet pass}+f_{planet}+f_{carrier}$':583.4}

def plot_multi_components_on_spectrum(f_dict):
    plt.figure(figsize=(10,5))
    filename = show_file_list[1]
    signal = np.loadtxt(os.path.join(data_dir, filename))[:,3]
    N = len(signal) # sample poingts
    transformed, f = dofft(signal, fs)
    rf = int(filename.split('_')[-1].split('.')[0])//2
    mf = rf/5*109.37 # meshing freq
    x_start, x_end = (int((mf-60)/fs*N),int((mf+60)/fs*N))
    mesh_range = list(range(x_start, x_end))
    f, fsignal = (f[mesh_range],transformed[mesh_range])
    y_max = np.max(fsignal)
    max_index = fsignal.argmax()

    # plot spectrum
    plt.plot(f, fsignal, lw=1.5)
    ax = plt.gca() 
    # plt.suptitle('Fourier spectrum of tooth clipped PG vibration signal with input shaft speed of %dHz'%(rf), **bigtitle_font)
    # plt.text(mf,0.9*y_max,'<--meshing frequency', color='r')
    xticks = list(plt.xticks()[0]) + [f[int(max_index)]]
    plt.xticks(xticks)
    tick = ax.xaxis.get_major_ticks()[-1]
    tick.label1.set_color('r')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=min(f), right=max(f))
    ax.set_xlabel('Frequency/Hz',fontsize=14)
    ax.set_ylabel('Amplitude',fontsize=14)
    plt.subplots_adjust(bottom=0.1)
    # plt.suptitle('Frequency band near meshing frequency',
    #             **bigtitle_font)
    for f_name,freq in f_dict_25Hz.items():
        freq = f[np.abs(f-freq)<0.1][0]
        f_index = np.argwhere(f==freq)[0]
        print(freq,f_index)
        plt.plot([freq,freq],[fsignal[f_index],0.6], lw=0.6, ls='dashed')
        plt.text(freq-3.5,(0.5+len(f_name)/180)*y_max, f_name, rotation='vertical',fontsize=13)
    plt.show()

def plot_spectrum():
    filename = show_file_list[1]
    signal = np.loadtxt(os.path.join(data_dir, filename))[:,3]
    N = len(signal)
    fsignal, f = dofft(signal, fs)
    fsignal, f = fsignal[:N//4], f[:N//4]
    plt.figure(figsize=(10,5))
    plt.plot(f, fsignal, lw=1)
    ax = plt.gca()
    # plt.suptitle('Signal spectrum with input shaft speed of %dHz'%(25), **bigtitle_font)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=min(f), right=max(f))
    ax.set_xlabel('Frequency/Hz', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.add_patch(
        patches.Rectangle(
            ((545.8-60),-0.001),
            120, 0.065,
            lw=1.5, facecolor='none', edgecolor="r",ls="--", zorder=10
        )
    )
    plt.annotate("MF",xy=(545,0.062),xytext=(50,10), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', facecolor='black', lw=0.8,
            connectionstyle='arc3,rad=0.15'),
            ha='center', va='bottom',color='blue')
    plt.annotate("RF",xy=(2050,0.085),xytext=(50,-10), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', facecolor='black', lw=0.8,
            connectionstyle='arc3,rad=-0.15'),
            ha='center', va='bottom',color='blue')
    plt.xticks(list(range(250,3000,250)))
    plt.show()

acc_list1 = [33.0,52.9,70.0,95.8]
acc_list2 = [44.0,76.9,89.7,99.6]
name1 = ["1","2","3","4"]

acc_list3 = [95.8,96.4,68.8,47.0,69.6]
acc_list4 = [99.6,98.0,99.8,98.2,98.8]
name2 = ["WC-A", "WC-B", "WC-C", "WC-D", "WC-E"]
def plot_bar_experiment1(list1,list2,name):
    
    width = 0.3
    ocapacity = 0.6
    x = np.arange(len(list1))

    plt.bar(x, list1, width=width, label='SCNN',fc='g',alpha=ocapacity)
    plt.bar(x+width, list2, width=width, label='DCNN',fc='r',alpha=ocapacity)
    plt.xticks(x+width/2,name)
    for ii in range(len(list1)):
        offset = (-0.1,1.5)
        plt.text(x[ii]+offset[0],list1[ii]+offset[1], str(list1[ii]))
        plt.text(x[ii]+width+offset[0],list2[ii]+offset[1], str(list2[ii]))
    # plt.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05))
    plt.legend()
    plt.xlabel("Number of training speed conditions", fontsize=14)
    plt.ylabel("Accuracy/%", fontsize=14)
    plt.ylim(top=110)
    plt.show()



if __name__ == '__main__':
    plot_tNf_for_signal()
    # plot_spectrum_difference()
    # plot_multi_components_on_spectrum(f_dict_25Hz)
    # plot_spectrum()
    # plot_bar_experiment1(acc_list1,acc_list2,name1)
    # print(file_list)
