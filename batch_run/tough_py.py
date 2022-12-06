# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:26:17 2020

@author: ASUS2
"""
# -*- coding: utf-8 -*-
#coding=utf-8
#-*- coding : utf-8-*-
# coding:unicode_escape
import csv
import numpy as np 
import os 
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import mcmc
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import subprocess
import math
import time
#网格数量
TimeYear=61
def HF_model(modelname, main_dir, X,idf):

    porosity = np.loadtxt('./porosity.txt', delimiter=', ', dtype=str,
                          skiprows=0)
    porosity = porosity.reshape(1, -1)
    pp = np.tile(porosity, (5, 1)).astype('float64')
    permeability = pow(10, -X)

    for j in range(0,5):
        workspace = main_dir + "/HF_in_chain" + str(idf[j])
        f = open(workspace+'/INCON1', 'r')
        f_new = open(workspace+'/INCON', 'w')
        i = 0
        while i < 2500:
            for line in f:
                if "aaaaaaaaaaaaaa" in line:
                    line = line.replace("aaaaaaaaaaaaaa",
                                        str("{0:.8E}".format(pp[j, i])))
                if "bbbbbbbbbbbbbb" in line:
                    line = line.replace("bbbbbbbbbbbbbb",
                                        str("{0:.8E}".format(
                                            permeability[j, i])))
                if "cccccccccccccc" in line:
                    line = line.replace("cccccccccccccc",
                                        str("{0:.8E}".format(
                                            permeability[j, i])))
                if "dddddddddddddd" in line:
                    line = line.replace("dddddddddddddd", str(
                        "{0:.8E}".format(permeability[j, i] / 10)))
                    i = i + 1
                f_new.write(line)
        f.close()
        f_new.close()


    TOUGHReactName = "run1.bat"
    subprocess.call(TOUGHReactName)
    time.sleep(100)
    print("22222")

    fy = np.zeros([5, TimeYear])
    for i in range(0,5):
        workspace = main_dir + "/HF_in_chain" + str(idf[i])
        path1 = workspace + '/Ttim.dat'
        data = pd.read_table(path1, header=4, sep='\s+')
        data = data.to_csv('f.csv', index=False, header=True)
        data = pd.read_csv('./f.csv', engine="python")
        data.iloc[:, 1] = data.iloc[:, 1] * 365.25
        data.iloc[:, 1] = data.iloc[:, 1].astype(np.int)
        data.columns
        data.rename(columns={'Time(yr)': 't', 't_26nds': 'con'},
                    inplace=True)
        data = data[['t', 'con']]
        groups = data.groupby(["t"]).max()
        groups = np.array(groups.reset_index('t'))
        xxx = []
        for j in range(0, groups.shape[0]):
            if groups[j, 0] % 1 == 0:
                xxx.append(groups[j, 1])
        yyy = np.array(xxx) * 100
        y1 = yyy[0:TimeYear]
        for ii in range(0, y1.shape[0]):
            fy[i, ii] = y1[ii]
    y = fy.reshape((5, -1))

    return y



def extract(workspace,X,idf,iter):
    if not os.path.exists(workspace):
        os.makedirs(workspace) #创建多级目录
    modelname="EGS"
    try:
        os.remove(os.path.join(workspace, 'MT3D001.UCN'))
        os.remove(os.path.join(workspace, modelname + '.hds'))
        os.remove(os.path.join(workspace, modelname + '.cbc'))
    except:
        pass

    print(iter)
    X=X[:,0:]
    y = HF_model(modelname, workspace, X, idf)
    return y


















