# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:12:54 2022

@author: 20160
"""

import csv
import numpy as np 
import os 
import shutil
import pandas as pd
import matplotlib 

import matplotlib.pyplot as plt
import subprocess
import math
import time

day_num=1100
t_array=np.arange(day_num)*86400
time_num=104
trail_num=int(day_num/time_num)+1
t_idx=0

for i in range(trail_num):
	
	source_dir='tough/'
	target_dir='tough_'+str(i)

	if os.path.exists(target_dir):
		shutil.rmtree(target_dir)
	shutil.copytree(source_dir,target_dir)
	print('copy tough to tough_'+str(i)+' finished!')
	
	workspace=target_dir
	f=open(source_dir+'/flow_wang.inp','r')
	f_new=open(workspace+'/flow.inp','w')
	
	
	for line in f:
		if 'ttttttttt'  in line:
			line=line.replace('ttttttttt',str('{0:.3E}'.format(t_array[np.min([t_idx+time_num,day_num-1])]+86400)))
		if 'xxx'  in line:

			if i <trail_num-1:
				line=line.replace('ttttttttt',str('{3d}'.format(trail_num)))	
			else:
				line=line.replace('ttttttttt',str('{3d}'.format(day_num%time_num)))
		if 'aaaaaaaaaa' in line :
			if t_idx<day_num:
				line=line.replace('aaaaaaaaaa',str('{0:.4E}'.format(t_array[t_idx])))
				t_idx+=1
			else:
				line=line.replace('aaaaaaaaaa','')
			
		if 'bbbbbbbbbb' in line:
			if t_idx<day_num:
				line=line.replace('bbbbbbbbbb',str('{0:.4E}'.format(t_array[t_idx])))
				t_idx+=1
			else:
				line=line.replace('bbbbbbbbbb','')
			
		if 'cccccccccc' in line:
			if t_idx<day_num:
				line=line.replace('cccccccccc',str('{0:.4E}'.format(t_array[t_idx])))
				t_idx+=1
			else:
				line=line.replace('cccccccccc','')
			
		if 'dddddddddd' in line:
			if t_idx<day_num:
				line=line.replace('dddddddddd',str('{0:.4E}'.format(t_array[t_idx])))
				t_idx+=1
			else:
				line=line.replace('dddddddddd','')
			
		if 'eeeeeeeeee' in line:
			if t_idx<day_num:
				line=line.replace('eeeeeeeeee',str('{0:.4E}'.format(t_array[t_idx])))
				t_idx+=1
			else:
				line=line.replace('eeeeeeeeee','')
						
		if 'ffffffffff' in line:
			if t_idx<day_num:
				line=line.replace('ffffffffff',str('{0:.4E}'.format(t_array[t_idx])))
				t_idx+=1
			else:
				line=line.replace('ffffffffff','')
			
		if 'gggggggggg' in line:
			if t_idx<day_num:
				line=line.replace('gggggggggg',str('{0:.4E}'.format(t_array[t_idx])))
				t_idx+=1
			else:
				line=line.replace('gggggggggg','')
			
		if 'hhhhhhhhhh' in line:
			if t_idx<day_num:
				line=line.replace('hhhhhhhhhh',str('{0:.4E}'.format(t_array[t_idx])))
				t_idx+=1
			else:
				line=line.replace('hhhhhhhhhh','')
			
			
		f_new.write(line)
	f.close()
	f_new.close()	

run_wang='run_wang1.bat'
f=open(run_wang,'w')
for i in range(trail_num):
	text='cd .\\tough_{}\n	start EOS1_OV.exe \ncd..\n\n'.format(i)
	f.write(text)
f.close()
subprocess.call(run_wang)
time.sleep(100)
print("22222")
	#shutil.rmtree(target_dir)



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


















