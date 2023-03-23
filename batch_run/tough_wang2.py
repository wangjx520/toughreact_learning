# -*- coding: utf-8 -*-
"""
Created on 3/23 20:03:54 2022

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

	#%%

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
	print('flow.inp 修改完毕')
	


run_wang='run_wang.bat'
f=open(run_wang,'w')
for i in range(trail_num):
	text='cd .\\tough_{}\n	start EOS1_OV.exe \ncd..\n\n'.format(i)
	f.write(text)
f.close()

#%% 可以实现tough运行结束之后再进行下面的操作
p=subprocess.Popen(run_wang,bufsize=2048,shell=True,stdin=subprocess.PIPE,\
				   stdout=subprocess.PIPE,close_fds=True)
p.wait()
result_str=p.stdout.read()
#%%
#subprocess.call(run_wang)

print("22222")
	#shutil.rmtree(target_dir)














