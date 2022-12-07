# -*- coding: utf-8 -*-
"""
Created on 2022/12/6 22:29:12 

@author: wjx
@email:  2016007238@qq.com
"""

import pandas as  pd
import numpy as np
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')  # 官方提供的一个'fast'模式，但是好像没用


#%% 绘图全局设置

#import configur
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams.update({"font.size":22})
plt.rcParams['savefig.dpi'] = 150 #图片像素
config = {
			#"font.family": 'serif',
			"font.size": 20,
			"mathtext.fontset": 'stix',  # 一种很像Times New Roman
			#"font.serif": ['SimSun'],
			"font.serif": ['SimHei'],
		 } # 初始化绘图参数
plt.rcParams.update(config)
plt.rcParams['axes.grid'] = False
plt.axis('off')

cmap = matplotlib.cm.turbo                     # 设置全局色条
font_title={'fontsize': 22,'fontweight': 1.8}  # 设置全局使用的标题字号
font_colorbar=16                               # 设置全局使用的色标字号

# 坐标系标签使用西文字体
ticklabels_style = {
	"fontname": "Times New Roman",
	"fontsize": 12,  # 小五号，9磅
}
plt.xticks(**ticklabels_style)
plt.yticks(**ticklabels_style)

# fig.tight_layout()
# plt.tight_layout()

#%%
def write_pkl(write_data,pkl_path):
	pickle.dump(write_data,open(pkl_path,'wb'))

def read_pkl(pkl_file_path):
	my_data=pickle.load(open(pkl_file_path,'rb'))
	return my_data
def make_dir(dir_path):
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

#%%
tough2pkl_data='tough_data/tough2pkl_data/'              # 用于存储模型使用的直接数据
tough_structured_data='tough_data/tough_structured_data/'  # 用于保存从tough输出数据中提取的有用数据，并保存模拟结果动画
tough_png_path='tough_data/tough_png_path/' # 用于生成tough模拟结构绘制动画所用的png图片

make_dir(tough2pkl_data)
make_dir(tough_structured_data)
make_dir(tough_png_path)

structured_conc_csv=tough_structured_data+'structured_conc.csv'
structured_mine_csv=tough_structured_data+'structured_mine.csv'
well_series_xlsx=tough_structured_data+'well_series.xlsx'  # 将井对应的信息提取出来

#%%
#'''
print('------------- start conc-------------')
#'''
f2=open(structured_conc_csv,'w')  #结构化的数据

workspace='tough_'+str(0)+'/'
columns=open(workspace+'co2d_conc.dat').readlines()[8]  
columns=columns.split()  # 按分隔符进行字符串分割
columns=','.join(columns)  
columns=columns.split(',')[1:-1]

columns.insert(0,'t')  #在列表的第一个位置加入't'
columns=np.array(columns)
idx=(columns!='')
columns=columns[idx]    #去掉是空格的元素
columns[1]='X'
 
f2.write(','.join(str(column) for column in columns)+'\n')
	
for dir_idx in range(11):
	workspace='tough_'+str(dir_idx)+'/'
	f1=open(workspace+'co2d_conc.dat').readlines()[9:]
	for line in f1:       # line是个189的str
		write_flag=True   # 如果line为数据行，为True，line为输出的时刻行
		line=line.split() # 默认的分割符号是空格，将line(字符串)分割成列表
		if line[0]=='ZONE':  # ZONE T= "0.273791E-01 yr"  F=POINT
			t=line[2][1:]   # 获得下面数据对应的时刻
			write_flag=False 
		
		new_line=t+','+','.join(line)  # 将列表中每一项加入 ‘,’，形成175的字符串
		new_line=new_line.strip(',')   # 移除头尾指定的字符
		new_line=new_line+'\n'         
		
		if write_flag:
			f2.write(new_line)

f2.close()

data_sheet=pd.read_csv(structured_conc_csv,encoding='utf-8',delimiter=',',header=0)
conc_data=data_sheet[['t','X','Y','P(bar)','pH','t_uo2+2']].values  # txyPHU

first_time=conc_data[0,0]
element_num=0
for i in range(conc_data.shape[0]):
	if conc_data[i,0]==first_time:
		element_num+=1
	else:
		print(f'共剖分了{element_num}个单元格')
		break

time_steps=int(conc_data.shape[0]/element_num)
print('共输出了{}个模拟时刻的数据'.format(time_steps))

conc_data_trans=conc_data.T  
conc_data=conc_data_trans.reshape(-1,time_steps,element_num)  #[5,110,2588]纵向不同时刻，横向不同
write_pkl(conc_data,tough2pkl_data+'txy_PHU.pkl')
#'''

#%%
#''' 处理文件表头
print('------------- start min-------------')
f2=open(structured_mine_csv,'w')
workspace='tough_'+str(0)+'/'
columns=open(workspace+'co2d_min.dat').readlines()[7]
columns=columns.split()
columns=','.join(columns)
columns=columns.split(',')[1:-1]

columns.insert(0,'t')  #在列表的第一个位置加入't'
columns=np.array(columns)
idx=(columns!='')
columns=columns[idx]    #去掉是空格的元素
columns[1]='X'
f2.write(','.join(str(column) for column in columns)+'\n')

#%%
for dir_idx in range(11):
	workspace='tough_'+str(dir_idx)+'/'
	f1=open(workspace+'co2d_min.dat').readlines()[8:] 

	for line in f1:       # line是个189的str
		write_flag=True   # 如果line为数据行，为True，line为输出的时刻行
		line=line.split() # 默认的分割符号是空格，将line(字符串)分割成列表
		if line[0]=='ZONE':  # ZONE T= "0.273791E-01 yr"  F=POINT
			t=line[2][1:]   # 获得下面数据对应的时刻
			write_flag=False 
		
		new_line=t+','+','.join(line)  # 将列表中每一项加入 ‘,’，形成175的字符串
		new_line=new_line.strip(',')   # 移除头尾指定的字符
		new_line=new_line+'\n'         
		
		if write_flag:
			f2.write(new_line)

f2.close()

data_sheet=pd.read_csv(structured_mine_csv,encoding='utf-8',delimiter=',',header=0) #=None)
mine_data=data_sheet[['t','X','Y','Permeabi.','uraninite']].values

mine_data_trans=mine_data.T
mine_data=mine_data_trans.reshape(-1,time_steps,element_num)  #[5,110,2588]纵向不同时刻，横向不同单元
write_pkl(mine_data,tough2pkl_data+'txy_kA.pkl')
#'''
#%% 对状态变量动态演化的动画绘制
#'''
import time
t0=time.time()
t =np.round(conc_data[0,:,1]*365)
x =conc_data[1,0,:]
y =conc_data[2,0,:]
P =conc_data[3,...]
pH=conc_data[4,...]
U =conc_data[5,...]
A =mine_data[4,...]

def var_levels(array):
	min_,max_=array.min(),array.max()
	levels=np.linspace(min_,max_,100)
	return levels
	
var_data_list  =[P,pH,U,A]
var_levels_list=[var_levels(P),var_levels(pH),var_levels(U),var_levels(A)]


for i in range(time_steps):

	import matplotlib.pyplot as plt
	fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(14,10))
	ax_list=list(np.array(axes).flatten())
	
	for var_idx in range(4):
		ax=ax_list[var_idx]
		ax.axis('off')
		ax.set_xticks([])
		ax.set_yticks([])
		pcm=ax.tricontourf(x,y,var_data_list[var_idx][i],\
					 extend='both',cmap=cmap,levels=var_levels_list[var_idx])
		cb=fig.colorbar(pcm,ax=ax)#,ticks=bar_ticks_P)
		cb.ax.tick_params(labelsize=font_colorbar)  # 设置色标刻度字体大小。
		cb.ax.locator_params(nbins=4)

	ax_list[0].set_title(r'$\mathrm{Tough\ P(bar)}$')
	ax_list[1].set_title(r'$\mathrm{Tough\ pH}$')
	ax_list[2].set_title(r'$\mathrm{Tough\ U(mol/L)}$')
	ax_list[3].set_title(r'$\mathrm{Tough\ A}$')

	fig.tight_layout()
	plt.tight_layout()
	print('第{%.4d}天的状态变量分布绘制完毕！'%(t[i]))

	plt.suptitle('第'+r'$\mathrm{%.4d}$'%(t[i])+'天分布',y=0.99)
	plt.savefig(tough_png_path+'fit_'+str(i+1))
	
	#plt.cla() #清除当前figure中的活动的axes,其他axes保持不变
	#plt.clf()  # 清除当前figure的所有axes，不关闭这个window，因此可以进行重复利用
	plt.close()    # 关闭window，如果没有指定，则当前fig
#'''

import cv2
fps=3
img=cv2.imread(tough_png_path+'fit_1.png')
size=(img.shape[1],img.shape[0])
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
videoWriter=cv2.VideoWriter('tough_data/tough_sim.mp4',fourcc,fps,size)
png_list=[]

for i in range(1200):
	png=tough_png_path+'fit_'+str(i+1)+'.png'
	#png_list.append(png)
	if os.path.exists(png):
		png_list.append(png)
		frame=cv2.imread(png)
		videoWriter.write(frame)

videoWriter.release()
print('视频合成完毕')
#'''

#%% ---------------- 将tough中的抽注水井对应的elements数据提取出来 -------------------
'''
well_info=pd.read_excel(source_DATA+'flux_K_well.xlsx',sheet_name='well_coord',header=0)
well_index=well_info[['实际','x','y']].values  # '实际'是tough定义的井号对应的真实井号，index翻译成指标
write_pkl(well_index,tough2pkl_data+'well_index.pkl')

idx_list=[]
for i in range(len(well_index)):
	dx_plus_dy=abs(well_index[i,1]-conc_data[1,0,:])+abs(well_index[i,2]-conc_data[2,0,:])
	idx=np.where(dx_plus_dy==dx_plus_dy.min())   # 获得井点所在的列索引 idx   [47]
	idx_list.append(idx)

def init_df():
	data=pd.DataFrame()
	data['time(yr)']=conc_data[0,:,0].flatten() 
	data['day']=np.round(conc_data[0,:,0].flatten()*365)
	return data

# conc数据
t_data    =init_df()
x_data    =init_df()
y_data    =init_df()
P_data    =init_df()  #初始化P_sheet
t_uo2_data=init_df()
pH_data   =init_df()
k_data    =init_df()
A_data    =init_df()

for i in range(len(idx_list)):
	idx=idx_list[i]
	well_name=well_index[i,0]
	t_data[well_name]    =conc_data[0,:,idx].flatten()
	x_data[well_name]    =conc_data[1,:,idx].flatten()
	y_data[well_name]    =conc_data[2,:,idx].flatten()
	P_data[well_name]    =conc_data[3,:,idx].flatten()
	pH_data[well_name]   =conc_data[4,:,idx].flatten()
	t_uo2_data[well_name]=conc_data[5,:,idx].flatten()
	k_data[well_name]    =mine_data[3,:,idx].flatten()
	A_data[well_name]    =mine_data[4,:,idx].flatten()

df=pd.DataFrame() #构造原始数据文件
df.to_excel(well_series_xlsx)#生成Excel,不然mode指定为'a'时会报错

writer=pd.ExcelWriter(well_series_xlsx,mode='a',engine='openpyxl')
t_data.to_excel(writer,sheet_name='t(yr)',index=False)
x_data.to_excel(writer,sheet_name='x(m)',index=False)
y_data.to_excel(writer,sheet_name='y(m)',index=False)
P_data.to_excel(writer,sheet_name='P(bar)',index=False)
pH_data.to_excel(writer,sheet_name='pH',index=False)
t_uo2_data.to_excel(writer,sheet_name='t_uo2',index=False)
k_data.to_excel(writer,sheet_name='k(m^2)',index=False)
A_data.to_excel(writer,sheet_name='A',index=False)

writer.save()
writer.close()

#'''
#%% ------------ 绘图(时间序列图) --------------
'''
P_sheet=pd.read_excel(well_series_xlsx,header=0,sheet_name='t_uo2',index_col='time(yr)')

P_data=P_sheet.values
well_names=P_sheet.columns

time=P_sheet.index
Z_wells_list=[]
Z_P_list=[]

for i in range(len(well_names)):
	well_name=well_names[i]
	if 'KZ' in  well_name:
		Z_wells_list.append(well_name)
		Z_P_list.append(P_sheet[well_name].values)


ticks_idx=np.linspace(0,len(time),5).astype(np.int32)
ticks_idx[-1]=ticks_idx[-1]-1  #最后一位
x_ticks=time[ticks_idx]
fig,axes=plt.subplots(ncols=4,nrows=3,figsize=(20,16))
ax_list=list(axes.flatten())

for i in range(len(ax_list)):
	well_name=Z_wells_list[i]
	ax=ax_list[i]
	ax.plot(time,Z_P_list[i])
	ax.set_title('${}$'.format(well_name))
	#ax.set_xticks(x_ticks)
	
	ax.set_xlabel('$time(year)$')
	ax.set_ylabel('$P(bar)$')
	
plt.tight_layout()
fig.tight_layout()
plt.savefig('Z_well_P_1')


fig,axes=plt.subplots(ncols=4,nrows=3,figsize=(20,16))
ax_list=list(axes.flatten())

for i in range(len(ax_list)):
	well_name=Z_wells_list[i+12]
	ax=ax_list[i]
	ax.plot(time,Z_P_list[i+12])
	ax.set_title('${}$'.format(well_name))
	
	ax.set_xlabel('$time(year)$')
	ax.set_ylabel('P(bar)')
	
plt.tight_layout()
fig.tight_layout()
plt.savefig('Z_well_P_2')

#'''


'''
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(14,10))
ax_list=list(np.array(axes).flatten())
	
for var_idx in range(4):
	ax=ax_list[var_idx]
	ax.axis('off')
	ax.set_xticks([])
	ax.set_yticks([])

[ax1,ax2,ax3,ax4]=ax_list
ax_list[0].set_title(r'$\mathrm{Tough\ P(bar)}$')
ax_list[1].set_title(r'$\mathrm{Tough\ pH}$')
ax_list[2].set_title(r'$\mathrm{Tough\ U(mol/L)}$')
ax_list[3].set_title(r'$\mathrm{Tough\ A}$')

array=var_data_list[0][0]
img1=ax1.tricontourf(x,y,array,\
				 extend='both',cmap=cmap,levels=var_levels_list[0])
img2=ax2.tricontourf([],[],[],\
				 extend='both',cmap=cmap,levels=var_levels_list[1])
img3=ax3.tricontourf([],[],[],\
				 extend='both',cmap=cmap,levels=var_levels_list[2])
img4=ax4.tricontourf([],[],[],\
				 extend='both',cmap=cmap,levels=var_levels_list[3])
	
for i in range(time_steps):
	
	pcm1=img1.set(x,y,array=var_data_list[0][i])
	pcm2=img1.set_data(x,y,var_data_list[1][i])
	pcm3=img1.set_data(x,y,var_data_list[2][i])
	pcm4=img1.set_data(x,y,var_data_list[3][i])
	
	cb1=fig.colorbar(pcm1,ax=ax1)#,ticks=bar_ticks_P)
	cb2=fig.colorbar(pcm2,ax=ax2)#,ticks=bar_ticks_P)
	cb3=fig.colorbar(pcm3,ax=ax3)#,ticks=bar_ticks_P)
	cb4=fig.colorbar(pcm4,ax=ax4)#,ticks=bar_ticks_P)
	
	cb1.ax.tick_params(labelsize=font_colorbar)  # 设置色标刻度字体大小。
	cb2.ax.tick_params(labelsize=font_colorbar)  # 设置色标刻度字体大小。
	cb3.ax.tick_params(labelsize=font_colorbar)  # 设置色标刻度字体大小。
	cb4.ax.tick_params(labelsize=font_colorbar)  # 设置色标刻度字体大小。
	cb1.ax.locator_params(nbins=4)
	
	fig.tight_layout()
	plt.tight_layout()

	plt.suptitle('第'+r'$\mathrm{%.4d}$'%(t[i])+'天分布',y=0.99)
	plt.savefig(tough_png_path+'fit_'+str(i+1))
	
	#plt.cla() #清除当前figure中的活动的axes,其他axes保持不变
	#plt.clf()  # 清除当前figure的所有axes，不关闭这个window，因此可以进行重复利用
	plt.close()    # 关闭window，如果没有指定，则当前fig

	
print('绘图共用了',time.time()-t0)
#'''






