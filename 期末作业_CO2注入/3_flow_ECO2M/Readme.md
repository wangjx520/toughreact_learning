
# 注入CO2
将1_mesh中的mesh文件和2_P_T中的save文件拷贝到这个文件夹，将save最后两行删掉，并留至少一个空行，改名为INCON（大小写都行），然后将input文件中的INCON关键字删除，让程序以根目录下的INCON文件作为条件。
