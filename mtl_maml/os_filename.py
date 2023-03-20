import os
import numpy as np

def get_filename(dir,train,test,perm,flag):

    cur_paths = os.listdir(dir)
    list1=[]
    for cur_path in cur_paths:
        list1.append(cur_path)
    list1 =[list1[i] for i in perm]

    if flag=='train':
        paths=list1[:train]
    elif flag=='test':
        paths=list1[train:train+test]

    return paths

if __name__=='__main__':
    # dir1 = '/hdd/zhanghonghu/D'
    dir1 = '../data/tian_x4/fine'
    dir2s = ['HR', 'LR']
    dir3s = ['GFS025', 'GFS050', 'era5_2020', 'era5_2000']
    dir4s = ['u10', 'v10', 'd2m', 'sp', 't2m', 'tp']
    dir='{}/{}/{}/{}'.format(dir1,dir2s[0],dir3s[0],dir4s[0])
    train=12
    test=8
    flag='train'
    perm = np.random.permutation(train+test)
    paths=get_filename(dir,train,test,perm=perm,flag=flag)

    print(paths)