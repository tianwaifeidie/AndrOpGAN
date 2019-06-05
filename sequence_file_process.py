#coding:utf-8
import numpy as np
import os
import math
'''
数据来源：经过压缩编码的opcode词典对应的opcode提取结果

数据结果：该程序用于新的特征提取，就是提取所有method的频度，然后对特征进行聚合,最后分别生成ben.npy和
mal.npy,共存放着44（opcode种类数）*31（多次聚合结果）=1364维度的特征，良性恶意各200个样本
'''

def get_filename(file_dir):
    for root,dirs,files in os.walk(file_dir):
        continue
    file_path = []
    for i in files:
        file_path.append(file_dir+i)
    return file_path


def matrix_conv(matrix,steps = 1):
    ret = []
    length = matrix.shape[0]
    stepsize = int(length/steps)
    for percent in np.arange(0, 1, 1 / float(steps)):
        idx = int(length * percent)
        add = np.sum(matrix[idx:idx + stepsize],axis=0)
        ret.append(add)
    # ret = np.array(ret)
    adds = []
    for i in range(int(math.sqrt(steps))):
        for j in range(int(steps/(2**(i+1)))):
            adds.append(np.sum(ret[j*(2**(i+1)):j*(2**(i+1))+(2**(i+1))] , axis = 0))
    ret = np.array(ret+adds)
    count = []
    for j in range(len(ret)):
        count.append([np.sum(ret[i])])
    count = np.array(count)
    count = np.concatenate((ret,count),axis = 1)
    final = []
    for i in range(int(ret.shape[0])):
        final.append(ret[i]/np.sum(ret[i]))

    func_size = []
    for j in range(len(ret)):
        function_size = int(np.sum(ret[j]))
        loglength = (int(min(8,max(1,math.log(function_size,8)))) - 1)/10
        func_size.append([loglength])
    func_size = np.array(func_size)

    final = np.array(final)
    final = np.append(final,func_size,axis = 1)

    final = np.reshape(final, int(final.shape[0] * final.shape[1]))
    final = np.expand_dims(final,axis=0)
    final = np.concatenate((count,final),axis = 0)
    return final

def main():

    dict = open('target_dict.txt')
    lines = dict.readlines()
    dic = []
    for line in lines:
        dic.append(line.split('\n')[0].split(' ')[1])

    #词典最后一个应该是最后一个计数的opcode，它的数值+1就是opcode的种类数，因为是从00开始的
    # opcode_number = int(dic[-1],16)+1

    opcode_number = int(max(dic),16)+1
    file_list = get_filename('sequence_file/')

    for file_name in file_list:
        try:
            print(file_name)
            file = open(file_name)
            rows = file.readlines()
            feature = []

            for row in rows:
                idx = int(len(row)/2)
                singal = np.zeros(opcode_number)
                for i in range(idx):
                    loc = int(row[2*i:2*(i+1)],16)
                    singal[loc] = singal[loc] +1
                feature.append(singal)
            feature = np.array(feature)
            # result = matrix_conv(feature)
            # all_feature.append(result)
            all_feature = matrix_conv(feature)
            file_name = file_name.split('/')[1].split('.')[0]
            np.save('npy/'+file_name+'.npy', all_feature)
            # all_feature = np.concatenate((all_feature,matrix_conv(feature)),axis=0)
        except:
            print("-------error!--------")
            print(file_name)
            print("-------error!--------")
            continue
    # all_feature = np.array(all_feature)


if __name__ == '__main__':
    main()