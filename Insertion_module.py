#coding:utf-8

import numpy as np
import os
from scipy.linalg import solve
from random import choice


def get_keys(d,values):
    return [k for k,v in d.items() if v == values]

def get_filename(file_dir):
    for root,dirs,file in os.walk(file_dir):
        continue
    file_path = []
    for i in file:
        file_path.append(file_dir+i)
    return file_path

def generate_code(change_vec,dict):
    #----load insert function----
    f = open('single_method.txt')
    lines = f.readlines()
    total = ''
    for line in lines:
        total = total + line
    total = total.split('##')

    #----load insert smali codes ----
    insert_list = []
    codes = open('Dalvik_insert_code.txt')
    lines = codes.readlines()
    for line in lines:
        insert_list.append(line)
    insert_list[-1] = insert_list[-1]+'\n'

    #----prepare the number and kinds of opcode ----
    goal = {}
    for i in range(len(change_vec)):
        num = change_vec[i]
        opcode  = choice(get_keys(dict,i))
        goal[opcode] = num

    #----generate the insert----
    insert_total = ''
    for i in range(len(goal.keys())):
        for j in range(goal[list(goal.keys())[i]]):
            insert = total[i]
            insert_total = insert_total + insert
            #TODO: Remember remove this break
            # break

    return insert_total

def insert_smali(decode_dir_path,change_vec,dict):
    decode_dir_path  = decode_dir_path + '/'
    #random select a smali file to insert
    smalis = []
    for root, dirs, fnames in os.walk(decode_dir_path):
        for fname in fnames:
            full_path = os.path.join(root, fname)
            if full_path.split('.')[-1] == 'smali':
                smalis.append(full_path)
    loc = np.random.randint(0,len(smalis)-1,1)
    insert_file = smalis[int(loc)]
    smali_file = open(insert_file,'a+')
    code = generate_code(change_vec,dict)
    smali_file.write(code)
    smali_file.close()
    print('insert finished!')
    print('target:{}'.format(insert_file))
    return 0


def insert(change_vec,decode_middle_path):
    dalvik_opcodes = {}

    # Reading Davlik opcodes into a dictionary
    with open('target_dict.txt') as fop:
        for linee in fop:
            (key, val) = linee.split()
            dalvik_opcodes[key] = int(val,16)
    # apks = get_filename(original_dir)
    # for i in range(len(apks)):
    #     apk_name = apks[i].split('/')[-1]
    #     out_file_location = decode_dir+'/'+ apks[i].split('/')[-1] + ".smali"
    out_file_location = decode_middle_path
    insert_smali(out_file_location,change_vec[0],dalvik_opcodes)
    apk_name = decode_middle_path.split('/')[-1].split('.')[0]
    #----build with apktool----
    cmd = 'apktool b '+out_file_location+' -o '+apk_name+'_new.apk'
    os.system(cmd)
    print('build finished!')
    return 0


def judge(vec):
    '''
    用于判断输入的一维矩阵中的元素是否都大于0
    本文件中用于筛选符合条件的中间向量
    '''
    for i in range(len(vec)):
        if vec[i] < 0:
            return False
    return True

def get_filename(file_dir):
    for root,dirs,files in os.walk(file_dir):
        continue
    file_path = []
    for i in files:
        file_path.append(file_dir+i)
    return file_path

def get_opcode_result(ori_vec,mal_lib):
    '''
    use matrix transform to get exchange vector
    '''
    change = []
    for i in range(len(mal_lib)):
        try:
            tar = mal_lib[i]
            a = np.ones(44)
            b = np.reshape(tar[:44], [44, 1]) * a
            j = 0
            for i in range(len(b)):
                b[i, j] = b[i, j] - 1
                j = j + 1
            c = ori_vec[0, :44] - tar[:44] * np.sum(ori_vec[0, :44])
            result = solve(b, c)
            if judge(result):
                change.append(result)
        except:
            continue
    change = np.array(change)
    value = []
    for i in range(len(change)):
        middle_vec = np.sum(change[i])
        value.append(middle_vec)
    locate_min = value.index(min(value))
    return change[locate_min]



def main():
    npy_list = get_filename('npy/')
    mal_vec_lib = np.load('result2_15000.npy')
    original_dir = 'orignal_apk/'
    decode_dir = 'middle_file/'
    for file in npy_list:
        apk_name = file.split('/')[1].split('.')[0]
        original_vec = np.load(file)
        final = get_opcode_result(original_vec, mal_vec_lib)
        final = final.astype(np.int)
        final = np.expand_dims(final, axis=0)
        print(final)
        decode_middle_path = decode_dir+apk_name+'.smali'
        insert(final,decode_middle_path)

if __name__ == '__main__':
    main()
