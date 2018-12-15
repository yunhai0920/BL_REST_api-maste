# -*- coding:utf-8 -*-
"""
@author:HuangJie
@time:18-12-9 下午3:01

"""
import re
import pandas as pd
from tgrocery import Grocery
from tgrocery.classifier import *

extra_addrs_dir = 'addrs_libs/full_address1.csv'
extra_lib = pd.read_csv(extra_addrs_dir, encoding='utf-8')
provinces = extra_lib[extra_lib['level'] == 1].loc[:, 'Name']
cities = extra_lib[extra_lib['level'] == 2].loc[:, 'Name']
grocery = Grocery('NameIdAdd_NLP')
model_name = grocery.name
text_converter = None
tgm = GroceryTextModel(text_converter, model_name)
tgm.load(model_name)
grocery.model = tgm


class Found(Exception):
    pass


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


def preprocess_ocr(result):
    name_total = ''
    id_total = ''
    for key in result:
        string1 = result[key][1]
        if len(string1) <= 8:
            continue
        string2 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*{}[]+", "", string1)
        no_digit = len(list(filter(str.isdigit, string2.encode('gbk'))))
        no_alpha = len(list(filter(is_alphabet, string2)))
        if len(set('法定代表人') & set(string2)) >= 2 or len(
                set('经营范围') & set(string2)) >= 2 or '资本' in string2 or '类型' in string2 or len(
                set('年月日') & set(string2)) >= 2 or len(set('登记机关') & set(string2)) >= 2 or '电话' in string2:
            predict_result = 'others'
        elif len(set('经营场所') & set(string2)) >= 3 or '住所' in string2 or len(set('营业场所') & set(string2)) >= 3:
            predict_result = 'company-address'
        elif len(set('统一社会信用代码') & set(string2)) >= 2 or ((no_digit + no_alpha) / len(string2) > 0.5 and no_digit > 8):
            predict_result = 'company-id'
        elif '名称' in string2:
            predict_result = 'company-name'
        else:
            predict_result = grocery.predict(string2)
        if str(predict_result) == 'company-name':
            name_total += string1
            break
        elif str(predict_result) == 'company-id':
            id_total += string1
        else:
            continue
    id_total = re.sub(r'\W', '', id_total)
    name_total = stupid_revise(name_total)
    return id_total, name_total


def may_cut_messy(data):
    cutted_data = data
    for site in provinces:
        flag = re.search(site, data)
        if flag is not None:
            cutted_data = data[flag.span()[0]:]
            return cutted_data
    for site in cities:
        flag = re.search(site, data)
        if flag is not None:
            cutted_data = data[flag.span()[0]:]
            return cutted_data
    return cutted_data


def re_prep(orig_data):
    orig_data = may_cut_messy(orig_data)
    punc = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！\-\-()，。？、~@#￥%……&*（）]+"
    if len(orig_data) > 5 and '省' in orig_data[-2:]:
        orig_data = orig_data[:-2] if orig_data[-2] == '省' else orig_data[:-1]
    if orig_data[-1] in ['.', '-', '、', '.', '*']:
        orig_data = orig_data[:-1]
    data_ = re.sub('.*?名称|.*?称', '', orig_data)
    data_ = re.sub('厅.*', '厅', data_)
    data_ = re.sub('店.*', '店', data_)
    data_ = re.sub('公司.*', '公司', data_)
    data_ = re.sub('馆.*', '馆', data_)
    try:
        data = ''
        for city in cities:
            pattern = u'(.*?' + str(city) + ')'
            data_split_ = re.split(pattern, data_)
            if not data_split_[0]:
                data += data_split_[1]
                raise Found
        for province in provinces:
            data = ''
            pattern = u'(.*?' + str(province) + ')'
            data_split = re.split(pattern, data_)
            if not data_split[0]:
                data += data_split[1]
                for city in cities:
                    pattern = u'(.*?'+str(city)+')'
                    data_split_ = re.split(pattern, data_split[2])
                    if not data_split[0]:
                        data += data_split_[1]
                        raise Found
    except Found:
        pass
    tail = re.sub(punc, '', orig_data)[len(data):]
    data = re.sub(punc, '', data)
    return data, tail


def stupid_revise(orig_data):
    data, tail = re_prep(orig_data)
    final = data + tail
    final = final.replace(' ', '')
    return final
