# -*- coding:utf-8 -*-
"""
@author:HuangJie
@time:18-11-6 上午10:33

"""
from __future__ import division
import ocr_whole
import time
import sys
import numpy as np
from PIL import Image
from tgrocery import Grocery
import re
from tgrocery.classifier import *
from stupid_name_rev import stupid_revise
reload(sys)
sys.setdefaultencoding("utf-8")


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


def demo_flask(image_file):
    grocery = Grocery('NameIdAdd_NLP')
    model_name = grocery.name
    text_converter = None
    tgm = GroceryTextModel(text_converter, model_name)
    tgm.load(model_name)
    grocery.model = tgm

    t = time.time()
    result_dir = './result'
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr_whole.model(image)
    output_file = os.path.join(result_dir, image_file.split('/')[-1])
    Image.fromarray(image_framed).save(output_file)
    name_total = ''
    id_total = ''
    for key in result:
        string1 = result[key][1]
        if len(string1) <= 8:
            continue
        string2 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*{}[]+", "", string1)
        no_digit = len(list(filter(str.isdigit, string2.encode('gbk'))))
        no_alpha = len(list(filter(is_alphabet, string2)))
        if len(set('法定代表人') & set(string2)) >= 2 or len(set('经营范围') & set(string2)) >= 2 or '资本' in string2 or '类型' in string2 or len(set('年月日') & set(string2)) >= 2 or len(set('登记机关') & set(string2)) >= 2 or '电话' in string2:
            predict_result = 'others'
        elif len(set('经营场所') & set(string2)) >= 3 or '住所' in string2 or len(set('营业场所') & set(string2)) >= 3:
            predict_result = 'company-address'
        elif len(set('统一社会信用代码') & set(string2)) >= 2 or ((no_digit+no_alpha) / len(string2) > 0.5 and no_digit > 8):
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
    print("Mission complete, it took {:.3f}s".format(time.time() - t))
    print('\nRecongition Result:\n')
    print(id_total)
    print(name_total)
    return output_file, id_total, name_total


if __name__ == "__main__":
    demo_flask('./test_images/test/7.jpg')
