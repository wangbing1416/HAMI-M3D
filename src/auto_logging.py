"""
    automatically convert log files to an excel table
"""
import os
import pandas as pd

path = '../log'
files = os.listdir(path)
item = ''
dic = []
for file_name in files:
    id = file_name[-13:-4]
    fp = open(os.path.join(path, file_name))
    try:
        data = fp.readlines()
        dic.append([id, float(data[-16][20:26]), float(data[-16][37:43]),
                    float(data[-12][17:23]), float(data[-12][27:33]), float(data[-12][37:43]),
                    float(data[-11][17:23]), float(data[-11][27:33]), float(data[-11][37:43]),
                    float(data[-8][37:43])])
    except:
        print('error id: {}'.format(file_name))
dic = sorted(dic, key=(lambda x:x[0]))  # sort by id

writer = pd.ExcelWriter('read result.xlsx')
data_pd = pd.DataFrame(dic)
data_pd.to_excel(writer, 'sheet1', float_format='%.4f', header=False, index=False)
writer.save()
writer.close()

print("finish")
