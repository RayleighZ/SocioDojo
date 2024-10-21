import pdb

import matplotlib.pyplot as plt
import json
import os
import polars as pl
from datetime import datetime

ACCOUNT_PATH = '/home/tione/notebook/rayleighz_prj/SocioDojo/Ckpts/llama3.1_8B_float_0.5day_USFin_verVI/Account'

if __name__ == '__main__':
    json_list = os.listdir(ACCOUNT_PATH)
    json_count = len(json_list)
    account_value_list = []
    account_list = []
    date2value = {}
    date2assets_ratio = {}
    for i in range(int(json_count / 2)):
        with open(f'{ACCOUNT_PATH}/default-{i * 2}.json', 'r') as f:
            account = json.load(f)
        account_list.append(account)
        account_value_list.append(account['total value'])
        date_str = account['protfolios']['default']['time'].split(' ')[0]
        date2value[date_str] = account['total value']
        total_value = account['total value']
        date2assets_ratio[date_str] = account['assets ratio']
        print(f'{date_str}: {total_value}')
    start_date = account_list[0]['protfolios']['default']['time']
    end_date = account_list[-1]['protfolios']['default']['time']
    final_lines = []
    # for i in open('./nasdq_utc.csv').readlines():
    #     if 'Date' in i:
    #         final_lines.append(i)
    #         continue
    #     date = i.split(',')[0]
    #     date_strip = date.split(' ')[0]
    #     residual = i.split(':00,')[-1]
    #     final_lines.append(f'{date_strip},{residual}')
    # open('./nasdq100_utc.csv', 'w').writelines(final_lines)
    # exit()
    df = pl.read_csv('./nasdq100_utc.csv')
    # nasdq = df.select(['Date', 'Close']).to_dict()
    # df = df.with_columns(
    #     pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d %H:%M:%S%z')
    # )
    # df = df.with_columns([
    #     pl.col("Date").dt.convert_time_zone('')
    # ])
    # lb = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S%z')
    # ub = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S%z')
    df = df.filter(
        pl.col('Date').is_in(date2value.keys())
    )
    nasdq_date = df.select('Date').to_numpy().flatten()
    nasdq_close = df.select('Close').to_numpy().flatten()
    nasdq_open = df.select('Open').to_numpy().flatten()
    triA_list = []
    assets_ratio = []
    for date in nasdq_date:
        triA_list.append(date2value[date])
        assets_ratio.append(date2assets_ratio[date])
    # for index in range(len(nasdq_open)):
    #     nasdq_list.append(nasdq_open[index])
    #     nasdq_list.append(nasdq_close[index])
    # base_money = account_value_list[0]
    # acc_return_value = [account_money / base_money for account_money in account_value_list]
    # length = min(len(account_value_list), len(nasdq_list))
    index = [i for i in range(len(nasdq_close))]
    base_triA = triA_list[0]
    acc_return_value = [triA / base_triA for triA in triA_list]
    # nasdq_list = [nasdq_list[i] for i in index]
    base_benchmark = nasdq_close[0]
    acc_nasdq_value = [nasdq_value / base_benchmark for nasdq_value in nasdq_close ]
    plt.subplots(figsize=(8, 4))
    plt.plot(index, acc_return_value,  label='Tri A')
    plt.plot(index, acc_nasdq_value, label='Nasdaq 100')
    # plt.plot(index, assets_ratio, label='assets ratio')
    plt.legend()
    for account in account_list:
        print(account['protfolios']['default']['time'])
        print(account['protfolios']['default']['assets'].keys())
        print(account['protfolios']['default']['assets'])
    # print(account_value_list)
    # print(acc_return_value)
    # print(acc_return_value[26])
    # print(acc_return_value[27])
    # print(acc_return_value[28])
    plt.savefig('./acc_return.png', dpi=300)
    plt.show()
