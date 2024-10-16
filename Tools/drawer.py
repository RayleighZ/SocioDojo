import pdb

import matplotlib.pyplot as plt
import json
import os
import polars as pl
from datetime import datetime

ACCOUNT_PATH = '/home/tione/notebook/rayleighz_prj/SocioDojo/Ckpts/llama3.1_8B_float_0.5day_USFin_verIV/Account'

if __name__ == '__main__':
    json_list = os.listdir(ACCOUNT_PATH)
    json_count = len(json_list)
    account_value_list = []
    account_list = []
    for i in range(json_count):
        with open(f'{ACCOUNT_PATH}/default-{i}.json', 'r') as f:
            account = json.load(f)
        account_list.append(account)
        account_value_list.append(account['total value'] - 980000)
    start_date = account_list[0]['protfolios']['default']['time']
    end_date = account_list[-1]['protfolios']['default']['time']
    df = pl.read_csv('./nasdq100.csv')
    df = df.with_columns(
        pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d %H:%M:%S%z')
    )
    lb = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S%z')
    ub = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S%z')
    df = df.filter(
        pl.col('Date').is_between(lb, ub)
    )
    nasdq_close = df.select('Close').to_numpy().flatten()
    nasdq_open = df.select('Open').to_numpy().flatten()
    nasdq_list = []
    for index in range(len(nasdq_open)):
        nasdq_list.append(nasdq_open[index])
        nasdq_list.append(nasdq_close[index])
    base_money = account_value_list[0]
    acc_return_value = [account_money / base_money for account_money in account_value_list]
    length = min(len(account_value_list), len(nasdq_list))
    index = [i for i in range(length)]
    acc_return_value = [acc_return_value[i] for i in index]
    nasdq_list = [nasdq_list[i] for i in index]
    base_benchmark = nasdq_list[0]
    acc_nasdq_value = [nasdq_value / base_benchmark for nasdq_value in nasdq_list ]
    plt.plot(index, acc_return_value)
    plt.plot(index, acc_nasdq_value)
    for account in account_list:
        print(account['protfolios']['default']['assets'].keys())
    plt.show()
