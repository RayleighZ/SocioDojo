import matplotlib.pyplot as plt
import polars as pl
from datetime import datetime


if __name__ == '__main__':
    stock_name = 'AMZN'
    start_date = '2021-11-30 00:00:00'
    end_date = '2022-01-31 00:00:00'
    stock_df_path = f'./TS/ICode_dfs/FIN2{stock_name}.csv'
    df = pl.read_csv(stock_df_path)
    print(df)
    df = df.with_columns(
        pl.col('datetime').str.strptime(pl.Date, format='%Y-%m-%d %H:%M:%S')
    )
    df = df.sort('datetime')
    print(df)
    lb = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    ub = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    print(lb)
    print(ub)
    df = df.filter(
        pl.col('datetime').is_between(lb, ub)
    )
    print(df)
    price_list = df.select('value').to_numpy().flatten()
    index = [i for i in range(len(price_list))]
    print(len(price_list))
    print()
    plt.plot(index[:36], price_list[:36], label=f'{stock_name} history')
    plt.plot(index[35:], price_list[35:], label=f'{stock_name} future')
    plt.legend()
    plt.show()
