import os
from tqdm import tqdm
import json

TS_ROOT_PATH = '../Corpus/TS_back/'
ICODE_BOOK_PATH = f'{TS_ROOT_PATH}/ICodebook.txt'

def is_us_exchange(icode_book_lines):
    for l in icode_book_lines:
        if 'Exchange: ' in l:
            l = l.strip()
            exchange = l.split(' ')[-1]
            if exchange in ['NYQ', 'PCX', 'NYM', 'NYS', 'CBT', 'PNK', 'DJI', 'NYB', 'CME', 'NMS']:
                return True
    return False

if __name__ == '__main__':
    stock_icode_list = []
    stock_icode_book_list = []
    exchange_set = set()
    with open(ICODE_BOOK_PATH, 'r', encoding='utf-8') as f:
        icode_book_total = f.read()
    for icode_book in icode_book_total.split('\n\n'):
        icode_book_lines = icode_book.split('\n')
        icode = icode_book_lines[0].split(' ')[-1]
        if 'Quotetype: EQUITY' in icode_book and is_us_exchange(icode_book_lines):
            stock_icode_list.append(icode)
            stock_icode_book_list.append(icode_book)
            stock_icode_book_list.append('\n\n')
    with open(f'{TS_ROOT_PATH}/metadata.json', 'r', encoding='utf-8') as file:
        metadata = json.load(file)
    filtered_dict = {k: metadata[k] for k in stock_icode_list if k in metadata}
    with open('./TS/metadata.json', 'w', encoding='utf-8') as file:
        json.dump(filtered_dict, file)
    open('./TS/ICodebook.txt', 'w').writelines(
        stock_icode_book_list
    )

    for ts_df in tqdm(os.listdir(f'{TS_ROOT_PATH}/ICode_dfs')):
        for icode in stock_icode_list:
            codes = icode.split(':')
            code = f'{codes[0]}2{codes[1]}'
            if code in ts_df:
                os.system(f'cp {TS_ROOT_PATH}/ICode_dfs/{ts_df} ./TS/ICode_dfs/')

    stock_icode_list = [f'{i}\n' for i in stock_icode_list]
    open('./TS/icode_list.txt', 'w').writelines(
        stock_icode_list
    )