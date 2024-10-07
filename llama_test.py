from langchain_core import messages
import transformers

from transformers import LlamaForCausalLM, AutoTokenizer
import Agent.prompts.analyst_instruct as PROMPT

pt = '''
Your role in the team is the analyst who watches and analyzes the latest information like news, articles, reports, and so on:
1. You will work with an actuator who is responsible for managing the hyperportfolio of the team and making buy or sell decisions.
2. Your task is to give a high-quality analysis for the actuator so that the actuator can make good decisions that optimize the hyperportfolio.
3. You need to find any indicator of potential movement in social, political, economic, or financial trends, from the given news.
4. You can give general suggestions, like "Apple stock price will go up", "It is time to sell Apple stock"
5. You can also give more precise buy or sell suggestions if you have confidence, "I think we should spend 10,000 on buying the GDP time series",
6. If you cannot see any opportunity, you should also indicate that "I cannot see any indicators", or "I think we should wait for now".
'''

function_call_exp = '''
If a you choose to call a function ONLY reply in the following format:
<{start_tag}={function_name}>{parameters}{end_tag}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{"example_name": "example_value"}</function>
'''

msg = [{'role': 'system', 'content': 'You are a helpful assistant who is working in a team playing a game of "hyperportfolio" that aims to analyze and predict the potential movement in real-life social, political, economic, or financial trends from real-time news and other information sources. \n\nThe rule of the game is as follows:\n1. Every team has an initial account of 1 million cash, your team can use this cash to buy or sell assets to build a hyperportfolio, the target of the game is to maximize the asset, which is the summation of the remaining cash and the value of an asset you own\n2. A hyperportfolio is composed of a set of assets that corresponds to the time series in different domains from real-life covers financial market close prices, economic time series, Google search trends, and political and public opinion poll trackers. \n3. The game begins on 2021-10-01 and ends on 2023-08-01, after beginning, the time will move forward, and you will consistently receive real-life news about what is happening in the world, newly released opinions from the internet or social network, or reports from research institutes, financial institutions, and so on.\n4. Your team may choose to "buy" or "sell" an asset during the game. Each asset corresponds to a time series, the buy price will be the latest value of the time series at the current time. \n5. You need to pay a commission when you buy or sell an asset, the amount is about 1% of the buy or sell value. \n6. The value of an asset you own will update over time, calculated as (current price/buy price)*(investment amount).\n7. For example, you may read news about the Apple company performing well for this season, Based on your analysis, you may think it is a good indicator that Apple stock price will increase and decide to invest 10,000 cash on the Apple stock time series.\n8. Each time series is marked by a ICode. The ICode has such format "[DOMAIN]:[CODE]". For example, the apple company stock price is "FIN:AAPL", FIN is the domain, AAPL is the code. There are five domains, "FIN", "WEB", "FTE", "FRD", "YGV", interpretations for them:\n    a) FIN: The close price time series of a financial instrument including stocks, ETFs, index funds, REITs, futures, currencies, indices, or cryptocurrencies\n    b) WEB: The Google trend time series of a keyword, such as "Apple", "iPhone", "Bitcoin", etc. \n    c) FTE: Political poll tracking time series, such as the president\'s approval rating, the generic ballot, etc.\n    d) FRD: Economic time series, such as GDP, PPI, unemployment rate, etc.\n    e) YGV: Public opinion poll tracking time series, such as support for universal health care, how sustainability policies impact Americans\' purchase behavior, etc.\n9. You may receive or pay overnight interest or fees if you hold an asset overnight, computed as rate*size*current_price/360, size=amount/buy_price. The rate varies for different assets.\n\nYour role in the team is the analyst who watches and analyzes the latest information like news, articles, reports, and so on:\n1. You will work with an actuator who is responsible for managing the hyperportfolio of the team and making buy or sell decisions.\n2. Your task is to give a high-quality analysis for the actuator so that the actuator can make good decisions that optimize the hyperportfolio.\n3. You need to find any indicator of potential movement in social, political, economic, or financial trends, from the given news.\n4. You can give general suggestions, like "Apple stock price will go up", "It is time to sell Apple stock"\n5. You can also give more precise buy or sell suggestions if you have confidence, "I think we should spend 10,000 on buying the GDP time series", \n6. If you cannot see any opportunity, you should also indicate that "I cannot see any indicators", or "I think we should wait for now".\n'}, {'role': 'system', 'content': 'Current time is 2021-10-01 04:10:00+00:00.'}, {'role': 'system', 'content': '\nYou can choose to read the full content of news, article, etc. or not based on the given metadata:\n    a) You will be given the metadata of the news, article, etc. like the title, author, date, etc. \n    b) If you think the news, article, etc. is not very useful for optimizing the hyperportfolio of the team based on the given metadata, you can choose to not read it, and you should make a function call notread. \n    c) Otherwise, you should call read function to read the full content.\n    d) If you cannot decide whether to read or not, such as the metadata do not give you useful information, you can call read function to read the full content.\nHere is the metadata of the news, article, etc.:\n\ndatetime: 2021-10-01 00:10:00 ET\n\n\nDo you want to read the full content according to the metadata?\n'}]
if __name__ == '__main__':
    model_dir = '/work/zhangyu/dev/LLM-models/Llama-3.1-8B-Instruct/'
    model = LlamaForCausalLM.from_pretrained(model_dir, load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, load_in_4bit=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map='auto',
    )
    # messages = [
    #     {'role': 'system', 'content': pt},
    #     {'role': 'user', 'content': 'Microsoft release Surface pro 10, but the cpu X1-E inside int seems weak, please generate a report for the impact of this news'}
    # ]
    messages = []
    messages.append({'role': 'system', 'content': f'Cutting Knowledge Date: 2021-10-1, and you should only make decisions based on the provided information'})
    messages.append({'role': 'system', 'content': f'you can use folling functions: {PROMPT.read_functions}, {function_call_exp}'})
    messages.extend(msg)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt'
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    sequences = model.generate(
        inputs,
        do_sample=True,
        top_p=10,
        temperature=1,
        eos_token_id=terminators,
        max_length=2000
    )
    response = sequences[0][inputs.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))
