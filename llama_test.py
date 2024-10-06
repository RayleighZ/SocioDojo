import transformers

from transformers import LlamaForCausalLM, AutoTokenizer
from accelerate import PartialState

pt = '''
Your role in the team is the analyst who watches and analyzes the latest information like news, articles, reports, and so on:
1. You will work with an actuator who is responsible for managing the hyperportfolio of the team and making buy or sell decisions.
2. Your task is to give a high-quality analysis for the actuator so that the actuator can make good decisions that optimize the hyperportfolio.
3. You need to find any indicator of potential movement in social, political, economic, or financial trends, from the given news.
4. You can give general suggestions, like "Apple stock price will go up", "It is time to sell Apple stock"
5. You can also give more precise buy or sell suggestions if you have confidence, "I think we should spend 10,000 on buying the GDP time series",
6. If you cannot see any opportunity, you should also indicate that "I cannot see any indicators", or "I think we should wait for now".
'''

if __name__ == '__main__':
    model_dir = '/home/tione/notebook/rayleighz_prj/Llama-3.1-8B-Instruct'
    model = LlamaForCausalLM.from_pretrained(model_dir, load_in_4bit=False, ignore_mismatched_sizes=True)
    distributed_state = PartialState()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, load_in_4bit=False, ignore_mismatched_sizes=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map='auto',
    )
    messages = [
        {'role': 'system', 'content': pt},
        {'role': 'user', 'content': 'Microsoft release Surface pro 10, but the cpu X1-E inside int seems weak, please generate a report for the impact of this news'}
    ]
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