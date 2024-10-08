import transformers

from transformers import AutoTokenizer, LlamaForCausalLM
import json


function_call_exp = '''
If a you choose to call a function, the function calling part of reply should in the following format:
<{start_tag}={function_name}>{parameters}{end_tag}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{"example_name": "example_value"}</function>
'''

DEBUG = True

class BaseLLMAget:
    def __init__(self, temperature: float, top_p: int):
        self.temperature = temperature
        self.top_p = top_p

    '''
    messgae: dict, contains these keys
        - role: system or user
        - content: prompt to role
    '''
    def inference(self, messages: list, device: str, date: str, function_call: list):
        raise NotImplementedError('LLMAgent pipeline not implemented')


def filter_function(response: str) -> list:
    function_list = []
    if '<function' not in response:
        return function_list
    function_fragment = response.split('<function')
    for f in function_fragment:
        # print(f'f is {f}')
        if not f.startswith('='):
            continue
        function = {}
        function_name = f.split('>')[0][1:]
        param_str = '{' + f.split('{')[-1].split('}')[0] + '}'
        try:
            param = json.loads(param_str) if param_str != '' else {}
        except json.JSONDecodeError:
            param = {}
        function = {
            'name': function_name,
            'parameter': param
        }
        function_list.append(function)
    return function_list

class Llama318BAgent(BaseLLMAget):
    def __init__(self, temperature: float, top_p: int, model_path: str):
        super().__init__(temperature, top_p)
        self.model = LlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, load_in_4bit=True)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def inference(self, messages: list, device: str, date: str, function_call: list = []) -> dict:
        input_message = []
        knowledge_cut = {'role': 'system', 'content': f'Cutting Knowledge Date: {date}, and you should only make decisions based on the provided information'}
        input_message.append(knowledge_cut)
        if len(function_call) != 0:
            function_call_message = {'role': 'system', 'content': f'you can choose to call folling functions if necessary: {function_call}, {function_call_exp}'}
            input_message.append(function_call_message)
        input_message.extend(messages)
        inputs = self.tokenizer.apply_chat_template(
            input_message,
            add_generation_prompt=True,
            return_tensors='pt'
        )
        sequences = self.model.generate(
            inputs,
            do_sample=True,
            top_p=self.top_p,
            eos_token_id=self.terminators,
            temperature=self.temperature,
            max_length=11451
        )
        response = sequences[0][inputs.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)

        result = {
            'content': response
        }
        function_list = filter_function(response)
        if len(function_list) == 1:
            result['function_call'] = function_list[0]
        elif len(function_list) != 0:
            result['function_call'] = function_list
        return result
