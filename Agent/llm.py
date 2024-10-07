import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer
import json


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
    def inference(self, message: list, device: str, date: str, function_call: dict):
        raise NotImplementedError('LLMAgent pipeline not implemented')


class Llama27bAgent(BaseLLMAget):
    def __init__(self, temperature: float, top_p: int, model_path: str):
        super().__init__(temperature, top_p)
        self.model = LlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, load_in_4bit=True)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def inference(self, message: list, device: str, date: str, function_call: dict = {}) -> dict:
        if DEBUG:
            return {'function_call': 'read'}
        input_message = []
        knowledge_cut = {'role': 'system', 'content': f'Cutting Knowledge Date: {date}, and you should only make decisions based on the provided information'}
        input_message.append(knowledge_cut)
        if len(function_call) != 0:
            function_call = {'role': 'system', 'content': f'you can choose to call folling functions if necessary: {function_call}, {function_call_exp}'}
            input_message.append(function_call)
        input_message.extend(message)
        inputs = self.tokenizer.apply_chat_template(
            input_message,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        sequences = self.model.generate(
            inputs,
            do_sample=True,
            top_k=self.top_p,
            eos_token_id=self.terminators,
            temperature=self.temperature,
            device=device
        )
        response = sequences[0][inputs.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)

        result = {
            'response': response
        }

        if '</function>' in response:
            function_info = response.split('<function=')[-1].split('</function>')[0]
            function_name = function_info.split('>')[0]
            paramater_str = function_info.split('>')[-1]
            paramater = json.loads(paramater_str) 
            result['function_call'] = {
                'name': function_name,
                'paramater': paramater
            }
        
        return result
