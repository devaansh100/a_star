import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from peft import LoraConfig, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
import torch
from transformers import BitsAndBytesConfig

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class ImprovedHeuristic(nn.Module):
    def __init__(self, base_model, device):
        super(ImprovedHeuristic, self).__init__()
        if device == 'cuda':
            bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype="auto", quantization_config=bnb_config, device_map='auto', attn_implementation="flash_attention_2")
            print('Loaded with optimizations')
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inference_tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side = 'left')
        self.inference_tokenizer.pad_token = self.tokenizer.eos_token
        config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=['q_proj', 'k_proj', 'v_proj', 'dense', 'embed_tokens'],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, config)
        print_trainable_parameters(self.model)

    def state_dict(self):
        state = self.model.state_dict()
        for name in list(state.keys()):
            if "lora" not in name:
                state.pop(name)
        return state
    
    def load_state_dict(self, state_dict, strict):
        self.model.load_state_dict(state_dict, strict = strict)

    def forward(self, batch):
        output = self.model(**batch)
        return output

class T5ImprovedHeuristic(nn.Module):
    def __init__(self, base_model, device):
        super(T5ImprovedHeuristic, self).__init__()
        if 'scratch' in base_model:
            base_model = base_model[-1]
            config = AutoConfig.from_pretrained(base_model)
            config._name_or_path = ''
            config.num_layers = 2
            config.num_decoder_layers = 2
            config.d_ff = 768
            config.d_model = 256
            model = AutoModelForSeq2SeqLM.from_config(config)
            init_model = AutoModelForSeq2SeqLM.from_pretrained(base_model, trust_remote_code=True)
            # model.lm_head.weight = nn.Parameter(init_model.lm_head.weight.clone())
            # model.shared.weight = nn.Parameter(init_model.shared.weight.clone())
            del init_model
            print_trainable_parameters(model)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.inference_tokenizer = self.tokenizer
        self.model = model
        
        print_trainable_parameters(self.model)

    def forward(self, batch):
        output = self.model(**batch)
        return output