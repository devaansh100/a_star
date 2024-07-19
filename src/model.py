import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, T5EncoderModel, AutoModel
from peft import LoraConfig, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
import torch
from transformers import BitsAndBytesConfig
from algorithm.utils import extract_differences
from multipledispatch import dispatch

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
	def __init__(self, params, base_model, device):
		super(ImprovedHeuristic, self).__init__()
		if params.loss == 'l2':
			raise NotImplementedError("Make appropriate changes in dataset and model before running decoder-only models with l2-loss")
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
	def __init__(self, params, base_model, device):
		super(T5ImprovedHeuristic, self).__init__()
		if params.loss == 'l2':
			model = AutoModel.from_pretrained(base_model)
			self.ffn = nn.Linear(model.config.d_model, 1)
			self.l2_loss = nn.MSELoss()
		else:
			model = AutoModelForSeq2SeqLM.from_pretrained(base_model, trust_remote_code=True)
		self.tokenizer = AutoTokenizer.from_pretrained(base_model)
		if self.tokenizer.bos_token_id is None:
			self.tokenizer.bos_token_id = self.tokenizer.pad_token_id
		self.inference_tokenizer = self.tokenizer
		self.model = model
		self.params = params
		self.detach_head = 'ss' in params.suffix
		self.max_new_tokens = 5
		
		print_trainable_parameters(self.model)

	def get_difference(self, prompts):
		difference = []
		if len(prompts) > 0:
			model_inputs = self.inference_tokenizer(prompts, return_tensors='pt', padding = True)
			model_inputs = {k: v.to(torch.device(self.model.device)) for k, v in model_inputs.items()}
			if self.params.loss == 'l2':
				difference = self.generate(**model_inputs)
			else:
				difference = self.generate(**model_inputs, num_return_sequences = self.params.self_consistency_seqs)
		return difference

	def generate(self, *args, **kwargs):
		if self.params.loss == 'l2':
			assert len(args) == 0
			kwargs['decoder_input_ids'] = (torch.ones(len(kwargs['input_ids']), 1) * self.tokenizer.bos_token_id).long().to(torch.device(self.params.device))
			diffs = self(kwargs)[1].tolist() # forward()
			diffs = [round(diff, 2) for diff in diffs]
		else:
			outputs = self.model.generate(*args, **kwargs, do_sample = True, top_k = 5, max_new_tokens = self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
			text_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)
			diffs = extract_differences(text_outputs)
			if 'num_return_sequences' in kwargs:
				diffs = [diffs[i : i + kwargs['num_return_sequences']] for i in range(0, len(diffs), kwargs['num_return_sequences'])]
				diffs = [max(set(x), key=x.count) for x in diffs]
		return diffs

	def forward(self, batch):
		if self.params.loss == 'l2':
			labels = batch.pop('labels') if 'labels' in batch else None
		output = self.model(**batch, output_hidden_states=True)
		if self.params.loss == 'l2':
			preds = self.ffn(output.last_hidden_state[:, 0])
			loss = self.l2_loss(preds.squeeze(-1), labels.squeeze(-1)) if labels is not None else None
			return (loss, preds.squeeze(-1))
		else:
			return output