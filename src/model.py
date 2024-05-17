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
			self.ffns = nn.ParameterList([nn.Linear(model.config.d_model, 1) for _ in range(params.num_heads)])
			self.l2_loss = nn.MSELoss(reduction='none')
		else:
			model = AutoModelForSeq2SeqLM.from_pretrained(base_model, trust_remote_code=True)
		self.tokenizer = AutoTokenizer.from_pretrained(base_model)
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
			# breakpoint()
			diffs = [round(diff, 2) for diff in diffs]
		else:
			outputs = self.model.generate(*args, **kwargs, do_sample = True, top_k = 5, max_new_tokens = self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
			text_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)
			# print(text_outputs)
			# breakpoint()
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
			reg_embs = []
			reg_embs.append(output.last_hidden_state[:, 0])
			if labels is None and self.params.num_heads == 2:
				preds = self.ffns[0](reg_embs[0])
				pred_tokens = self.tokenizer([str(round(pred.item(), 2)) + ', ' for pred in preds], padding = True, add_special_tokens = False, return_tensors = 'pt').input_ids
				batch['decoder_input_ids'] = torch.cat((batch['decoder_input_ids'], pred_tokens.to(torch.device(self.params.device))), dim = 1)
				batch['decoder_attention_mask'] = torch.where(batch['decoder_input_ids'] == self.tokenizer.pad_token_id, 0, 1)
				output = self.model(**batch)
			if self.params.num_heads == 2:
				attention_mask = batch['decoder_attention_mask']
				last_idx = (attention_mask.sum(dim = 1) - 1)
				reg_embs.append(output.last_hidden_state[torch.arange(batch['input_ids'].shape[0]), last_idx])
			if labels is None:
				loss = None
				if self.params.num_heads == 2:
					preds = [preds, self.ffns[1](reg_embs[1])]
					# print(preds)
					# breakpoint()
				else:
					preds = [self.ffns[0](reg_embs[0])]
			else:
				# if self.detach_head:
				# 	preds = [head(reg_emb.detach()) for head, reg_emb in zip(self.ffns, reg_embs)]
				# else:
				# 	preds = [head(reg_emb) for head, reg_emb in zip(self.ffns, reg_embs)]
				preds = [head(reg_emb) for head, reg_emb in zip(self.ffns, reg_embs)]
				loss = 0
				for i in range(len(preds)):
					loss_batch = self.l2_loss(preds[i].squeeze(-1), labels[:, i].squeeze(-1))
					loss_mask = labels[:, i].squeeze(-1) != -100
					if torch.any(loss_mask):
						loss += torch.mean(loss_batch[loss_mask])
			return (loss, torch.cat(preds, dim = -1).sum(dim = 1))
		else:
			return output
		# elif self.params.loss != 'l2' and -100 in batch['labels']:
		# 	# https://github.com/huggingface/transformers/blob/1872bde7fc6a5d6796bd742bc2dc38eaf8069c5d/src/transformers/models/t5/modeling_t5.py#L1765
		# 	last_hidden_state = output.decoder_hidden_states[-1]
		# 	if self.model.config.tie_word_embeddings:
		# 		last_hidden_state = last_hidden_state * (self.model.config.d_model**-0.5)
		# 	lm_logits = self.model.lm_head(last_hidden_state)
		# 	loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), batch['labels'].view(-1))
		# 	breakpoint()
		# 	return (loss, )
		# else:
		# 	return output