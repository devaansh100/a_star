import torch
import numpy as np
import os
import random
import argparse
from algorithm import *
from data import *
from torch.utils.data import DataLoader
from model import ImprovedHeuristic, T5ImprovedHeuristic
from runner import Runner
from ddp import *

model_dict = {'phi2': 'microsoft/phi-2', 'codellama': 'codellama/CodeLlama-7b-hf', 'flan-t5': 'google/flan-t5-base', 'code-t5': 'Salesforce/codet5-small', 'scratch-t5': ('scratch', 'Salesforce/codet5p-220m')}
def init_seed(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

def main(params):
	os.environ['TOKENIZERS_PARALLELISM'] = "false"
	model = ImprovedHeuristic(params, model_dict[params.base_model], params.device) if 't5' not in params.base_model else T5ImprovedHeuristic(params, model_dict[params.base_model], params.device)
	if params.num_gpus > 1:
		init_distributed()
	else:
		torch.cuda.set_device(params.local_rank)
	data = read_data(params, model.tokenizer)
	train_dataset = HeuristicDataset(params, data['train']['tokenized'], data['train']['raw'], model.tokenizer) if 't5' not in params.base_model else T5HeuristicDataset(params, data['train']['tokenized'], data['train']['raw'], model.tokenizer)
	val_dataset = HeuristicDataset(params, data['val']['tokenized'], data['val']['raw'], model.tokenizer, val = True) if 't5' not in params.base_model else T5HeuristicDataset(params, data['val']['tokenized'], data['val']['raw'], model.tokenizer, val = True)
	if params.num_gpus > 1:
		local_rank = int(os.environ['LOCAL_RANK'])
		model = nn.parallel.DistributedDataParallel(model, device_ids = [local_rank], output_device = [local_rank], find_unused_parameters=True)
		train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
		val_sampler = DistributedSampler(dataset=val_dataset, shuffle=False)
		train_dl = DataLoader(train_dataset, batch_size = params.batch_size,
							num_workers = 4, pin_memory = True, collate_fn = train_dataset.collate_fn_train, sampler=train_sampler)
		val_dl = DataLoader(val_dataset, batch_size = 2*params.batch_size,
						num_workers = 4, pin_memory = True, collate_fn = val_dataset.collate_fn_test, sampler=val_sampler)
	else:
		train_dl = DataLoader(train_dataset, batch_size = params.batch_size, num_workers = 4, pin_memory = params.device == 'cuda', shuffle = True, collate_fn = train_dataset.collate_fn_train)
		val_dl = DataLoader(val_dataset, batch_size = params.batch_size, num_workers = 4, pin_memory = params.device == 'cuda', shuffle = False, collate_fn = train_dataset.collate_fn_test)
		if len(params.create_gb_data):
			train_dataset.val = True
			val_dl = DataLoader(train_dataset, batch_size = params.batch_size, num_workers = 4, pin_memory = params.device == 'cuda', shuffle = False, collate_fn = train_dataset.collate_fn_test)
			assert params.test, "Only use --create-gb-data in the test loop"
	runner = Runner(params, train_dl, val_dl, data['test'])
	runner.train(model)

if __name__ == '__main__':
	init_seed(43)
	parser = argparse.ArgumentParser()
	parser.add_argument('--job', default = 'fixed_data') # testing
	parser.add_argument('--data-dir', default = '../datasets')
	parser.add_argument('--model-dir', default = '../models')
	parser.add_argument('--domain', choices = ['maze', 'sokoban'], default = 'sokoban')
	parser.add_argument('--dataset', default = 'boxoban-small', choices = ['maze-grade', 'maze-eval', 'maze-large', 'maze-small', 'boxoban-astar', 'boxoban-grade', 'boxoban-astar-dec', 'boxoban-small', 'boxoban-rand-small', 'boxoban-astar-bs', 'boxoban-eval','boxoban-astar-opt', 'boxoban-astar-rand', 'boxoban-astar-large', 'maze-fixed', 'maze-fixed-2', 'boxoban-fixed'])
	# boxoban-length-gen - with 7k means puzzles requiring < 7k iters and without it means puzzles requires > 7k, less than 14k iterations
	parser.add_argument('--create-data', default = '0', nargs = '+', type = int, help = 'args should be the values for arguments of create_data functions in data.utils')
	parser.add_argument('--prompt-file', default = '../datasets/prompt.txt')
	parser.add_argument('--grad-step', default = 1, type = int)
	parser.add_argument('--base-model', default = 'code-t5')
	parser.add_argument('--test-after',  default = 3000, type = int)
	parser.add_argument('--bs', dest = 'batch_size', default = 64, type = int)
	parser.add_argument('--lr', dest = 'learning_rate', default = 1e-4, type = float)
	parser.add_argument('--lm', dest = 'load_model', default = '', type = str)
	parser.add_argument('--test', action = 'store_true')
	parser.add_argument('--rt', dest = 'refresh_training', action = 'store_true')
	parser.add_argument('--save-to-bs', dest='save_to_bootstrap', type = str, default = '')
	parser.add_argument('--create-gb-data', type = str, default = '')
	parser.add_argument('--gb', action = 'store_true')
	parser.add_argument('--test-ilr', nargs = '+', type = str, default=['test'])
	parser.add_argument('--num-epochs', default = 30, type = int)
	parser.add_argument('--device', choices = ['cuda', 'cpu'], default = 'cuda')
	parser.add_argument('--sampled-nodes', default = 4, type = int)
	parser.add_argument('--sample', choices = ['random', 'deception', 'rand_dec', 'optimal', 'optimal_easy', 'optimal_med', 'optimal_easy_med', 'optimal_hard', 'opt_dec', 'rand_opt', 'opt_dec_rand'], default = 'deception')
	parser.add_argument('--num-gpus', default = 1, type = int)
	parser.add_argument('--local-rank', default = 0, type = int)
	parser.add_argument('--target', default = '', choices = ['', 'dec'], type = str)
	parser.add_argument('--loss', default = 'ce', choices = ['ce', 'l2'], type = str)
	parser.add_argument('--train-files', nargs = '*', default = []) # alg_sokoban_2/alg_sokoban for sokoban
	parser.add_argument('--val-files', nargs = '*', default = [])
	parser.add_argument('--alg-files', nargs = '*', default = [])
	parser.add_argument('--sc', dest = 'self_consistency_seqs', default = 3)
	parser.add_argument('--bootstrap-data', default = '0', nargs = '+', type = int, help = 'args should be the values for arguments of create_data functions in data.utils')
	parser.add_argument('--num-heads', default = 1, type = int)
	parser.add_argument('--train-seqs', default = 15, type = int)
	parser.add_argument('--val-seqs', default = 5, type = int)

	params = parser.parse_args()
	if len(params.val_files) == 0:
		params.val_files = params.train_files
	creator_func = {'maze': create_maze_dataset, 'sokoban': create_sokoban_dataset}
	if params.create_data:
		creator_func[params.domain](params, *params.create_data)
		create_supervision(params)
		exit()
	
	# if params.bootstrap_data:
	# 	solver = get_improved_heuristic_solver(AStar_maze if params.domain == 'maze' else AStar_sokoban)
	# 	model = ImprovedHeuristic(model_dict[params.base_model], params.device) if 't5' not in params.base_model else T5ImprovedHeuristic(model_dict[params.base_model], params.device)
	# 	filename = f"{params.model_dir}/{params.domain}/{params.job}/{params.base_model}/model_best_test.pth"
	# 	checkpoint = torch.load(filename, map_location = torch.device(params.device))
	# 	model.load_state_dict(checkpoint['model'], strict = False)
	# 	model = model.to(params.device)
	# 	prompt = open(params.prompt_file).read()
	# 	class PickleableModelAStar(solver):
	# 		def __init__(self, *args, **kwargs):
	# 			super().__init__(*args, **kwargs)
	# 	solver = PickleableModelAStar
	# 	get_alg = lambda puzzle, **kwargs : solver(
	# 							puzzle, 
	# 							model = model, 
	# 							prompt = prompt, 
	# 							domain = params.domain, 
	# 							device = params.device, 
	# 							num_return_sequences = params.self_consistency_seqs,
	# 							**kwargs
	# 						)
	# 	creator_func[params.domain](params, *params.bootstrap_data, solver = get_alg)
	# 	create_supervision(params, solver = get_alg)
	# 	exit()
	
	main(params)