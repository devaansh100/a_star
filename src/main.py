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
	MODEL, DATASET = (T5ImprovedHeuristic, T5HeuristicDataset) if 't5' in params.base_model else (ImprovedHeuristic, HeuristicDataset)
	model = MODEL(params, model_dict[params.base_model], params.device)
	data = read_data(params, model.tokenizer)
	train_dataset = DATASET(params, data['train']['tokenized'], data['train']['raw'], model.tokenizer)
	val_dataset = DATASET(params, data['val']['tokenized'], data['val']['raw'], model.tokenizer, val = True)
	train_dl = DataLoader(train_dataset, batch_size = params.batch_size, num_workers = 4, pin_memory = params.device == 'cuda', shuffle = True, collate_fn = train_dataset.collate_fn_train)
	val_dl = DataLoader(val_dataset, batch_size = params.batch_size, num_workers = 4, pin_memory = params.device == 'cuda', shuffle = False, collate_fn = val_dataset.collate_fn_test)
	if len(params.create_gb_data):
		assert params.test or len(params.test_ilr) > 1, "Only use --create-gb-data in the test loops"
		if params.test:
			train_dataset.val = True
			val_dl = DataLoader(train_dataset, batch_size = params.batch_size, num_workers = 4, pin_memory = params.device == 'cuda', shuffle = False, collate_fn = train_dataset.collate_fn_test)
	runner = Runner(params, train_dl, val_dl, data['test'])
	runner.train(model)

if __name__ == '__main__':
	init_seed(43)
	parser = argparse.ArgumentParser()
	parser.add_argument('--job', required=True)
	parser.add_argument('--data-dir', default = '../datasets')
	parser.add_argument('--model-dir', default = '../models')
	parser.add_argument('--domain', choices = ['maze', 'sokoban'], default = 'sokoban')
	parser.add_argument('--dataset', required =  True, choices = ['maze-grade', 'maze-eval', 'maze-multipath-eval', 'maze-large', 'maze-multipath-small', 'maze-multipath-long', 'maze-small', 'maze-multipath', 'maze-tiny', 'boxoban-tiny', 'boxoban-grade', 'boxoban-long', 'maze-long', 'boxoban-eval', 'boxoban-large', 'boxoban-small'])
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
	parser.add_argument('--create-gb-data', type = str, default = '')
	parser.add_argument('--suffix', type = str, default = '')
	parser.add_argument('--mix-gb-data', action = 'store_true')
	parser.add_argument('--retokenize', action = 'store_false')
	parser.add_argument('--gb', action = 'store_true')
	parser.add_argument('--test-ilr', nargs = '+', type = str, default=['test'])
	parser.add_argument('--num-epochs', default = 40, type = int)
	parser.add_argument('--device', choices = ['cuda', 'cpu'], default = 'cuda')
	parser.add_argument('--sample', choices = ['optimal', 'optimal_dist', 'optimal_easy', 'optimal_med', 'optimal_hard', 'optimal_easy_med', 'optimal_easy_hard', 'optimal_med_hard'], default = 'optimal')
	parser.add_argument('--dist-factor', default = 2, type = float)
	parser.add_argument('--loss', default = 'ce', choices = ['ce', 'l2', 'ft'], type = str)
	parser.add_argument('--train-files', nargs = '*', default = []) # alg_sokoban_2/alg_sokoban for sokoban
	parser.add_argument('--val-files', nargs = '*', default = [])
	parser.add_argument('--alg-files', nargs = '*', default = [])
	parser.add_argument('--sc', dest = 'self_consistency_seqs', default = 3)
	parser.add_argument('--bootstrap-data', default = '0', nargs = '+', type = int, help = 'args should be the values for arguments of create_data functions in data.utils')
	parser.add_argument('--num-heads', default = 1, type = int)
	parser.add_argument('--train-seqs', default = 8, type = int)
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