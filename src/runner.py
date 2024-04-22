import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor
from tqdm import tqdm
from algorithm import *
import os
from torch.cuda.amp import GradScaler
import random
from collections import Counter
from algorithm.utils import extract_differences
import pickle as pkl
from ddp import *
import time

class Runner():
    def __init__(self, params, train_dl, test_dl, test_puzzles):
        self.train_dl = train_dl
        self.val_dl = test_dl
        self.params = params
        self.test_puzzles = test_puzzles
        self.model_dir = params.model_dir + '/' + params.domain + '/' + params.job + '/' + params.base_model
        
        self.prompt = open(params.prompt_file).read()
        # legend = {'maze': "@ - player, # - wall, . - empty cell, X - goal", 'sokoban': "@ - player, # - wall, . - empty docks, ' ' - empty cell, $ - box, X - box on dock, O - player on dock"}
        # self.prompt = self.prompt.replace('{puzzle_legend}', legend[self.params.domain]).replace('{domain}', self.params.domain)

        self.best_test = float('inf')
        if not params.test:
            os.makedirs(self.model_dir, exist_ok = True)
    
    def save_model(self, model, epoch, filename):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_test': self.best_test
        }
        try:
            torch.save(checkpoint, self.model_dir + '/' + filename)
            print(f'Saving at {self.model_dir}/{filename}')
        except:
            print('Saving failed')

    def load_model(self, model, filename, load_opt = True):
        print(f'Loading from {self.model_dir}/{filename}')
        checkpoint = torch.load(self.model_dir + '/' + filename, map_location = torch.device(self.params.device))
        model.load_state_dict(checkpoint['model'], strict = False)
        # self.best_test = checkpoint['best_test']
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        return model, checkpoint['epoch']

    def fit_one_epoch(self, model, epoch):
        model.train()
        total_loss = 0
        p_bar = tqdm(self.train_dl, desc = f'Epoch {epoch}')
        for step, batch in enumerate(p_bar):
            raw_data = batch.pop('raw')
            batch = {k: v.to(torch.device(self.params.device)) for k, v in batch.items()}
            outputs = model(batch)
            loss = outputs.loss
            if torch.isnan(loss):
                if is_main_process():
                    print('NaN loss encountered. Training further not useful. Try testing model_best_test.pth')
                exit()
            if self.params.num_gpus > 1:
                loss_collated = [torch.zeros_like(loss).cuda() for _ in range(params.num_gpus)]
                dist.all_gather(loss_collated, loss)
                total_loss += sum(loss_collated).item()
            else:
                total_loss += loss.item()
            p_bar.set_postfix({'loss': loss.item()})
            loss.backward()
            if (step + 1) % self.params.grad_step == 0 or (step + 1) == len(self.train_dl):
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            if (step + 1) % self.params.test_after == 0:
                self.test(model, epoch)
                model.train()
        train_loss = round(total_loss / len(self.train_dl), 4)
        if is_main_process():
            print(f'Train Loss: {train_loss}')
    
    def prepare_model_for_astar(self, model, prompt): # NOTE: DO NOT USE KV_CACHE FOR BATCHED OUTPUTS
        if self.params.device == 'cuda':
            legend = {'maze': "@ - player, # - wall, . - empty cell, X - goal", 'sokoban': "@ - player, # - wall, . - empty docks, ' ' - empty cell, $ - box, X - box on dock, O - player on dock"}
            prompt = prompt.replace('{puzzle_legend}', legend[self.params.domain]).replace('{domain}', self.params.domain)
            source_idx = prompt.index('puzzle_str')
            prompt_inputs = prompt[:source_idx]
            kv_cache = {}
            for i in tqdm(range(1, 5), desc = 'Creating KV Cache'):
                model_inputs = model.tokenizer([prompt_inputs] * self.params.self_consistency_seqs * i, return_tensors='pt').input_ids.to(torch.device(self.params.device))
                kv_cache[i] = output = model({'input_ids': model_inputs}).past_key_values
            prompt = prompt[source_idx:]
        else:
            kv_cache = None
            prompt = self.prompt
        return model, kv_cache, prompt
        
    def get_ilr_metrics(self, ilr_p_list, swc_p_list, time_p, time_ref):
        ilr, swc = 0, 0
        ilr_improved, ilr_worsened, ilr_optimal = 0, 0, 0
        n_better, n_worse, n_optimal = 0, 0, 0

        itr, itr_optimal = 0, 0
        for i, (ilr_p, swc_p) in enumerate(zip(ilr_p_list, swc_p_list)):
            ilr += ilr_p
            swc += swc_p
            if ilr_p > 1:
                ilr_improved += ilr_p
                n_better += 1
            elif ilr_p < 1:
                ilr_worsened += ilr_p
                n_worse += 1
            if swc_p == 1:
                ilr_optimal += ilr_p
                n_optimal += 1
            
            itr_p = time_ref[i]/time_p[i]
            itr += itr_p
            if swc_p == 1:
                itr_optimal += itr_p
        
        ilr /= len(self.test_puzzles['raw'])
        ilr = round(ilr, 4)

        if n_better > 0:
            ilr_improved /= n_better
            ilr_improved = round(ilr_improved, 4)
        else:
            ilr_improved = -1

        if n_worse > 0:
            ilr_worsened /= n_worse
            ilr_worsened = round(ilr_worsened, 4)
        else:
            ilr_worsened = -1

        if n_optimal > 0:
            ilr_optimal /= n_optimal
            ilr_optimal = round(ilr_optimal, 4)

            itr_optimal /= n_optimal
            itr_optimal = round(itr_optimal, 6)
        else:
            ilr_optimal, itr_optimal = -1, -1

        itr /= len(self.test_puzzles['raw'])
        itr = round(itr, 6)

        swc /= len(self.test_puzzles['raw'])
        swc = round(swc, 4)

        n_better /= len(self.test_puzzles['raw'])
        n_better = round(100 * n_better, 2)

        n_worse /= len(self.test_puzzles['raw'])
        n_worse = round(100 * n_worse, 2)

        n_optimal /= len(self.test_puzzles['raw'])
        n_optimal = round(100 * n_optimal, 2)

        return ilr, swc, ilr_improved, ilr_worsened, ilr_optimal, n_better, n_worse, n_optimal, itr, itr_optimal

    def test_ilr(self, model):
        assert self.params.num_gpus == 1, 'Only 1 GPU for ILR right now'
        algs, ilr_p, swc_p = [], [], []
        time_p, time_ref = [], []
        solver = get_improved_heuristic_solver(AStar_maze if self.params.domain == 'maze' else AStar_sokoban)
        # solver = get_random_heuristic_solver(AStar_maze if self.params.domain == 'maze' else AStar_sokoban)
        with torch.no_grad():
            model.eval()
            # model, kv_cache, prompt = self.prepare_model_for_astar(model, self.prompt)
            p_bar = tqdm(zip(self.test_puzzles['raw'], self.test_puzzles['alg']),  total = len(self.test_puzzles['raw']))
            for i, (puzzle, ref_alg) in enumerate(p_bar):
                alg = solver(puzzle, model = model, prompt = self.prompt, domain = self.params.domain, device = self.params.device, num_return_sequences = self.params.self_consistency_seqs)
                # alg = solver(puzzle, domain = self.params.domain, strategy = 'uniform')
                start = time.time()
                alg.search()
                end = time.time() 
                time_p.append((end - start) * 1000)
                algs.append(alg)
                ilr_p.append(len(ref_alg.closed)/len(alg.closed))
                if self.params.dataset == 'boxoban-astar' or self.params.dataset == 'maze-generated' or self.params.dataset == 'boxoban-astar-large': # generated before bug was resolved
                    swc_p.append(len(ref_alg.optimal_plan[1:])/len(alg.optimal_plan))
                else:
                    swc_p.append(ref_alg.optimal_plan[-1].g/alg.optimal_plan[-1].g)
                p_bar.set_postfix({'swc': round(sum(swc_p)/(i + 1), 2), 'ilr': round(sum(ilr_p)/(i + 1), 4), 'ilr_p': round(ilr_p[-1], 2), 'swc_p': round(swc_p[-1], 2)})

        # time_ref = self.get_astar_runtimes()
        time_ref = [0] * len(time_p)
        ilr, swc, ilr_improved, ilr_worsened, ilr_optimal, n_better, n_worse, n_optimal, itr, itr_optimal = self.get_ilr_metrics(ilr_p, swc_p, time_p, time_ref)

        print(f'ILR-on-solved: {ilr}')
        print(f'ILR-on-optimal: {ilr_optimal}')
        print(f'ILR-on-improved: {ilr_improved}')
        print(f'Improved %: {n_better}')
        print(f'ILR-on-worsened: {ilr_worsened}')
        print(f'Worsened %: {n_worse}')
        print(f'SWC: {swc}')
        print(f'Optimal %: {n_optimal}')
        print(f'Same %: {round(100 - n_better - n_worse, 2)}')
        print(f'ITR-on-solved: {itr}')
        print(f'ITR-on-optimal: {itr_optimal}')

    
    def get_astar_runtimes(self):
        time_ref = []
        solver = AStar_maze if self.params.domain == 'maze' else AStar_sokoban
        for puzzle in tqdm(self.test_puzzles['raw']):
            alg = solver(puzzle)
            start = time.time()
            alg.search()
            end = time.time()
            time_ref.append((end - start) * 1000)
        return time_ref
    
    def get_metrics(self, preds, gts):
        overestimated, underestimated = [0, []], [0, []]
        for diff, gt_diff in zip(preds, gts):
            if diff > gt_diff:
                overestimated[0] += 1
                overestimated[1].append(diff - gt_diff)
            elif diff < gt_diff:
                underestimated[0] += 1
                underestimated[1].append(gt_diff - diff)
        overestimated_p, overestimated_a = round(100 * overestimated[0] / len(self.val_dl.dataset), 2), round(sum(overestimated[1])/len(overestimated[1]), 2) if len(overestimated[1]) > 0 else 0
        underestimated_p, underestimated_a = round(100 * underestimated[0] / len(self.val_dl.dataset), 2), round(sum(underestimated[1])/len(underestimated[1]), 2) if len(underestimated[1]) > 0 else 0
        optimal_p = round(100 - overestimated_p - underestimated_p, 2)
        avg_diff = round((sum(overestimated[1]) + sum(underestimated[1]))/len(gts), 2)
        return avg_diff, overestimated_p, overestimated_a, underestimated_p, underestimated_a, optimal_p
    
    def test(self, model, epoch):
        model.eval()
        with torch.no_grad():
            p_bar = tqdm(self.val_dl, desc = f'Val {epoch}')
            preds = []
            gts = []
            get_score = lambda pred, gt: round(sum([abs(diff - gt_diff) for diff, gt_diff in zip(pred, gt)])/len(gt), 2)
            typecast = float if self.params.domain == 'maze' else int
            for step, batch in enumerate(p_bar):
                raw_data = batch.pop('raw')
                batch = {k: v.to(torch.device(self.params.device)) for k, v in batch.items()}
                outputs = model.model.generate(**batch, do_sample = True, top_k = 5, num_beams = 1, max_new_tokens = 5, pad_token_id=model.tokenizer.eos_token_id)
                text_outputs = model.tokenizer.batch_decode(outputs, skip_special_tokens = True)
                diffs = extract_differences(text_outputs)
                if len(raw_data[0]) == 3:
                    gt_diffs = [round(typecast(d[2] - d[1]), 2) for d in raw_data]
                elif len(raw_data[0]) == 5:
                    gt_diffs = [round(typecast(d[3] - d[2]), 2) for d in raw_data]
                preds.extend(diffs)
                gts.extend(gt_diffs)
                disp_diffs = random.sample([(d,g) for d, g in zip(diffs, gt_diffs)], min(4, len(diffs)))
                p_bar.set_postfix({'diffs': disp_diffs})
        
        if self.params.num_gpus > 1:
            preds_collated = [None for _ in range(params.num_gpus)]
            dist.all_gather_object(preds_collated, preds)

            gts_collated = [None for _ in range(params.num_gpus)]
            dist.all_gather_object(gts_collated, gts)
        else:
            preds_collated, gts_collated = preds, gts
        avg_diff, overestimated_p, overestimated_a, underestimated_p, underestimated_a, optimal_p = self.get_metrics(preds_collated, gts_collated)
        self.best_test = min(avg_diff, self.best_test)
        if is_main_process():
            print(f'Epoch {epoch}:')
            print(f'Average Diff from GT: {avg_diff}')
            print(f'Overestimated {overestimated_p}% by {overestimated_a}')
            print(f'Underestimated {underestimated_p}% by {underestimated_a}')
            print(f'Optimally predicted {optimal_p}% points')
            print(f'Avg Diff with: {tuple((i, get_score([i] * len(gts), gts)) for i in range(11 if self.params.domain == "sokoban" else 9))}')
            print(f'Dist: {Counter(preds)}')
            if self.best_test == avg_diff and not self.params.test:
                self.save_model(model, epoch, 'model_best_test.pth') # NOTE: Never load from model_best_test.pth to continue training
    
    
    def train(self, model):
        model = model.to(torch.device(self.params.device))
        last_epoch = -1
        if self.params.base_model == 'phi2':
            self.optimizer = AdamW(model.parameters(), lr=self.params.learning_rate)
        else:
            self.optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.params.learning_rate)


        if len(self.params.load_model):
            model, last_epoch = self.load_model(model, self.params.load_model, load_opt = not (self.params.test or self.params.test_ilr))
        
        if self.params.test:
            self.test(model, last_epoch)
            exit()
        
        if len(self.params.test_ilr) > 1:
            self.test_ilr(model)
            exit()

        steps_per_epoch = len(self.train_dl) // self.params.grad_step + int(len(self.train_dl) % self.params.grad_step == 0)
        if self.params.base_model == 'phi2':
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=int(steps_per_epoch * 0.5),
                num_training_steps=(steps_per_epoch * self.params.num_epochs),
            )
        else:
            class IdentityScheduler():
                @classmethod
                def step(self):
                    return
            self.lr_scheduler = IdentityScheduler

        for _ in range((last_epoch + 1) * steps_per_epoch):
            self.lr_scheduler.step()

        for epoch in range(last_epoch, self.params.num_epochs - 1):
            if self.params.num_gpus > 1:
                self.train_dl.sampler.set_epoch(epoch)
                self.test_dl.sampler.set_epoch(epoch)
            self.fit_one_epoch(model, epoch + 1)
            if is_main_process():
                self.save_model(model, epoch + 1, 'model_latest.pth')
            self.test(model, epoch + 1)


