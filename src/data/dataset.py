from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import random
from collections import Counter, defaultdict

class HeuristicDataset(Dataset):
    def __init__(self, params, datapoints, raw_datapoints, tokenizer, val = False):
        super().__init__()
        self.datapoints = [(datapoints[0][i], datapoints[1][i]) for i in range(len(datapoints[0]))]
        self.tokenizer = tokenizer
        self.raw_datapoints = raw_datapoints
        self.val = val
        if self.val and len(self.datapoints) > 5000:
            self.idxs = random.sample(range(len(self)), 5000)
            # print(Counter([int(self.raw_datapoints[x][2] - self.raw_datapoints[x][1]) for x in self.idxs]))
        
        if not self.val and params.domain == 'maze':
            self.idxs = []
            groups = defaultdict(list)
            for i, (_, heuristic, optimal_cost) in enumerate(self.raw_datapoints):
                groups[optimal_cost - heuristic - 1].append(i)
            for v in groups.values():
                if len(v) < 3500:
                    self.idxs.extend(v)
                else:
                    self.idxs.extend(random.sample(v, 3500))
            dist = Counter([int(self.raw_datapoints[x][2] - self.raw_datapoints[x][1]) for x in self.idxs])
            # print(dist)
            # print(len(self.idxs))
            # exit()
        # dist = Counter([int(x[2] - x[1]) for x in self.raw_datapoints]) 
        # print(dist)
        # exit()
        

    def __len__(self):
        length = len(self.datapoints) if not hasattr(self, 'idxs') else len(self.idxs)
        return length if length > 0 else 1
    
    def __getitem__(self, idx):
        out = []
        if hasattr(self, 'idxs'):
            idx = self.idxs[idx]
        input_ids, labels = self.datapoints[idx]
        if not self.val:
            labels += [self.tokenizer.eos_token_id]
            input_ids, labels = input_ids + labels, [-100] * len(input_ids) + labels
            labels = torch.tensor(labels).long()
            out.append(labels)
        input_ids = torch.tensor(input_ids).long()
        out.append(input_ids)
        out.append(self.raw_datapoints[idx])
        return out

    def collate_fn_train(self, batch):
        input_ids, labels, raw_data = [], [], []
        out = {}
        for b in batch:
            input_ids.append(b[1])
            labels.append(b[0])
            raw_data.append(b[2])

        out = {
            'input_ids': pad_sequence(input_ids, batch_first=True, padding_value = self.tokenizer.pad_token_id),
            'labels': pad_sequence(labels, batch_first=True, padding_value = -100),
            'raw': raw_data
        }
        out['attention_mask'] = torch.where(out['input_ids'] == self.tokenizer.pad_token_id, 0, 1)
        return out

    def collate_fn_test(self, batch): # left_padded tensor for batched generation
        input_ids, raw_data = [], []
        out = {}
        for b in batch:
            input_ids.append(b[0].flip(0))
            raw_data.append(b[1])
        out = {
            'input_ids': pad_sequence(input_ids, batch_first=True, padding_value = self.tokenizer.pad_token_id).flip(-1),
            'raw': raw_data
        }
        out['attention_mask'] = torch.where(out['input_ids'] == self.tokenizer.pad_token_id, 0, 1)
        return out

class T5HeuristicDataset(HeuristicDataset):
    def __init__(self, params, datapoints, raw_datapoints, tokenizer, val = False):
        super().__init__(params, datapoints, raw_datapoints, tokenizer, val)

    def __getitem__(self, idx):
        out = []
        if hasattr(self, 'idxs'):
            idx = self.idxs[idx]
        input_ids, labels = self.datapoints[idx]
        if not self.val:
            out.append(torch.tensor(labels).long())
        input_ids = torch.tensor(input_ids).long()
        out.append(input_ids)
        out.append(self.raw_datapoints[idx])
        return out

    def collate_fn_test(self, batch): # left_padded tensor for batched generation
        input_ids, raw_data = [], []
        out = {}
        for b in batch:
            input_ids.append(b[0])
            raw_data.append(b[1])
        out = {
            'input_ids': pad_sequence(input_ids, batch_first=True, padding_value = self.tokenizer.pad_token_id),
            'raw': raw_data
        }
        out['attention_mask'] = torch.where(out['input_ids'] == self.tokenizer.pad_token_id, 0, 1)
        return out