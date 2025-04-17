import random
import torch
import os
import sys
import json
from tqdm import tqdm
from transformers import AutoTokenizer

class LLMEvolutionSearcher:
    """Evolution search for GLoRA configurations on language models"""

    def __init__(self, args, device, model, tokenizer, val_dataset, output_dir):
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.args = args
        self.max_epochs = args.evolution_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {}
        self.keep_top_k[self.select_num] = []
        self.keep_top_k[50] = []
        self.epoch = 0
        self.candidates = []
        self.top_metrics = []
        
        # Define configuration choices based on GLoRA paper
        self.choices = {
            'A': [f'LoRA_{args.rank}', 'vector', 'constant', 'none'],
            'B': [f'LoRA_{args.rank}', 'vector', 'constant', 'none'],
            'C': [f'LoRA_{args.rank}', 'vector', 'none'],
            'D': ['vector', 'constant', 'none'],
            'E': ['vector', 'constant', 'none']
        }
        
        # Create validation dataloader with a smaller batch size for evaluation
        from torch.utils.data import DataLoader
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=1
        )
        
        print(f"Initialized LLM evolution search with {self.population_num} population size")

    def save_checkpoint(self):
        """Save the current state of the evolution search"""
        info = {
            'top_metrics': self.top_metrics,
            'memory': self.memory,
            'candidates': self.candidates,
            'vis_dict': {k: v for k, v in self.vis_dict.items() if isinstance(k, str)},
            'keep_top_k': {
                k: v for k, v in self.keep_top_k.items() 
                if isinstance(k, int) and isinstance(v, list)
            },
            'epoch': self.epoch
        }
        
        checkpoint_path = os.path.join(self.output_dir, f"evolution-checkpoint-{self.epoch}.json")
        
        # We can't directly JSON serialize tuples, so convert candidates to string representation
        serializable_info = {
            'top_metrics': info['top_metrics'],
            'memory': [[str(cand) for cand in epoch_mem] for epoch_mem in info['memory']],
            'candidates': [str(cand) for cand in info['candidates']],
            'vis_dict': info['vis_dict'],
            'keep_top_k': {
                str(k): [str(cand) for cand in v] 
                for k, v in info['keep_top_k'].items()
            },
            'epoch': info['epoch']
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(serializable_info, f, indent=2)
        
        # Also save the best configuration
        if len(self.keep_top_k[self.select_num]) > 0:
            best_config = self.keep_top_k[self.select_num][0]
            best_config_path = os.path.join(self.output_dir, "best_evolution_config.json")
            self.save_config(best_config, best_config_path)
            
        print(f'Saved evolution checkpoint to {checkpoint_path}')

    def save_config(self, config, path):
        """Save a configuration to a file"""
        config_dict = []
        for i, layer_config in enumerate(config):
            config_dict.append({
                'layer': i,
                'A': layer_config['A'],
                'B': layer_config['B'],
                'C': layer_config['C'],
                'D': layer_config['D'],
                'E': layer_config['E']
            })
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def set_config(self, config):
        """Set the configuration for evaluation"""
        linear_layers = [module for name, module in self.model.named_modules() 
                         if isinstance(module, torch.nn.Linear)]
        
        # Filter to only include attention layers if needed
        attention_layers = []
        for i, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, torch.nn.Linear) and any(attn_name in name for attn_name 
                                                          in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                                             'gate_proj', 'up_proj', 'down_proj']):
                attention_layers.append((i, name, module))
        
        # Set config for each layer
        for i, layer_config in enumerate(config):
            if i < len(attention_layers):
                _, name, module = attention_layers[i]
                if hasattr(module, 'eval_config'):
                    module.eval_config = layer_config
                    
    def get_param_tensor(self, config, in_feature, out_feature, name):
        """Calculate the number of parameters for a tensor given its configuration"""
        if 'A' in name or 'B' in name or 'C' in name:
            if 'C' in name:
                out_feature = in_feature
                in_feature = 1
            if 'LoRA' in config:
                try:
                    rank = int(config.split('_')[1])
                except:
                    rank = self.args.rank
                param = out_feature*rank + in_feature*rank
            elif 'vector' in config:
                param = out_feature
            elif 'constant' in config:
                param = 1
            elif 'none' in config:
                param = 0
            else:
                raise ValueError(f"Unknown config: {config}")
        else:
            if 'vector' in config:
                param = out_feature
            elif 'constant' in config:
                param = 1
            elif 'none' in config:
                param = 0
            else:
                raise ValueError(f"Unknown config: {config}")
        return param
    
    def get_param(self, configs):
        """Calculate the total number of parameters for a configuration"""
        params = 0
        
        # Get all attention layers
        attention_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(attn_name in name for attn_name 
                                                         in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                                            'gate_proj', 'up_proj', 'down_proj']):
                attention_layers.append((name, module))
        
        # Only calculate parameters for the number of configs provided
        for i, (name, module) in enumerate(attention_layers):
            if i >= len(configs):
                break
                
            out_channel = module.out_features
            in_channel = module.in_features
            
            for sup_tnsr in ['A', 'B', 'C', 'D', 'E']:
                params += self.get_param_tensor(configs[i][sup_tnsr], in_channel, out_channel, sup_tnsr)
                
        return params

    def is_legal(self, cand):
        """Check if a candidate is legal and evaluate it"""
        assert isinstance(cand, tuple)
        
        # Add to visited dictionary if not already present
        if str(cand) not in self.vis_dict:
            self.vis_dict[str(cand)] = {}
            
        info = self.vis_dict[str(cand)]
        
        # Skip if already visited
        if 'visited' in info:
            return False
            
        # Calculate parameter count
        n_parameters = self.get_param(configs=cand)
        info['params'] = n_parameters / 10.**6
        
        # Check parameter constraints
        if info['params'] > self.parameters_limits:
            print('parameters limit exceeded')
            return False
            
        if info['params'] < self.min_parameters_limits:
            print('under minimum parameters limit')
            return False
        
        # Evaluate the candidate
        eval_loss = self.evaluate(config=cand)
        info['loss'] = eval_loss
        info['score'] = -eval_loss  # Negative loss as score (higher is better)
        info['visited'] = True
        
        return True

    def evaluate(self, config):
        """Evaluate a configuration on the validation set"""
        self.model.eval()
        self.set_config(config)
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                
                batch_size = batch['input_ids'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def update_top_k(self, candidates, *, k, key, reverse=True):
        """Update the top-k candidates based on a key function"""
        assert k in self.keep_top_k
        print('Selecting top candidates...')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        """Generate random candidates in batches"""
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if str(cand) not in self.vis_dict:
                    self.vis_dict[str(cand)] = {}
            for cand in cands:
                yield cand

    def get_random_cand(self):
        """Generate a random candidate configuration"""
        # Count the number of attention layers in the model
        attention_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(attn_name in name for attn_name 
                                                         in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                                            'gate_proj', 'up_proj', 'down_proj']):
                attention_layers.append((name, module))
        
        depth = len(attention_layers)
        cand_tuple = []
        
        for i in range(depth):
            cand_tuple.append({
                'A': random.choice(self.choices['A']),
                'B': random.choice(self.choices['B']),
                'C': random.choice(self.choices['C']),
                'D': random.choice(self.choices['D']),
                'E': random.choice(self.choices['E'])
            })
            
        return tuple(cand_tuple)

    def get_random(self, num):
        """Generate a set of random candidates"""
        print('Generating random candidates...')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print(f'random {len(self.candidates)}/{num}, loss: {self.vis_dict[str(cand)]["loss"]:.4f}')
            
        print(f'Generated {len(self.candidates)} random candidates')

    def get_mutation(self, k, mutation_num, m_prob):
        """Generate mutated candidates from the top-k configurations"""
        assert k in self.keep_top_k
        print('Generating mutations...')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            """Generate a mutated candidate"""
            cand = list(random.choice(self.keep_top_k[k]))
            final = []
            
            for i in range(len(cand)):
                final_layer = {}
                for key in ['A', 'B', 'C', 'D', 'E']:
                    random_s = random.random()
                    if random_s < m_prob:
                        final_layer[key] = random.choice(self.choices[key])
                    else:
                        final_layer[key] = cand[i][key]
                final.append(final_layer)
                
            return tuple(final)

        cand_iter = self.stack_random_cand(random_func)
        iters = 0
        
        while len(res) < mutation_num and iters < max_iters:
            iters += 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print(f'mutation {len(res)}/{mutation_num}, loss: {self.vis_dict[str(cand)]["loss"]:.4f}')
            
        print(f'Generated {len(res)} mutations')
        return res

    def get_crossover(self, k, crossover_num):
        """Generate crossover candidates from the top-k configurations"""
        assert k in self.keep_top_k
        print('Generating crossovers...')
        res = []
        max_iters = crossover_num * 10

        def random_func():
            """Generate a crossover candidate"""
            cand_1 = list(random.choice(self.keep_top_k[k]))
            cand_2 = list(random.choice(self.keep_top_k[k]))
            final = []
            
            for i in range(min(len(cand_1), len(cand_2))):
                final_layer = {}
                for key in ['A', 'B', 'C', 'D', 'E']:
                    final_layer[key] = random.choice([cand_1[i][key], cand_2[i][key]])
                final.append(final_layer)
                
            return tuple(final)

        cand_iter = self.stack_random_cand(random_func)
        iters = 0
        
        while len(res) < crossover_num and iters < max_iters:
            iters += 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print(f'crossover {len(res)}/{crossover_num}, loss: {self.vis_dict[str(cand)]["loss"]:.4f}')
            
        print(f'Generated {len(res)} crossovers')
        return res

    def search(self):
        """Run the evolutionary search process"""
        print(f'Starting evolutionary search with:')
        print(f'  Population size: {self.population_num}')
        print(f'  Selection size: {self.select_num}')
        print(f'  Mutation size: {self.mutation_num}')
        print(f'  Crossover size: {self.crossover_num}')
        print(f'  Random size: {self.population_num - self.mutation_num - self.crossover_num}')
        print(f'  Max epochs: {self.max_epochs}')

        # Generate initial population
        self.get_random(self.population_num)

        # Run for specified number of epochs
        while self.epoch < self.max_epochs:
            print(f'Epoch {self.epoch}/{self.max_epochs}')

            # Save current population to memory
            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
            
            # Update top candidates
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[str(x)]['score'])
            
            # Update top 50 candidates
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[str(x)]['score'])

            # Print results
            print(f'Epoch {self.epoch}: top {len(self.keep_top_k[self.select_num])} results')
            tmp_metrics = []
            
            for i, cand in enumerate(self.keep_top_k[self.select_num]):
                loss = self.vis_dict[str(cand)]['loss']
                params = self.vis_dict[str(cand)]['params']
                print(f'No.{i+1} Loss = {loss:.4f}, params = {params:.2f}M')
                tmp_metrics.append(loss)
                
            self.top_metrics.append(tmp_metrics)

            # Generate new population through mutation and crossover
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            # Add random candidates to maintain population size
            self.get_random(self.population_num)

            self.epoch += 1
            self.save_checkpoint()

        # Save best configuration
        best_config = self.keep_top_k[self.select_num][0]
        best_path = os.path.join(self.output_dir, "best_glora_config.json")
        self.save_config(best_config, best_path)
        
        return best_config