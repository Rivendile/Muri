import os
import pickle
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from dataloader import KVReader
# import torch.profiler
from transformers import AutoTokenizer

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}

class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, sargs, file_path: str, block_size=512):
        directory, filename = os.path.split(file_path)
        cached_features_file = sargs["model_name"] + "_cached_lm_" + str(block_size) + "_" + filename
        cached_features_file_path = os.path.join(
            directory, cached_features_file
        )
        text = None
        if args.use_hdfs:
            # Use HDFS datasets
            print("Loading features from HDFS")
            self.reader = KVReader(directory, args.num_readers)
            keys = self.reader.list_keys()
            k_index_map = {k.strip('/'): k for k in keys}
            if cached_features_file in k_index_map and not args.overwrite_cache:
                # logger.info("Loading features from cached file %s", cached_features_file)
                data = self.reader.read_many([k_index_map[cached_features_file]])[0]
                self.examples = pickle.loads(data)
            else:
                data = self.reader.read_many([k_index_map[filename]])[0]
                text = data.decode('utf-8')
            cached_features_file_path = os.path.join(
                args.output_dir, cached_features_file
            )
        else:
            assert os.path.isfile(file_path)
            print("Loading features from local file")
            if os.path.exists(cached_features_file_path) and not args.overwrite_cache:
                # logger.info("Loading features from cached file %s", cached_features_file_path)
                with open(cached_features_file_path, "rb") as handle:
                    self.examples = pickle.load(handle)
            else:
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
        if text:
            # Do writting
            # logger.info("Creating features from dataset file at %s", directory)
            self.examples = []
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            
            # logger.info("Saving features into cached file %s", cached_features_file_path)
            with open(cached_features_file_path, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples(args, sargs, tokenizer):
    file_path = sargs["train_dir"]
    return TextDataset(tokenizer, args, sargs, file_path=file_path, block_size=args.block_size)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels



class NLPModel:
    def __init__(self, idx, args, sargs):
        self.idx = idx
        self.args = args
        self.sargs = sargs # specific args for this model
        self.args.do_train = True
        self.args.overwrite_output_dir = True
    
    def prepare(self, hvd):
        '''
        prepare dataloader, model, optimizer for training
        '''
        self.sargs["local_rank"] = hvd.local_rank()
        if self.sargs["model_name"] in ["bert", "roberta", "distilbert", "camembert"]:
            self.sargs["mlm"] = True
        else:
            self.sargs["mlm"] = False
        if (os.path.exists(self.args.output_dir)
            and os.listdir(self.args.output_dir)
            and self.args.do_train
            and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    self.args.output_dir
                )
            )
        
        # Setup CUDA, GPU & distributed training
        self.device = torch.device("cuda", self.sargs["local_rank"])
        self.sargs["n_gpu"] = 1

        # Load pretrained model and tokenizer
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.sargs["model_name"]]

        config = config_class()

        if self.sargs["model_name"] == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.sargs["model_name"])

        self.args.block_size = self.tokenizer.max_len_single_sentence

        # logger.info("Training new nlp model from scratch")
        self.model = model_class(config=config)

        self.model.to(self.device)

        train_dataset = load_and_cache_examples(self.args, self.sargs, self.tokenizer)

        def collate(examples: List[torch.Tensor]):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        self.train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        self.train_loader = DataLoader(
            train_dataset, sampler=self.train_sampler, batch_size=self.sargs["batch_size"], collate_fn=collate,
            prefetch_factor=self.sargs["prefetch_factor"], num_workers=self.sargs["num_workers"]
        )

        t_total = self.sargs["iters"]
        self.sargs["num_train_epochs"] = self.sargs["iters"] // len(self.train_loader) + 1
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        
        self.optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=self.model.named_parameters(prefix='model'+str(self.idx)))
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_iters, num_training_steps=t_total
        )

        model_to_resize = self.model.module if hasattr(self.model, "module") else self.model  # Take care of distributed/parallel training
        model_to_resize.resize_token_embeddings(len(self.tokenizer))
        self.model.zero_grad()
        self.model.train()

        self.cur_epoch = 0
        self.batch_idx = -1

        self.dataloader_iter = iter(self.train_loader)

    def get_data(self):
        '''
        get data
        '''
        try:
            data = next(self.dataloader_iter)
        except StopIteration:
            self.cur_epoch += 1
            self.train_sampler.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.train_loader)
            data = next(self.dataloader_iter)
            self.batch_idx = -1
        self.batch_idx +=1
        inputs, labels = mask_tokens(data, self.tokenizer, self.args) if self.sargs["mlm"] else (data, data)
        
        return inputs, labels
    
    def forward_backward(self, thread):
        '''
        forward, calculate loss and backward
        '''
        thread.join()
        inputs, labels = thread.get_result()
        if self.args.cuda:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs, masked_lm_labels=labels) if self.sargs["mlm"] else self.model(inputs, labels=labels)
        loss = outputs[0]
        loss.backward()


    def comm(self):
        '''
        sync for communication
        '''
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def print_info(self):
        print("Model ", self.idx, ": ", self.sargs["model_name"], "; batch size: ", self.sargs["batch_size"], "; block size: ", self.args.block_size)

    def data_size(self):
        if self.sargs['model_name']=='bert':
            per_data_size = 10684/4517
        else:
            per_data_size = 10684/2361
        return self.sargs['batch_size']*per_data_size