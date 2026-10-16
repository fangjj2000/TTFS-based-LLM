import copy
import logging
import os
import random
import shutil
import time
from argparse import Namespace
from typing import Tuple, List, Dict

import torch
from datasets import load_dataset
from torch import nn, Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


def get_pile(nameLogger: str,
             tokenizer,
             train_size: int,
             val_size: int,
             seed: int,
             seqlen: int,
             test_only: bool = False) -> Tuple[List, List] | Tensor:
    r"""

    :param nameLogger:
    :type nameLogger: str
    :param tokenizer:
    :type tokenizer:
    :param train_size:
    :type train_size: int
    :param val_size:
    :type val_size: int
    :param seed:
    :type seed: int
    :param seqlen:
    :type seqlen: int
    :param test_only:
    :type test_only: bool
    :return:
    :rtype: Tuple[List, List]
    """
    logger = logging.getLogger(nameLogger)
    logger.info("获取 【pile】 数据集")
    traindata = load_dataset("mit-han-lab/pile-val-backup", split="validation")  # 官方数据集只有 validation 分划。
    traindata = traindata.shuffle(seed=seed)
    valdata = copy.deepcopy(traindata)

    if test_only:
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] >= seqlen + 1:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)

        return valenc

    random.seed(seed)
    val_sample_ratio = 0.9
    trainloader = []
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_wikitext2(nameLogger: str,
                  tokenizer,
                  train_size: int,
                  val_size: int,
                  seed: int,
                  seqlen: int,
                  test_only: bool) -> Tuple[List, List]:
    r"""

    :param nameLogger:
    :type nameLogger: str
    :param tokenizer:
    :type tokenizer:
    :param train_size:
    :type train_size: int
    :param val_size:
    :type val_size: int
    :param seed:
    :type seed: int
    :param seqlen:
    :type seqlen: int
    :param test_only:
    :type test_only: bool
    :return:
    :rtype: Tuple[List, List]
    """
    logger = logging.getLogger(nameLogger)
    logger.info("获取 【wikitext2】 数据集")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
                           trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, valloader


def get_c4(nameLogger: str,
           tokenizer,
           train_size: int,
           val_size: int,
           seed: int,
           seqlen: int,
           test_only: bool) -> Tuple[List, List] | Tensor:
    r"""

    :param nameLogger:
    :type nameLogger:
    :param tokenizer:
    :type tokenizer:
    :param train_size:
    :type train_size:
    :param val_size:
    :type val_size:
    :param seed:
    :type seed:
    :param seqlen:
    :type seqlen:
    :param test_only:
    :type test_only:
    :return:
    :rtype: Tuple[List, List]
    """
    logger = logging.getLogger(nameLogger)
    logger.info("获取 【c4】 数据集")

    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation')

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    if test_only:
        return valenc

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            # i = random.randint(int(len(traindata)*val_sample_ratio),len(traindata)-1)
            i = random.randint(0, len(valdata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            # if trainenc.input_ids.shape[1] >= seqlen+1:
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))

    return trainloader, valloader


def get_redpajama(nameLogger: str,
                  tokenizer,
                  train_size: int,
                  val_size: int,
                  seed: int,
                  seqlen: int,
                  test_only: bool) -> Tuple[List, List] | Tensor:
    r"""

    :param nameLogger:
    :type nameLogger: str
    :param tokenizer:
    :type tokenizer:
    :param train_size:
    :type train_size: int
    :param val_size:
    :type val_size: int
    :param seed:
    :type seed: int
    :param seqlen:
    :type seqlen: int
    :return:
    :rtype: Tuple[List, List]
    """
    logger = logging.getLogger(nameLogger)
    logger.info("获取 【redpajama】 数据集")

    traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split='train')

    valdata = copy.deepcopy(traindata)

    if test_only:
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] >= seqlen + 1:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)

        return valenc

    random.seed(seed)
    traindata = traindata.shuffle(seed=seed)
    trainloader = []
    val_sample_ratio = 0.9
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_ptb(nameLogger: str,
            tokenizer,
            train_size: int,
            val_size: int,
            seed: int,
            seqlen: int,
            test_only: bool) -> Tuple[List, List] | Tensor:
    r"""

    :param nameLogger:
    :type nameLogger: str
    :param tokenizer:
    :type tokenizer:
    :param nsamples:
    :type nsamples: int
    :param seed:
    :type seed: int
    :param seqlen:
    :type seqlen: int
    :return:
    :rtype: Tuple[List, List]
    """
    logger = logging.getLogger(nameLogger)
    logger.info("获取 【ptb】 数据集")
    traindata = load_dataset("ptb-text-only/ptb_text_only", 'penn_treebank', split='train')
    testdata = load_dataset("ptb-text-only/ptb_text_only", 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    if test_only:
        return testenc
    random.seed(seed)

    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
                           trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_loaders(name: str,
                nameLogger: str,
                tokenizer,
                train_size: int = 128,
                val_size: int = 64,
                seed: int = 0,
                seqlen: int = 2048,
                test_only: bool = False
                ):
    if 'c4' in name:
        return get_c4(nameLogger, tokenizer, train_size, val_size, seed, seqlen, test_only)

    elif 'pile' in name:
        return get_pile(nameLogger, tokenizer, train_size, val_size, seed, seqlen, test_only)

    elif 'ptb' in name:
        return get_ptb(nameLogger, tokenizer, train_size, val_size, seed, seqlen, test_only)

    elif 'redpajama' in name:
        return get_redpajama(nameLogger, tokenizer, train_size, val_size, seed, seqlen, test_only)

    elif 'wikitext2' in name:
        return get_wikitext2(nameLogger, tokenizer, train_size, val_size, seed, seqlen, test_only)

    else:
        raise NotImplementedError(f"输入的数据集 {name} 不受支持！")


@torch.no_grad()
def test_ppl(args: Namespace,
             logger: logging.Logger,
             model: nn.Module,
             tokenizer,
             prefixed_key_values=None,
             datasets=['wikitext2']) -> Dict:
    r"""

    :param args:
    :type args: Namespace
    :param nameLogger:
    :type nameLogger: str
    :param model:
    :type model: nn.Module
    :param tokenizer:
    :type tokenizer:
    :param prefixed_key_values:
    :type prefixed_key_values:
    :param datasets:
    :type datasets:
    :return:
    :rtype: Dict
    """
    results = {}
    total_ppl = list()
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            logger.name,
            tokenizer,
            seed=0,
            seqlen=args.ppl_seqlen,
            test_only=True
        )
        if "c4" in dataset or "redpajama" in dataset or "pile" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = args.ppl_seqlen
        nsamples = testenc.numel() // seqlen
        model.eval()
        nlls = []
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)]
            labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)]
            batch = batch.to(model.device)
            labels = labels.to(model.device)
            outputs = model(batch, labels=labels, past_key_values=prefixed_key_values)
            neg_log_likelihood = outputs.loss * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        results[dataset] = ppl.item()
        total_ppl.append(ppl)
        logger.info(f'{dataset} 的 ppl 测试结果: {ppl}\n')
    avg_ppl = torch.mean(torch.tensor(total_ppl)).item()
    logger.info(f"平均的 ppl 测试结果：{avg_ppl}\n\n")
    results["avg_ppl"] = avg_ppl
    return results


class BlockTrainDataset(Dataset):
    def __init__(self, size, seqlen, hidden_size, batch_size, dtype, cache_path='./cache/', off_load_to_disk=False):
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        assert size % batch_size == 0

        if self.off_load_to_disk:
            self.cache_path = self.get_cache_path(cache_path)
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            self.data = torch.zeros((self.size // self.batch_size, self.batch_size, self.seqlen, self.hidden_size),
                                    dtype=self.dtype)

    def _initialize_data_on_disk(self):
        for idx in range(self.size // self.batch_size):
            tensor = torch.zeros((self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)
            filepath = self._get_file_path(idx)
            torch.save(tensor, filepath)

    def _get_file_path(self, idx):
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            tensor = torch.load(filepath)
        else:
            tensor = self.data[idx]
        return tensor

    def update_data(self, idx, new_data):
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            torch.save(new_data.to(self.dtype), filepath)
        else:
            self.data[idx] = new_data

    def get_subset(self, num):
        data = []
        iter = num // self.batch_size
        for idx in range(iter):
            if self.off_load_to_disk:
                filepath = self._get_file_path(idx)
                tensor = torch.load(filepath)
            else:
                tensor = self.data[idx]
            data.append(tensor)
        return torch.cat(data, dim=0)

    def delete_cache(self):
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)

    def get_cache_path(self, parent_dir):
        new_cache_path = None
        while new_cache_path is None or os.path.exists(new_cache_path):
            flag = time.time()
            new_cache_path = os.path.join(parent_dir, flag)
            time.sleep(1)  # avoid same flag


def replace_last_directory_level(original_path, new_directory):
    parts = original_path.split(os.sep)
    parts[-1] = new_directory
    new_path = os.path.join(*parts)
    return new_path


def copy_block_dataset(dataset: BlockTrainDataset, cache_dir=None):
    new_cache_path = None
    if dataset.off_load_to_disk:
        old_cache_path = dataset.cache_path
        while new_cache_path is None or os.path.exists(new_cache_path):
            flag = time.time()
            new_cache_path = replace_last_directory_level(old_cache_path, flag)
            time.sleep(1)  # avoid same flag
        shutil.copytree(old_cache_path, new_cache_path)
    new_dataset = BlockTrainDataset(dataset.size, dataset.seqlen,
                                    dataset.hidden_size, dataset.batch_size,
                                    dataset.dtype, new_cache_path,
                                    dataset.off_load_to_disk)
    if not dataset.off_load_to_disk:
        for index, data in enumerate(dataset):
            new_dataset.update_data(index, data)
    return new_dataset
