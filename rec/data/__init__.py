import os
import sys
import numpy as np
import skimage
import paddle
import signal
import random

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import copy
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import paddle.distributed as dist

from .imaug import transform, create_operators
from .simple_dataset import SimpleDataSet
from .lmdb_dataset import LMDBDataSet

__all__ = [
    'build_dataloader', 'transform', 'create_operators', 'set_signal_handlers'
]


def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


def set_signal_handlers():
    pid = os.getpid()
    try:
        pgid = os.getpgid(pid)
    except AttributeError:
        # In case `os.getpgid` is not available, no signal handler will be set,
        # because we cannot do safe cleanup.
        pass
    else:
        # XXX: `term_mp` kills all processes in the process group, which in 
        # some cases includes the parent process of current process and may 
        # cause unexpected results. To solve this problem, we set signal 
        # handlers only when current process is the group leader. In the 
        # future, it would be better to consider killing only descendants of 
        # the current process.
        if pid == pgid:
            # support exit using ctrl+c
            signal.signal(signal.SIGINT, term_mp)
            signal.signal(signal.SIGTERM, term_mp)



def build_dataloader(config, mode, device, logger, seed=None):
    config = copy.deepcopy(config)

    support_dict = [
        'SimpleDataSet',
        'LMDBDataSet',
    ]
    module_name = config[mode]['dataset']['name']
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."
    dataset = eval(module_name)(config, mode, logger, seed)
    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    if 'use_shared_memory' in loader_config.keys():
        use_shared_memory = loader_config['use_shared_memory']
    else:
        use_shared_memory = True

    if mode == "Train":
        # Distribute data to multiple cards
        if 'sampler' in config[mode]:
            config_sampler = config[mode]['sampler']
            sampler_name = config_sampler.pop("name")
            batch_sampler = eval(sampler_name)(dataset, **config_sampler)
        else:
            batch_sampler = DistributedBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last)
    else:
        # Distribute data to single card
        batch_sampler = BatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

    if 'collate_fn' in loader_config:
        from . import collate_fn
        collate_fn = getattr(collate_fn, loader_config['collate_fn'])()
    else:
        collate_fn = None
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory,
        collate_fn=collate_fn)

    return data_loader