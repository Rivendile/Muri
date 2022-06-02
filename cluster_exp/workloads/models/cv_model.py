import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
# import torch.profiler
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models


class CVModel:
    def __init__(self, idx, args, sargs):
        self.idx = idx
        self.args = args
        self.sargs = sargs # specific args for this model
    
    def prepare(self, hvd):
        '''
        prepare dataloader, model, optimizer for training
        '''
        self.device = torch.device("cuda")

        train_dataset = \
            datasets.ImageFolder(self.sargs["train_dir"],
                            transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.sargs["batch_size"],
            sampler=self.train_sampler, num_workers=self.sargs["num_workers"],
            prefetch_factor=self.sargs["prefetch_factor"])


        self.model = getattr(models, self.sargs["model_name"])(num_classes=self.args.num_classes)

        if self.args.cuda:
            self.model.cuda()
        
        optimizer = optim.SGD(self.model.parameters(), lr=(self.args.base_lr),
                    momentum=self.args.momentum, weight_decay=self.args.wd)
        compression = hvd.Compression.fp16 if self.args.fp16_allreduce else hvd.Compression.none
        self.optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=self.model.named_parameters(prefix='model'+str(self.idx)),
            compression=compression,
            op=hvd.Adasum if self.args.use_adasum else hvd.Average)
        
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        self.dataloader_iter = iter(self.train_loader)

        self.cur_epoch = 0
        self.batch_idx = -1

        self.model.train()

    def get_data(self):
        '''
        get data
        '''
        try:
            data,target = next(self.dataloader_iter)
        except StopIteration:
            self.cur_epoch += 1
            self.train_sampler.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.train_loader)
            data,target = next(self.dataloader_iter)
            self.batch_idx = -1
        self.batch_idx +=1
        
        return data,target
    
    def forward_backward(self, thread):
        '''
        forward, calculate loss and backward
        '''
        thread.join()
        data, target = thread.get_result()
        if self.args.cuda:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

    def comm(self):
        '''
        sync for communication
        '''
        self.optimizer.step()
    
    def print_info(self):
        print("Model ", self.idx, ": ", self.sargs["model_name"], "; batch size: ", self.sargs["batch_size"])

    def data_size(self):
        # each image is 108.6kb on average
        return self.sargs["batch_size"] * 108.6