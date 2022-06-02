import argparse
import os

# Training settings: common
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-iters', type=float, default=0,
                    help='number of warmup iters')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--this-dir', type=str, default='./',
                    help='the path of this file')

# common for CV
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')

# common for nlp



# specific
parser.add_argument('--train-dir0', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--train-dir1', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--train-dir2', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--train-dir3', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')

parser.add_argument('--prefetch-factor0', type=int, default=2, help='prefatch factor for dataloder.')
parser.add_argument('--prefetch-factor1', type=int, default=2, help='prefatch factor for dataloder.')
parser.add_argument('--prefetch-factor2', type=int, default=2, help='prefatch factor for dataloder.')
parser.add_argument('--prefetch-factor3', type=int, default=2, help='prefatch factor for dataloder.')

parser.add_argument('--num-workers0', type=int, default=2, help='number of workers for dataloder.')
parser.add_argument('--num-workers1', type=int, default=2, help='number of workers for dataloder.')
parser.add_argument('--num-workers2', type=int, default=2, help='number of workers for dataloder.')
parser.add_argument('--num-workers3', type=int, default=2, help='number of workers for dataloder.')

parser.add_argument('--model0', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--model1', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--model2', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--model3', type=str, default='resnet50',
                    help='model to benchmark')
                    
parser.add_argument('--batch-size0', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--batch-size1', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--batch-size2', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--batch-size3', type=int, default=32,
                    help='input batch size for training')

parser.add_argument('--iters0', type=int, default=100,
                    help='number of iters to train')
parser.add_argument('--iters1', type=int, default=100,
                    help='number of iters to train')
parser.add_argument('--iters2', type=int, default=100,
                    help='number of iters to train')
parser.add_argument('--iters3', type=int, default=100,
                    help='number of iters to train')


# common for NLP
parser.add_argument(
    "--output_dir",
    type=str,
    default='output',
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
)
parser.add_argument(
    "--cache_dir",
    default=None,
    type=str,
    help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
)
parser.add_argument(
    "--block_size",
    default=-1,
    type=int,
    help="Optional input sequence length after tokenization."
    "The training dataset will be truncated in block of this size for training."
    "Default to the model max input length for single sentence inputs (take into account special tokens).",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--num_train_epochs", default=1000.0, type=float, help="Total number of training epochs to perform."
)
parser.add_argument(
    "--save_total_limit",
    type=int,
    default=None,
    help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
)
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
)
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
)
parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument(
    "--use_hdfs",
    action="store_true",
    help="Whether to use HDFS to load data",
)
parser.add_argument(
    "--num_readers",
    type=int,
    default=4,
    help="Number of HDFS data readers, should be larger than 0",
)

# common for RL
#DQN
parser.add_argument('--play-times', type=int, default=50, help='DQN: play times before each iter')
#A2C
parser.add_argument('--rollout-length', type=int, default=5, help='A2C: rollout length for trajectory')

# common for rpc
parser.add_argument('--scheduler-ip', type=str, required=True, help='IP of scheduler')
parser.add_argument('--scheduler-port', type=int, default=9011, help='port of scheduler')
parser.add_argument('--trainer-port', type=int, required=True, help='port of trainer')

parser.add_argument('--job-id0', type=int, default=-1, help='the id of job0')
parser.add_argument('--job-id1', type=int, default=-1, help='the id of job1')
parser.add_argument('--job-id2', type=int, default=-1, help='the id of job2')
parser.add_argument('--job-id3', type=int, default=-1, help='the id of job3')

