from argparse import ArgumentParser as ap
from trainer import trainVRP

parser = ap(description='Combinatorial Optimization')
parser.add_argument('--seed', default=12345, type=int)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--task', default='vrp')
parser.add_argument('--nodes', dest='num_nodes', default=10, type=int)
parser.add_argument('--actor_lr', default=5e-4, type=float)
parser.add_argument('--critic_lr', default=5e-4, type=float)
parser.add_argument('--max_grad_norm', default=2., type=float)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--layers', dest='num_layers', default=1, type=int)
parser.add_argument('--train-size',default=1000000, type=int)
parser.add_argument('--valid-size', default=1000, type=int)

args = parser.parse_args()

#print('NOTE: SETTING CHECKPOINT: ')
#args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
#print(args.checkpoint)

print(args)
trainVRP(args)
