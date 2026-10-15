import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch_stu', type=int, default=50, help='Max number of epochs.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--lr', type=float, default=None, help='Initial learning rate. DB1 uses 0.002 and DB2 uses 0.006 when omitted.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--tau', type=float, default=0.5, help='Contrastive temperature.')
parser.add_argument('--alpha', type=float, default=0.5, help='Weight of the LSA-to-GSA contrastive loss.')
parser.add_argument('--beta', type=float, default=0.5, help='Weight of the GSA-to-LSA contrastive loss.')
parser.add_argument('--layers', type=int, default=2, help='Number of GAT layers in each block.')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
parser.add_argument('--K', type=int, default=2, help='Number of interaction blocks.')
parser.add_argument('--dataset', type=str, default='miRNA', help='Dataset name.')
parser.add_argument('--save_dir', type=str, default='./result', help='Directory for metrics and predictions.')
parser.add_argument('--emb_dim', type=int, default=128, help='Input embedding dimension.')
parser.add_argument('--no_cuda', action='store_false', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
args = parser.parse_args()
if args.lr is None:
    args.lr = 0.006 if args.dataset.lower() == 'mirna2' else 0.002
args.device = torch.device('cuda:0' if args.no_cuda and torch.cuda.is_available() else 'cpu')
