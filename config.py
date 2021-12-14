import argparse

parser = argparse.ArgumentParser(description='DTA with pretrain')
parser.add_argument('--device', type=str, default="0",
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 512)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--dataset', type=str, default='davis', choices=['davis', 'kiba', 'PDBBind'],
                    help='Dataset')

parser.add_argument('--penc', type=str, default='davis', choices=['ESM','ESM1v','ESM1b','ESMMSA','ESM_Conv', 'CNN'],
                    help='Protein encoder ')
parser.add_argument('--esmdim', type=int, default=768,
                    help='Protein ESM dim ')
parser.add_argument('--pencdim', type=int, default=128,
                    help='Protein encoder ')
parser.add_argument('--ppipretrain', dest='ppipretrain', action='store_true')
parser.set_defaults(ppipretrain=False)

parser.add_argument('--denc', type=str, default='davis', choices=['Chemberta', 'GIN'],
                    help='Dataset')
parser.add_argument('--ccipretrain', dest='ccipretrain', action='store_true')
parser.set_defaults(ccipretrain=False)
parser.add_argument('--dencdim', type=int, default=64,
                    help='Protein encoder ')
parser.add_argument('--infograph', dest='infograph', action='store_true')
parser.set_defaults(infograph=False)
# parser.add_argument('--dim', type=int, default=128,
#                     help='hidden dim of GIN encoder')

parser.add_argument('--model', type=int, default=0,
                    help='Model')

parser.add_argument('--setting', type=str, default='3', choices=['1', '2', '3', '4'],
                    help='Setting 1: warm, setting 2: cold-target, setting 3: cold-drug, setting 4: cold-drug-target'
                    )

parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--notrain', dest='train', action='store_false')
parser.set_defaults(train=True)

parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--pretrainprojection', dest='pretrainprojection', action='store_true')
parser.add_argument('--notest', dest='test', action='store_false')
parser.set_defaults(test=True)

parser.add_argument('--freeze_de', dest='freeze_de', action='store_true')
parser.set_defaults(freeze_de=False)
parser.add_argument('--name', type=str, default="",
                    help='Addition name')
args = parser.parse_args()
