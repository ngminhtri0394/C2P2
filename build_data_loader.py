from torch.utils.data import DataLoader as PTDataLoader
from torch_geometric.data import DataLoader as GMDataLoader


def build_data_loader(args, train_data, valid_data, test_data):
    if 'GIN' in args.denc:
        train_loader = GMDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        valid_loader = GMDataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
        test_loader = GMDataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = PTDataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=False)
        valid_loader = PTDataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                                    drop_last=False)
        test_loader = PTDataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader
