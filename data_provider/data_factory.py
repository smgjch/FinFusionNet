from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_btc, m4Dataset_btc, m4Dataset_btc_CGNN, m4Dataset_btc_block, mDataset_btc, mDataset_btc_CGNN, mDataset_btc_block
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'custom': Dataset_Custom,
    'btc': Dataset_btc,
    "mbtc": mDataset_btc,
    "mbtc_block": mDataset_btc_block,
    "mbtc_CGNN": mDataset_btc_CGNN,
    "m4btc_CGNN": m4Dataset_btc_CGNN,
    "m4btc": m4Dataset_btc,
    "m4btc_block": m4Dataset_btc_block
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    # shuffle_flag = False
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        # print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
            
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        # print(batch_size)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader



# Example usage:
# Assuming args is an object with necessary attributes
# dataset = Dataset_Custom(args, root_path='path/to/data', flag='train', size=[96, 48, 96], features='M', data_path='btc.tsv')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
