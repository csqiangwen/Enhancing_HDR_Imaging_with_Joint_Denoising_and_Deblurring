import torch.utils.data
from data.img_train_dataset import HDRDataset as Train_Dataset
from data.img_test_dataset import HDRDataset as Test_Dataset
from data.img_test_dataset_full import HDRDataset as Test_Dataset_Full

def CreateDataLoader(opt):
    if opt.isTrain:
        img_train_dataset = Train_Dataset(opt)
        img_test_dataset = Test_Dataset(opt)
        
        img_train_loader = torch.utils.data.DataLoader(img_train_dataset, batch_size=opt.batchSize, shuffle=opt.shuffle,
                                                       num_workers=int(opt.nThreads), drop_last=True)

        img_test_loader = torch.utils.data.DataLoader(img_test_dataset, batch_size=1, shuffle=False,
                                                       num_workers=int(opt.nThreads), drop_last=True)
        
        return img_train_loader, img_test_loader
    else:
        img_test_dataset = Test_Dataset(opt)
        img_test_loader = torch.utils.data.DataLoader(img_test_dataset, batch_size=1, shuffle=False,
                                                       num_workers=int(opt.nThreads), drop_last=True)

        return img_test_loader
