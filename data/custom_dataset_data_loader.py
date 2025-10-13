import torch.utils.data
from data.img_train_dataset_static import HDRDataset as Train_Dataset_static
from data.img_train_dataset_dynamic import HDRDataset as Train_Dataset_dynamic
from data.img_test_dataset_HIDD import HDRDataset as Test_Dataset_HIDD
from data.img_test_dataset_custom import HDRDataset as Test_Dataset_Custom

def CreateDataLoader(opt):
    if opt.isTrain:
        img_train_dataset_static = Train_Dataset_static(opt)
        img_train_dataset_dynamic = Train_Dataset_dynamic(opt)
        img_test_dataset_HIDD = Test_Dataset_HIDD(opt)
        
        img_train_loader_static = torch.utils.data.DataLoader(img_train_dataset_static, batch_size=opt.batchSize, shuffle=opt.shuffle,
                                                              num_workers=int(opt.nThreads), drop_last=True)
        img_train_loader_dynamic = torch.utils.data.DataLoader(img_train_dataset_dynamic, batch_size=opt.batchSize, shuffle=opt.shuffle,
                                                               num_workers=int(opt.nThreads), drop_last=True)
        
        return img_train_loader_static, img_train_loader_dynamic
    else:
        if opt.isCustomData:
            img_test_dataset_Custom = Test_Dataset_Custom(opt)
            img_test_loader_Custom = torch.utils.data.DataLoader(img_test_dataset_Custom, batch_size=1, shuffle=False,
                                                                num_workers=1, drop_last=True)

            return img_test_loader_Custom
        else:
            img_test_dataset_HIDD = Test_Dataset_HIDD(opt)
            img_test_loader_HIDD = torch.utils.data.DataLoader(img_test_dataset_HIDD, batch_size=1, shuffle=False,
                                                            num_workers=1, drop_last=True)
            return img_test_loader_HIDD
        
