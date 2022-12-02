import torch.utils.data
from data.img_train_dataset_static import HDRDataset as Train_Dataset_static
from data.img_train_dataset_dynamic import HDRDataset as Train_Dataset_dynamic
from data.img_test_dataset_static import HDRDataset as Test_Dataset_static
from data.img_test_dataset_dynamic import HDRDataset as Test_Dataset_dynamic
from data.img_test_dataset_sensetime import HDRDataset as Test_Dataset_sensetime
from data.img_test_dataset_mix import HDRDataset as Test_Dataset_mix

def CreateDataLoader(opt):
    if opt.isTrain:
        img_train_dataset_static = Train_Dataset_static(opt)
        img_train_dataset_dynamic = Train_Dataset_dynamic(opt)
        img_test_dataset_mix = Test_Dataset_mix(opt)
        
        img_train_loader_static = torch.utils.data.DataLoader(img_train_dataset_static, batch_size=opt.batchSize, shuffle=opt.shuffle,
                                                              num_workers=int(opt.nThreads), drop_last=True)
        img_train_loader_dynamic = torch.utils.data.DataLoader(img_train_dataset_dynamic, batch_size=opt.batchSize, shuffle=opt.shuffle,
                                                               num_workers=int(opt.nThreads), drop_last=True)

        img_test_loader_mix = torch.utils.data.DataLoader(img_test_dataset_mix, batch_size=1, shuffle=False,
                                                          num_workers=int(opt.nThreads), drop_last=True)
        
        return img_train_loader_static, img_train_loader_dynamic, img_test_loader_mix
    else:
        img_test_dataset_static = Test_Dataset_static(opt)
        img_test_dataset_dynamic = Test_Dataset_dynamic(opt)
        img_test_dataset_sensetime = Test_Dataset_sensetime(opt)
        img_test_dataset_mix = Test_Dataset_mix(opt)
        img_test_loader_static = torch.utils.data.DataLoader(img_test_dataset_static, batch_size=1, shuffle=False,
                                                             num_workers=int(opt.nThreads), drop_last=True)
        img_test_loader_dynamic = torch.utils.data.DataLoader(img_test_dataset_dynamic, batch_size=1, shuffle=False,
                                                              num_workers=int(opt.nThreads), drop_last=True)
        img_test_loader_sensetime = torch.utils.data.DataLoader(img_test_dataset_sensetime, batch_size=1, shuffle=False,
                                                                num_workers=int(opt.nThreads), drop_last=True)
        img_test_loader_mix = torch.utils.data.DataLoader(img_test_dataset_mix, batch_size=1, shuffle=False,
                                                                num_workers=int(opt.nThreads), drop_last=True)

        return img_test_loader_static, img_test_loader_dynamic, img_test_loader_sensetime, img_test_loader_mix
