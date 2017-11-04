import glob
import os
import torch
import torch.utils.data as data
import torchvision.datasets.folder as folder

        
class TestFolder(data.Dataset):
    """
    Just like ImageFolder https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    But loads training datasets where there are no class folders
    
    """
    def __init__(self, root, ext, transform):
        self.root = root
        self.transform = transform
        self.img_paths = []
        self.loader = folder.default_loader
        
        for filename in glob.glob(os.path.join(root, '*.{0}'.format(ext))):
            self.img_paths.append(filename)
        
    def __getitem__(self, index):
        path = self.img_paths[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
        
    def __len__(self):
        return len(self.img_paths)
    
# Reference: https://github.com/pytorch/examples/blob/master/imagenet/main.py

def encode_state(model_name, model_state_dict, optimizer_state_dict, epoch_num, best_precision):
    state = dict()
    state['model_name'] = model_name
    state['model_state_dict'] = model_state_dict
    state['optimizer_state_dict'] = optimizer_state_dict
    state['epoch_num'] = epoch_num
    state['best_precision'] = best_precision
    return state
    
def decode_state(state):
    model_name = state['model_name']
    model_state_dict = state['model_state_dict']
    optimizer_state_dict = state['optimizer_state_dict']
    epoch_num = state['epoch_num']
    best_precision = state['best_precision']
    return model_name, model_state_dict, optimizer_state_dict, epoch_num, best_precision
    

# We save the model and the best model on filesystem
def save_model(checkpoint_folder, state, epoch_num):
    filename = 'checkpoint_{0}.pth.tar'.format(epoch_num)
    filepath = os.path.join(checkpoint_folder, filename)
    torch.save(state, filepath)
    return filepath
        
def load_model(checkpoint_file):
    state = torch.load(checkpoint_file)
    return decode_state(state)
        
def accuracy(output, target, batch_size, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count