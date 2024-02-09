import math   
import matplotlib.pyplot as plt
    
class LRHelper:
    def __init__(self, default_lr, warmup_iters, decay_lr_iters, min_lr):
        self.default_lr = default_lr
        self.warmup_iters = warmup_iters
        self.decay_lr_iters = decay_lr_iters
        self.min_lr = min_lr
        
    def set_default_lr(self, optimizer):      
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.default_lr    
        
    def set_cousine_annealing_lr(self, optimizer, epoch):
        if epoch < self.warmup_iters:
            lr = self.default_lr * epoch / self.warmup_iters
        elif epoch > self.decay_lr_iters:
            lr = self.min_lr
        else:
            decay_ratio = (epoch - self.warmup_iters) / (self.decay_lr_iters - self.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.default_lr - self.min_lr)
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
class LRViewer:
    def __init__(self):
        self.param_groups = [{'lr': 0.0}]
        
    def view_lr_helper(self, lr_helper, num_epochs):
        lrs = []
        for epoch in range(num_epochs):
            lr_helper.set_cousine_annealing_lr(self, epoch)
            lrs.append(self.param_groups[0]['lr'])
        
        plt.plot(range(num_epochs), lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Cosine Annealing Learning Rate Schedule')
        plt.show()