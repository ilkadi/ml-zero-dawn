import torch
from contextlib import nullcontext

class HardwareHelper:
    torch_datatype_by_string = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    context_by_device = {
        'cuda': lambda device_type, torch_datatype: torch.amp.autocast(device_type=device_type, dtype=torch_datatype),
        'cpu': lambda device_type, torch_datatype:  nullcontext()
    }
    data_to_device = {
        'cuda': lambda device, x, y: (x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)),
        'cpu': lambda device, x, y: (x.to(device), y.to(device))
    }
    
    def __init__(self, data_type, device_type, device):
        print("Setting up hardware..")
        
        self.data_type = data_type
        self.device_type = device_type
        self.device = device
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.dtype = self.torch_datatype_by_string[self.data_type]
        
        self.context = self.context_by_device[self.device_type](self.device, self.dtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.data_type == 'float16'))
    
    def send_to_device(self, x, y):
        return self.data_to_device[self.device_type](self.device, x, y)