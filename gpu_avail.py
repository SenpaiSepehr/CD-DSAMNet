# Find available gpu

import torch

def get_unused_gpu():
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        device_ids = []

        for i in range(total_gpus):
            try:
                torch.cuda.set_device(i)
                device = torch.device('cuda')

                allocated_memory = torch.cuda.memory_allocated(i)
                total_memory = torch.cuda.max_memory_allocated(i)
                utilization = allocated_memory / total_memory

                if utilization < 0.6: device_ids.append(i)

            except RuntimeError: continue # Check for the next gpu
        
        if not device_ids: device_ids.append(-1)
        
    else: device_ids = [-1]

    return device_ids
