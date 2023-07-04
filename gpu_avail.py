# Find available gpu
import subprocess
import torch

def get_unused_gpu():
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        device_ids = []

        for i in range(total_gpus):
            try:
                torch.cuda.set_device(i)
                device = torch.device('cuda')
                utilization = get_gpu_utilization(i)

                if utilization < 60: device_ids.append(i)

            except RuntimeError:
                continue # Check for the next gpu
        
        if not device_ids:
            device_ids.append(-1)
        
    else:
        device_ids = [-1]

    return device_ids

def get_gpu_utilization(device_id):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
        '--format=csv,noheader', '-i', str(device_id)],
        stdout=subprocess.PIPE, text=True)

        utilization = int(result.stdout.strip().split('\n')[0].replace(' %', ''))
    except subprocess.CalledProcessError:
        utilization = 0

    return utilization