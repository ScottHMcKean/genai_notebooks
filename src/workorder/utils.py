import torch
import gc

def get_gpu_utilization() -> float:
  free, total = torch.cuda.mem_get_info()
  return 1-free/total

def print_gpu_utilization() -> None:
  print(f"% GPU Utilization: {get_gpu_utilization()*100:.1f}")  

def clear_gpu(objects:list):
  for obj_to_del in objects:
    try:
      del model
    except:
      pass

  gc.collect()
  torch.cuda.empty_cache()