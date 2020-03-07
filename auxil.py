#autil.py
import glob 
import torch

def accum_all_track(load_path):
  list = glob.glob(load_path+'/*')
  
  initial_load = torch.load(list[0])
  accum_track = dict([(k, torch.zeros_like(v.cpu())) for k, v in initial_load.items()])

  for path in list:
    track_load = torch.load(path)

    for k, v in track_load.items():
      v_ = v.cpu()
      accum_track[k] += accum_track[k]+v_.detach().numpy()
  
  torch.save(accum_track, load_path+'/track.pth')
  print('Save accum all track...')
  return accum_track



def make_mask_sub(tracked_bin, mixed_portion):
  
  size = tracked_bin.size()
  
  flat = tracked_bin.reshape([-1, ])
  p = int(len(flat)*mixed_portion)
  v, idx = flat.topk(k=p)
  threshold = v[p-1]
  
  mask = torch.where(flat > threshold, torch.ones_like(flat), torch.zeros_like(flat))

  return mask.reshape(size)

def make_mask(load_path, mixed_portion):
  bin_change = torch.load(load_path+'/track.pth')

  mask = dict([(k, make_mask_sub(v[1], mixed_portion)) for k, v in enumerate(bin_change.items())])


  torch.save(mask, load_path+'/mask.pth')
  print('Make mask..')
  
