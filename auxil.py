#autil.py
import glob 
import torch
import os


def accum_section_track(section, load_path):

  initial_load = torch.load(load_path+'/track1.pth')
  accum_track = dict([(k, torch.zeros_like(v)) for k, v in initial_load.items()])

  for i in range(section[0]+1,  section[1]+1):
    path = load_path + '/track'+str(i)+'.pth'
    track_load = torch.load(path)
    for k, v in track_load.items():
      accum_track[k] += v
  
  
  print(accum_track.keys())
  return accum_track


def make_mask_sub(tracked_bin, mixed_portion):
  
  size = tracked_bin.size()
  
  flat = tracked_bin.reshape([-1, ])
  p = int(len(flat)*mixed_portion)
  v, idx = flat.topk(k=p)
  threshold = v[p-1]
  
  mask = torch.where(flat > threshold, torch.ones_like(flat), torch.zeros_like(flat))

  return mask.reshape(size)

def make_mask(section, load_path):
  bin_change = accum_section_track(section, load_path)

  save_path = load_path+'/track_section'+str(section[0])+'_'+str(section[1])
  if not os.path.isdir(save_path):
    os.mkdir(save_path)

  for i in range(1, 10):
    mixed_portion = 0.1*i
    mask = dict([(k, make_mask_sub(v[1], mixed_portion)) for k, v in enumerate(bin_change.items())])
    torch.save(mask, save_path+'/mask_{:.1f}.pth' .format(mixed_portion))
  print('Make mask..')

