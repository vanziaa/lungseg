import numpy
import pywt
import SimpleITK as sitk
from skimage import exposure, io, util
import six
from six.moves import range
import os
from time import time

def left(ct_path,ct_file,seg_path,seg_file,n):
    ct = sitk.ReadImage(os.path.join(ct_path, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    seg = sitk.ReadImage(os.path.join(seg_path, seg_file), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    left_lung_mask=seg_array
    left_lung_mask[left_lung_mask<1.5]=0
    left_lung_mask[left_lung_mask>1.5]=1
    left_lung=ct_array*left_lung_mask
    
    
    target_vol=left_lung
    target_vol[target_vol==0]=-2000
    target_vol[target_vol>=n]=1
    target_vol[target_vol<n]=0
    
    return target_vol

def right(ct_path,ct_file,seg_path,seg_file,n):
    ct = sitk.ReadImage(os.path.join(ct_path, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    seg = sitk.ReadImage(os.path.join(seg_path, seg_file), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    left_lung_mask=seg_array
    left_lung_mask[left_lung_mask<1.5]=0
    left_lung_mask[left_lung_mask>1.5]=1
    left_lung=ct_array*left_lung_mask
    ct = sitk.ReadImage(os.path.join(ct_path, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    seg = sitk.ReadImage(os.path.join(seg_path, seg_file), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    whole_lung_mask=seg_array
    whole_lung_mask[whole_lung_mask>1]=1
    whole_lung=ct_array*whole_lung_mask
    right_lung=whole_lung-left_lung
    
    target_vol=right_lung
    target_vol[target_vol==0]=-2000
    target_vol[target_vol>=n]=1
    target_vol[target_vol<n]=0
    
    return target_vol

ct_path = '/root/liver_spleen_seg/abdominal-multi-organ-segmentation-master/lung/img/'
seg_path = '/root/liver_spleen_seg/abdominal-multi-organ-segmentation-master/lung/mask/'

#save
left_path = '/root/liver_spleen_seg/abdominal-multi-organ-segmentation-master/lung/left_seg/'
right_path = '/root/liver_spleen_seg/abdominal-multi-organ-segmentation-master/lung/right_seg/'


for file_index, file in enumerate(os.listdir(ct_path)):
    
    start_time = time()
    
    ct = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    seg = sitk.ReadImage(os.path.join(seg_path, file.replace('img', 'mask')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    print(file,ct_array.shape)
    print(file.replace('img', 'mask'),seg_array.shape)

    # 将CT读入内存
    left_lung_seg_10h=left(ct_path,file,seg_path,file.replace('img', 'mask'),-1100)#1
    left_lung_seg_8h=left(ct_path,file,seg_path,file.replace('img', 'mask'),-900)#2
    left_lung_seg_5h=left(ct_path,file,seg_path,file.replace('img', 'mask'),-500)#3
    left_lung_seg_3h=left(ct_path,file,seg_path,file.replace('img', 'mask'),-100)#4
    left_lung_seg_1h=left(ct_path,file,seg_path,file.replace('img', 'mask'),100)#5
    whole_left=left_lung_seg_10h+left_lung_seg_8h+left_lung_seg_5h+left_lung_seg_3h+left_lung_seg_1h
    print(whole_left.shape)
    left_lung_seg = sitk.GetImageFromArray(whole_left)
    seg = sitk.ReadImage(os.path.join(seg_path, file.replace('img', 'mask')), sitk.sitkUInt8)

    left_lung_seg.SetDirection(seg.GetDirection())
    left_lung_seg.SetOrigin(seg.GetOrigin())
    left_lung_seg.SetSpacing(seg.GetSpacing())
    sitk.WriteImage(left_lung_seg, os.path.join(left_path, file.replace('img', 'left_seg')))
    print("left saved",os.path.join(left_path, file.replace('img', 'left_seg')))
    
    right_lung_seg_10h=right(ct_path,file,seg_path,file.replace('img', 'mask'),-1100)
    right_lung_seg_8h=right(ct_path,file,seg_path,file.replace('img', 'mask'),-900)
    right_lung_seg_5h=right(ct_path,file,seg_path,file.replace('img', 'mask'),-500)
    right_lung_seg_3h=right(ct_path,file,seg_path,file.replace('img', 'mask'),-100)
    right_lung_seg_1h=right(ct_path,file,seg_path,file.replace('img', 'mask'),100)
    whole_right=right_lung_seg_10h+right_lung_seg_8h+right_lung_seg_5h+right_lung_seg_3h+right_lung_seg_1h
    print(whole_right.shape)
    right_lung_seg = sitk.GetImageFromArray(whole_right)
    seg = sitk.ReadImage(os.path.join(seg_path, file.replace('img', 'mask')), sitk.sitkUInt8)
    right_lung_seg.SetDirection(seg.GetDirection())
    right_lung_seg.SetOrigin(seg.GetOrigin())
    right_lung_seg.SetSpacing(seg.GetSpacing())
    sitk.WriteImage(right_lung_seg, os.path.join(right_path, file.replace('img', 'right_seg')))
    print("right saved",os.path.join(right_path, file.replace('img', 'right_seg')))
    speed = time() - start_time
    print('this case use {:.3f} s'.format(speed))