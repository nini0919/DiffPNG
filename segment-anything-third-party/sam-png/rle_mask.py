# rle编码的格式是：

# 格式是像素位置，长度（这了包含其实位置的像素的计数）

# 根据大佬的说法是，rle编码能节省内存

import numpy as np

def rle2mask(rle_dict):
    height, width = rle_dict["size"]
    mask = np.zeros(height * width, dtype=np.uint8)

    rle_array = np.array(rle_dict["counts"])
    starts = rle_array[0::2]
    lengths = rle_array[1::2]

    current_position = -1
    for start, length in zip(starts, lengths):
     #   current_position += start
        mask[start-1:start-1 + length] = 1
      #  current_position += length

    mask = mask.reshape((height, width), order='F')
    return mask


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 
    1 - mask, 
    0 - background
    
    Returns run length as string formated
    '''
 #   print("看下输入的img",img)
    pixels= img.T.flatten()#转置后看图像
 #   print("pixels进行flatten以后=",pixels)
# pixels进行flatten以后= [1 1 0 0 0 0 0 0 0 0 0 0 1 1]#14位
    pixels = np.concatenate([[0], pixels, [0]])
  #  print("pixels=",pixels)
#                 pixels = [0 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0]#16位
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
 #   print("runs=",runs)#这个记录的是bit值开始变化的位置,这里+1是为了位置的调整
    if len(runs) % 2 != 0:
        runs = np.append(runs, len(pixels))
    
    runs[1::2] -= runs[::2]
    #这句代码写得很抽象,其实是在进行编码.
    #运行前的结果是：
    # runs= [ 1  3 13  15]   #runs中的每个数值都代表像素值发生变化的位置
    # 运行后的结果是:
    # runs= [ 1  2 13  2]
    # 意思是第1个位置算起，共有2个bit是相同的，所以用3-1得到
    # 意思是第13个位置算起，共有2个bit是相同的，所以用15-13得到。
    # 对应上面头部和末尾的两个11
 
    #  print("runs=",runs)
    seg=[]
    
    for x in runs:
        
        seg.append(int(x))
    size=[]
    for x in img.shape:
         size.append(int(x))
    result=dict()
    result['counts']=seg
    result['size']=size
    return result



# array = np.array([[1, 0, 1, 0, 1],
#                   [1, 1, 0, 1, 0],
#                   [1, 0, 1, 0, 1],
#                   [1, 1, 0, 1, 0],
#                   [1, 0, 1, 0, 1]])
array = np.array([[True, True, False],
                  [True, True, False],])
array = np.array([[1, 1, 0],
                  [1, 1, 0],])
print(array)
rle_result=mask2rle(array)
print(rle_result)
mask_result=rle2mask(rle_result)
print(mask_result)
