import cv2
import numpy as np

def get_energy_map(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, mask1 = cv2.threshold(src=img1, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(src=img2, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
    # 获取重叠区域的模板
    mask = mask1 & mask2
    mask = mask//255
    # 获取两张图片的模板
    img1_overlap = img1*mask
    img2_overlap = img2*mask

    # 必定属于img1的部分,与img1相连的权重设置为无穷大
    img_pixel1 = mask1-mask*255
    cv2.imshow("img_pixel1",img_pixel1)

    # 必定属于img2的部分，除去img1的算作img2的部分，不会影响最终的结果，与img2的权重设置为无穷大
    back_ground = np.ones_like(mask,dtype=np.uint8)
    back_ground = back_ground*255
    img_pixel2 = back_ground-mask*255-img_pixel1
    cv2.imshow("img_pixel2",img_pixel2)
    # 第三部分的权重，也即相交部分的权重设置为0
    # 构建权重输入：
    # 左到右的边的权重
    pre = img1_overlap-img2_overlap
    pre = pre.astype(np.float)
    pre = pre*pre
    # 从左到右的权重
    left = (pre+np.concatenate((pre[:,1:],np.expand_dims(pre[:,0],1)),axis=1))/2
    # 从右到左的权重
    right = (pre+np.concatenate((np.expand_dims(pre[:,img2_overlap.shape[1]-1],1),pre[:,:img2_overlap.shape[1]-1]),axis=1))/2
    # 从上到下的权重
    up = (pre+np.concatenate((pre[1:,:],np.expand_dims(pre[0,:],0)),axis=0))/2
    # 从下到上的权重
    down = (pre+np.concatenate((np.expand_dims(pre[img2_overlap.shape[0]-1,:],axis=0),pre[:img2_overlap.shape[0]-1,:]),axis=0))/2
    cv2.imshow("pre",pre)
    cv2.imshow("left",left)

    # 计算相交部分的平滑项
    return img_pixel1,img_pixel2,left,right,up,down
