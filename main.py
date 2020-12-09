import cv2
import maxflow
from get_energy_map.energy import get_energy_map
import numpy as np
if __name__ == '__main__':
    src = cv2.imread("image/img1.jpg")
    dst = cv2.imread("image/img2.jpg")

    img_pixel1,img_pixel2,left,right,up,down = get_energy_map(src,dst)

    g = maxflow.GraphFloat()
    img_pixel1 = img_pixel1.astype(float)
    img_pixel1 = img_pixel1*1e10
    img_pixel2 = img_pixel2.astype(float)
    img_pixel2 = img_pixel2*1e10
    nodeids = g.add_grid_nodes(img_pixel1.shape)
    print(img_pixel1.shape)
    g.add_grid_tedges(nodeids,img_pixel1,img_pixel2)
    structure_left = np.array([[0,0,0],
                               [0,0,1],
                               [0,0,0]])
    g.add_grid_edges(nodeids,weights=left,structure=structure_left,symmetric=False)
    structure_right = np.array([[0,0,0],
                               [1,0,0],
                               [0,0,0]])
    g.add_grid_edges(nodeids,weights=right,structure=structure_right,symmetric=False)
    structure_up = np.array([[0,0,0],
                               [0,0,0],
                               [0,1,0]])
    g.add_grid_edges(nodeids,weights=up,structure=structure_up,symmetric=False)
    structure_down = np.array([[0,1,0],
                               [0,0,0],
                               [0,0,0]])
    g.add_grid_edges(nodeids,weights=down,structure=structure_down,symmetric=False)
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)

    # The labels should be 1 where sgm is False and 0 otherwise.
    img2 = np.int_(np.logical_not(sgm))
    src_mask = img2.astype(np.uint8)
    dst_mask = np.logical_not(img2).astype(np.uint8)
    src_mask = np.stack((src_mask,src_mask,src_mask),axis=-1)
    dst_mask = np.stack((dst_mask,dst_mask,dst_mask),axis=-1)

    src = src*src_mask
    dst = dst*dst_mask

    result =src+dst

    cv2.imshow("src",src)
    cv2.imshow("dst",dst)
    cv2.imshow("result",result)
    cv2.imwrite("result.jpg",result)
    print(type(img2))
    # Show the result.
    from matplotlib import pyplot as ppl

    ppl.imshow(img2)
    ppl.show()
    cv2.waitKey(0)

