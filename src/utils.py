import torch as th



def calculate_iou(target_bbox: th.Tensor, gr_bbox: th.Tensor):

    inner_bbox = []
    for (tar_box, gr_box) in zip(target_bbox, gr_bbox):
        
        a = max(tar_box[0].item(), gr_box[0].item())
        b = max(tar_box[1].item(), gr_box[1].item())
        c = min(tar_box[2].item(), gr_box[2].item())
        d = min(tar_box[3].item(), gr_box[3].item())
        inner_bbox.append([a, b, c, d])

    iner_bbox = th.Tensor(inner_bbox)
    iner_area = (iner_bbox[:, 2] - iner_bbox[:, 0]) * (iner_bbox[:, 3] - iner_bbox[:, 1])
    tar_area = (target_bbox[:, 2] - target_bbox[:, 0]) * (target_bbox[:, 3] - target_bbox[:, 1])
    gr_area = (gr_bbox[:, 2] - gr_bbox[:, 0]) * (gr_bbox[:, 3] - gr_bbox[:, 1])
    
    return iner_area / (tar_area + gr_area)

