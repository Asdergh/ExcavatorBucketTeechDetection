import torch as th
import json as js
import os
from torchvision.io import read_image
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize
)
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert

_transforms_ = {
    "resize": Resize,
    "normalize": Normalize
}

class TeethsDetectionSet(Dataset):

    def __init__(self, confs: dict) -> None:

        self.params = confs
        self._root_ = os.path.join(
            self.params["source_path"], 
            self.params["split"]
        )
        
        self.tf = None
        self.resize = False
        if "transforms" in self.params:
            tf_confs = self.params["transforms"]
            if "resize" in tf_confs:
                self.new_size = tf_confs["resize"]["size"]
                self.resize = True

            if len(tf_confs) != 0:
                tfs = []
                for tf in tf_confs:
                    tf = _transforms_[tf](**tf_confs[tf])
                    tfs.append(tf)
                
                self.tf = Compose(tfs)
            
            
        
        annots = os.path.join(self._root_, "_annotations.coco.json")
        with open(annots, "r") as annots_f:
            annots = js.load(annots_f)
        
        self.images = annots["images"]
        self.targets = annots["annotations"]
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[th.Tensor]:

        img_confs = self.images[idx]
        tar_confs = self.targets[idx]

        img = read_image(os.path.join(
            self._root_, 
            img_confs["file_name"]
        )).to(th.float32) / 255.0
        bbox = th.Tensor(tar_confs["bbox"])

        if self.tf is not None:

            img = self.tf(img)
            if self.resize:
                
                scale_x = self.new_size[0] / img_confs["width"]
                scale_y = self.new_size[1] / img_confs["height"]
                bbox[0] *= scale_x
                bbox[2] *= scale_x
                bbox[1] *= scale_y
                bbox[3] *= scale_y
        
        return (img, bbox)
            



if __name__ == "__main__":

    dataset = TeethsDetectionSet({
        "source_path": "C:\\Users\\1\\Downloads\\tooth excava.v2i.coco",
        "split": "train",
        "transforms": {
            "resize": {
                "size": (252, 252),
                "max_size": None,
                "antialias": True
            },
            "normalize": {
                "mean": [0.67, 0.67, 0.67],
                "std": [0.45, 0.45, 0.45]
            }
        },
    })
    loader = DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True
    )
    sample, bboxes = next(iter(loader))
    bboxes = box_convert(bboxes, in_fmt="xywh", out_fmt="xyxy")
    print(bboxes)
    print(bboxes[:, 0].size(), bboxes.size())
    sample = draw_bounding_boxes(sample[0], bboxes[0].unsqueeze(dim=0), width=5)
    sample = sample.permute(1, 2, 0).numpy()
    cv2.imshow("test", sample)
    cv2.waitKey(0)
    # for frame in sample:

    #     cv2.imshow("test", frame)
    #     if cv2.waitKey(1) == ord("q"):
    #         break
    
        