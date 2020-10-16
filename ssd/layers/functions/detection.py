import torch
from torch.autograd import Function
from torchvision.ops.boxes import batched_nms
from ..box_utils import decode, batch_decode, nms


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms_threshold must be non negative.")
        self.conf_thresh = conf_thresh
        self.variance = cfg["variance"]

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        breakpoint()
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        from time import perf_counter

        # Decode predictions into bboxes.
        decoded_boxes = batch_decode(loc_data, prior_data, self.variance)
        boxes_ = (
            decoded_boxes.unsqueeze(2)
            .expand(-1, num_priors, self.num_classes, 4)
            .contiguous()
        )
        boxes_ = boxes_.view(-1, 4)
        scores_ = conf_preds.contiguous().view(-1)

        rows = torch.arange(num, dtype=torch.long)[:, None]
        cols = torch.arange(self.num_classes, dtype=torch.long)[None, :]
        idxs = rows * self.num_classes + cols
        idxs = idxs.unsqueeze(1).expand(num, num_priors, self.num_classes)
        idxs = idxs.to(scores_).view(-1)
        mask = scores_ > self.conf_thresh
        boxesf = boxes_[mask].contiguous()
        scoresf = scores_[mask].contiguous()
        idxsf = idxs[mask].contiguous()
        start = perf_counter()

        keep = batched_nms(boxesf, scoresf, idxsf, 0.00015)

        print("Batch duration {}".format(perf_counter() - start))
        start = perf_counter()
        boxes_k = boxesf[keep]
        scores_k = scoresf[keep]
        labels = idxsf[keep] % self.num_classes
        batch_index = idxsf[keep] // self.num_classes
        print("Batch duration {}".format(perf_counter() - start))

        start = perf_counter()
        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            # print('decoded boxes ', decoded_boxes)
            # print('conf scores', conf_scores)
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class

                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1
                )
        print("Batch duration {}".format(perf_counter() - start))
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        keep = batched_nms()
        return output, boxes, scores
