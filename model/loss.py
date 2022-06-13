import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class MyLoss(nn.Module):
    def __init__(self, grid_cells, bbox_num, l_coord, l_noobj):
        super(MyLoss, self).__init__()
        self.grid_cells = grid_cells
        self.bbox_num = bbox_num
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        # box is a tensor:[[x1,y1,x2,y2],[],...]
        n = box1.size(0)
        m = box2.size(0)
        # lt为左上角最大的坐标，tensor(n,m,2)
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(n, m, 2),  # (N, 2) -> (N, 1, 2) -> (N, M, 2)
            box2[:, :2].unsqueeze(0).expand(n, m, 2)  # (N, 2) -> (1, M, 2) -> (N, M, 2)
        )
        # rb为右下角最小的坐标，tensor(n,m,2)
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(n, m, 2),
            box2[:, 2:].unsqueeze(0).expand(n, m, 2)
        )
        wh = rb - lt  # wh: tensor(n,m,2)
        wh[wh < 0] = 0  # wh<0意味着没有交集
        intersection = wh[:, :, 0] * wh[:, :, 1]
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        area1 = area1.unsqueeze(1).expand_as(intersection)
        area2 = area2.unsqueeze(0).expand_as(intersection)

        iou = intersection / (area2 + area1 - intersection)
        return iou

    def forward(self, pred_tensor, ground_truth):
        # pred_tensor 和 grouond_truth的size均为[batchsize,7,7,30]
        batch_size = pred_tensor.size()[0]
        contain_obj_mask = ground_truth[:, :, :, 4] > 0
        no_obj_mask = ground_truth[:, :, :, 4] == 0
        # 重新把两个mask扩展成为输入一样的尺寸 (batchsize, 7, 7) -> (batchsize, 7, 7, 1) -> (batchsize, 7, 7, 30)
        contain_obj_mask.unsqueeze(-1).expand_as(ground_truth)
        no_obj_mask.unsqueeze(-1).expand_as(ground_truth)

        pred_contain_obj = pred_tensor[contain_obj_mask].view(-1, 30)
        box_pred_contain_obj = pred_contain_obj[:, :10].contiguous().view(-1, 5)
        class_pred_contain_obj = pred_contain_obj[:, 10:]

        ground_contain_obj = ground_truth[contain_obj_mask].view(-1, 30)  # (N_pred_contain_obj, 30)
        box_ground_contain_obj = ground_contain_obj[:, 0:10].contiguous().view(-1, 5)  # (2*N_pred_contain_obj, 5)
        class_ground_contain_obj = ground_contain_obj[:, 10:]  # (N_pred_contain_obj,20]
        '''
        compute not contain obj loss
        '''
        pred_no_obj = pred_tensor[no_obj_mask].view(-1, 30)  # (N_pred_no_obj, 30)
        ground_no_obj = ground_truth[no_obj_mask].view(-1, 30)
        no_obj_mask1 = torch.ByteTensor(pred_no_obj.size())  # a new mask to select confidence only

        no_obj_mask1.zero_()
        no_obj_mask1[:, 4] = 1
        no_obj_mask1[:, 9] = 1  # confidence1 and confidence2 set 1
        confidence_pred_no_obj = pred_no_obj[no_obj_mask1]
        confidence_ground_no_obj = ground_no_obj[no_obj_mask1]

        # no obj loss
        no_obj_loss = F.mse_loss(confidence_pred_no_obj, confidence_ground_no_obj, size_average=False)
        '''
        compute contain obj loss
        '''
        contain_obj_response_box_mask = torch.ByteTensor(box_ground_contain_obj.size())
        contain_obj_response_box_mask.zero_()  # mask for bbox responsible for object
        contain_obj_irresponse_box_mask = torch.ByteTensor(box_ground_contain_obj.size())
        contain_obj_irresponse_box_mask.zero_()  # mask for bbox not responsible for object

        boxIOU_pred_ground = torch.zeros(box_ground_contain_obj.size())
        for i in range(0, box_ground_contain_obj.size()[0], 2):  # for every grid that should contain object
            box1_pred_contain_obj = box_pred_contain_obj[i, i + 2]  # box1_pred_contain_obj.size() = (2, 5)
            box1_pcon_xyxy = Variable(
                torch.FloatTensor(box1_pred_contain_obj.size()))  # [x, y, w, h] -> [x1, y1, x2, y2]
            box1_pcon_xyxy[:, :2] = box1_pred_contain_obj[:, :2] - 0.5 * self.grid_cells * box1_pred_contain_obj[:, 2:4]
            box1_pcon_xyxy[:, 2:4] = box1_pred_contain_obj[:, :2] + 0.5 * self.grid_cells * box1_pred_contain_obj[:,
                                                                                            2:4]
            box2_ground_contain_obj = box_ground_contain_obj[i].view(-1, 5)  # 因为ground_truth的两个box是一样的，所以只需要取一个
            box2_gcon_xyxy = Variable(torch.FloatTensor(box2_ground_contain_obj.size()))
            box2_gcon_xyxy[:, :2] = box2_gcon_xyxy[:, :2] - 0.5 * self.grid_cells * box2_ground_contain_obj[:, 2: 4]
            box2_gcon_xyxy[:, 2:4] = box2_gcon_xyxy[:, :2] - 0.5 * self.grid_cells * box2_ground_contain_obj[:, 2: 4]

            iou = self.compute_iou(box1_pcon_xyxy[:, :4], box2_gcon_xyxy[:, :4])  # iou:(2,1)
            max_iou, max_index = iou.max(0)  # iou.max() only returns value, iou.max(0) returns value, index
            contain_obj_response_box_mask[i + max_index] = 1
            contain_obj_irresponse_box_mask[i + 1 - max_index] = 1
            boxIOU_pred_ground[i + max_index, 4] = max_iou
        response_box_pred_contain_obj = box_pred_contain_obj[contain_obj_response_box_mask].view(-1, 5)
        response_box_ground_contain_obj = box_ground_contain_obj[contain_obj_irresponse_box_mask].view(-1, 5)
        response_box_IOU = boxIOU_pred_ground[contain_obj_response_box_mask].view(-1, 5)
        coordinates_loss = F.mse_loss(response_box_pred_contain_obj[:, :2], response_box_ground_contain_obj[:, :2],
                                      size_average=False) + F.mse_loss(
            torch.sqrt(response_box_pred_contain_obj[:, 2:4]), torch.sqrt(response_box_ground_contain_obj[:, 2:4]),
            size_average=False)
        response_loss = F.mse_loss(response_box_pred_contain_obj[:, 4], response_box_IOU[:, 4], size_average=False)
        # irresponse loss
        irresponse_box_pred_contain_obj = box_pred_contain_obj[contain_obj_irresponse_box_mask].view(-1, 5)
        irresponse_box_ground_contain_obj = box_ground_contain_obj[contain_obj_irresponse_box_mask].view(-1, 5)
        irresponse_box_ground_contain_obj[:, 4] = 0
        irresponse_loss = F.mse_loss(irresponse_box_pred_contain_obj[:, 4], irresponse_box_ground_contain_obj[:, 4],
                                     size_average=False)
        # class loss
        class_loss = F.mse_loss(class_pred_contain_obj, class_ground_contain_obj, size_average=False)
        return (
                       self.l_coord * coordinates_loss + response_loss + self.l_noobj * no_obj_loss + class_loss + response_loss + irresponse_loss) / batch_size
