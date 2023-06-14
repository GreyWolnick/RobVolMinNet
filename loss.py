import torch

def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        # print("y_true")
        # print(y_true)
        #
        # print("y_pred")
        # print(y_pred)
        #
        # print("________________________")
        y_true_1 = y_true
        y_pred_temp = y_pred

        y_true_temp = y_true
        y_pred_2 = y_pred

        # y_pred_1 = torch.clamp(y_pred_1, 1e-7, 1.0)
        # y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0)
        # y_pred_1 = torch.clamp(y_pred_1, min=1e-7, max=1.0)
        # y_true_2 = torch.clamp(y_true_2, min=1e-4, max=1.0)
        y_pred_1 = torch.clamp(y_pred_temp, min=0.5, max=1.0)
        y_true_2 = torch.clamp(y_true_temp, min=0.6, max=1.0)

        print(y_true_temp)
        print("y_true_2")
        print(torch.clamp(y_true_temp.float(), min=0.6, max=1.0))

        print("y_pred_1")
        print(y_pred_1)

        exit()

        return alpha * torch.mean(-torch.sum(y_true_1 * torch.log(y_pred_1), dim=-1)) + beta * torch.mean(-torch.sum(y_pred_2 * torch.log(y_true_2), dim=-1))
    return loss
