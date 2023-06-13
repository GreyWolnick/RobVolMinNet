import torch

def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        print("y_true")
        print(y_true)

        print("y_pred")
        print(y_pred)
        exit()
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = torch.clamp(y_pred_1, 1e-7, 1.0)
        y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0)

        return alpha * torch.mean(-torch.sum(y_true_1 * torch.log(y_pred_1), dim=-1)) + beta * torch.mean(-torch.sum(y_pred_2 * torch.log(y_true_2), dim=-1))
    return loss
