import torch
import tensorflow as tf

def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return alpha*tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.log(y_pred_1), axis = -1)) + beta*tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis = -1))
    return loss

# def symmetric_cross_entropy(alpha, beta):
#     def loss(y_true, y_pred):
#         print("y_true")
#         print(y_true)
#
#         print("y_pred")
#         print(y_pred)
#
#         print("________________________")
#         y_true_1 = y_true
#         y_pred_1 = y_pred
#
#         y_true_2 = y_true
#         y_pred_2 = y_pred
#
#         y_pred_1 = torch.clamp(y_pred_1, 1e-7, 1.0)
#         y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0)
#
#         print("y_true_2")
#         print(y_true)
#
#         print("y_pred_1")
#         print(y_pred_1)
#
#         exit()
#
#         return alpha * torch.mean(-torch.sum(y_true_1 * torch.log(y_pred_1), dim=-1)) + beta * torch.mean(-torch.sum(y_pred_2 * torch.log(y_true_2), dim=-1))
#     return loss
