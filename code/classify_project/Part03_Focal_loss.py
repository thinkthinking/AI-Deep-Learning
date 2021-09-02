import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=30, alpha=0.25, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(torch.ones(class_num, 1)*alpha)

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        # 不论inputs是多少维的，新建的class_mask的type和device都与inputs保持一致
        class_mask = inputs.data.new(N, C).fill_(0)
        # tensor 张量封装成Variable变量
        class_mask = Variable(class_mask)

        #在PyTorch中view函数作用为重构张量的维度，相当于numpy中的resize()的功能
        # 变成n行1列
        ids = targets.view(-1, 1)

        # scatter() 和 scatter_() 的作用是一样的，只不过 scatter() 不会直接修改原来的 Tensor，而 scatter_() 会
        # PyTorch 中，一般函数加下划线代表直接在原来的 Tensor 上修改
        # scatter(dim, index, src)的参数有3个
        # dim：沿着哪个维度进行索引
        # index：用来scatter的元素索引
        # src：用来scatter的源元素，可以是一个标量或一个张量
        class_mask.scatter_(1, ids.data, 1.)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()


        # ids.data.view(-1)  变成1行N列
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss




if __name__ == '__main__':
    # m = 3
    # n = 2
    # inputs = torch.randn(m, n)
    # new_inputs = inputs.new()
    # new_inputs2 = torch.Tensor.new(inputs).fill_(0)
    # print('---------inputs--------',inputs)
    #
    # print("--------new_inputs-------",new_inputs)
    # print(new_inputs.type(), inputs.type())
    # print("----------new_inputs--------",new_inputs2)




    # import torch
    # tt1 = torch.tensor([-0.3623, -0.6115, 0.7283, 0.4699, 2.3261, 0.1599])
    # result = tt1.view(-1, 1)
    # print("------result--------",result)




    import torch

    mini_batch = 4
    out_planes = 6
    out_put = torch.rand(mini_batch, out_planes)
    softmax = torch.nn.Softmax(dim=1)
    out_put = softmax(out_put)

    print(out_put)
    label = torch.tensor([1, 3, 3, 2])
    one_hot_label = torch.zeros(mini_batch, out_planes).scatter_(0, label.unsqueeze(1), 1)
    print(one_hot_label)
