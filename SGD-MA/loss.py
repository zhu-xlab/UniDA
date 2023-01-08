import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for labels and noises.
    """

    def __init__(self, method):
        """
        Class initializer.
        """
        super().__init__()
        self.how = method
        if method == 'kld':
            self.loss = nn.KLDivLoss(reduction='batchmean')
            self.log_softmax = nn.LogSoftmax(dim=1)
        elif method == 'l1':
            self.loss = nn.L1Loss()
        elif method == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(method)

    def forward(self, output, target):
        """
        Forward propagation.
        """
        if self.how == 'kld':
            output = self.log_softmax(output)
        return self.loss(output, target)


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
        
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # type: Tensor
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


# single layer
def MMD_S(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    num_class = int(source.size()[0])
    loss_a = 0.0
    loss = 0.0
    for ii in range(num_class):
        source1 = source[ii]
        target1 = target[ii]
        source1 = source1.view(source1.size(0), -1)
        target1 = target1.view(target1.size(0), -1)
        kernels = guassian_kernel(source1.clone(), target1.clone(), kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        batch_size = int(source1.size()[0])
        for i in range(batch_size):
            s1, s2 = i, (i+1)%batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
            loss1 = loss / float(batch_size)
        loss_a += loss1
    return loss_a/num_class
