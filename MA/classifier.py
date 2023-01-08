import os

import numpy as np
import torch
from torch import optim

from data import *
from net import *
from lib import *
import datetime
from tqdm import tqdm
from util import utils as ut

DEVICE = None

class cla_network(nn.Module):
    def __init__(self):
        super(cla_network, self).__init__()
        self.feature_extractor = ResNet50Fc(args.model.pretrained_resnet)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)


    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)

        return y


def classifier(network, loss_cl):
    network.train()
    num_batches_source = args.data.dataloader.num_batches_source
    optimizer = torch.optim.SGD(network.parameters(), lr = args.train.lr, momentum=args.train.momentum,
                                    weight_decay=args.train.weight_decay, nesterov=args.train.nesterov)
    train_source = tqdm(source_train_dl, desc='\r')
    #print(len(source_train_dl.dataset))
    batch_size = args.data.dataloader.batch_size
    loss_cla = []
    for i, (im_source, label_source) in enumerate(train_source):
        im_source = im_source.to(DEVICE)
        label_source = label_source.to(DEVICE)
        optimizer.zero_grad()
        output = network(im_source)
        loss  = loss_cl(output, label_source.long())
        loss.backward()
        optimizer.step()

        loss_cla.append(loss.item())

    return network


def validation(network, loss_cl):
    network.eval()
    tbar_source = tqdm(source_test_dl, desc='\r')
    test_loss = 0.0
    correct = 0
    for i, (image_test, label_test) in enumerate(tbar_source):
        image_test = image_test.to(DEVICE)
        label_test = label_test.to(DEVICE)
        with torch.no_grad():
            output_test = network(image_test)
        loss = loss_cl(output_test, label_test)
        test_loss += loss.item()
        tbar_source.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
        #pred = output_test.data.cpu().numpy()
        pred = output_test.argmax(axis=1)
        correct += pred.eq(label_test.data.view_as(pred)).sum()
    corr = 100. * correct / len(source_test_dl.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(source_test_dl.dataset), corr))
    return test_loss, corr




def main(out_path, index=0):
    """
    Main function for training a generator.
    """
    global DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ut.set_seed(seed=2019 + index)

    # num_epochs = args.data.dataloader.num_epochs
    num_epochs_source = args.data.dataloader.num_epochs_source
    save_every = 50


    assert num_epochs_source >= save_every
    

    cls_network = cla_network().to(DEVICE)
    
    path_loss = os.path.join(out_path, 'loss-cla.txt')
    dir_model = os.path.join(out_path, 'classifier')
    dir_model_best = os.path.join(out_path, 'best')
    path_model = None

    os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)
    with open(path_loss, 'w') as f:
        f.write('Epoch\tLossSum\tAccuracy\n')

    loss_cl = nn.CrossEntropyLoss().to(DEVICE)
    best_corr = 0.0
    
    for epoch in range(1, num_epochs_source + 1):
        model_cla = classifier(cls_network, loss_cl)
        test_loss, corr = validation(model_cla, loss_cl)

        with open(path_loss, 'a') as f:
            f.write(str(epoch))
            f.write('\t'+ str(test_loss))
            f.write('\t' + str(corr) + '\n')

        if epoch % save_every == 0:
            path = os.path.join(dir_model,'{}.pth.tar'.format(epoch))
            ut.save_checkpoints(model_cla, path)
            path_model = path

        new_corr = corr
        if new_corr > best_corr:
            best_corr = new_corr
            path_best = os.path.join(dir_model_best, 'best.pth.tar')
            ut.save_checkpoints(model_cla, path_best)
    print('Finished training the classifier (index={})'.format(index))

if __name__ == "__main__":
    main('Path/to/pretrained/')

