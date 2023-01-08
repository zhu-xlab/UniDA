from data import *
from net import *
from lib import *
import datetime
import models_DG
from tqdm import tqdm
from torch import optim
from util import utils as ut
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import utils as gen_utils
import loss as gen_loss
from vgg import Vgg16
cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

# output_device = None
# global output_device
import os
os.environ["CUDA VISIBLE_DEVICES"] = "0"

if args.misc.gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    gpu_ids = select_GPUs(args.misc.gpus)
    output_device = gpu_ids[0]

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

log_dir = '{}/{}'.format(args.log.root_dir,now)

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc
}


class ClassNet(nn.Module):
    def __init__(self):
        super(ClassNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f1, feature, f2, predict_prob = self.classifier(f)

        return f1, feature, f2, predict_prob


# ===================domain adaptation
dataset = args.data.dataset.name
out_path = '/Path/SGD-MA/output/'
num_epochs =  args.data.dataloader.num_epochs
viz_every = args.data.dataloader.viz_every
save_every = args.data.dataloader.save_every

images_size = 256
num_channels = 3
nx = images_size*images_size*num_channels
source_num_classes = len(source_classes)
target_num_classes = len(target_classes)
num_noises = 10
dec_layers = 5
alpha = 2
beta = 1
gama = 0
STYLE_WEIGHT = 1e0
CONTENT_WEIGHT = 1e0


cls_path = args.Data_G.cls_path
Classifier_pre = ClassNet()    
ut.load_checkpoints(Classifier_pre, cls_path)
gen_network_pre = gen_utils.init_generator(source_num_classes, num_channels, num_noises)
vgg = Vgg16()

Classifier = nn.DataParallel(Classifier_pre, device_ids=gpu_ids, output_device=output_device).train(False)
gen_network = nn.DataParallel(gen_network_pre, device_ids=gpu_ids, output_device=output_device).train(True)
vgg = nn.DataParallel(vgg, device_ids=gpu_ids, output_device=output_device).train(False)

path_loss = os.path.join(out_path, 'loss_gen.txt')
dir_model = os.path.join(out_path, 'generator')
path_model = None
os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)

best_acc = 0.0


discriminator_da = AdversarialNetwork(256)
discriminator_separate = AdversarialNetwork(256)

feature_extractor = nn.DataParallel(Classifier_pre.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
classifier = nn.DataParallel(Classifier_pre.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator_da = nn.DataParallel(discriminator_da, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator_separate = nn.DataParallel(discriminator_separate, device_ids=gpu_ids, output_device=output_device).train(True)
# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator_da.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_discriminator_separate = OptimWithSheduler(
    optim.SGD(discriminator_separate.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)

global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 0
n_classes = gen_network_pre.num_classes
n_noises = gen_network_pre.num_noises
batch_size = args.data.dataloader.batch_size
while global_step < args.train.min_step:

    iters = zip(source_train_dl, target_train_dl)
    epoch_id += 1

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(output_device)
        label_target = label_target.to(output_device)
        label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source = im_source.to(output_device)
        im_target = im_target.to(output_device)

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        domain_prob_discriminator_source = discriminator_da.forward(feature_source)
        domain_prob_discriminator_target = discriminator_da.forward(feature_target)

        domain_prob_discriminator_source_separate = discriminator_separate.forward(feature_source.detach())
        domain_prob_discriminator_target_separate = discriminator_separate.forward(feature_target.detach())

        source_share_weight = get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s, domain_temperature=1.0, class_temperature=10.0)
        source_share_weight = normalize_weight(source_share_weight)
        target_share_weight = get_target_share_weight(domain_prob_discriminator_target_separate, fc2_t, domain_temperature=1.0, class_temperature=1.0)
        target_share_weight = normalize_weight(target_share_weight)
            
        # ==============================compute loss
        adv_loss = torch.zeros(1, 1).to(output_device)
        adv_loss_separate = torch.zeros(1, 1).to(output_device)

        tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)
        tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)

        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))
        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate, torch.zeros_like(domain_prob_discriminator_target_separate))

        # ============================== cross entropy loss
        # source_label = torch.topk(label_source, 1)[1].squeeze(1)
        ce = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label_source)
        ce = torch.mean(ce, dim=0, keepdim=True)

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_discriminator_separate]):
            loss = ce + adv_loss + adv_loss_separate
            loss.backward()

        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            logger.add_scalar('adv_loss', adv_loss, global_step)
            # logger.add_scalar('ce', ce, global_step)
            logger.add_scalar('adv_loss_separate', adv_loss_separate, global_step)
            # logger.add_scalar('acc_train', acc_train, global_step)

        if global_step % args.test.test_interval == 0:

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            with TrainingModeManager([feature_extractor, classifier, discriminator_separate], train=False) as mgr, \
                 Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax', 'target_share_weight']) as target_accumulator, \
                 torch.no_grad():

                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing')):
                    im = im.to(output_device)
                    label = label.to(output_device)

                    feature = feature_extractor.forward(im)
                    feature, __, before_softmax, predict_prob = classifier.forward(feature)
                    domain_prob = discriminator_separate.forward(__)

                    target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                                  class_temperature=1.0)

                    for name in target_accumulator.names:
                        globals()[name] = variable_to_numpy(globals()[name])

                    target_accumulator.updateData(globals())

            for x in target_accumulator:
                globals()[x] = target_accumulator[x]

            def outlier(each_target_share_weight):
                return each_target_share_weight < args.test.w_0

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]

            for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label,
                                                                                 target_share_weight):
                if each_label in common_classes:
                    counters[each_label].Ntotal += 1.0
                    each_pred_id = np.argmax(each_predict_prob)
                    if not outlier(each_target_share_weight[0]) and each_pred_id == each_label:
                        counters[each_label].Ncorrect += 1.0
                else:
                    counters[-1].Ntotal += 1.0
                    if outlier(each_target_share_weight[0]):
                        counters[-1].Ncorrect += 1.0

            acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
            acc_test = torch.ones(1, 1) * np.mean(acc_tests)

            path_acc = './acc.txt'
            with open(path_acc, 'a') as f:
                f.write('\t{}\n'.format(acc_tests))
                f.write('\t{}\n'.format(acc_test))

            logger.add_scalar('acc_test', acc_test, global_step)
            clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'discriminator': discriminator_da.state_dict() if not isinstance(discriminator_da, Nonsense) else 1.0,
                'discriminator_separate': discriminator_separate.state_dict(),
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)