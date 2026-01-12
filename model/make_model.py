import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from .backbones.resnet import ResNet18
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
#####
class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0
        #选择模型
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        
        #if pretrain_choice == 'imagenet':
            #self.base.load_param(model_path)
            #print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier_train1_ = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)   
            self.classifier_train2_ = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.classifier_train3_ = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)              
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier_train1_ = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)    
            self.classifier_train2_ = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)  
            self.classifier_train3_ = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)        
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier_train1_ = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.classifier_train2_ = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.classifier_train3_ = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier_train1_ = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.classifier_train2_ = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.classifier_train3_ = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        
            #self.classifier_train1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            #self.classifier_train1.apply(weights_init_classifier)
            #self.classifier_train2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            #self.classifier_train2.apply(weights_init_classifier)
            #self.classifier_train3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            #self.classifier_train3.apply(weights_init_classifier)
        self.classifier_train1 = ResNet18(self.in_planes, self.num_classes)
        self.classifier_train2 = ResNet18(self.in_planes, self.num_classes)
        self.classifier_train3 = ResNet18(self.in_planes, self.num_classes)

        #self.bottleneck3C = nn.BatchNorm1d(self.in_planes*3)
        #self.bottleneck3C.bias.requires_grad_(False)
        #self.bottleneck3C.apply(weights_init_kaiming)

        #self.bottleneck1C = nn.LayerNorm(self.in_planes)
        #self.bottleneck1C.bias.requires_grad_(False)
        #self.bottleneck1C.apply(weights_init_kaiming)

        self.bottleneck1C1 = nn.Sequential(
            nn.LayerNorm(self.in_planes),
            nn.Dropout(0.1),  # 可选，减少过拟合和梯度爆炸
            nn.Linear(self.in_planes, self.in_planes, bias=False)
        )
        self.bottleneck1C2 = nn.Sequential(
            nn.LayerNorm(self.in_planes),
            nn.Dropout(0.1),  # 可选，减少过拟合和梯度爆炸
            nn.Linear(self.in_planes, self.in_planes, bias=False)
        )
        self.bottleneck1C3 = nn.Sequential(
            nn.LayerNorm(self.in_planes),
            nn.Dropout(0.1),  # 可选，减少过拟合和梯度爆炸
            nn.Linear(self.in_planes, self.in_planes, bias=False)
        )

        self.bottleneck3Cto1C = nn.Sequential(
            nn.LayerNorm(self.in_planes*3),
            nn.Dropout(0.1),  # 可选，减少过拟合和梯度爆炸
            nn.Linear(self.in_planes*3, self.in_planes*2, bias=False),
            nn.LayerNorm(self.in_planes*2),
            nn.Dropout(0.1),  # 可选，减少过拟合和梯度爆炸    
            nn.Linear(self.in_planes*2, self.in_planes, bias=False)        
        )
    def forward(self, label=None, x=None, y=None, mode='train'):
        #print("embed层数是多少", self.in_planes)
        device = x.device if x is not None else next(self.parameters()).device
        label = label.to(device)
        if mode == 'train':
            xyc, xt, yt, lorthx_sum, lorthy_sum, lc_sum = self.base(camids=label, x=x, y=y, mode=mode)
            feat1, _, score1 = self.classifier_train1(xyc)#nobn, bn, score
            feat2, _, score2 = self.classifier_train2(xt)
            feat3, _, score3 = self.classifier_train3(yt)
            global_feat = [feat1, feat2, feat3]

            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = []
                cls_score.append(self.classifier_train1_(feat1, label))
                cls_score.append(self.classifier_train2_(feat2, label))
                cls_score.append(self.classifier_train3_(feat3, label))
                #cls_score = [self.classifier_train(f, label) for f in feat]
            else:
                cls_score = [score1, score2, score3]
            return cls_score, global_feat, lorthx_sum, lorthy_sum, lc_sum  # global feature for triplet loss
        #if mode == 'train':暂时只用softmax，不把score做成list↑
            #xyc, xt, yt, lorthx_sum, lorthy_sum, lc_sum = self.base(camids=label, x=x, y=y, mode=mode)
            #global_feat = [xyc, xt, yt]
            #feat = [self.bottleneck1C(f) for f in global_feat]

            #if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                #cls_score = [self.classifier_train(f, label) for f in feat]
            #else:
                #cls_score = [self.classifier_train(f) for f in feat]

            #return cls_score, global_feat, lorthx_sum, lorthy_sum, lc_sum  # global feature for triplet loss
        elif mode == 'test-x':
            xt = self.base(camids=label, x=x, y=None, mode=mode)
            feat2, feat2_bn, _= self.classifier_train2(xt)
            if self.neck_feat == 'after':
                print("Test with feature after BN")
                return feat2_bn
            else:
                print("Test with feature before BN")
                return feat2
            
        elif mode == 'test-y':
            yt = self.base(camids=label, x=None, y=y, mode=mode)
            feat3, feat3_bn, _= self.classifier_train3(yt)
            if self.neck_feat == 'after':
                print("Test with feature after BN")
                return feat3_bn
            else:
                print("Test with feature before BN")
                return feat3
            
        elif mode == 'test-xy':
            xyc = self.base(camids=label, x=x, y=y, mode=mode)
            feat1, feat1_bn, _= self.classifier_train1(xyc)
            if self.neck_feat == 'after':
                print("Test with feature after BN")
                return feat1_bn
            else:
                print("Test with feature before BN")
                return feat1

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))




__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}
"""
def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
"""
def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model

