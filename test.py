import os
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, models
from model import *
import pretrainedmodels

#DATA_ROOT = './datasets/xuelang_round1_test_a_20180709'
#DATA_ROOT = './datasets/xuelang_round1_test_b'
DATA_ROOT = './datasets/xuelang_round2_test_a_20180809'
RESULT_FILE = 'result.csv'

def test_and_generate_result(epoch_num, model_name='resnet101', img_size=320, is_multi_gpu=False):
    data_transform = transforms.Compose([
        transforms.Resize(img_size, Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize([0.53744068, 0.51462684, 0.52646497], [0.06178288, 0.05989952, 0.0618901])
    ])

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    is_use_cuda = torch.cuda.is_available()

    if  'resnet152' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = resnet152.MyResNet152(model_ft)
        del model_ft
    elif 'resnet50' == model_name.split('_')[0]:
        model_ft = models.resnet50(pretrained=True)
        my_model = resnet50.MyResNet50(model_ft)
        del model_ft
    elif 'resnet101' == model_name.split('_')[0]:
        model_ft = models.resnet101(pretrained=True)
        my_model = resnet101.MyResNet101(model_ft)
        del model_ft
    elif 'densenet121' == model_name.split('_')[0]:
        model_ft = models.densenet121(pretrained=True)
        my_model = densenet121.MyDenseNet121(model_ft)
        del model_ft
    elif 'densenet169' == model_name.split('_')[0]:
        model_ft = models.densenet169(pretrained=True)
        my_model = densenet169.MyDenseNet169(model_ft)
        del model_ft
    elif 'densenet201' == model_name.split('_')[0]:
        model_ft = models.densenet201(pretrained=True)
        my_model = densenet201.MyDenseNet201(model_ft)
        del model_ft
    elif 'densenet161' == model_name.split('_')[0]:
        model_ft = models.densenet161(pretrained=True)
        my_model = densenet161.MyDenseNet161(model_ft)
        del model_ft
    elif 'ranet' == model_name.split('_')[0]:
        my_model = ranet.ResidualAttentionModel_92()
    elif 'senet154' == model_name.split('_')[0]:
        model_ft = pretrainedmodels.models.senet154(num_classes=1000, pretrained='imagenet')
        my_model = MySENet154(model_ft)
        del model_ft
    else:
        raise ModuleNotFoundError

    state_dict = torch.load('./checkpoint/' + model_name + '/Models_epoch_' + epoch_num + '.ckpt', map_location=lambda storage, loc: storage.cuda())['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        my_model.load_state_dict(new_state_dict)
    else:
        my_model.load_state_dict(state_dict)

    if is_use_cuda:
        my_model = my_model.cuda()
    my_model.eval()

    with open(os.path.join('checkpoint', model_name, model_name+'_'+str(img_size)+'_'+RESULT_FILE), 'w', encoding='utf-8') as fd:
        fd.write('filename|defect,probability\n')
        test_files_list = os.listdir(DATA_ROOT)
        for _file in test_files_list:
            file_name = _file
            if '.jpg' not in file_name:
                continue
            file_path = os.path.join(DATA_ROOT, file_name)
            img_tensor = data_transform(Image.open(file_path).convert('RGB')).unsqueeze(0)
            if is_use_cuda:
                img_tensor = Variable(img_tensor.cuda(), volatile=True)
            output = F.softmax(my_model(img_tensor), dim=1)
            defect_prob = round(output.data[0, 1], 6)
            if defect_prob == 0.:
                defect_prob = 0.000001
            elif defect_prob == 1.:
                defect_prob = 0.999999
            target_str = '%s,%.6f\n' % (file_name, defect_prob)
            fd.write(target_str)

def test_and_generate_result_round2(epoch_num, model_name='resnet101', img_size=320, is_multi_gpu=False):
    data_transform = transforms.Compose([
        transforms.Resize(img_size, Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize([0.53744068, 0.51462684, 0.52646497], [0.06178288, 0.05989952, 0.0618901])
    ])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    is_use_cuda = torch.cuda.is_available()

    if  'resnet152' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = resnet152.MyResNet152(model_ft)
        del model_ft
    elif 'resnet152-r2' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = resnet152.MyResNet152_Round2(model_ft)
        del model_ft
    elif 'resnet152-r2-2o' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = resnet152.MyResNet152_Round2_2out(model_ft)
        del model_ft
    elif 'resnet152-r2-2o-gmp' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = resnet152.MyResNet152_Round2_2out_GMP(model_ft)
        del model_ft
    elif 'resnet152-r2-hm-r1' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = resnet152.MyResNet152_Round2_HM_round1(model_ft)
        del model_ft
    elif 'resnet50' == model_name.split('_')[0]:
        model_ft = models.resnet50(pretrained=True)
        my_model = resnet50.MyResNet50(model_ft)
        del model_ft
    elif 'resnet101' == model_name.split('_')[0]:
        model_ft = models.resnet101(pretrained=True)
        my_model = resnet101.MyResNet101(model_ft)
        del model_ft
    elif 'densenet121' == model_name.split('_')[0]:
        model_ft = models.densenet121(pretrained=True)
        my_model = densenet121.MyDenseNet121(model_ft)
        del model_ft
    elif 'densenet169' == model_name.split('_')[0]:
        model_ft = models.densenet169(pretrained=True)
        my_model = densenet169.MyDenseNet169(model_ft)
        del model_ft
    elif 'densenet201' == model_name.split('_')[0]:
        model_ft = models.densenet201(pretrained=True)
        my_model = densenet201.MyDenseNet201(model_ft)
        del model_ft
    elif 'densenet161' == model_name.split('_')[0]:
        model_ft = models.densenet161(pretrained=True)
        my_model = densenet161.MyDenseNet161(model_ft)
        del model_ft
    elif 'ranet' == model_name.split('_')[0]:
        my_model = ranet.ResidualAttentionModel_92()
    elif 'senet154' == model_name.split('_')[0]:
        model_ft = pretrainedmodels.models.senet154(num_classes=1000, pretrained='imagenet')
        my_model = MySENet154(model_ft)
        del model_ft
    else:
        raise ModuleNotFoundError

    state_dict = torch.load('./checkpoint/' + model_name + '/Models_epoch_' + epoch_num + '.ckpt', map_location=lambda storage, loc: storage.cuda())['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        my_model.load_state_dict(new_state_dict)
    else:
        my_model.load_state_dict(state_dict)

    if is_use_cuda:
        my_model = my_model.cuda()
    my_model.eval()

    with open(os.path.join('checkpoint', model_name, model_name+'_'+str(img_size)+'_'+RESULT_FILE), 'w', encoding='utf-8') as fd:
        fd.write('filename|defect,probability\n')
        test_files_list = os.listdir(DATA_ROOT)
        for _file in test_files_list:
            file_name = _file
            if '.jpg' not in file_name:
                continue
            file_path = os.path.join(DATA_ROOT, file_name)
            img_tensor = data_transform(Image.open(file_path).convert('RGB')).unsqueeze(0)
            if is_use_cuda:
                img_tensor = Variable(img_tensor.cuda(), volatile=True)
            _, output, _ = my_model(img_tensor)
            #output = my_model(img_tensor)
            output = F.softmax(output, dim=1)
            for k in range(11):
                defect_prob = round(output.data[0, k], 6)
                if defect_prob == 0.:
                    defect_prob = 0.000001
                elif defect_prob == 1.:
                    defect_prob = 0.999999
                target_str = '%s,%.6f\n' % (file_name + '|' + ('norm' if 0 == k else 'defect_'+str(k)), defect_prob)
                fd.write(target_str)

if __name__ == '__main__':
    #test_and_generate_result('10', 'resnet152_2018073100', 416, True)
    #test_and_generate_result('2', 'resnet50_2018072500', 416, True)
    #test_and_generate_result('7','resnet101_2018072600', 416, True)
    #test_and_generate_result_round2('14','resnet152-r2-2o-gmp_2018081600', 600, True)
    #test_and_generate_result_round2('14', 'resnet152-r2-2o_2018081300', 600, True)
    #test_and_generate_result('12', 'densenet161_new_stra', 352, True)
    #test_and_generate_result('25', 'ranet_2018072400', 416, True)
    #test_and_generate_result('8', 'senet154_2018072500', 416, True)
    test_and_generate_result_round2('9','resnet152-r2-hm-r1_2018082000', 576, True)
