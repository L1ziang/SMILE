import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from resnet50_scratch_dag import Resnet50_scratch_dag as resnet50

from facenet_pytorch import MTCNN, training
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
from my_target_models import get_input_resolution, get_model

import torchvision.transforms.functional as TF


from my_utils import crop_and_resize, normalize, ALL_MEANS, ALL_STDS

from tqdm import tqdm

import os
import pickle

class Normalize(torch.nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image_tensor):
        image_tensor = (image_tensor-torch.tensor(self.mean, device=image_tensor.device)[:, None, None])/torch.tensor(self.std, device=image_tensor.device)[:, None, None]
        return image_tensor


def train(model, train_loader, optimizer, epoch, device, prefix=100):
     model.train()
     for batch_idx, (data, target) in enumerate(train_loader):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(data)
          loss = F.cross_entropy(output, target)
          loss.backward()
          optimizer.step()
          if batch_idx % prefix == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())) 

features = {}
def test(args, model, testset, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)

            feature_data = []
            fs = features['fs'].cpu().numpy()
            img_path = testset.imgs[batch_idx]
            feature_data.append({
                'dataset': args.dataset,
                'label': target.item(),
                'features': fs,
                'img_path': img_path
            })
            
            original_path = feature_data[0]['img_path'][0]
            base_dir = os.path.dirname(original_path)
            file_name = os.path.basename(original_path)

            new_file_name = file_name.replace('.jpg', '.latent')
            new_path = os.path.join(base_dir.replace('train', f'eval/{args.arch_name}'), new_file_name)

            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            with open(new_path, 'wb') as f:
                pickle.dump(feature_data, f)

            if args.arch_name == 'sphere20a':
                output = output[0]
            test_loss += 0
            _, predicted = output.max(1)

            correct += predicted.eq(target.view_as(predicted)).sum().item()

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss / len(test_loader.dataset), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
class CropImgForSphereFace:
    def __call__(self, img):

        assert len(img.shape) == 3 or len(img.shape) == 4
        img = TF.resize(img, (256, 256))
        return img[..., 16:226, 38:218]
    
def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    torch.manual_seed(666)
    net = get_model(args.arch_name, device)
    net = net.to(device)

    args.resolution = get_input_resolution(args.arch_name)

    if args.dataset == "vggface2":
        Mean = ALL_MEANS[args.arch_name]
        Std = ALL_STDS[args.arch_name]

        if args.arch_name == 'resnet50':
            T_resize = 360
            test_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(T_resize),
            transforms.CenterCrop(args.resolution),
            Normalize(Mean, Std)
            ])
            def hook_fn(module, input, output):
                features['fs'] = output
            layer_to_hook = net.pool5_7x7_s1
            hook = layer_to_hook.register_forward_hook(hook_fn)

        elif args.arch_name == 'inception_resnetv1_vggface2':
            T_resize = 210
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                Normalize(Mean, Std)
                ])
            def hook_fn(module, input, output):
                features['fs'] = output
            layer_to_hook = net.last_bn
            hook = layer_to_hook.register_forward_hook(hook_fn)

        elif args.arch_name == 'mobilenet_v2':
            T_resize = 360
            RESIZE = 224
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        elif args.arch_name == 'efficientnet_b0':
            T_resize = 360
            RESIZE = 256
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        elif args.arch_name == 'inception_v3':
            T_resize = 360
            RESIZE = 342
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        elif args.arch_name == 'swin_transformer':
            T_resize = 360
            RESIZE = 260
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        elif args.arch_name == 'vgg16':
            T_resize = 360
            RESIZE = 224
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        elif args.arch_name == 'vision_transformer':
            T_resize = 360
            RESIZE = 224
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        totalset = torchvision.datasets.ImageFolder("../vggface2_eval/train", transform=test_transform)

        test_loader = DataLoader(totalset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
    elif args.dataset == "CASIA":
        Mean = ALL_MEANS[args.arch_name]
        Std = ALL_STDS[args.arch_name]
        if args.arch_name == 'inception_resnetv1_casia': 
            T_resize = 225
            test_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(T_resize),
            transforms.CenterCrop(args.resolution),
            Normalize(Mean, Std)
            ])
            def hook_fn(module, input, output):
                features['fs'] = output
            layer_to_hook = net.last_bn
            hook = layer_to_hook.register_forward_hook(hook_fn)
            totalset = torchvision.datasets.ImageFolder("/root/autodl-tmp/CASIA-WebFace_eval/train", transform=test_transform)
        if args.arch_name == 'efficientnet_b0_casia': 
            T_resize = 225
            RESIZE = 256
            test_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(T_resize),
            transforms.CenterCrop(args.resolution),
            transforms.Resize(RESIZE),
            Normalize(Mean, Std)
            ])
            def hook_fn(module, input, output):
                features['fs'] = output
            layer_to_hook = net.avgpool
            hook = layer_to_hook.register_forward_hook(hook_fn)
            totalset = torchvision.datasets.ImageFolder("/root/autodl-tmp/CASIA-WebFace_eval/train", transform=test_transform)
        if args.arch_name == 'sphere20a':
            def custom_transform(image, args):
                if image.size != args.resolution:
                    image = CropImgForSphereFace()(image)
                    image = transforms.Resize(args.resolution)(image)

                return image
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Lambda(lambda img: custom_transform(img, args)),
                Normalize(Mean, Std)
            ])
            def hook_fn(module, input, output):
                features['fs'] = output
            layer_to_hook = net.fc5
            hook = layer_to_hook.register_forward_hook(hook_fn)
            totalset = torchvision.datasets.ImageFolder("/root/autodl-tmp/CASIA-WebFace_eval/train", transform=test_transform)
        test_loader = DataLoader(totalset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
    else:
        raise ValueError("Unknown dataset!")

    print(f"Testset length: {len(totalset)}")

    test_loader = DataLoader(totalset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    with torch.no_grad():
        test(args, net, totalset, test_loader, device)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')
    parser.add_argument('--log-interval', type=int, default=10, metavar='')
    parser.add_argument('--arch_name', default='efficientnet_b0_casia', type=str, choices=['resnet50', 'inception_resnetv1_vggface2', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3', 'swin_transformer', 'vgg16', 'vision_transformer', 'vgg16bn', 'inception_resnetv1_casia', 'sphere20a', 'efficientnet_b0_casia'], help='model name from torchvision or resnet50v15')
    parser.add_argument('--dataset', default='CASIA', type=str, choices=['vggface2', 'vggface', 'CASIA'], help='test dataset')
    args = parser.parse_args()

    main(args)