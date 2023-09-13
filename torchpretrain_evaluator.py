import argparse
import tqdm
import os
import torch
import torchvision 
import torchvision.transforms as transforms
import numpy as np
import pandas as pd




class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def parse_gt_dict(gt_csv_path):
    gt_csv = pd.read_csv(gt_csv_path)
    gt_arr = np.asarray(gt_csv)

    name2gtidx = dict()
    for i in gt_arr:
        name2gtidx.update({i[0]: {"idx": i[1], "label": i[2]}})
            
    metaid2gtidx = dict()
    for i in gt_arr:
        metaid2gtidx.update({i[2]: {"idx": i[1], "label": i[2]}})

    return name2gtidx, metaid2gtidx

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='gpu', default=6)
    parser.add_argument('--seed', help='seed', default=0)
    parser.add_argument('--root', help='ImageNet data root', default="/data/ImageNet1k/")
    parser.add_argument('--gt_path', help='ImageNet ground truth root', default="./ILSVRC2012_torchindex.csv")
    parser.add_argument('--pretrained_model', help='Choose target model', default="vgg16", choices=['vgg16', 'resnet18'])

    args = parser.parse_args()

    # For reproducibility
    random_seed = int(args.seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    image_transforms = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_data = ImageFolderWithPaths(root=os.path.join(args.root, "val"), transform=image_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    if args.pretrained_model == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif args.pretrained_model == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)

    model.cuda()
    model.eval()

    name2gtidx, metaid2gtidx = parse_gt_dict(args.gt_path)

    correct = 0
    pbar = tqdm.tqdm(test_loader)
    with torch.no_grad():
        for it, (inputs, labels, paths) in enumerate(pbar):
            curr_pred = model(inputs.cuda())
            curr_pred_idx = torch.argmax(curr_pred).item()

            curr_name = paths[0].split("/")[-1]
            curr_gt_idx = name2gtidx[curr_name]['idx']

            correct += int(curr_pred_idx == curr_gt_idx)

            if (it+1)%1000==0:
                desc = "Evaluating... (Model: {}, Current Acc: {:6.3f})".format(args.pretrained_model, correct/(it+1)*100)
                pbar.set_description(desc)
    
    print("Final Accuracy: {:6.3f}".format(correct/len(test_loader)*100))
    print("Check the public link for pretrained models: https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights")