import argparse
import yaml
import os
import pickle
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoencoderKL
import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd
import shutil
from torch.amp import GradScaler
import logging
from utils import *
parser = argparse.ArgumentParser(description=globals()["__doc__"])
parser.add_argument("--seed", type=int, default = 1234, help="Random seed") #orl is 1234
parser.add_argument('--JobID', type=int, default=1, help='JobID')
parser.add_argument('--setting_path', type=str, default='cancer.yml', help='Path to setting')
parser.add_argument("--train", action="store_true", help="Whether to test the model")
args = parser.parse_args()
from sklearn.metrics import precision_score, recall_score, f1_score
logging.basicConfig(level=logging.WARNING)
# 全局变量缓存完整数据集
_full_train_dataset = None
_full_test_dataset = None

# Get SLURM task ID
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
NUM_IMAGES = 2500  # 从CIFAR10 TRAIN中选择1000个样本
PERCENTAGE = 50
GEN_IMAGES = 2

# 动态生成文件路径
def get_task_specific_path(base_name):
    return f"{base_name}_task_{task_id}"

# 1. Load CIFAR10 data with only 5 classes
class CIFAR10Subset(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, classes=None):
        super(CIFAR10Subset, self).__init__(root=root, train=train, download=download, transform=transform,
                                            target_transform=target_transform)
        if classes is not None:
            self.data, self.targets = self._filter_classes(classes)

    def _filter_classes(self, classes):
        indices = np.where(np.isin(self.targets, classes))[0]
        data = self.data[indices]
        targets = np.array(self.targets)[indices]
        target_map = {original_class: new_class for new_class, original_class in enumerate(classes)}
        targets = [target_map[t] for t in targets]
        return data, targets
def load_cifar10():
    global _full_train_dataset, _full_test_dataset
    if _full_train_dataset is None or _full_test_dataset is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB image
        ])
        desired_classes = [0, 1, 2, 3, 4]  # 只选择5类：airplane, automobile, bird, cat, deer
        _full_train_dataset = CIFAR10Subset(root="./data", train=True, download=True, transform=transform,
                                            classes=desired_classes)
        _full_test_dataset = CIFAR10Subset(root="./data", train=False, download=True, transform=transform,
                                           classes=desired_classes)
    return _full_train_dataset, _full_test_dataset
# 2. Randomly select a subset from the training set
def select_subset(dataset, config,num_samples=1000):
    indices_file = os.path.join(config.resultpath,"indices.npy")
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
    else:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        np.save(indices_file, indices)
    subset = Subset(dataset, indices)
    return subset
# 3. Save CIFAR10 dataset images with index and label in filenames
def save_cifar10_images(dataset, output_dir,config, max_images=None):
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(config.resultpath,output_dir)
    if max_images is not None:
        max_images = min(len(dataset), max_images)
    else:
        max_images = len(dataset)
    for i in range(max_images):
        img, label = dataset[i]
        file_name = f"{i}_{label}.png"
        img = img.permute(1, 2, 0).numpy()
        img = ((img * 0.5) + 0.5) * 255  # Denormalize
        img = np.clip(img, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_dir, file_name))
# 4. Use Stable Diffusion to generate images
def generate_images11(prompt, train_dataset, pipeline, gen_images=1):
    generated_images = []
    generated_labels = []
    ground_truth_mapping = []  # To keep track of which ground truth image each generated image corresponds to

    for idx, item in enumerate(train_dataset):
        label = item[1]
        img = item[0]
        img = to_pil_image(img)
        img = img.convert("RGB").resize((512, 512))
        for _ in range(gen_images):  # For each selected CIFAR10 image, generate GEN_IMAGES images
            output = pipeline(prompt=prompt, image=img, num_inference_steps=30)
            gen_img = output["images"][0].resize((32, 32))  # Resize to 32x32
            generated_images.append(gen_img)
            generated_labels.append(label)  # Keep the label
            ground_truth_mapping.append(idx)  # Map to the current ground truth image index

    return generated_images, generated_labels, ground_truth_mapping
# 5. Encode images into latent space
def encode_to_latent(vae, images, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae.eval()
    vae.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    images = [img.convert("RGB").resize((512, 512)) for img in images]
    latents = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            #print(i)
            batch = images[i:i + batch_size]
            batch_tensor = torch.stack([transform(img) for img in batch]).to(device)

            latent_output = vae.encode(batch_tensor)
            latent_dist = latent_output.latent_dist

            scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
            latent = latent_dist.sample() * scaling_factor
            latents.append(latent.cpu().numpy())

    if latents:
        latents = np.concatenate(latents, axis=0)
    else:
        latents = np.array([])

    return latents
# 6. Calculate Wasserstein distance in latent space
def calculate_wasserstein_distance(latent1, latent2):
    latent1_flat = latent1.flatten()
    latent2_flat = latent2.flatten()
    return wasserstein_distance(latent1_flat, latent2_flat)
# 7. Filter generated images by specific ground truth
def filter_generated_images(vae, generated_images, generated_labels, ground_truth_images, ground_truth_mapping,
                            percentage=PERCENTAGE):
    # Encode ground truth images into latent space
    ground_truth_latents = encode_to_latent(vae, ground_truth_images)

    from collections import defaultdict
    generated_groups = defaultdict(list)
    for i, gt_idx in enumerate(ground_truth_mapping):
        generated_groups[gt_idx].append((generated_images[i], generated_labels[i]))

    filtered_images = []
    filtered_labels = []
    for gt_idx, group in generated_groups.items():
        # Get the latent of the specific ground truth image
        gt_latent = ground_truth_latents[gt_idx]
        # Encode generated images in this group
        group_images = [img for img, _ in group]
        group_latents = encode_to_latent(vae, group_images)
        # Calculate distances to the specific ground truth latent
        distances = [calculate_wasserstein_distance(gen_latent, gt_latent) for gen_latent in group_latents]
        # Determine the threshold for the top percentage within this group
        distances_array = np.array(distances)
        threshold = np.percentile(distances_array, percentage)
        # Select images within the group that are closer than the threshold
        selected_indices = [i for i, dist in enumerate(distances) if dist <= threshold]
        selected_images = [group_images[i] for i in selected_indices]
        selected_labels = [group[i][1] for i in selected_indices]
        filtered_images.extend(selected_images)
        filtered_labels.extend(selected_labels)

    print(f"Selected {len(filtered_images)} images out of {len(generated_images)} based on Wasserstein distance.")
    return filtered_images, filtered_labels
# 8. Save generated images
def save_generated_images(images, labels,config, output_dir):
    output_dir = os.path.join(config.resultpath,output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for idx, (img, label) in enumerate(zip(images, labels)):
        img.save(os.path.join(output_dir, f"gen_{label}_{idx}.png"))
# 9. Define ResNet-20 CNN classifier suitable for CIFAR10
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):  # 修改为5类
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64*14*14, num_classes)  #参数选取

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
def ResNet20(num_classes=7):  # 修改为5类
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
# 10. Training function (保持不变)
def train_model_old(model, train_loader, test_loader, epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model.to(device)
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler(enabled=True)
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            with amp.autocast(device_type='cuda'):
                outputs = model(batch)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            with amp.autocast(device_type='cuda'):
                outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy, train_losses
#3Criteria Precision，Recall，F1 Score with Macro Average
def train_model(model, train_loader, test_loader, epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Move model to device and enable benchmarking
    model.to(device)
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler(enabled=True)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Training loop
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()

            with amp.autocast(device_type='cuda'):
                outputs = model(batch)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Evaluation on test set
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            with amp.autocast(device_type='cuda'):
                outputs = model(batch)
            _, predicted = torch.max(outputs, 1)

            # Collect predictions and true labels
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = torch.tensor(all_labels)
    all_predictions = torch.tensor(all_predictions)

    # Calculate metrics
    accuracy = (all_predictions == all_labels).sum().item() / len(all_labels)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")

    result_list={
        'accuracy'  :accuracy,
        'precision' :precision,
        'recall'    :recall,
        'f1'        :f1,
        'train_losses':train_losses,
    }

    return result_list
# 11. Convert images to TensorDataset
def images_to_dataset(images, labels):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensors = [transform(img) for img in images]

    return TensorDataset(torch.stack(tensors), torch.tensor(labels))
# CIFAR10类别列表（只保留5类）
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer"]
# 定义Prompt模板
#prompt_template = "An image of a [object]  with [specific details]."
prompt_template = "This is a skin cancer picture with category []"
# 动态生成Prompt
def generate_prompt(object_class, specific_details="clear details"):
    return prompt_template.replace("[object]", object_class).replace("[specific details]", specific_details)
# 生成图像时使用动态Prompt
def generate_images(prompt_template, train_dataset, pipeline, gen_images=1):
    generated_images = []
    generated_labels = []
    ground_truth_mapping = []
    int2label_dict = [
        'Melanoma','Melanocytic Nevus','Basal Cell Carcinoma','Actinic Keratosis/Intraepithelial Carcinoma',
        'Benign Keratosis-like Lesions','Dermatofibroma','Vascular Lesions',
    ]
    for idx, item in enumerate(train_dataset):
        img = item[0]
        label = item[1]
        prompt = f"This is a skin cancer picture with category {int2label_dict[label]}"
        img = to_pil_image(img)
        img = img.convert("RGB").resize((512, 512))
        for _ in range(gen_images):
            output = pipeline(prompt=prompt, image=img, num_inference_steps=30,strength=0.15)
            gen_img = output["images"][0].resize((450,450))  # Resize to 32x32
            generated_images.append(gen_img)
            generated_labels.append(label)
            ground_truth_mapping.append(idx)

    return generated_images, generated_labels, ground_truth_mapping
# 12. Main program flow (修改基线模型部分)

def cancer_select(label_list,num,image_path):
    transform = transforms.Compose([
        transforms.Resize((450,450)),  # 调整图像大小到224x224
        transforms.ToTensor(),  # 将PIL图像转换为张量
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
    ])
    selected_images = {}
    labels = label_list.columns[1:]
    for label in labels:
        group = label_list[label_list[label] == 1.0]
        selected_images[label] = group.head(num)['image'].tolist()
    list = []
    label_dict={
        'MEL'   :0,
        'NV'    :1,
        'BCC'   :2,
        'AKIEC' :3,
        'BKL'   :4,
        'DF'    :5,
        'VASC'  :6,
    }
    img_orl = []
    label_orl = []
    for label, image_names in selected_images.items():
        for image_name in image_names:
            try:
                #image_path = f'./data/ISIC2018_Task3_Training_Input/{image_name}.jpg'  # 根据实际图片格式调整
                image_path1 = os.path.join(image_path,f'{image_name}.jpg')
                img = Image.open(image_path1)
                img_tensor = transform(img)
                temp = (img_tensor,label_dict[f'{label}'])
                img_orl.append(img)
                label_orl.append(label)
                list.append(temp)
            except FileNotFoundError:
                print(f"图片 {image_name} 未找到")
    return list,img_orl,label_orl
def cancer(config):
    label_train = pd.read_csv('./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
    label_test = pd.read_csv('./data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv')
    image_path_train = f'./data/ISIC2018_Task3_Training_Input/'
    image_path_test = f'./data/ISIC2018_Task3_Test_Input/'
    # 按标签分组并选取training图片
    train_dataset,train_img_orl,train_label_orl = cancer_select(label_train,config.Model.train_num,image_path_train)
    test_dataset,test_img_orl,test_label_orl = cancer_select(label_test,config.Model.test_num,image_path_test)
    #cache_directory = "./"
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16, verbose=False, #cache_dir=cache_directory
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    # 生成动态Prompt模板
    prompt_template = "An image of a [object]  with [specific details]."
    # 根据task_id生成不同数量的图片
    gen_images_per_sample = config.Model.genimg * config.Model.task  # 每个图片生成两个
    generated_images, generated_labels, ground_truth_mapping = generate_images(
        prompt_template, train_dataset, pipeline, gen_images=gen_images_per_sample
    )

    # 保存生成的图像
    generated_images_dir = get_task_specific_path("generated_images")
    save_generated_images(generated_images, generated_labels,config, output_dir=generated_images_dir)

    #-----------------------------------------------------------------------------------------------------------------------------
    print('With wasserstein filter')
    ground_truth_images = train_img_orl
    # 加载VAE模型 #这里暂时放在 c 盘
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        subfolder="vae"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 筛选生成的图像
    filtered_images, filtered_labels = filter_generated_images(
        vae, generated_images, generated_labels, ground_truth_images, ground_truth_mapping, percentage=PERCENTAGE
    )
    print('Filtering images finished')
    filtered_images_dir = get_task_specific_path("filtered_images")
    save_generated_images(filtered_images, filtered_labels,config, output_dir=filtered_images_dir)

    # 将筛选出的图像转换为TensorDataset
    filtered_dataset = images_to_dataset(filtered_images, filtered_labels)

    # 创建一个包含原始和筛选图像的新数据集
    original_tensors = torch.stack([item[0] for item in train_dataset], dim=0)
    original_labels = torch.tensor([item[1] for item in train_dataset])

    combined_tensors = torch.stack([item[0] for item in filtered_dataset], dim=0)
    combined_labels = torch.tensor([item[1] for item in filtered_dataset])

    combined_tensors = torch.cat([original_tensors, combined_tensors], dim=0)
    combined_labels = torch.cat([original_labels, combined_labels], dim=0)


    combined_dataset = TensorDataset(combined_tensors, combined_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=config.Model.batch_size, shuffle=True)

    # 定义测试集的DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config.Model.batch_size, shuffle=False)

    # 基线模型的准确率文件路径
    BASELINE_ACCURACY_PATH = os.path.join(config.resultpath,f"baseline_accuracy_task_{task_id}.npy")
    #--------------------------------------------------------------------------------------------------------------
    if not os.path.exists(BASELINE_ACCURACY_PATH):
        print("Training baseline model...")
        baseline_model = ResNet20()
        result_baseline= train_model(baseline_model, DataLoader(train_dataset,batch_size=config.Model.batch_size, shuffle=True), test_loader, epochs=config.Model.epoch)
        # 使用 pickle 保存字典
        with open(BASELINE_ACCURACY_PATH, 'wb') as f:
            pickle.dump(result_baseline, f)
    else:
        print("Loading baseline model accuracy...")
        # 使用 pickle 加载字典
        with open(BASELINE_ACCURACY_PATH, 'rb') as f:
            result_baseline = pickle.load(f)

    # 训练增强模型
    print("Training augmented model with wass...")
    augmented_model = ResNet20()
    result_augmented_wass = train_model(augmented_model, combined_loader,
                                                       test_loader, epochs=config.Model.epoch)
    # 比较结果
    #results = {
    #    "Model": ["Baseline", "Augmented"],
    #    "Accuracy": [baseline_accuracy, augmented_accuracy_wass]
    #}
    #results_df = pd.DataFrame(results)
    # 保存结果到CSV文件
    #results_csv_path = os.path.join(config.resultpath,f"model_accuracies_{task_id}_without_wass.csv")
    #results_df.to_csv(results_csv_path, index=False)
    #print(f"Results saved to {results_csv_path}")
    #----------------------------------------------------------------------------------------------------------------------------------
    print('Without wasserstein filter')
    filtered_dataset = images_to_dataset(generated_images,generated_labels)
    original_tensors = torch.stack([item[0] for item in train_dataset], dim=0)
    original_labels = torch.tensor([item[1] for item in train_dataset])

    combined_tensors = torch.stack([item[0] for item in filtered_dataset], dim=0)
    combined_labels = torch.tensor([item[1] for item in filtered_dataset])

    combined_tensors = torch.cat([original_tensors, combined_tensors], dim=0)
    combined_labels = torch.cat([original_labels, combined_labels], dim=0)

    combined_dataset = TensorDataset(combined_tensors, combined_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=config.Model.batch_size, shuffle=True)
    # 定义测试集的DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config.Model.batch_size, shuffle=False)
    # 训练增强模型
    print("Training augmented model without wass...")
    augmented_model = ResNet20()
    result_augmented = train_model(augmented_model, combined_loader, test_loader, epochs=config.Model.epoch)

    # 比较结果
    results = {
        "Model":        ["Baseline", "Augmented",'Augmented_wass'],
        "accuracy":     [result_baseline["accuracy"],       result_augmented["accuracy"],       result_augmented_wass["accuracy"]],
        "precision":    [result_baseline["precision"],     result_augmented["precision"],     result_augmented_wass["precision"]],
        "recall":       [result_baseline["recall"],         result_augmented["recall"],         result_augmented_wass["recall"]],
        "f1":           [result_baseline["f1"],             result_augmented["f1"],             result_augmented_wass["f1"]],
        "train_losses": [result_baseline["train_losses"],   result_augmented["train_losses"],   result_augmented_wass["train_losses"]],
    }
    results_df = pd.DataFrame(results)

    # 保存结果到CSV文件
    results_csv_path = os.path.join(config.resultpath,f"model_accuracies_{task_id}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")
    print("No filtered images available for training.")

def change_config(config, setting, ls):
    print(f'Keys of id_setting is {ls.keys()}')
    config.Model.task = setting['task'][ls['id_task']]
    #config.Model.wass = setting['wass'][ls['id_wass']]
    return config
if __name__ == "__main__":
    # load base config
    args.config = os.path.join('configs', args.setting_path)
    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)  # 训练情况下yml配置的导入
        setting = config['setting']
        config = dict2namespace(config)  # 这个函数的作用是将dict转化为namespace变量，可以被.方法索引
    # setting
    #setting_path = os.path.join('configs', args.setting_path)
    #with open(os.path.join(setting_path), "r") as f:
    #    setting = yaml.safe_load(f)
    # Job
    ls = []
    for i, key in enumerate(setting):
        length = len(setting[f'{key}'])
        ls.append(length)
    Jobs = assig(ls)
    # Set resultpath
    ls = {}
    #print(Jobs)
    resultpath = ''
    for i, key in enumerate(setting):
        id = f'id_{key}'
        # id_value = Jobs[ i , 2 - 1] - 1
        id_value = Jobs[i, args.JobID - 1] - 1
        ls[f'{id}'] = id_value
        resultpath = resultpath + f"{setting[key][id_value]}_"
    print(resultpath)
    # match JobID
    config = change_config(config, setting, ls)
    # set resultpath and loggingpath
    config.resultpath = './results/Models/' + resultpath
    config.modelpath  = './results/Models/data'
    config.loggingpath = './results/log/' + resultpath
    # set logging
    #setlogging(args, config.loggingpath)
    #logging.info(f'\nThe process is the Job {args.JobID}.')
    #logging.info(f'\nSetting is {args.setting_path}')
    # set result
    if not os.path.exists(os.path.join(config.resultpath)):
        os.makedirs(os.path.join(config.resultpath))
    # training or testing
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(config.device)
    if args.train:
        cancer(config)
    elif args.test:
        pass
    # 调试代码 python main.py --train