import argparse
import yaml
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
from torchvision import transforms, models
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoencoderKL
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd
import logging
import gc
import re
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist

# 日志设置
def set_logging(args, logging_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handler1 = logging.StreamHandler()
    os.makedirs(logging_path, exist_ok=True)
    handler2 = logging.FileHandler(
        os.path.join(logging_path, "stdout.txt"),
        mode='w' if not args.resume_training else 'a'
    )
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(logging.INFO)

# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 全局数据集缓存
_full_train_dataset = None
_full_test_dataset = None

# 木薯叶病变类别
cassava_classes = [
    "Cassava_Bacterial_Blight", "Cassava_Brown_Streak_Disease", 
    "Cassava_Green_Mottle", "Cassava_Mosaic_Disease", "Healthy"
]

class CassavaLeafDataset(Dataset):
    def __init__(self, root, csv_file="train.csv", transform=None, split="train"):
        self.root = root
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []
        
        df = pd.read_csv(csv_file)
        # 不再添加 train_cassava_images，直接用 root
        for _, row in df.iterrows():
            img_name = row['image_id']  # 直接用 CSV 中的 image_id
            img_path = os.path.join(root, img_name)
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(row['label'])
            else:
                logging.warning(f"Image not found: {img_path}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def load_cassava_dataset():
    global _full_train_dataset, _full_test_dataset
    if _full_train_dataset is None or _full_test_dataset is None:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Kaggle 常用尺寸
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        new_root = '/zlf/results/Models/1_0.2_20_500'
        _full_train_dataset = CassavaLeafDataset(root=new_root, csv_file="/home/tjxy/zlf/leaf/data/train.csv", transform=train_transform, split="train")
        _full_test_dataset = CassavaLeafDataset(root=new_root, csv_file="/home/tjxy/zlf/leaf/data/train.csv", transform=test_transform, split="test")
        logging.info(f"Loaded Cassava Dataset: {len(_full_train_dataset)} train, {len(_full_test_dataset)} test samples")
    return _full_train_dataset, _full_test_dataset


# 按类别选择子集（上限 100 张）
def select_balanced_subset(dataset, num_samples, prefix="subset", base_indices=None):
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
    indices_file = f"{prefix}_indices_task_{task_id}_{num_samples}.npy"
    
    num_classes = 5
    samples_per_class = min(num_samples // num_classes, 100)  # 每个类别最多 100 张
    
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        if max(indices) >= len(dataset):
            logging.warning(f"Loaded indices exceed dataset size {len(dataset)}. Regenerating.")
            indices = generate_balanced_indices(dataset, num_samples, samples_per_class, base_indices)
            np.save(indices_file, indices)
    else:
        indices = generate_balanced_indices(dataset, num_samples, samples_per_class, base_indices)
        np.save(indices_file, indices)
    
    return Subset(dataset, indices)

def generate_balanced_indices(dataset, num_samples, samples_per_class, base_indices=None):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    selected_indices = set(base_indices) if base_indices is not None else set()
    for label in range(5):
        available_indices = [i for i in class_indices[label] if i not in selected_indices]
        if len(available_indices) <= samples_per_class:
            selected_indices.update(available_indices)
        else:
            selected = np.random.choice(available_indices, samples_per_class, replace=False)
            selected_indices.update(selected)
    
    if len(selected_indices) < num_samples:
        remaining = num_samples - len(selected_indices)
        all_indices = list(range(len(dataset)))
        remaining_indices = [i for i in all_indices if i not in selected_indices]
        if remaining_indices and remaining > 0:
            extra_indices = np.random.choice(remaining_indices, min(remaining, len(remaining_indices)), replace=False)
            selected_indices.update(extra_indices)
    
    return np.array(list(selected_indices))

# 按类别选择所有可用图像（上限 100 张）
def select_all_balanced_subset(dataset, prefix="subset"):
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
    indices_file = f"{prefix}_indices_task_{task_id}_all.npy"
    
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
    else:
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        indices = []
        for label in range(5):
            available_indices = class_indices[label]
            if len(available_indices) <= 100:
                indices.extend(available_indices)
            else:
                selected = np.random.choice(available_indices, 100, replace=False)
                indices.extend(selected)
        indices = np.array(indices)
        np.save(indices_file, indices)
    
    logging.info(f"Selected {len(indices)} images for full train dataset")
    return Subset(dataset, indices)

def save_cassava_images(dataset, output_dir, config):
    output_dir = os.path.join(config.resultpath, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        img = img * std[:, None, None] + mean[:, None, None]
        img = img.clamp(0, 1)
        img = to_pil_image(img)
        img.save(os.path.join(output_dir, f"{i}_{label}.png"))

def generate_prompt(class_name):
    return (
        f"A highly detailed, photorealistic image of a cassava leaf affected by {class_name}, "
        f"captured in natural lighting and environment. The image should faithfully preserve "
        f"all the distinctive features of {class_name} on a cassava leaf."
    )

def generate_images(prompt_template, cassava_images, labels, pipeline, gen_images, config, output_dir="generated_images", resume_from=492):
    output_dir = os.path.join(config.resultpath, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    generated_labels = []
    ground_truth_mapping = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    
    batch_size = 50
    strengths = config.setting.strength
    images_per_strength = gen_images // len(strengths)
    
    # 计算已生成的图像数量
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("gen_idx_")]
    if existing_files:
        last_file = max(existing_files, key=lambda x: int(x.split('_')[2]))
        last_idx = int(last_file.split('_')[2])  # 提取 gen_idx
        generated_labels = [int(f.split('_')[4]) for f in existing_files]
        ground_truth_mapping = [int(f.split('_')[6]) for f in existing_files]
        logging.info(f"Resuming from gen_idx {last_idx + 1}")
    else:
        last_idx = -1
    
    for batch_start in range(0, len(cassava_images), batch_size):
        batch_end = min(batch_start + batch_size, len(cassava_images))
        batch_images = cassava_images[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]
        
        for idx, (img, label) in enumerate(zip(batch_images, batch_labels)):
            truth_idx = batch_start + idx
            if truth_idx < resume_from:  # 跳过前 492 张
                continue
            
            object_class = cassava_classes[label]
            prompt = generate_prompt(object_class)
            img = img.convert("RGB").resize((512, 512))
            
            # 从 truth_492 开始，检查 strength 和 i
            start_strength_idx = 0
            start_i = 0
            if truth_idx == resume_from:
                # 检查已生成的 strength 和 i
                existing_for_truth = [f for f in existing_files if f"truth_{truth_idx}" in f]
                if existing_for_truth:
                    last_strength = float(existing_for_truth[-1].split('_')[-1].replace('.png', ''))
                    start_strength_idx = strengths.index(last_strength) if last_strength in strengths else len(strengths)
                    start_i = len([f for f in existing_for_truth if f"strength_{last_strength}" in f])
            
            for s_idx, strength in enumerate(strengths[start_strength_idx:], start=start_strength_idx):
                for i in range(start_i if s_idx == start_strength_idx else 0, images_per_strength):
                    with torch.no_grad():
                        output = pipeline(
                            prompt=prompt,
                            image=img,
                            strength=strength,
                            guidance_scale=7.5
                        )
                    gen_img = output["images"][0].resize((224, 224))
                    gen_idx = last_idx + 1 + len(generated_labels)
                    filename = f"gen_idx_{gen_idx}_label_{label}_truth_{truth_idx}_strength_{strength}.png"
                    gen_img.save(os.path.join(output_dir, filename))
                    generated_labels.append(label)
                    ground_truth_mapping.append(truth_idx)
                    del output
                    torch.cuda.empty_cache()
    
    logging.info(f"Generated {len(generated_labels)} new images from {len(cassava_images)} original training images")
    return None, generated_labels, ground_truth_mapping

def load_generated_images(output_dir, config):
    # output_dir = os.path.join(config.resultpath, output_dir)  # 注释掉动态路径
    output_dir = "/home/tjxy/zlf/leaf/results/Models/1_0.2_20_500/generated_images"  # 硬编码固定路径
    pattern = re.compile(r"gen_idx_(\d+)_label_(\d+)_truth_(\d+)_strength_([\d.]+)\.png")
    file_info = []
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            idx, label, truth, strength = match.groups()
            file_info.append((int(idx), int(label), int(truth), float(strength)))
    file_info.sort(key=lambda x: x[0])
    images = [Image.open(os.path.join(output_dir, f"gen_idx_{i}_label_{l}_truth_{t}_strength_{s}.png"))
              for i, l, t, s in file_info]
    labels = [l for _, l, _, _ in file_info]
    ground_truth = [t for _, _, t, _ in file_info]
    return images, labels, ground_truth

def encode_to_latent(vae, images, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.eval()
    vae.to(device)
    transform = transforms.ToTensor()
    latents = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = [img.convert("RGB").resize((512, 512)) for img in images[i:i + batch_size]]
            batch_tensor = torch.stack([transform(img) for img in batch]).to(device)
            latent = vae.encode(batch_tensor).latent_dist.sample() * vae.config.scaling_factor
            latents.append(latent.cpu().numpy())
            del batch_tensor
    return np.concatenate(latents, axis=0) if latents else np.array([])

def calculate_wasserstein_distance(latent1, latent2):
    return wasserstein_distance(latent1.flatten(), latent2.flatten())

def calculate_mmd(latent1, latent2, gamma=None):
    latent1_flat = latent1.flatten().reshape(1, -1)
    latent2_flat = latent2.flatten().reshape(1, -1)

    if gamma is None:
        combined = np.vstack([latent1_flat, latent2_flat])
        dists = cdist(combined, combined, metric='euclidean')
        median = np.median(dists[dists != 0])
        gamma = 1.0 / (2 * median**2 + 1e-8)  # 防止除0

    xx = rbf_kernel(latent1_flat, latent1_flat, gamma=gamma)
    yy = rbf_kernel(latent2_flat, latent2_flat, gamma=gamma)
    xy = rbf_kernel(latent1_flat, latent2_flat, gamma=gamma)
    mmd = xx + yy - 2 * xy  # 返回 (1, 1)

    return float(mmd[0, 0])  # 转换为标量

def filter_generated_images(generated_labels, config, distance_type="wasserstein", percentage=50):
    distances = np.load(os.path.join("./results/Models/all", f"distances_{distance_type}.npy"))
    threshold = np.percentile(distances, percentage)
    selected_indices = [i for i, dist in enumerate(distances) if dist <= threshold]
    selected_labels = [generated_labels[i] for i in selected_indices]
    logging.info(f"Selected {len(selected_indices)} images with {distance_type}, tol={percentage}")
    return selected_indices, selected_labels

def images_to_dataset(images, labels):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensors = [transform(img) for img in images]
    return TensorDataset(torch.stack(tensors), torch.tensor(labels))

# 使用 EfficientNet-B0（参考 Kaggle 前排方案）
def get_efficientnet_b0(num_classes=5):
    model = models.efficientnet_b0(weights="EfficientNet_B0_Weights.DEFAULT")
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False  # 冻结特征提取层
        else:
            if "weight" in name:
                nn.init.xavier_normal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
            param.requires_grad = True
    return model

def cache_generated_latents(vae, generated_images, config):
    latent_dir = "./results/Models/all"
    latent_path = os.path.join(latent_dir, "generated_latents.npy")
    os.makedirs(latent_dir, exist_ok=True)  # 确保目录存在
    if not os.path.exists(latent_path):
        logging.info("Caching latent representations for generated images...")
        generated_latents = encode_to_latent(vae, generated_images)
        np.save(latent_path, generated_latents)
        logging.info(f"Latent representations saved to {latent_path}")
    return np.load(latent_path)

def precompute_distances(vae, generated_images, generated_labels, full_dataset, config, distance_type):
    distance_path = os.path.join("./results/Models/all", f"distances_{distance_type}.npy")
    os.makedirs(os.path.dirname(distance_path), exist_ok=True)
    if not os.path.exists(distance_path):
        logging.info(f"Precomputing {distance_type} distances...")
        
        # 按类别收集原始图像
        class_original_images = defaultdict(list)
        for img, label in full_dataset:
            img = img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            img = img.clamp(0, 1)
            img = to_pil_image(img)
            label = int(label.item())
            class_original_images[label].append(img)
        
        # 计算原始图像的均值潜在表示
        class_original_latents = {}
        for label in class_original_images:
            class_original_latents[label] = encode_to_latent(vae, class_original_images[label]).mean(axis=0)
            logging.info(f"Mean original latents for label {label}: {class_original_latents[label].shape}")
        
        # 获取生成图像的潜在表示
        generated_latents = cache_generated_latents(vae, generated_images, config)
        logging.info(f"Generated latents shape: {generated_latents.shape}")
        
        # 计算距离
        distances = []
        for gen_latent, gen_label in zip(generated_latents, generated_labels):
            original_latent = class_original_latents[gen_label]
            if distance_type == "wasserstein":
                dist = calculate_wasserstein_distance(gen_latent, original_latent)
            elif distance_type == "tv":
                dist = calculate_tv_divergence(gen_latent, original_latent)
            elif distance_type == "mmd":
                dist = calculate_mmd(gen_latent, original_latent)
            distances.append(dist)
        
        distances = np.array(distances, dtype=np.float32)
        logging.info(f"Distances shape: {distances.shape}, dtype: {distances.dtype}")
        np.save(distance_path, distances)
        logging.info(f"Distances for {distance_type} saved to {distance_path}")
    return np.load(distance_path)

def train_model(model, train_loader, test_loader, config, model_type="Baseline", train_num=0, filter_method="None", filter_tolerance="N/A", generated_used=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Using device: {device}")
    
    total_samples = len(train_loader.dataset)
    is_baseline = (model_type == "Baseline")
    
    epochs = 50 if is_baseline else 150
    batch_size = config.Model.batch_size  # 直接使用配置中的 batch_size，不放大
    base_lr = 0.0001
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, weight_decay=0.01 if not is_baseline else 0.0)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scaler = amp.GradScaler()
    
    patience = 40
    min_delta = 0.001
    best_accuracy = 0.0
    patience_counter = 0
    
    # 验证集划分
    val_size = int(0.3 * total_samples)
    train_size = total_samples - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])
    
    # 优化 num_workers，适配 1-2 张 GPU
    num_workers = min(4, os.cpu_count() or 4)  # 使用 CPU 核心数但上限为 4
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    warmup_epochs = 5
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            with amp.autocast(device_type='cuda'):
                outputs = model(batch)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        val_labels, val_preds = [], []
        with torch.no_grad():
            for batch, labels in val_loader:
                batch, labels = batch.to(device), labels.to(device)
                with amp.autocast(device_type='cuda'):
                    outputs = model(batch)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
        
        val_accuracy = (np.array(val_labels) == np.array(val_preds)).mean()
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, LR: {current_lr:.6f}")
        
        if val_accuracy > best_accuracy + min_delta:
            best_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1} with best validation accuracy: {best_accuracy:.4f}")
                break
    
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            with amp.autocast(device_type='cuda'):
                outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = (np.array(all_labels) == np.array(all_preds)).mean()
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    logging.info(
        f"Final Test Metrics for {model_type} (Train Size: {train_num}, Total Samples: {total_samples}) "
        f"- Filtered by: {filter_method} (Tolerance: {filter_tolerance}), "
        f"Generated Images Retained: {generated_used} "
        f"- Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def cassava_leaf_disease(config):
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
    
    # 加载训练集
    cassava_images_dir = "/home/tjxy/zlf/leaf/results/Models/1_0.2_20_500/train_cassava_images"
    cassava_image_files = sorted(os.listdir(cassava_images_dir), key=lambda x: int(x.split('_')[0]))
    cassava_images = [Image.open(os.path.join(cassava_images_dir, f)) for f in cassava_image_files]
    cassava_labels = [int(f.split('_')[1].split('.')[0]) for f in cassava_image_files]
    logging.info(f"Loaded {len(cassava_images)} saved images from {cassava_images_dir}")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_tensors = [train_transform(img) for img in cassava_images]
    train_dataset = TensorDataset(torch.stack(train_tensors), torch.tensor(cassava_labels))
    logging.info(f"Train dataset label distribution: {Counter(train_dataset.tensors[1].tolist())}")
    
    # 加载测试集
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = CassavaLeafDataset(
        root="/home/tjxy/zlf/leaf/data/test_images_500",
        csv_file="/home/tjxy/zlf/leaf/data/test_500.csv",
        transform=test_transform,
        split="train"
    )
    logging.info(f"Loaded test dataset: {len(test_dataset)} samples")
    
    # 生成图像
    if config.Model.regen:
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16
        )
        generate_images(
            "A photo of a cassava leaf with [object]", cassava_images, cassava_labels,
            pipeline, config.Model.genimg, config, "generated_images", resume_from=492
        )
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
    
    generated_images, generated_labels, ground_truth_mapping = load_generated_images("generated_images", config)
    logging.info(f"Generated labels distribution: {Counter(generated_labels)}")
    
    # 创建 DataLoader
    num_gpus = 2
    test_batch_size = config.Model.batch_size * num_gpus
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4 * num_gpus)
    
    # VAE 和距离计算（只执行一次）
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="vae")
    vae.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cache_generated_latents(vae, generated_images, config)
    for dist_type in ["wasserstein", "tv", "mmd"]:
        precompute_distances(vae, generated_images, generated_labels, train_dataset, config, dist_type)
    
    # 训练模型
    results = {}
    base_indices = None
    for train_num in config.setting.train_num:
        train_subset = select_balanced_subset(train_dataset, train_num, prefix=f"train_{train_num}", base_indices=base_indices)
        logging.info(f"Train subset size for train_num={train_num}: {len(train_subset)}, labels: {Counter([train_subset.dataset.tensors[1][i].item() for i in train_subset.indices])}")
        base_indices = train_subset.indices
        
        # Baseline 模型
        baseline_model = get_efficientnet_b0(num_classes=5)
        baseline_result = train_model(
            baseline_model, 
            DataLoader(train_subset, batch_size=config.Model.batch_size, shuffle=True),
            test_loader, 
            config,
            model_type="Baseline",
            train_num=train_num,
            filter_method="None",
            filter_tolerance="N/A",
            generated_used=0
        )
        results[f"Baseline_{train_num}"] = baseline_result
        del baseline_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # 选择对应 train_num 的生成图像
        path = cassava_images_dir  # 修复 path 未定义问题
        train_indices = [int(f.split('_')[0]) for f in sorted(os.listdir(path), key=lambda x: int(x.split('_')[0]))][:train_num]
        selected_generated_images = []
        selected_generated_labels = []
        selected_ground_truth_mapping = []
        for idx, gen_img, gen_label in zip(ground_truth_mapping, generated_images, generated_labels):
            if idx in train_indices:
                selected_generated_images.append(gen_img)
                selected_generated_labels.append(gen_label)
                selected_ground_truth_mapping.append(idx)
        
        # 增强模型
        distance_types = ["wasserstein", "tv", "mmd", "none"]
        filtered_indices_dict = {}
        for dist_type in distance_types:
            if dist_type != "none":
                filtered_indices_dict[dist_type] = {}
                for wass_tol in config.setting.wass_tol:
                    selected_indices, selected_labels = filter_generated_images(
                        generated_labels, config, distance_type=dist_type, percentage=wass_tol
                    )
                    filtered_indices = [i for i in selected_indices if ground_truth_mapping[i] in train_indices]
                    filtered_labels = [generated_labels[i] for i in filtered_indices]
                    filtered_indices_dict[dist_type][wass_tol] = (filtered_indices, filtered_labels)
            
            for wass_tol in (config.setting.wass_tol if dist_type != "none" else [None]):
                try:
                    if dist_type == "none":
                        filtered_images = selected_generated_images
                        filtered_labels = selected_generated_labels
                        generated_used = len(filtered_images)
                        filter_method = "None"
                        filter_tolerance = "N/A"
                    else:
                        selected_indices, selected_labels = filtered_indices_dict[dist_type][wass_tol]
                        filtered_images = [generated_images[idx] for idx in selected_indices]
                        filtered_labels = selected_labels
                        generated_used = len(filtered_images)
                        filter_method = dist_type.capitalize()
                        filter_tolerance = wass_tol
                    
                    filtered_dataset = images_to_dataset(filtered_images, filtered_labels)
                    combined_tensors = torch.cat([torch.stack([item[0] for item in train_subset]), 
                                                 torch.stack([item[0] for item in filtered_dataset])])
                    combined_labels = torch.cat([torch.tensor([item[1] for item in train_subset]), 
                                                torch.tensor(filtered_labels)])
                    combined_dataset = TensorDataset(combined_tensors, combined_labels)
                    combined_loader = DataLoader(combined_dataset, batch_size=config.Model.batch_size, shuffle=True)
                    
                    augmented_model = get_efficientnet_b0(num_classes=5)
                    result_augmented = train_model(
                        augmented_model, 
                        combined_loader, 
                        test_loader, 
                        config,
                        model_type="Augmented",
                        train_num=train_num,
                        filter_method=filter_method,
                        filter_tolerance=filter_tolerance,
                        generated_used=generated_used
                    )
                    key = f"Augmented_{dist_type}_{train_num}" if dist_type == "none" else f"Augmented_{dist_type}_tol_{wass_tol}_{train_num}"
                    results[key] = result_augmented
                
                except Exception as e:
                    logging.error(f"Augmented training failed for {dist_type}, tol={wass_tol}: {str(e)}")
                    raise
                finally:
                    if 'filtered_dataset' in locals():
                        del filtered_dataset
                    if 'combined_dataset' in locals():
                        del combined_dataset
                    if 'augmented_model' in locals():
                        del augmented_model
                    torch.cuda.empty_cache()
                    gc.collect()
    
    # 生成 CSV
    results_data = []
    for train_num in config.setting.train_num:
        baseline_key = f"Baseline_{train_num}"
        results_data.append({
            "Model": f"Baseline_{train_num}",
            "Train_Size": train_num,
            "Generated_Used": 0,
            "Filter_Method": "None",
            "Filter_Tolerance": "N/A",
            "Accuracy": results[baseline_key]["accuracy"],
            "Precision": results[baseline_key]["precision"],
            "Recall": results[baseline_key]["recall"],
            "F1": results[baseline_key]["f1"]
        })

        train_indices = [int(f.split('_')[0]) for f in sorted(os.listdir(path), key=lambda x: int(x.split('_')[0]))][:train_num]
        total_generated = sum(1 for idx in ground_truth_mapping if idx in train_indices)

        distance_types = ["wasserstein", "tv", "mmd", "none"]
        for dist_type in distance_types:
            if dist_type == "none":
                filtered_count = total_generated
                key = f"Augmented_{dist_type}_{train_num}"
                results_data.append({
                    "Model": key,
                    "Train_Size": train_num,
                    "Generated_Used": filtered_count,
                    "Filter_Method": "None",
                    "Filter_Tolerance": "N/A",
                    "Accuracy": results[key]["accuracy"],
                    "Precision": results[key]["precision"],
                    "Recall": results[key]["recall"],
                    "F1": results[key]["f1"]
                })
            else:
                for wass_tol in config.setting.wass_tol:
                    key = f"Augmented_{dist_type}_tol_{wass_tol}_{train_num}"
                    selected_indices, _ = filtered_indices_dict[dist_type][wass_tol]
                    filtered_count = len(selected_indices)
                    results_data.append({
                        "Model": key,
                        "Train_Size": train_num,
                        "Generated_Used": filtered_count,
                        "Filter_Method": dist_type.capitalize(),
                        "Filter_Tolerance": wass_tol,
                        "Accuracy": results[key]["accuracy"],
                        "Precision": results[key]["precision"],
                        "Recall": results[key]["recall"],
                        "F1": results[key]["f1"]
                    })

    results_df = pd.DataFrame(results_data)
    results_csv_path = os.path.join(config.resultpath, f"model_accuracies_{task_id}.csv")
    try:
        os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
        results_df.to_csv(results_csv_path, index=False)
        logging.info(f"Results saved to {results_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")
        raise

    del vae
    torch.cuda.empty_cache()
    gc.collect()

class Dict2Namespace:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Namespace(value))
            else:
                setattr(self, key, value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cassava Leaf Disease Classification")
    parser.add_argument("--seed", type=int, default=66, help="Random seed")
    parser.add_argument('--JobID', type=int, default=1, help='JobID')
    parser.add_argument('--setting_path', type=str, default='config/cassava_leaf.yml', help='Path to setting')
    parser.add_argument("--train", action="store_true", help="Whether to train the model")
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training")
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    with open(args.setting_path, "r") as f:
        config = yaml.safe_load(f)
        setting = config['setting']
        config = Dict2Namespace(config)
    
    ls = [len(setting[key]) for key in setting]
    Jobs = np.array(np.meshgrid(*[range(1, l+1) for l in ls])).T.reshape(-1, len(ls))
    num_start = (args.JobID - 1) * config.Job_num
    num_end = min(len(Jobs), args.JobID * config.Job_num)
    logging.info(f"Processing jobs from {num_start} to {num_end-1}")
    
    for Job in range(num_start, num_end):
        args.JobID = Job + 1
        ls = {f'id_{key}': Jobs[Job, i] - 1 for i, key in enumerate(setting)}
        resultpath = '_'.join(f"{setting[key][ls[f'id_{key}']]}" for key in setting)
        config.resultpath = f'./results/Models/{resultpath}'
        config.loggingpath = f'./results/log/{resultpath}'
        set_logging(args, config.loggingpath)
        logging.info(f"Starting Job {args.JobID} with setting {args.setting_path}")
        
        config.Model.task = setting['task'][ls['id_task']]
        config.Model.strength = setting['strength'][ls['id_strength']]
        config.Model.wass_tol = setting['wass_tol'][ls['id_wass_tol']]
        config.Model.train_num = setting['train_num'][ls['id_train_num']]
        
        if args.train:
            cassava_leaf_disease(config)
            logging.info(f"Job {Job} completed successfully!")