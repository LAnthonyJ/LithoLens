import torchvision
import torchvision.transforms as transforms
from .custom_dataset import MyDataset
def old_build_dataset(data_path, mode="train"):
    "The project is a binary classification problem. Thus, fix num_classes to 2."
    if mode == "train":
        train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Limit the crop scale to avoid empty patches
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),                # Vertical flip
    transforms.RandomRotation(degrees=180),              # Rotate to arbitrary angles
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Vary color and lighting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))   # Random erasing for regularization
])

        # Update paths and set is_test=False
        train_dataset = MyDataset(
            img_dir=data_path + "/train_images",
            csv_path=data_path + "/train_labels.csv",
            transform=train_transform,
            is_test=False
        )
        return train_dataset, 2

    elif mode == "test":
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Use MyDataset to read test_images and sample_submission.csv
        test_dataset = MyDataset(
            img_dir=data_path + "/test_images",
            csv_path=data_path + "/sample_submission.csv",
            transform=val_transform,
            is_test=True
        )
        return test_dataset, 2

    raise NotImplementedError
def build_dataset(data_path, mode="train"):
    "The project is a binary classification problem. Thus, fix num_classes to 2."
    if mode == "train":
        train_transform = transforms.Compose([
            # Enhanced Data Augmentation
            transforms.RandomResizedCrop(
                size=480,
                scale=(0.3, 1.0),
                ratio=(0.75, 1.33)
            ),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            # Warping for watermarks / artifacts / perspective distortion
            transforms.RandomPerspective(
                distortion_scale=0.3,
                p=0.6
            ),

            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=10
            ),

            transforms.ToTensor(),

            # Random erasing after ToTensor
            transforms.RandomErasing(
                p=0.7,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.0),
                value=0,
                inplace=False
            ),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        train_dataset = MyDataset(
            img_dir=data_path + "/train_images",
            csv_path=data_path + "/train_labels.csv",
            transform=train_transform,
            is_test=False
        )
        return train_dataset, 2

    elif mode == "test":
        val_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(480),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        test_dataset = MyDataset(
            img_dir=data_path + "/test_images",
            csv_path=data_path + "/sample_submission.csv",
            transform=val_transform,
            is_test=True
        )
        return test_dataset, 2

    raise NotImplementedError
def build_all_datasets(data_path, args, val_ratio=0.1, seed=35):
    """
    Return train, val, test datasets.

    Binary classification:
        num_classes = 2

    Train transform follows the final report:
        RandomResizedCrop(480, scale=(0.3, 1.0), ratio=(0.75, 1.33))
        RandomHorizontalFlip(0.5)
        RandomVerticalFlip(0.5)
        RandomPerspective(distortion_scale=0.3, p=0.6)
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)
        ToTensor
        RandomErasing(p=0.7, scale=(0.02, 0.2), ratio=(0.3, 3.0), value=0)
        Normalize(ImageNet mean/std)

    Val/Test transform:
        Resize(512)
        CenterCrop(480)
        ToTensor
        Normalize(ImageNet mean/std)
    """
    import numpy as np
    from torch.utils.data import Subset
    from torchvision import transforms

    num_classes = 2


    if args.model == "dinov2":
        size = 448
    elif args.model == "dinov3":
        size = 512
    else:
        size = 480

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=size,
            scale=(0.3, 1.0),
            ratio=(0.75, 1.33)
        ),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        transforms.RandomPerspective(
            distortion_scale=0.3,
            p=0.6
        ),

        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=10
        ),

        transforms.ToTensor(),

        transforms.RandomErasing(
            p=0.7,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.0),
            value=0,
            inplace=False
        ),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


    val_test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


    full_train_dataset_for_train = MyDataset(
        img_dir=data_path + "/train_images/train_images",
        csv_path=data_path + "/train_labels.csv",
        transform=train_transform,
        is_test=False
    )

    #  only transform is different
    full_train_dataset_for_val = MyDataset(
        img_dir=data_path + "/train_images/train_images",
        csv_path=data_path + "/train_labels.csv",
        transform=val_test_transform,
        is_test=False
    )

    test_dataset = MyDataset(
        img_dir=data_path + "/test_images/test_images",
        csv_path=data_path + "/sample_submission.csv",
        transform=val_test_transform,
        is_test=True
    )

    if hasattr(full_train_dataset_for_train, "df") and "label" in full_train_dataset_for_train.df.columns:
        labels = np.array(full_train_dataset_for_train.df["label"].values)
    elif hasattr(full_train_dataset_for_train, "labels"):
        labels = np.array(full_train_dataset_for_train.labels)
    elif hasattr(full_train_dataset_for_train, "targets"):
        labels = np.array(full_train_dataset_for_train.targets)
    else:
        raise AttributeError(
            "Cannot find labels from dataset. Expected one of: "
            "dataset.df['label'], dataset.labels, or dataset.targets."
        )

    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    # stratified
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)

        num_val = int(len(class_indices) * val_ratio)

        val_indices.extend(class_indices[:num_val])
        train_indices.extend(class_indices[num_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_dataset = Subset(full_train_dataset_for_train, train_indices)
    val_dataset = Subset(full_train_dataset_for_val, val_indices)

    return train_dataset, val_dataset, test_dataset, num_classes