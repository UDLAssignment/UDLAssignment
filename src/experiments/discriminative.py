import torch
import torch.optim as optim
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose
from torch.utils.data import ConcatDataset
from models.coreset import RandomCoreset
from util.experiment_utils import run_point_estimate_initialisation, run_task
from util.transforms import Flatten, Scale, Permute
from util.datasets import NOTMNIST
from util.outputs import save_model
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import wandb

def permuted_mnist(config):
    """
    Runs the 'Permuted MNIST' experiment from the VCL paper, in which each task
    is obtained by applying a fixed random permutation to the pixels of each image.
    """

    writer = wandb.init(
        project="udl" if not config.test else "test_udl",
        name=config.name
    )
    wandb.config = config

    # flattening and permutation used for each task
    torch.manual_seed(9)
    transforms = [Compose([Flatten(), Scale(), Permute(torch.randperm(config.mnist_flattened_dim))]) for _ in range(config.n_tasks)]

    if config.transform_swaps is not None:
        swap_idxs = [int(idx) for idx in config.transform_swaps.split(":")]
        transform = transforms[swap_idxs[0]]
        transforms[swap_idxs[0]] = transforms[swap_idxs[1]]
        transforms[swap_idxs[1]] = transform

    # create model, single-headed in permuted MNIST experiment
    if config.module_type == "old":
        from models.vcl_nn import DiscriminativeVCL
        model = DiscriminativeVCL(
            config,
        ).to(config.device)
    else:
        from models.contrib import DiscriminativeVCL
        model = DiscriminativeVCL(
            config,
            x_dim=config.mnist_flattened_dim
        ).to(config.device)
    
    coreset = RandomCoreset(size=config.coreset_size)

    if config.data == "mnist":
        dataset = MNIST
    elif config.data == "fashion":
        dataset = FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {config.data}")
    
    mnist_train = ConcatDataset(
        [dataset(root="data", train=True, download=True, transform=t) for t in transforms]
    )
    task_size = len(mnist_train) // config.n_tasks
    train_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(config.n_tasks)]
    )

    mnist_test = ConcatDataset(
        [dataset(root="data", train=False, download=True, transform=t) for t in transforms]
    )
    task_size = len(mnist_test) // config.n_tasks
    test_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(config.n_tasks)]
    )

    if config.run_mle_init:
        run_point_estimate_initialisation(model=model, data=mnist_train,
                                        epochs=config.epochs, batch_size=config.batch_size,
                                        device=config.device, lr=config.lr,
                                        multiheaded=config.multiheaded,
                                        task_ids=train_task_ids)

        save_model(model, config.name + '/model_post_init.pth')
    
    model.start_variational_phase()
        
    # each task is classification of MNIST images with permuted pixels
    for task in range(config.n_tasks):
        run_task(
            model=model, train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids, task_idx=task,
            coreset=coreset, config=config,
            device=config.device, lr=config.lr, summary_writer=writer
        )


def split_mnist(config):
    """
    Runs the 'Split MNIST' experiment from the VCL paper, in which each task is
    a binary classification task carried out on a subset of the MNIST dataset.
    """

    writer = wandb.init(
        project="udl_sm" if not config.test else "test_udl_sm",
        name=config.name
    )
    wandb.config = config

    transform = Compose([Flatten(), Scale()])

    # download dataset
    mnist_train = MNIST(root="data", train=True, download=True, transform=transform)
    mnist_test = MNIST(root="data", train=False, download=True, transform=transform)

    # create model, single-headed in permuted MNIST experiment
    if config.module_type == "old":
        from models.vcl_nn import DiscriminativeVCL
        model = DiscriminativeVCL(
            config,
        ).to(config.device)
    else:
        from models.contrib import DiscriminativeVCL
        model = DiscriminativeVCL(
            config,
            x_dim=config.mnist_flattened_dim
        ).to(config.device)

    coreset = RandomCoreset(size=config.coreset_size)

    label_to_task_mapping = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3, 7: 3,
        8: 4, 9: 4,
    }

    if isinstance(mnist_train[0][1], int):
        train_task_ids = torch.Tensor([label_to_task_mapping[y] for _, y in mnist_train])
        test_task_ids = torch.Tensor([label_to_task_mapping[y] for _, y in mnist_test])
    elif isinstance(mnist_train[0][1], torch.Tensor):
        train_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in mnist_train])
        test_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in mnist_test])

    # each task is a binary classification task for a different pair of digits
    binarize_y = lambda y, task: (y == (2 * task + 1)).long()

    if config.run_mle_init:
        run_point_estimate_initialisation(model=model, data=mnist_train,
                                        epochs=config.epochs, batch_size=config.batch_size,
                                        device=config.device, lr=config.lr,
                                        multiheaded=config.multiheaded,
                                        task_ids=train_task_ids,  y_transform=binarize_y)

    for task_idx in range(config.n_tasks):
        run_task(
            model=model, train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids, task_idx=task_idx,
            coreset=coreset, config=config,
            device=config.device, lr=config.lr, summary_writer=writer, y_transform=binarize_y
        )



def split_not_mnist(config):
    """
    Runs the 'Split not MNIST' experiment from the VCL paper, in which each task
    is a binary classification task carried out on a subset of the not MNIST
    character recognition dataset.
    """
    writer = wandb.init(
        project="udl_snm" if not config.test else "test_udl_snm",
        name=config.name
    )
    wandb.config = config

    transform = Compose([Flatten(), Scale()])

    not_mnist_train = NOTMNIST(train=True, overwrite=False, transform=transform)
    not_mnist_test = NOTMNIST(train=False, overwrite=False, transform=transform)

    # create model, single-headed in permuted MNIST experiment
    if config.module_type == "old":
        from models.vcl_nn import DiscriminativeVCL
        model = DiscriminativeVCL(
            config,
        ).to(config.device)
    else:
        from models.contrib import DiscriminativeVCL
        model = DiscriminativeVCL(
            config,
            x_dim=config.mnist_flattened_dim
        ).to(config.device)

    coreset = RandomCoreset(size=config.coreset_size)

    # The y classes are integers 0-9.
    label_to_task_mapping = {
        0: 0, 1: 1,
        2: 2, 3: 3,
        4: 4, 5: 0,
        6: 1, 7: 2,
        8: 3, 9: 4,
    }

    train_task_ids = torch.from_numpy(np.array([label_to_task_mapping[y] for _, y in not_mnist_train]))
    test_task_ids = torch.from_numpy(np.array([label_to_task_mapping[y] for _, y in not_mnist_test]))

    summary_logdir = os.path.join("logs", "disc_s_n_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # each task is a binary classification task for a different pair of digits
    # binarize_y(c, n) is 1 when c is is the nth digit - A for task 0, B for task 1
    binarize_y = lambda y, task: (y == task).long()

    if config.run_mle_init:
        run_point_estimate_initialisation(model=model, data=not_mnist_train,
                                        epochs=config.epochs, batch_size=config.batch_size,
                                        device=config.device, lr=config.lr,
                                        multiheaded=config.multiheaded,
                                        task_ids=train_task_ids,  y_transform=binarize_y)

    for task_idx in range(config.n_tasks):
        run_task(
            model=model, train_data=not_mnist_train, train_task_ids=train_task_ids,
            test_data=not_mnist_test, test_task_ids=test_task_ids, task_idx=task_idx,
            coreset=coreset, config=config,
            device=config.device, lr=config.lr, summary_writer=writer, y_transform=binarize_y
        )
