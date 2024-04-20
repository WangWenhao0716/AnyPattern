# Written by Yixiao Ge
from .distributed_identity_sampler import (
    DistributedIdentitySampler,
    DistributedJointIdentitySampler,
)
from .distributed_slice_sampler import (
    DistributedJointSliceSampler,
    DistributedSliceSampler,
)

__all__ = ["build_train_sampler", "build_test_sampler"]


def build_train_sampler(num_instances, datasets, epoch=0):

    num_instances = num_instances
    shuffle = True

    if num_instances > 0:
        # adopt PxK sampler
        # for a single dataset
        return DistributedIdentitySampler(
            datasets,
            num_instances=num_instances,
            shuffle=shuffle,
            epoch=epoch,
        )

    else:
        # adopt normal random sampler
        if isinstance(datasets, (tuple, list)):
            # for a list of individual datasets
            samplers = []
            for dataset in datasets:
                samplers.append(
                    DistributedSliceSampler(dataset.data, shuffle=shuffle, epoch=epoch,)
                )
            return samplers
        else:
            # for a single dataset
            return DistributedSliceSampler(datasets.data, shuffle=shuffle, epoch=epoch,)


def build_test_sampler(cfg, datasets):

    if isinstance(datasets, (tuple, list)):
        # for a list of individual datasets
        samplers = []
        for dataset in datasets:
            samplers.append(DistributedSliceSampler(dataset.data, shuffle=False,))
        return samplers

    else:
        # for a single dataset
        return DistributedSliceSampler(datasets.data, shuffle=False,)