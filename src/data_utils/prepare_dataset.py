from data_utils.extend import ExtendedDataset
from data_utils.split import split_indices
from datasets.retina_areds import RetinaAREDSDataset, RetinaAREDSSubset
from datasets.retina_ucsf import RetinaUCSFDataset, RetinaUCSFSubset, RetinaUCSFSegDataset, RetinaUCSFSegSubset
from datasets.brain_ms import BrainMSDataset, BrainMSSubset, BrainMSSegDataset, BrainMSSegSubset, brain_MS_split
from datasets.brain_gbm import BrainGBMDataset, BrainGBMSubset, BrainGBMSegDataset, BrainGBMSegSubset
from datasets.synthetic import SyntheticDataset, SyntheticSubset
from datasets.synthetic import get_time as synth_get_time
from datasets.synthetic3d import Synthetic3DDataset, Synthetic3DSubset
from datasets.synthetic3d import get_time_3d as synth3d_get_time
from datasets.brain_adni import BrainADNIDataset, BrainADNISubset
from datasets.brain_adni import get_time_adni as adni_get_time
# from datasets.brain_metastases import BrainMetastasesDataset, BrainMetastasesSubset
from torch.utils.data import DataLoader
from utils.attribute_hashmap import AttributeHashmap


def prepare_dataset(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for predicting one future timepoint from one earlier timepoint.
    '''

    # Read dataset.
    if config.dataset_name == 'retina_areds':
        dataset = RetinaAREDSDataset(eye_mask_folder=config.eye_mask_folder)
        Subset = RetinaAREDSSubset

    elif config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.target_dim)
        Subset = BrainGBMSubset

    elif config.dataset_name == 'brain_metastases':
        dataset = BrainMetastasesDataset(target_dim=config.target_dim)
        Subset = BrainMetastasesSubset

    elif config.dataset_name == 'synthetic':
        dataset = SyntheticDataset(base_path=config.dataset_path,
                                image_folder=config.image_folder,
                                target_dim=config.target_dim)
        Subset = SyntheticSubset

    elif config.dataset_name == 'synthetic3d':
        dataset = Synthetic3DDataset(base_path=config.dataset_path,
                                     image_folder=config.image_folder,
                                     target_dim=config.target_dim)
        Subset = Synthetic3DSubset

    elif config.dataset_name == 'brain_adni':
        dataset = BrainADNIDataset(base_path=config.dataset_path,
                                   target_dim=config.target_dim)
        Subset = BrainADNISubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # For synthetic dataset, ensure max_t is computed to avoid division by zero later.
    if config.dataset_name == 'synthetic':
        try:
            if not hasattr(dataset, 'max_t') or dataset.max_t == 0:
                computed_max_t = 0
                for image_list in getattr(dataset, 'image_by_patient', []):
                    for p in image_list:
                        try:
                            computed_max_t = max(computed_max_t, synth_get_time(p))
                        except Exception:
                            pass
                # Fallback to 1.0 if still zero to avoid division by zero downstream
                dataset.max_t = computed_max_t if computed_max_t > 0 else 1.0
        except Exception:
            # Defensive fallback
            dataset.max_t = 1.0

    # For synthetic3d dataset, ensure max_t is computed
    if config.dataset_name == 'synthetic3d':
        try:
            if not hasattr(dataset, 'max_t') or dataset.max_t == 0:
                computed_max_t = 0
                for volume_list in getattr(dataset, 'volumes_by_subject', []):
                    for p in volume_list:
                        try:
                            computed_max_t = max(computed_max_t, synth3d_get_time(p))
                        except Exception:
                            pass
                dataset.max_t = computed_max_t if computed_max_t > 0 else 1.0
        except Exception:
            dataset.max_t = 1.0

    # For brain_adni dataset, ensure max_t is computed
    if config.dataset_name == 'brain_adni':
        try:
            if not hasattr(dataset, 'max_t') or dataset.max_t == 0:
                computed_max_t = 0
                for volume_list in getattr(dataset, 'volumes_by_subject', []):
                    for p in volume_list:
                        try:
                            computed_max_t = max(computed_max_t, adni_get_time(p))
                        except Exception:
                            pass
                dataset.max_t = computed_max_t if computed_max_t > 0 else 1.0
        except Exception:
            dataset.max_t = 1.0

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))

    if hasattr(config, 'subject_idx') and config.subject_idx is not None:
        # Handle both single int (if passed differently) or list
        subject_indices = config.subject_idx
        if isinstance(subject_indices, int):
            subject_indices = [subject_indices]
        
        # Ensure all are ints
        subject_indices = [int(x) for x in subject_indices]

        for subject_idx in subject_indices:
            if subject_idx < 0 or subject_idx >= len(dataset):
                raise ValueError(f"Subject index {subject_idx} out of bounds. Dataset size: {len(dataset)}")
        
        # Get standard split first
        train_indices, val_indices, _ = split_indices(indices=indices, splits=ratios, random_seed=1)
        
        # Override test set
        test_indices = list(subject_indices)
        
        # Remove from other sets if present
        for subject_idx in subject_indices:
            if subject_idx in train_indices:
                train_indices.remove(subject_idx)
            if subject_idx in val_indices:
                val_indices.remove(subject_idx)
    else:
        train_indices, val_indices, test_indices = \
            split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_aug = None
    if len(transforms_list) == 4:
        transforms_train, transforms_val, transforms_test, transforms_aug = transforms_list
    else:
        transforms_train, transforms_val, transforms_test = transforms_list

    # Some dataset Subset classes (e.g., SyntheticSubset, Synthetic3DSubset) do not accept
    # `transforms` / `transforms_aug` kwargs. Only pass them when available.
    if config.dataset_name in ['synthetic', 'synthetic3d', 'brain_adni', 'brain_metastases']:
        train_set = Subset(main_dataset=dataset,
                           subset_indices=train_indices,
                           return_format='one_pair')
        val_set = Subset(main_dataset=dataset,
                         subset_indices=val_indices,
                         return_format='all_pairs')
        
        test_return_format = 'all_pairs'
        if getattr(config, 'test_min_max', False) and config.dataset_name in ['synthetic3d', 'brain_adni']:
            test_return_format = 'min_max_pair'
            
        test_set = Subset(main_dataset=dataset,
                          subset_indices=test_indices,
                          return_format=test_return_format)
    else:
        train_set = Subset(main_dataset=dataset,
                           subset_indices=train_indices,
                           return_format='one_pair',
                           transforms=transforms_train,
                           transforms_aug=transforms_aug)
        val_set = Subset(main_dataset=dataset,
                         subset_indices=val_indices,
                         return_format='all_pairs',
                         transforms=transforms_val)
        test_set = Subset(main_dataset=dataset,
                          subset_indices=test_indices,
                          return_format='all_pairs',
                          transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set), min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=1,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_npt(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for predicting one future timepoint from potentially multiple earlier timepoints.
    '''

    # Read dataset.
    if config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.target_dim)
        Subset = BrainGBMSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='all_subsequences',
                       transforms=transforms_train)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_subsequences',
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_subsequences',
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set), min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=1,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_full_sequence(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for iterating over all full sequences.
    '''

    # Read dataset.
    if config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.target_dim)
        Subset = BrainGBMSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    full_set = Subset(main_dataset=dataset,
                      subset_indices=indices,
                      return_format='full_sequence',
                      transforms=None)

    full_set = DataLoader(dataset=full_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return full_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_all_subarrays(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for iterating over all subarrays.
    This means we will not consider subsequences with dropped out intermediate values.
    '''

    # Read dataset.
    if config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.target_dim)
        Subset = BrainGBMSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='all_subarrays',
                       transforms=transforms_train)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_subarrays',
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_subarrays',
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set), min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=1,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_segmentation(config: AttributeHashmap, transforms_list = [None, None, None]):
    # Read dataset.
    if config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFSegDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSegSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSSegDataset(target_dim=config.target_dim)
        Subset = BrainMSSegSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMSegDataset(target_dim=config.target_dim)
        Subset = BrainGBMSegSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    if config.dataset_name == 'brain_ms':
        train_indices, val_indices, test_indices = brain_MS_split()

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       transforms=transforms_train)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set) / config.batch_size, min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=config.batch_size,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel()
