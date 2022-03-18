import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.preprocessing import Normalizer


class FMRIDataset(Dataset):
    """
    Wrapper for paired fMRI-images data.
    """
    def __init__(self, fmri, images, transform=None):
        self.fmri = fmri.astype(np.float32)
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform:
            return self.fmri[idx], self.transform(self.images[idx])
        else:
            return self.fmri[idx], self.images[idx]


def load_69_fmri(root):
    data = loadmat(root / "69dataset.mat")
    fmri = data['Y']
    images = data['X']

    # Reshape second dimension to rectangular images
    # transpose to bring images to the correct orientation
    images = np.transpose(images.reshape(-1, 28, 28), axes=(0, 2, 1))

    # train-test split
    train_images = np.concatenate([images[0:45], images[50:95]])
    test_images = np.concatenate([images[45:50], images[95:100]])
    train_fmri = np.concatenate([fmri[0:45], fmri[50:95]])
    test_fmri = np.concatenate([fmri[45:50], fmri[95:100]])

    # normalize
    scaler = Normalizer()
    train_fmri = scaler.fit_transform(train_fmri)
    test_fmri = scaler.transform(test_fmri)

    return train_fmri, test_fmri, train_images, test_images


def load_brains_data(root, subject="", visual_area=""):
    images = loadmat(root / "Y_brains.mat")  # trials x pixels
    # keep only relevant data
    images = images['Y']  # trials x pixels
    # Shape is (360, 3136)

    # Reshape second dimension to rectangular images
    # transpose to bring images to the correct orientation
    images = np.transpose(images.reshape(-1, 56, 56), axes=(0, 2, 1))

    # Provides data for train-test split
    splitting = loadmat(root / "train_testset.mat")

    # without flatten it adds one unnecessary dimension
    train_idx = splitting['trainidx'].flatten() - 1  # number of samples:288
    test_idx = splitting['testidx'].flatten() - 1  # number of samples:72
    # We subtract 1 due to indexing starting to 1

    # FMRI-data
    subjects_v1 = []  # list that contains FMRI V1 data for each subject
    subjects_v2 = []
    for i in range(1, 4):
        # path to subject folder
        subject_data = root / ("sub-0" + str(i))

        V1_data = loadmat(subject_data / ("XS0" + str(i) + "_V1.mat"))
        V2_data = loadmat(subject_data / ("XS0" + str(i) + "_V2.mat"))

        # keep only relevant data
        subjects_v1.append(V1_data['X'])  # trials x voxels
        subjects_v2.append(V2_data['X_V2'])  # trials x voxels

    # Train-test splitting
    train_img, test_img = images[train_idx], images[test_idx]

    # Subject 3: V1, V2
    train_S3V1, test_S3V1 = subjects_v1[2][train_idx], subjects_v1[2][test_idx]
    train_S3V2, test_S3V2 = subjects_v2[2][train_idx], subjects_v2[2][test_idx]

    # concatenate visual areas
    train_fmri = np.concatenate((train_S3V1, train_S3V2), axis=1)
    test_fmri = np.concatenate((test_S3V1, test_S3V2), axis=1)

    # normalize
    scaler = Normalizer()
    train_fmri = scaler.fit_transform(train_fmri)
    test_fmri = scaler.transform(test_fmri)

    return train_fmri, test_fmri, train_img, test_img


def load_neuron(root):
    fmri = np.load(root / "X.npy")  # shape: (720, 5438)
    images = (np.load(root / "y.npy").reshape(-1, 10, 10) * 255).astype('uint8')  # shape: (720, 100)

    test_idx = [0, 1, 2, 3, 4, 
                6, 7, 8, 9, 10, 
                12, 13, 14, 15, 16, 
                18, 19, 20, 21, 22, 
                24, 25, 29, 27, 28, 
                719, 718, 717, 716, 715, 
                701, 700, 699, 698, 697, 
                713, 712, 711, 710, 709, 
                707, 706, 705, 704, 703, 
                695, 694, 693, 692, 691]

    test_fmri = fmri[test_idx]
    train_fmri = np.delete(fmri, test_idx, axis=0)
    test_images = images[test_idx]
    train_images = np.delete(images, test_idx, axis=0)

    # normalize
    scaler = Normalizer()
    train_fmri = scaler.fit_transform(train_fmri)
    test_fmri = scaler.transform(test_fmri)

    return train_fmri, test_fmri, train_images, test_images
