import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
from scipy.io import loadmat
from slicegan.util import read_mat

def batch(data,type,l, sf):
    """
    Dataset used in the original paper: https://github.com/bispl-kaist/SliceGAN_AdaIN
    :param data: data path
    :param type: data type
    :param l: image size
    :param sf: scale factor
    :return:
    """
    Testing = False  # If true, visualize data before training starts
    if type == 'png' or type == 'jpg':
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            if len(img.shape)>2:
                img = img[:,:,0]
            img = img[::sf,::sf]
            x_max, y_max= img.shape[:]
            phases = np.unique(img)
            data = np.empty([32 * 900, len(phases), l, l])
            for i in range(32 * 900):
                x = np.random.randint(1, x_max - l-1)
                y = np.random.randint(1, y_max - l-1)
                # create one channel per phase for one hot encoding
                for cnt, phs in enumerate(phases):
                    img1 = np.zeros([l, l])
                    img1[img[x:x + l, y:y + l] == phs] = 1
                    data[i, cnt, :, :] = img1

            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :]+2*data[j, 1, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='tif':
        datasetxyz=[]
        # img = np.array(tifffile.imread(data[0]))
        img = read_mat(data[0])
        img = img[::sf,::sf,::sf]
        ## Create a data store and add random samples from the full image
        x_max, y_max, z_max = img.shape[:]
        print('training image shape: ', img.shape)
        vals = np.unique(img)
        for dim in range(3):
            data = np.empty([32 * 900, len(vals), l, l])
            print('dataset ', dim)
            for i in range(32*900):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                z = np.random.randint(0, z_max - l)
                # create one channel per phase for one hot encoding
                lay = np.random.randint(img.shape[dim]-1)
                for cnt,phs in enumerate(list(vals)):
                    img1 = np.zeros([l,l])
                    if dim==0:
                        img1[img[lay, y:y + l, z:z + l] == phs] = 1
                    elif dim==1:
                        img1[img[x:x + l,lay, z:z + l] == phs] = 1
                    else:
                        img1[img[x:x + l, y:y + l,lay] == phs] = 1
                    data[i, cnt, :, :] = img1[:,:]
                    # data[i, (cnt+1)%3, :, :] = img1[:,:]

            if Testing:
                for j in range(2):
                    plt.imshow(data[j, 0, :, :] + 2 * data[j, 1, :, :])
                    plt.pause(1)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='colour':
        ## Create a data store and add random samples from the full image
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            img = img[::sf,::sf,:]
            ep_sz = 32 * 900
            data = np.empty([ep_sz, 3, l, l])
            x_max, y_max = img.shape[:2]
            for i in range(ep_sz):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                # create one channel per phase for one hot encoding
                data[i, 0, :, :] = img[x:x + l, y:y + l,0]
                data[i, 1, :, :] = img[x:x + l, y:y + l,1]
                data[i, 2, :, :] = img[x:x + l, y:y + l,2]
            print('converting')
            if Testing:
                datatest = np.swapaxes(data,1,3)
                datatest = np.swapaxes(datatest,1,2)
                for j in range(5):
                    plt.imshow(datatest[j, :, :, :])
                    plt.pause(0.5)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='grayscale':
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = img/img.max()
            img = img[::sf, ::sf]
            x_max, y_max = img.shape[:]
            data = np.empty([32 * 900, 1, l, l])
            for i in range(32 * 900):
                x = np.random.randint(1, x_max - l - 1)
                y = np.random.randint(1, y_max - l - 1)
                subim = img[x:x + l, y:y + l]
                data[i, 0, :, :] = subim
            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)
    return datasetxyz


def batch_multi(data_list, codes, type, l, sf):
    """ Used for SliceGAN-AdaIN
    Simply concatenates data with different 'codes' to make up the whole dataset
        :param data_list: list of data path
        :param codes: list of scalars representing granular size
        :param type: data type
        :param l: image size
        :param sf: scale factor
        :return:
    """
    sz = 32 * 250
    if type == 'tif':
        mult_data = []
        for idx, code in enumerate(codes):
            datasetxyz = []
            img = read_mat(data_list[idx])
            img = img[::sf, ::sf, ::sf]
            ## Create a data store and add random samples from the full image
            x_max, y_max, z_max = img.shape[:]
            print('training image shape: ', img.shape)
            vals = np.unique(img)
            for dim in range(3):
                data = np.empty([sz, len(vals), l, l])
                print('dataset ', dim)
                for i in range(sz):
                    x = np.random.randint(0, x_max - l)
                    y = np.random.randint(0, y_max - l)
                    z = np.random.randint(0, z_max - l)
                    # create one channel per phase for one hot encoding
                    lay = np.random.randint(img.shape[dim] - 1)
                    for cnt, phs in enumerate(list(vals)):
                        img1 = np.zeros([l, l])
                        if dim == 0:
                            img1[img[lay, y:y + l, z:z + l] == phs] = 1
                        elif dim == 1:
                            img1[img[x:x + l, lay, z:z + l] == phs] = 1
                        else:
                            img1[img[x:x + l, y:y + l, lay] == phs] = 1
                        data[i, cnt, :, :] = img1[:, :]
                data = torch.FloatTensor(data)
                datasetxyz.append(data)
            sincode = torch.ones(8000, 128) * code * 1e-2
            datasetxyz.append(sincode)
            mult_data.append(datasetxyz)
        multi_datasetxyz = multidataset(mult_data[0], mult_data[1], mult_data[2])

    return multi_datasetxyz


class multidataset(torch.utils.data.Dataset):
    ''' Used for SliceGAN-AdaIN
    data1, data2, data3 are stacked tensors with different code vectors
        returns: Dataset divided into 3 different 2D planes, and code vector.
    '''
    def __init__(self, data1, data2, data3):
        self.datax = torch.cat((data1[0], data2[0], data3[0]), dim=0)
        self.datay = torch.cat((data1[1], data2[1], data3[1]), dim=0)
        self.dataz = torch.cat((data1[2], data2[2], data3[2]), dim=0)
        self.code = torch.cat((data1[3], data2[3], data3[3]), dim=0)

    def __len__(self):
        return self.datax.shape[0]

    def __getitem__(self, idx):
        return self.datax[idx], self.datay[idx], self.dataz[idx], self.code[idx]



