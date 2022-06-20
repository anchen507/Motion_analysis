from utils.data_utils import *
import matplotlib.pyplot as plt
class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir,patch_size, input_frame):
        super(TrainSetLoader).__init__()
        self.dir = dataset_dir
        self.patch_size=patch_size
        self.input_frame=input_frame
        with open(dataset_dir+'/sep_trainlist.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        np.random.shuffle(self.train_list)
        self.tranform = augumentation()
    def __getitem__(self, idx):
        HR = []
        FLOW=[]
        strat_frame_idx=random.randint(0,7-self.input_frame)
        for i in range(strat_frame_idx,strat_frame_idx+self.input_frame):
            img_hr = Image.open(self.dir + '/sequences/' +self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_hr = img_hr.transpose(2,0,1)
            HR.append(img_hr)
        HR = np.stack(HR, 1)
        combine_data=HR
        combine_data = random_crop(combine_data, self.patch_size)

        #combine_data = self.tranform(combine_data)

        combine_data = torch.from_numpy(np.ascontiguousarray(combine_data))

        HR = combine_data[0:3,:,:,:]

        return  HR
    def __len__(self):
        return len(self.train_list)

class ValidSetLoader(Dataset):
    def __init__(self, dataset_dir,patch_size, input_frame):
        super(ValidSetLoader).__init__()
        self.dir = dataset_dir
        self.patch_size=patch_size
        self.input_frame=input_frame
        with open(dataset_dir+'/sep_testlist.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.tranform = augumentation()
    def __getitem__(self, idx):
        HR = []
        strat_frame_idx=random.randint(0,7-self.input_frame)
        for i in range(strat_frame_idx,strat_frame_idx+self.input_frame):
            img_hr = Image.open(self.dir + '/sequences/' + self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_hr = img_hr.transpose(2,0,1)
            HR.append(img_hr)

        HR = np.stack(HR, 1)

        #HR = random_crop(HR, self.patch_size)
        #HR = self.tranform(HR)

        HR = torch.from_numpy(np.ascontiguousarray(HR))

        return HR

    def __len__(self):
        return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir,patch_size, input_frame):
        super(TestSetLoader).__init__()
        self.dir = dataset_dir
        self.patch_size=patch_size
        self.input_frame=input_frame
        with open(dataset_dir+'/test_HFS.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.tranform = augumentation()
    def __getitem__(self, idx):
        HR = []
        strat_frame_idx=0
        for i in range(strat_frame_idx,strat_frame_idx+self.input_frame):
            img_hr = Image.open(self.dir + '/SynMVD/' + self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_hr = img_hr.transpose(2,0,1)
            C,H,W=np.shape(img_hr)
            new_H=(H//64)*64
            new_W=(W//64)*64
            HR.append(img_hr[:,:new_H,:new_W])

        HR = np.stack(HR, 1)

        HR = torch.from_numpy(np.ascontiguousarray(HR))

        return HR


    def __len__(self):

        return len(self.train_list)

class augumentation(object):
    def __call__(self, input):
        if random.random()<0.5:
            input = input[::-1, :, :]
        if random.random()<0.5:
            input = input[:, ::-1, :]
        if random.random()<0.5:
            input = input.transpose(0, 1, 3, 2)#C N H W
        return input
