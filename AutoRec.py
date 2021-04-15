# AutoRec--单隐层神经网络推荐模型
import torch
import torch.nn as nn
import numpy as np
import math
import time
import argparse
import torch.utils.data as Data
import torch.optim as optim


class AutoRec(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(AutoRec, self).__init__()

        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_units = args.hidden_units   # 隐藏层神经元个数
        self.lambda_value = args.lambda_value

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.hidden_units),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_units, self.num_items),
        )

    def forward(self, torch_input):
        encoder = self.encoder(torch_input)
        decoder = self.decoder(encoder)
        return decoder

    def loss(self, decoder, input, optimizer, mask_input):
        cost = 0
        temp2 = 0

        # 预测值decoder减去实际值input的均方误差
        cost += ((decoder-input) * mask_input).pow(2).sum()
        rmse = cost

        # 正则项
        for i in optimizer.param_groups:
            for j in i['params']:
                if j.data.dim() == 2:
                    temp2 += torch.t(j.data).pow(2).sum()
        cost += temp2 * self.lambda_value * 0.5
        return cost, rmse


def train(epoch, loader):
    RMSE = 0
    cost_all = 0
    for step, (batch_x, batch_mask_x, batch_y) in enumerate(loader):    # batch_y用不到
        batch_x = batch_x.type(torch.FloatTensor)
        batch_mask_x = batch_mask_x.type(torch.FloatTensor)
        if args.cuda:
            batch_x = batch_x.cuda()
            batch_mask_x = batch_mask_x.cuda()

        decoder = autoRec(batch_x)
        loss, rmse = autoRec.loss(decoder=decoder, input=batch_x, optimizer=optimizer, mask_input=batch_mask_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cost_all += loss
        RMSE += rmse

    RMSE = np.sqrt(RMSE.detach().cpu().numpy() / (train_mask_r == 1).sum())
    print('epch ', epoch, 'train RMSE: ', RMSE)


def test(epoch):
    test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor)
    test_mask_r_tensor = torch.from_numpy(test_mask_r).type(torch.FloatTensor)
    if args.cuda:
        test_r_tensor = test_r_tensor.cuda()
        test_mask_r_tensor = test_mask_r_tensor.cuda()

    decoder = autoRec(test_r_tensor)

    unseen_user_test_list = list(user_test_set - user_train_set)
    unseen_item_test_list = list(item_test_set - item_train_set)

    for user in unseen_user_test_list:
        for item in unseen_item_test_list:
            if test_mask_r[user, item] == 1:
                decoder[user, item] = 3

    mse = ((decoder - test_r_tensor) * test_mask_r_tensor).pow(2).sum()
    RMSE = mse.detach().cpu().numpy() / (test_mask_r == 1).sum()
    RMSE = np.sqrt(RMSE)

    print('epoch ', epoch, 'test RMSE: ', RMSE)


def get_data(path, num_users, num_items, num_total_ratings, train_ratio):
    fp = open(path + 'ratings.dat')
    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    train_r = np.zeros((num_users, num_items))
    test_r = np.zeros((num_users, num_items))

    train_mask_r = np.zeros((num_users, num_items)) # 用来标记user对item评分了
    test_mask_r = np.zeros((num_users, num_items))  # 用来标记user对item评分了

    random_perm_idx = np.random.permutation(num_total_ratings)
    train_idx = random_perm_idx[0:int(num_total_ratings*train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings*train_ratio):]

    lines = fp.readlines()

    '''Train'''
    for itr in train_idx:
        line = lines[itr]
        user, item, rating, _ = line.split('::')
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        train_r[user_idx, item_idx] = int(rating)
        train_mask_r[user_idx, item_idx] = 1

        user_train_set.add(user_idx)
        item_train_set.add(item_idx)

    '''Test'''
    for itr in test_idx:
        line = lines[itr]
        user, item, rating, _ = line.split('::')
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_r[user_idx, item_idx] = int(rating)
        test_mask_r[user_idx, item_idx] = 1

        user_test_set.add(user_idx)
        item_test_set.add(item_idx)

    return train_r, train_mask_r, test_r, test_mask_r, \
        user_train_set, item_train_set, user_test_set, item_test_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='I-AutoRec ')
    parser.add_argument('--hidden_units', type=int, default=500)
    parser.add_argument('--lambda_value', type=float, default=1)
    parser.add_argument('--train_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
    parser.add_argument('--grad_clip', type=bool, default=False)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")
    parser.add_argument('--random_seed', type=int, default=1000)
    parser.add_argument('--display_step', type=int, default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    np.random.seed(args.random_seed)
    num_users = 6040
    num_items = 3952
    num_total_ratings = 1000209
    train_ratio = 0.9   # 用来划分训练集和测试集，90%为训练集

    data_name = 'ml-1m'
    path = './%s' % data_name + '/'

    train_r, train_mask_r, test_r, test_mask_r, \
        user_train_set, item_train_set, user_test_set, item_test_set = \
        get_data(path, num_users, num_items, num_total_ratings, train_ratio)

    autoRec = AutoRec(args, num_users, num_items)
    if args.cuda:
        autoRec.cuda()

    optimizer = optim.Adam(autoRec.parameters(), lr=args.base_lr, weight_decay=1e-4)
    num_batch = int(math.ceil(num_users/args.batch_size))
    torch_dataset = Data.TensorDataset(torch.from_numpy(train_r), torch.from_numpy(train_mask_r), torch.from_numpy(train_r))
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    for epoch in range(args.train_epoch):
        train(epoch=epoch, loader=loader)
        test(epoch=epoch)