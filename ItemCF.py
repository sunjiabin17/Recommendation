# 基于物品的协同过滤推荐算法
import random
import math
from operator import itemgetter

class ItemBasedCF():
    # 初始化相关参数
    def __init__(self):
        # 找到与目标电影相似的20个电影，为其推荐10部电影
        self.n_sim_movie = 20
        self.n_rec_movie = 10

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 物品相似度矩阵
        self.movie_sim_matrix = {}
        self.movie_popular = {} # movie_popular[movieId] = n 看过某个电影的人数
        self.movie_count = {}

        print('Similar movie number = %d' % self.n_sim_movie)
        print('Recommend movie number = %d' % self.n_rec_movie)
    
    # 读文件得到‘用户-电影’数据
    def get_dataset(self, filename, pivot=0.75):
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split('::')
            if random.random() < pivot: # 训练集
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                trainSet_len += 1
            else:                       # 测试集
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1
        print('Split trainSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)

    
    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)

    # 计算电影之间的相似度
    def calc_movie_sim(self):
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        self.movie_count = len(self.movie_popular)
        print('Total movie number = %d' % self.movie_count)

        print('Build movie co-rated movie matrix ...')
        for user, movies in self.trainSet.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    self.movie_sim_matrix.setdefault(m1, {})
                    self.movie_sim_matrix[m1].setdefault(m2, 0)
                    self.movie_sim_matrix[m1][m2] += 1
        print('Build movie co-rated movies matrix success!')

        # 计算电影之间的相似度
        print('Calculating movie similarity matrix ...')
        for m1, related_movies in self.movie_sim_matrix.items():
            for m2, count in related_movies.items():
                # 某电影的用户数为0
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    self.movie_sim_matrix[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
        print('Calculate movie similarity matrix success!')

    # 针对目标用户U，找到与其最相似的K个用户，产生N个推荐
    def recommend(self, user):
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainSet[user]    # 该用户已经看过的电影，排除推荐

        # 根据用户看过的电影，寻找与之相似的电影
        for movie, rating in watched_movies.items():
            # related_movie: 相似的电影
            # w: 相似的电影的相似度
            for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[0:K]:
                if related_movie in watched_movies: # 已经看过
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += w * float(rating)    # 相似度乘以评分求和
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]
        
    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print('Evaluation start ...')
        N = self.n_rec_movie

        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0

        # 覆盖率
        all_rec_movies = set()

        for i, user in enumerate(self.trainSet):
            test_movies = self.testSet.get(user, {})
            rec_movies = self.recommend(user)
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_movies)
        
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f'% (precision, recall, coverage))

if __name__ == '__main__':
    rating_file = 'data/ml-1m/ratings.dat'
    itemCF = ItemBasedCF()
    itemCF.get_dataset(rating_file)
    itemCF.calc_movie_sim()
    itemCF.evaluate()