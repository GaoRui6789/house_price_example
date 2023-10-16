import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

"""
初始的算法程序
"""
def show_data_summary(input_data):
    """数据探索"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("统计信息:")
    print(input_data.describe())

    print("前十行数据：")
    print(input_data.head(10))
    print("....")


def data_hist(input_data):
    """数据频度可视化"""
    input_data.hist(bins=100, figsize=(20, 12))
    plt.savefig("./image/data_hist.png")
    plt.show()


def data_scatter(input_data):
    """地理位置分布可视化"""
    input_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.savefig("./image/data_scatter.png")
    plt.show()


def encode_label(data):
    """字符串转换"""
    encoder = LabelEncoder()
    data["ocean_proximity"] = encoder.fit_transform(data["ocean_proximity"])


def imputer_by_median(data):
    """处理缺失数据"""
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(data)
    return pd.DataFrame(X, columns=data.columns)


def scale_data(data):
    """最大最小归一"""
    scalar = MinMaxScaler(feature_range=(0, 100), copy=False)
    scalar.fit_transform(data)


def compare_scale_data(origin, scaled):
    """数据缩放前后对比"""
    plt.subplot(2, 1, 1)
    plt.scatter(x=origin["longitude"], y=origin["latitude"],
                c=origin["median_house_value"], cmap="viridis", alpha=0.1)
    plt.subplot(2, 1, 2)
    plt.scatter(x=scaled["longitude"], y=scaled["latitude"],
                c=origin["median_house_value"], cmap="viridis", alpha=0.1)
    plt.savefig("./image/compare_scale_data.png")
    plt.show()


def show_predict_result(test_data, test_value, predict_value, n):
    """测试集数据对比"""
    ax = plt.subplot(221)
    plt.scatter(x=test_data["longitude"], y=test_data["latitude"],
                s=test_value, c="dodgerblue", alpha=0.5)
    plt.subplot(222)
    plt.hist(test_value, color="dodgerblue")

    plt.subplot(223)
    plt.scatter(x=test_data["longitude"], y=test_data["latitude"],
                s=predict_value, c="lightseagreen", alpha=0.5)
    plt.subplot(224)
    plt.hist(predict_value, color="lightseagreen")
    plt.savefig("./image/show_predict_result_n_{}.png".format(n))
    plt.show()


def split_house_value(data):
    value = data["median_house_value"].copy()
    return data.drop(["median_house_value"], axis=1), value


def eval_metrics(actual, pred):
    """指标度量"""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    print("rmse：{}， mae：{}， r2: {}".format(rmse, mae, r2))
    return rmse, mae, r2


def show_cv_scores(k_range, cv_scores):
    """可视化不同超参数下的交叉验证得分情况"""
    plt.plot(k_range, cv_scores)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.savefig("./image/show_cv_scores.png")
    plt.show()


if __name__ == "__main__":
    input_data = pd.read_csv("./data/housing.csv")
    # show_data_summary(input_data)
    # data_hist(input_data)
    # data_scatter(input_data)

    encode_label(input_data)
    # print(input_data['ocean_proximity'].describe())
    input_data = imputer_by_median(input_data)
    # print(input_data['total_bedrooms'].describe())
    # show_data_summary(input_data)

    scale_data(input_data)
    compare_scale_data(pd.read_csv("./data/housing.csv"), input_data)

    train_set, test_set = train_test_split(input_data,
                                           test_size=0.2, random_state=59)
    train_data, train_value = split_house_value(train_set)
    test_data, test_value = split_house_value(test_set)
    # show_data_summary(test_set)



    cv_scores = []
    k_range = range(1, 31)
    for n in k_range:
        print("knn n=", n)
        knn_reg = neighbors.KNeighborsRegressor(n)
        knn_reg.fit(train_data, train_value)
        predict_value = knn_reg.predict(test_data)
        eval_metrics(test_value, predict_value)
        show_predict_result(test_data, test_value, predict_value, n)
        scores = cross_val_score(knn_reg, train_data, train_value, cv=10)
        print("cross_val_scores: {}".format(scores))
        # 算数平均值作为数据指标进行可视化
        cv_scores.append(scores.mean())

    show_cv_scores(cv_scores)
