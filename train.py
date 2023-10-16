#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mlflow
import click
import logging
import mlflow.sklearn
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

@click.command()
@click.option("--min_n", "-min", type=int, default="1", help="neighbors最小n值")
@click.option("--max_n", "-max", type=int, default="31", help="neighbors最大n值")
@click.option("--training_data", "-td", type=str, default="./data/housing.csv", help="所需数据")
def train(min_n, max_n, training_data):

    input_data = pd.read_csv(training_data)

    # 数据预处理
    encode_label(input_data)
    logging.info(input_data['ocean_proximity'].describe())
    input_data = imputer_by_median(input_data)
    logging.info(input_data['total_bedrooms'].describe())
    show_data_summary(input_data)

    scale_data(input_data)
    compare_scale_data(pd.read_csv("./data/housing.csv"), input_data)

    # 保存处理后的数据
    input_data.to_csv('./data/result.csv')

    # 拆分数据
    train_set, test_set = train_test_split(input_data,
                                           test_size=0.2, random_state=59)
    train_data, train_value = split_house_value(train_set)
    test_data, test_value = split_house_value(test_set)
    show_data_summary(test_set)

    # 在平台中记录父运行的参数
    mlflow.log_param("split", 0.2)
    mlflow.log_param("random", True)
    mlflow.log_param("scale", "MinMaxScaler")
    mlflow.log_param("algorithm", "k-nearest neighbors")
    mlflow.log_param("Imputation", "median")

    # 建模训练
    cv_scores = []
    k_range = range(min_n, max_n)
    for n in k_range:
        # 根据超参数的不同，拆分为不同的运行
        with mlflow.start_run(nested=True):

            # 在平台中记录当前子运行超参数值
            mlflow.log_param("n", n)

            logging.info("knn n={}".format(n))
            knn_reg = neighbors.KNeighborsRegressor(n)
            knn_reg.fit(train_data, train_value)

            predict_value = knn_reg.predict(test_data)
            show_predict_result(test_data, test_value, predict_value, n)

            # 指标度量
            rmse, mae, r2 = eval_metrics(test_value, predict_value)

            # 交叉验证
            scores = cross_val_score(knn_reg, train_data, train_value, cv=10)
            logging.info("cross_val_scores: {}".format(scores))
            # 算数平均值作为数据指标进行可视化
            cv_scores.append(scores.mean())

            # 保存文件目录到模型库中
            mlflow.log_artifacts("./image/", "image")
            mlflow.log_artifacts("./data/", "data")

            # 在平台中记录指标
            mlflow.log_metric("r2", float(r2))
            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("mae", float(mae))
            mlflow.log_metric("cv_score_mean", float(scores.mean()))

            # 保存当前模型文件
            mlflow.sklearn.log_model(sk_model=knn_reg, artifact_path="model_n_{}".format(n), conda_env="conda.yaml")

            print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    show_cv_scores(k_range, cv_scores)


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
    # plt.show()


def data_scatter(input_data):
    """地理位置分布可视化"""
    input_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.savefig("./image/data_scatter.png")
    # plt.show()


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
    # plt.show()


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
    # plt.show()


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
    # plt.show()


if __name__ == "__main__":
    # 数据探索
    # input_data = pd.read_csv("./data/housing.csv")
    # show_data_summary(input_data)
    # data_hist(input_data)
    # data_scatter(input_data)
    train()

    # # 数据预处理
    # encode_label(input_data)
    # logging.info(input_data['ocean_proximity'].describe())
    # input_data = imputer_by_median(input_data)
    # logging.info(input_data['total_bedrooms'].describe())
    # show_data_summary(input_data)
    #
    # scale_data(input_data)
    # compare_scale_data(pd.read_csv("./data/housing.csv"), input_data)
    #
    # # 拆分数据
    # train_set, test_set = train_test_split(input_data,
    #                                        test_size=0.2, random_state=59)
    # train_data, train_value = split_house_value(train_set)
    # test_data, test_value = split_house_value(test_set)
    # show_data_summary(test_set)
    #
    # # 建模训练
    # cv_scores = []
    # k_range = range(1, 31)
    # for n in k_range:
    #     logging.info("knn n=".format(n))
    #     knn_reg = neighbors.KNeighborsRegressor(n)
    #     knn_reg.fit(train_data, train_value)
    #
    #     predict_value = knn_reg.predict(test_data)
    #     show_predict_result(test_data, test_value, predict_value, n)
    #
    #     # 指标度量
    #     eval_metrics(test_value, predict_value)
    #
    #     # 交叉验证
    #     scores = cross_val_score(knn_reg, train_data, train_value, cv=10)
    #     logging.info("cross_val_scores: {}".format(scores))
    #     # 算数平均值作为数据指标进行可视化
    #     cv_scores.append(scores.mean())
    #
    # show_cv_scores(cv_scores)
