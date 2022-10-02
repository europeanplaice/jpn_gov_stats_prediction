import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing, NBEATSModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
import datetime
from scipy.stats import zscore
from darts.utils.likelihood_models import GaussianLikelihood
from tqdm import tqdm
import copy
from darts.metrics.metrics import smape
import os, shutil


def old_data():
    l = []
    for t in ["出荷", "在庫", "在庫率"]:
        tmp = pd.read_excel("b2015_sgs1j.xlsx", sheet_name=t, header=2)
        tmp = tmp[tmp["業種名"] != "接続係数"]
        tmp = tmp.set_index("業種名")
        tmp.columns = list(map(lambda x: t + "__" + x, tmp.columns))
        l.append(tmp)
    df = pd.concat(l, axis=1)
    df.index = list(
        map(lambda x: datetime.datetime(int(str(x)[:4]), int(str(x)[4:]), 1), df.index)
    )
    df = df.reset_index(names="time")
    return df


def latest_data():
    l = []
    for group_key, group_val in {"業種別": "b2015_gsm1j", "品目別": "b2015_hsm1j"}.items():
        for t in ["生産", "出荷", "在庫", "在庫率"]:
            tmp = pd.read_excel(f"{group_val}.xlsx", sheet_name=t, header=2)
            tmp["品目名称"] = tmp["品目名称"].apply(lambda x: group_key + "__" + x + "__" + t)
            if t == "生産":
                del tmp["付加生産ウエイト"]
            elif t == "出荷":
                del tmp["出荷ウエイト"]
            elif t == "在庫":
                del tmp["在庫ウエイト"]
            elif t == "在庫率":
                del tmp["在庫率ウエイト"]
            del tmp["品目番号"]
            del tmp["p 202208"]
            tmp = tmp.set_index("品目名称")
            l.append(tmp)
    df = pd.concat(l)
    df.index.name = None
    df.columns = list(
        map(
            lambda x: datetime.datetime(int(str(x)[:4]), int(str(x)[4:]), 1), df.columns
        )
    )
    df = df.T
    df = df.reset_index(names="time")
    return df


def prepare_timeseries(df: pd.DataFrame, category: list[str], split: bool):
    train_group = []
    val_group = []
    group = []
    for i in category:
        try:
            df[i] = zscore(df[i].astype(float).values)
        except (TypeError, ValueError):
            print("error", i)
            continue
        series = TimeSeries.from_dataframe(df, time_col="time", value_cols=i)
        if split:
            train, val = series.split_before(pd.Timestamp("2021-01-01"))
            assert len(train) == 96
            assert len(val) == 19
            assert len(series) == 115
            train_group.append(train)
            val_group.append(val)
        group.append(series)
    return group, train_group, val_group


def main():
    df_old = old_data()
    category_old = df_old.columns[df_old.columns != "time"]
    df = latest_data()
    category = df.columns[df.columns != "time"]
    assert len(category) == len(list(set(category))), set(
        [x for x in category if category.count(x) > 1]
    )
    n_epochs_pre = 1
    n_epochs = 1
    n_epochs_finetune = 1
    group_pre, _, _ = prepare_timeseries(df_old, category_old, split=False)
    group, train_group, val_group = prepare_timeseries(df, category, split=True)
    model = NBEATSModel(
        input_chunk_length=12,
        output_chunk_length=6,
        save_checkpoints=True,
        model_name="pretrain",
        force_reset=True,
        n_epochs=0,
        batch_size=1024,
        likelihood=GaussianLikelihood(),
        num_blocks=2
    )
    model.fit(group_pre, verbose=True, epochs=n_epochs_pre)
    # shutil.copytree("darts_logs/pretrain", "darts_logs/train", dirs_exist_ok=True)
    # model = NBEATSModel.load_from_checkpoint("train", best=False)
    model.fit(
        train_group, verbose=True, epochs=n_epochs_pre + n_epochs
    )
    # shutil.copytree("darts_logs/train", "darts_logs/finetuning", dirs_exist_ok=True)
    prediction_on_val = model.predict(
        len(val_group[0]), series=train_group, num_samples=100
    )
    # model = NBEATSModel.load_from_checkpoint("finetuning", best=False)
    model.fit(group, verbose=True, epochs=n_epochs_pre + n_epochs + n_epochs_finetune)
    future_prediction = model.predict(18, series=group, num_samples=100)
    smape_val: list[float] = []
    for i in tqdm(range(len(val_group))):
        smape_val.append(smape(val_group[i], prediction_on_val[i]))
        plt.figure(figsize=(12, 6))
        group[i].plot(label="actual")
        prediction_on_val[i].plot(label="prediction as validation")
        pd.concat(
            [
                prediction_on_val[i].quantiles_df(),
                future_prediction[i].quantiles_df()
            ]).to_excel(f"data/sheets/{category[i]}.xlsx")
        future_prediction[i].plot(label="future prediction")
        plt.legend()
        plt.savefig(f"data/high_quality/{category[i]}.jpg", dpi=300)
        plt.savefig(f"data/low_quality/{category[i]}.jpg", dpi=72)
        plt.close()
    pd.DataFrame({"cat": category, "smape": smape_val}).to_excel("stats.xlsx")
    # plt.show()


if __name__ == "__main__":
    main()
