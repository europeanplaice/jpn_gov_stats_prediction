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
        map(
            lambda x: datetime.datetime(int(str(x)[:4]), int(str(x)[4:]), 1), df.index
        )
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
        # df[i] = (df[i] - df[i].mean()) / df[i].std()
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
    assert len(category) == len(list(set(category))), set([x for x in category if category.count(x) > 1])
    # model_checkpoint = ModelCheckpoint(monitor="val_loss")
    pl_trainer_kwargs = {"callbacks": [EarlyStopping("val_loss", patience=3)]}
    n_epochs_pre = 2
    n_epochs = 4
    group_pre, _, _ = prepare_timeseries(df_old, category_old, split=False)
    group, train_group, val_group = prepare_timeseries(df, category, split=True)
    model = NBEATSModel(
        input_chunk_length=12,
        output_chunk_length=6,
        # pl_trainer_kwargs=pl_trainer_kwargs,
        save_checkpoints=True,
        model_name="a",
        force_reset=True,
        n_epochs=0,
        batch_size=1024,
        likelihood=GaussianLikelihood()
        # dropout=0.2,
        # optimizer_kwargs={"lr": 1e-4},
    )
    model.fit(group_pre, verbose=True, epochs=n_epochs_pre)
    model = NBEATSModel.load_from_checkpoint("a", best=False)
    model.fit(train_group, val_series=val_group, verbose=True, epochs=n_epochs_pre + n_epochs)
    prediction_on_val = model.predict(len(val_group[0]), series=train_group, num_samples=100)
    prediction_on_val.to_pickle("prediction_on_val.pickle")
    model = NBEATSModel.load_from_checkpoint("a", best=False)
    model.fit(group, verbose=True, epochs=n_epochs_pre + n_epochs + 2)
    future_prediction = model.predict(18, series=group, num_samples=100)
    future_prediction.to_pickle("future_prediction.pickle")
    for i in tqdm(range(len(val_group))):
        plt.figure(figsize=(12, 6))
        group[i].plot(label="actual")
        # import pdb; pdb.set_trace()
        # prediction[i].pd_series().plot(label="forecast", lw=3)
        prediction_on_val[i].plot(label="prediction as validation")
        future_prediction[i].plot(label="future prediction")
        plt.legend()
        plt.savefig(f"data/high/{category[i]}.jpg", dpi=300)
        plt.savefig(f"data/low/{category[i]}.jpg", dpi=72)
        plt.close()
        # plt.show()


if __name__ == "__main__":
    main()
