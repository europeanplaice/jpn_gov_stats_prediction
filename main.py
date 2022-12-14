import datetime
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts.metrics.metrics import smape
from darts.models.forecasting.nbeats import NBEATSModel
from darts.timeseries import TimeSeries
from darts.utils.likelihood_models import GaussianLikelihood
from tqdm import tqdm


def old_data() -> pd.DataFrame:
    """
    Gets data before 2015.
    The format is different from the one after 2015.
    The data is used for additional training.
    """
    datalist = []
    for t in ["出荷", "在庫", "在庫率"]:
        tmp = pd.read_excel("b2015_sgs1j.xlsx", sheet_name=t, header=2)
        tmp = tmp[tmp["業種名"] != "接続係数"]
        tmp = tmp.set_index("業種名")
        tmp.columns = list(map(lambda x: t + "__" + x, tmp.columns))
        datalist.append(tmp)
    df = pd.concat(datalist, axis=1)
    df.index = list(
        map(lambda x: datetime.datetime(int(str(x)[:4]), int(str(x)[4:]), 1), df.index)
    )
    df = df.reset_index(names="time")
    return df


def latest_data() -> pd.DataFrame:
    """
    Gets data after 2015.
    """
    datalist = []
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
            datalist.append(tmp)
    df = pd.concat(datalist)
    df.index.name = None
    df.columns = list(
        map(
            lambda x: datetime.datetime(int(str(x)[:4]), int(str(x)[4:]), 1), df.columns
        )
    )
    df = df.T
    df = df.reset_index(names="time")
    return df


def prepare_timeseries(
    df: pd.DataFrame, category: list[str], split: bool
) -> Tuple[list[TimeSeries], list[TimeSeries], list[TimeSeries], list[str]]:
    """
    Convert pandas dataframe to darts timeseries
    """
    train_group: list[TimeSeries] = []
    val_group: list[TimeSeries] = []
    group: list[TimeSeries] = []
    invalid = []
    for i in category:
        try:
            df[i] = df[i].astype(float).values / 100.0
            series = TimeSeries.from_dataframe(df, time_col="time", value_cols=i)
        except (TypeError, ValueError):
            print("error", i)
            invalid.append(i)
            continue
        if split:
            train, val = series.split_before(pd.Timestamp("2021-01-01"))
            assert len(train) == 96
            assert len(val) == 19
            assert len(series) == 115
            train_group.append(train)
            val_group.append(val)
        group.append(series)
    for i in invalid:
        category.remove(i)
    return group, train_group, val_group, category


def main() -> None:
    df_old = old_data()
    category_old = df_old.columns[df_old.columns != "time"].tolist()
    df = latest_data()
    category: list[str] = df.columns[df.columns != "time"].tolist()
    assert len(category) == len(list(set(category))), set(
        [x for x in category if category.count(x) > 1]
    )
    n_epochs_pretrain = 1
    n_epochs = 1
    n_epochs_finetune = 1
    pretrain_group, _, _, category_old = prepare_timeseries(
        df_old, category_old, split=False
    )
    group, train_group, val_group, category = prepare_timeseries(
        df, category, split=True
    )
    likelihood = GaussianLikelihood()
    model = NBEATSModel(
        input_chunk_length=18,
        output_chunk_length=6,
        save_checkpoints=True,
        model_name="train",
        force_reset=True,
        n_epochs=0,
        batch_size=32,
        likelihood=likelihood,
        num_blocks=1,
        pl_trainer_kwargs={"accelerator": "auto"},
        generic_architecture=True,
        optimizer_kwargs={"lr": 1e-4},
    )
    model.fit(pretrain_group, verbose=True, epochs=n_epochs_pretrain)
    model.fit(train_group, verbose=True, epochs=n_epochs)
    prediction_on_val = model.predict(
        len(val_group[0]), series=train_group, num_samples=200
    )
    global_smape = smape(val_group, prediction_on_val)
    print("global smape", np.mean(global_smape))
    model.fit(group, verbose=True, epochs=n_epochs_finetune)
    future_prediction = model.predict(18, series=group, num_samples=200)
    smape_val: list[float] = []
    os.makedirs("data/sheets", exist_ok=True)
    os.makedirs("data/high_quality", exist_ok=True)
    os.makedirs("data/low_quality", exist_ok=True)
    for i in tqdm(range(len(val_group))):
        smape_val.append(smape(val_group[i], prediction_on_val[i]))
        plt.figure(figsize=(12, 6))
        group[i].plot(label="actual")
        prediction_on_val[i].plot(label="prediction as validation")
        pd.concat(
            [prediction_on_val[i].quantiles_df(), future_prediction[i].quantiles_df()]
        ).to_excel(f"data/sheets/{category[i]}.xlsx")
        future_prediction[i].plot(label="future prediction")
        plt.legend()
        plt.savefig(f"data/high_quality/{category[i]}.jpg", dpi=300)
        plt.savefig(f"data/low_quality/{category[i]}.jpg", dpi=72)
        plt.close()
    pd.DataFrame({"cat": category, "smape": smape_val}).to_excel("stats.xlsx")


if __name__ == "__main__":
    main()
