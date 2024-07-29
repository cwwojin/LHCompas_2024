import pandas as pd
import os.path as path
from compas.inference import ForecastModel
import plotly.express as px
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True, help="path to the model")
    return parser.parse_args()


def main():
    args = get_args()
    model = ForecastModel(args.model_path)
    df = pd.read_csv("./data/dataset_v1.2.csv")
    emd_target = list(df["EMD_CD"].unique())

    # # 1. sejong-city total dataset
    df_sejong = df.copy()
    df_sejong["vacancy_rate"] = df_sejong["vacancy_rate"] * df_sejong["bld_tot_area"]
    df_sejong = (
        df_sejong.groupby("STD_YM")
        .agg(
            {
                "vacancy_rate": "sum",
                "move_pop": "sum",
                "area_pop": "sum",
                "service_type_count": "max",
                "biz_opens": "sum",
                "biz_closures": "sum",
                "bld_tot_area": "sum",
                "bld_area_small": "sum",
                "bld_area_midlarge": "sum",
                "bld_area_complex": "sum",
                "maxgrid_lat": "mean",
                "maxgrid_lon": "mean",
                "call_rate": "first",
                "novel_balance_COFIX": "first",
                "bld_loan_complex": "first",
                "novel_trade_COFIX": "first",
                "avg_comp_stock": "first",
                "balance_COFIX": "first",
                "avg_treasury_10yrs": "first",
                "bld_loan_small": "first",
                "avg_treasury_5yrs": "first",
                "CPI": "first",
                "bld_loan_midlarge": "first",
                "avg_treasury_3yrs": "first",
                "CD_91": "first",
                "standard_interest": "first",
            }
        )
        .reset_index()
        .sort_values("STD_YM")
    )
    df_sejong["vacancy_rate"] = df_sejong["vacancy_rate"] / df_sejong["bld_tot_area"]
    df_sejong_out = model.forecast(df_sejong, steps=19)[["vacancy_rate"]]

    # 2. inference per-EMD, post process sum
    results = []
    for emd_cd in emd_target:
        df_sample = df.loc[df["EMD_CD"] == emd_cd].sort_values(by="STD_YM")
        df_out = model.forecast(df_sample, steps=19)
        df_out = df_out[["vacancy_rate", "bld_tot_area"]]
        df_out["EMD_CD"] = emd_cd
        results.append(df_out)
    pd.concat(results, axis=0).reset_index().rename(columns={"index": "STD_YM"}).to_csv(
        "inference_result_emd.csv",
        index=False,
    )
    df_out = (
        pd.concat(results, axis=0)[["vacancy_rate", "bld_tot_area"]]
        .reset_index()
        .rename(columns={"index": "STD_YM"})
    )
    df_out["vacancy_rate"] = df_out["vacancy_rate"] * df_out["bld_tot_area"]
    df_out = df_out.groupby("STD_YM").agg("sum")
    df_out["vacancy_rate"] = df_out["vacancy_rate"] / df_out["bld_tot_area"]
    df_out = df_out[["vacancy_rate"]]

    result_df = pd.concat(
        [
            df_out.rename(columns={"vacancy_rate": "vac_EMD"}),
            df_sejong_out.rename(columns={"vacancy_rate": "vac_Sejong"}),
        ],
        axis=1,
    )
    result_df = pd.concat(
        [
            pd.concat(
                [
                    df_sejong.set_index("STD_YM")["vacancy_rate"],
                    result_df["vac_EMD"],
                ]
            ).rename("vacancy_rate"),
            # pd.concat([df_sejong.set_index('STD_YM')['vacancy_rate'],result_df['vac_Sejong']]).rename('Sejong-total'),
        ],
        axis=1,
    )
    result_df.reset_index().rename(columns={"index": "STD_YM"}).to_csv(
        "./inference_result_sejong.csv",
        index=False,
    )

    fig = px.line(
        result_df,
        title="Vacancy Rate Prediction",
        labels={"index": "Time", "value": "Vacancy Rate"},
    )
    fig.add_vline(x="2024-05", line_dash="dash", line_color="green")
    fig.update_layout(showlegend=False)
    fig.show()


if __name__ == "__main__":
    main()
