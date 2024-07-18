import pandas as pd
from compas.inference import ForecastModel

model = ForecastModel(
    model_path="/Users/leeseungjun/Desktop/projects/LHCompas_2024/.saved_models/CNN1D_in_1_out_1_train_306_val_6_2024-07-18_15-07-1721284331"
)
df = pd.read_csv(
    "/Users/leeseungjun/Desktop/projects/LHCompas_2024/data/custom/tmp_dataset_with_total_population2024.csv"
)

model.forecast(df, steps=12)
