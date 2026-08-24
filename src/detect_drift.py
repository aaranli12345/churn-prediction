import pandas as pd
from data_prep import clean
from evidently import Report
from evidently.presets import DataDriftPreset
import json


reference = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
current = pd.read_csv("../batches/batch_11.csv")


reference = clean(reference)
current = clean(current)

report = Report([DataDriftPreset()])
my_eval = report.run(current_data=current, reference_data=reference)
my_eval.save_html("drift_report_batch11.html")
result_dict = my_eval.dict()
print(json.dumps(result_dict, indent=2)[:3000])

important_columns = ["Contract", "MonthlyCharges", "tenure"]

def check_important_drift(result_dict, important_columns):
    for metric in result_dict["metrics"]:
        name = metric.get("metric_name", "")
        for col in important_columns:
            if f"column={col}," in name:
                value = metric["value"]
                threshold = metric["config"].get("threshold", 0.1)
                if value > threshold:
                    print(f"⚠️ Drift detected in business-critical column: {col} (score={value:.3f})")
                else:
                    print(f"✅ No drift in {col} (score={value:.3f})")

check_important_drift(result_dict, important_columns)
