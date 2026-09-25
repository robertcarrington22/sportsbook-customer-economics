"""Run the whole pipeline.

simulate -> SQL -> models -> LTV projection -> test design -> one day of scoring -> report -> dashboard

The TabPFN comparison (model_tabpfn.py) is separate because it takes 10 to 40
minutes on CPU; its cached results are picked up by build_report.py.
"""

import simulate
import analyze
import model
import ltv_forecast
import experiment_design
import score
import build_report
import build_dashboard

if __name__ == "__main__":
    for step in (simulate, analyze, model, ltv_forecast, experiment_design):
        print(f"\n######## {step.__name__}")
        step.main()
    print("\n######## score (sample day)")
    score.main("2025-06-15")
    for step in (build_report, build_dashboard):
        print(f"\n######## {step.__name__}")
        step.main()
