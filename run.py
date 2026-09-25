"""Run the whole pipeline: simulate -> SQL -> models -> LTV projection -> test design -> report.

The TabPFN comparison (model_tabpfn.py) is separate because it takes about
15 minutes on CPU; its cached results are picked up by build_report.py.
"""

import simulate
import analyze
import model
import ltv_forecast
import experiment_design
import build_report

if __name__ == "__main__":
    for step in (simulate, analyze, model, ltv_forecast, experiment_design, build_report):
        print(f"\n######## {step.__name__}")
        step.main()
