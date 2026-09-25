"""Run the whole pipeline: simulate -> SQL -> models -> report."""

import simulate
import analyze
import model
import build_report

if __name__ == "__main__":
    for step in (simulate, analyze, model, build_report):
        print(f"\n######## {step.__name__}")
        step.main()
