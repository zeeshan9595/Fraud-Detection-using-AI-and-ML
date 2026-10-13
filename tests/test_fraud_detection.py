import numpy as np
import pandas as pd
import pytest
from fraud_detection import chronological_split, features, choose_f1, choose_precision, threshold_metrics, cost_sensitivity

def test_chronological_split():
    df=pd.DataFrame({"step":range(100),"isFraud":[i%2 for i in range(100)]})
    tr,va,te=chronological_split(df,"isFraud","step")
    assert len(tr)==60 and len(va)==20 and len(te)==20
    assert tr.step.max()<va.step.min()<te.step.max()

def test_single_class_partition_rejected():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"step":range(10),"isFraud":[0]*10}),"isFraud","step")

def test_paysim_leakage_exclusions():
    df=pd.DataFrame(columns=["isFraud","step","amount","nameOrig","nameDest","isFlaggedFraud","newbalanceOrig","newbalanceDest","oldbalanceOrg"])
    assert list(features(df,"paysim","isFraud","step").columns)==["amount","oldbalanceOrg"]

def test_threshold_metrics_consistent():
    m=threshold_metrics(np.array([0,0,1,1]),np.array([.1,.8,.7,.9]),.75)
    assert (m["true_positives"],m["false_positives"],m["false_negatives"])==(1,1,1)
    assert m["precision"]==pytest.approx(.5) and m["recall"]==pytest.approx(.5) and m["f1"]==pytest.approx(.5)

def test_threshold_at_precision_reports_constraint():
    choice=choose_precision([0,0,1,1],[.05,.2,.75,.95],.9)
    assert choice["validation_constraint_met"] and choice["validation_precision"]>=.9

def test_f1_threshold_range():
    result=choose_f1([0,0,1,1],[.1,.2,.8,.9])
    assert 0<=result["threshold"]<=1

def test_cost_is_scenario_not_roi():
    result=cost_sensitivity({"false_positives":3,"false_negatives":2},10,100)
    assert result["total_illustrative_cost"]==230
    assert "not measured operational cost" in result["interpretation"]

def test_cost_inputs_paired():
    with pytest.raises(ValueError):
        cost_sensitivity({"false_positives":1,"false_negatives":1},10,None)
