# [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Introduction

이 데이터셋은 유럽 사람들의 2013년 9월 중 이틀간의 신용카드 거래 데이터입니다. 

총 284,807건의 거래 중 492건만 사기 거래입니다. 데이터셋이 매우 불균형하며, 양성 클래스(사기 거래)는 모든 거래의 약 0.172%를 차지합니다.

데이터셋에는 주성분 분석(PCA) 변환의 결과로 얻어진 숫자형 입력 변수만 포함되어 있습니다. 
'Time'과 'Amount'라는 두 개의 특성은 PCA로 변환되지 않았습니다. 
'Time'은 각 거래와 데이터셋의 첫 거래 사이의 경과 시간(초)을 나타내며, 'Amount'는 거래 금액을 나타냅니다. 
'Class' 특성은 반응 변수로, 사기 거래인 경우 1을 나타내고 그렇지 않은 경우 0을 가집니다.

## Evaluation Metric

이 데이터셋은 데이터 불균형이 심하고 False Positive보다 False Negative가 더 치명적입니다.

따라서 AUPRC (Area Under the Precision-Recall Curve)와 F2 점수를 고려하려고 합니다.

## TODO

1. optuna 활용하여 hyperparameter tuning 