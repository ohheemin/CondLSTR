# CondLSTR

1. tensor가 sparse하게 들어가야 하지만 stride한 문제 해결 X
2. gpu 메모리 2.8GB로 segment해서 들어가야 하기 때문에 transformer 기반 모델 학습 어려움
3. 5. vram 큰 gpu로 다시 돌려볼 예정

# 수정사항

1. trainer.py (engine),
2. train.py (lane)
3. resnet10.py 추가 (34 layers -> 10 layers)
4. TuSimple dataset (24GB) 추가

