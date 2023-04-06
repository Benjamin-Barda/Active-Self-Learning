# Active-Self-Learning

## Usage
```
pip install requirements.txt
```
### Pretrain 

```
python pretrain.py --config=pretrain.config.json --model=resnet18 --batch_size=512 --pred_dim=512 --encoder_dim=2048 --num_epochs=800 --device=gpu --num_workers=1
```

### Linear Classifier 
```
python linear_classifier.py --config=lcls.config.json --num_epochs=100 --batch_size=256 --device=gpu
```

### Base Line Evaluation

```
python linear_classifier.py --config=lcls.config.json --num_epochs=100 --batch_size=128 --device=gpu --base_line_eval=True
```

