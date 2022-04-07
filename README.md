# time-series

1. Train a model:

    `python -m main`

2. Check the result of one line:

    `python -m inference --line_num 29 --model_name model_reg_cls_alpha=10.pth`

3. Check the metric of all lines:

    `python -m result --horizon 12 --model_name model_reg_cls_alpha=10.pth --threshold 0.5`
