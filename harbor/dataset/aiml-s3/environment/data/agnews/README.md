# agnews

AG News topic classification (Zhang et al. 2015), 4 classes: 0 World, 1 Sports, 2 Business, 3 Sci/Tech.

- train.csv: 40000 articles: id, text, target
- The evaluator holds the 7,600 official test articles ($EVAL_DIR/agnews/inputs.csv: id, text at verification).
- Metric: error rate. Comparator for `prediction` claims: TF-IDF (1-2 grams) + logistic regression trained on train.csv (about 8% error).
