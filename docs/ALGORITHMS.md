# Subject-aware ML pipelines

`patient_split` uses GroupShuffleSplit on supplied patient identifiers.
`make_pipeline` constructs scaled SVM, logistic regression or random forest
estimators with class balancing. `fit_training_partition` optionally fits a
pipeline only on training rows and returns held-out indices without reporting scores.
`cross_validation_indices` supplies stratified patient-group folds.

Feature table shape is `(windows, features)`; labels are 0/1; groups are true patient
identities, not filenames. Confirm cross-session identity relationships in CHB-MIT.
The demo constructs an unfitted pipeline. A tiny synthetic unit test checks that
the scaler learns training statistics only. No dataset classifier was trained.
Group stratification is approximate and may yield a single-class fold; inspect
folds before training and tune hyperparameters in a separate nested training split.

## References

- [Method or implementation reference](https://scikit-learn.org/stable/modules/cross_validation.html)
- [CHB-MIT 1.0.0](https://physionet.org/content/chbmit/1.0.0/)
- [Siena 1.0.0](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
