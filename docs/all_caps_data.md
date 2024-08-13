### Overview
We now have cap signal data for the cap0-m6A class (`data__cap_0m6A.csv`). To retrain our classifier, we need cap signal data for all classes, and not just cap0-m6A class:

- Cap 0 (need data for this cap)
- Cap 1 (need data for this cap)
- Cap 2 (need data for this cap)
- Cap 2-1 (need data for this cap)
- Cap 0-m6A (you made this data `data__cap_0m6A.csv` in previous steps)

### Why Retrain?
Capfinder uses retraining instead of transfer learning. We believe retraining the classifier from scratch is preferable to transfer learning for the following reasons:

1. It's a simpler approach
2. It avoids potential biases from the pretrained model
3. It allows the model to learn optimal representations for all classes simultaneously

### Steps to Prepare Data

1. Download cap signal data for existing caps (Cap 0, cap 1, cap 2, and cap 2-1):
    <!-- TODO: Upload cap signal data and update link -->
    Download the cap signal data for existing classes from [this link](https://).

2. Create a data directory:
    Create a new directory to store all cap signal data files.

3. Extract downloaded data:
    Extract the downloaded files into your new data directory.

4. Add new cap0-m6A data:
    Place the `data__cap_0m6A.csv` file in the same data directory.

5. Verify data:
    Ensure your data directory now contains CSV files for all cap classes:
    - `data__cap_0_run1.csv`
    - `data__cap_0_run2.csv`
    - `data__cap_1_run1.csv`
    - `data__cap_2_run1.csv`
    - `data__cap_2-1_run1.csv`
    - `data__cap_2-1_run2.csv`
    - `data__cap_0m6A.csv`

The suffix `run1` and `run2` shows that the data was acquired from two different sequencing runs. If later on, you acquire more data for `cap0-m6A` class, you can rename the two files as `data__cap_0m6A_run1.csv` and `data__cap_0m6A_run1.csv`


## Next Steps
With all cap signal data prepared in a single directory, you're now ready to proceed with retraining the Capfinder classifier. We will next use a training pipeline that processes all these files in batches, does hyperparameter tuning, and final training.
