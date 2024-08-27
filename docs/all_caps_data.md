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

1. Download cap signal data for existing caps (Cap 0, cap 1, cap 2, and cap 2-1) from these two links below:

    - [capfinder_training_data_p1](https://mega.nz/folder/YdJQ1azA#bvcKe4z8nCbFjqcYteh9WQ)
    - [capfinder_training_data_p2](https://mega.nz/folder/drcyFKIS#wRSzygVZGaFFljTS0X2oqA)

2. Create a data directory:

    Create a new directory (name it as, lets say, `caps_data_dir`)

3. Extract downloaded zipped data:

    Extract the downloaded files -- `capfinder_training_data_p1.zip` and `capfinder_training_data_p2.zip` -- into `caps_data_dir` you created in step 2.

    Ensure that this directory has all 21 parts of the data (`data.tar.gz.part00.gpg` -- `data.tar.gz.part20.gpg`). These 21 parts are encrypted and you need a password to decrypt them first.

4. Getting the password:

    For now we only want to share data with people who want to collaborate. If you wish you collaborate, please send an email to [Eivind Valen](https://www.mn.uio.no/ibv/personer/vit/edv/) and you will be sent the password.

5. Decrypting the data:

    To decrypt the data use the following script:

    ??? example "script"

        ```sh
        #!/bin/bash

        # Function to read password securely
        read_password() {
            read -s -p "Enter password for decryption: " password
            echo
        }

        # Function to get directory path
        get_directory() {
            read -p "Enter the path to the directory containing encrypted files: " directory
            if [ ! -d "$directory" ]; then
                echo "The specified path does not exist or is not a directory."
                exit 1
            fi
        }

        # Main script
        echo "Welcome to the Decrypt, Extract, and Cleanup script!"

        # Get directory path
        get_directory

        # Get password
        read_password

        # Decrypt files
        echo "Decrypting files..."
        for file in "$directory"/data.tar.gz.part*.gpg; do
            if [ -f "$file" ]; then
                gpg --batch --yes --passphrase "$password" --decrypt "$file" > "${file%.gpg}"
                if [ $? -eq 0 ]; then
                    echo "Decrypted: $file"
                    rm "$file"
                else
                    echo "Failed to decrypt: $file"
                    exit 1
                fi
            fi
        done

        # Extract files
        echo "Extracting files..."
        cat "$directory"/data.tar.gz.part* | tar xzvf - --transform='s|.*/||' -C "$directory"

        # Check if extraction was successful
        if [ $? -eq 0 ]; then
            echo "Extraction completed successfully."

            # Remove the decrypted compressed files
            echo "Removing decrypted compressed files..."
            rm "$directory"/data.tar.gz.part*

            if [ $? -eq 0 ]; then
                echo "Decrypted compressed files have been removed."
            else
                echo "Failed to remove some or all of the decrypted compressed files."
            fi
        else
            echo "Extraction failed. Decrypted files have not been removed."
            exit 1
        fi

        echo "Process completed successfully."
        ```

    Just download the script and run it. It will ask for the password for decrytion and the path of the directory where encrypted data is currently residing. The script will decrypt, combine, and extract the tar files.

    If you are successful, you should see the following contents in your `caps_data_dir`:

    - `data__cap_0_run1.csv`
    - `data__cap_0_run2.csv`
    - `data__cap_1_run1.csv`
    - `data__cap_2_run1.csv`
    - `data__cap_2-1_run1.csv`
    - `data__cap_2-1_run2.csv`

4. Add new cap0-m6A data:

    Place the `data__cap_0m6A.csv` file in the same data directory.

5. Verify data:

    Ensure your `caps_data_dir` directory now contains CSV files for all cap classes:

    - `data__cap_0_run1.csv`
    - `data__cap_0_run2.csv`
    - `data__cap_1_run1.csv`
    - `data__cap_2_run1.csv`
    - `data__cap_2-1_run1.csv`
    - `data__cap_2-1_run2.csv`
    - `data__cap_0m6A.csv`

The suffix `run1` and `run2` shows that the data was acquired from two different sequencing runs. If later on, you acquire more data for `cap0-m6A` class, you can rename the two files as `data__cap_0m6A_run1.csv` and `data__cap_0m6A_run2.csv`


## Next Steps
With all cap signal data prepared in a single directory, you're now ready to proceed with retraining the Capfinder classifier. We will next use a training pipeline that processes all these files in batches, does hyperparameter tuning, and final training.
