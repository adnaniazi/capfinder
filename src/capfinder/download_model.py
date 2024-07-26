import glob
import os

from comet_ml import API


def rename_downloaded_model(
    output_dir: str, orig_model_name: str, new_model_name: str
) -> None:
    """
    Renames the downloaded model file to a new name.

    Parameters:
    output_dir (str): The directory where the model file is located.
    orig_model_name (str): The original name of the model file.
    new_model_name (str): The new name to rename the model file to.

    Returns:
    None
    """
    # Construct the new full path
    orig_path = os.path.join(output_dir, orig_model_name)
    new_path = os.path.join(output_dir, new_model_name)
    os.rename(orig_path, new_path)


def create_version_info_file(output_dir: str, version: str) -> None:
    """
    Create a file to store the version information. If any file with a name starting
    with "v" already exists in the output directory, delete it before creating a new one.

    Parameters:
    output_dir (str): The directory where the version file will be created.
    version (str): The version string to be written to the file.

    Returns:
    None
    """
    version_file = os.path.join(output_dir, f"v{version}")

    # Find and delete any existing version file
    existing_files = glob.glob(os.path.join(output_dir, "v*"))
    for file in existing_files:
        os.remove(file)

    # Create a new version file
    with open(version_file, "w") as f:
        f.write(version)


def download_comet_model(
    workspace: str,
    model_name: str,
    version: str,
    output_dir: str = "./",
    force_download: bool = False,
) -> None:
    """
    Download a model from Comet ML using the official API.

    Parameters:
    workspace (str): The Comet ML workspace name
    model_name (str): The name of the model
    version (str): The version of the model to download (use "latest" for the most recent version)
    output_dir (str): The local directory to save the downloaded model (default is current directory)
    force_download (bool): If True, download the model even if it already exists locally

    Returns:
    str: The path to the model file (either existing or newly downloaded), or None if download failed
    """

    os.makedirs(output_dir, exist_ok=True)
    api = API()
    model = api.get_model(workspace, model_name)
    model.download(version, output_dir, expand=True)
    orig_model_name = model._get_assets(version)[0]["fileName"]
    rename_downloaded_model(output_dir, orig_model_name, f"{model_name}.keras")
    create_version_info_file(output_dir, version)


if __name__ == "__main__":
    workspace = "adnaniazi"
    model_name = "cnn_lstm-classifier"
    version = "1.1.0"
    output_dir = "/export/valenfs/data/processed_data/MinION/9_madcap/a_code/capfinder/src/capfinder/model"
    force_download = False
    download_comet_model(workspace, model_name, version, output_dir, force_download)
