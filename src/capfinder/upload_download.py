import hashlib
import json
import os
import random
import shutil
import tarfile
import tempfile
import threading
import time
from typing import List, Optional, Tuple

import comet_ml
from loguru import logger


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate the SHA256 hash of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The hexadecimal representation of the file's SHA256 hash.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class CometArtifactManager:
    """
    Manages the creation, uploading, and downloading of dataset artifacts using Comet ML.
    """

    def __init__(self, project_name: str, dataset_dir: str) -> None:
        """
        Initialize the CometArtifactManager.

        Args:
            project_name (str): The name of the Comet ML project.
            dataset_dir (str): The directory containing the dataset.
        """
        self.project_name = project_name
        self.dataset_dir = dataset_dir
        self.artifact_name = "cap_data"
        self.experiment = self.initialize_comet_ml_experiment()
        self.artifact: Optional[comet_ml.Artifact] = None
        self.tmp_dir: Optional[str] = None
        self.chunk_files: List[str] = []
        self.upload_lock = threading.Lock()
        self.upload_threads: List[threading.Thread] = []

    def initialize_comet_ml_experiment(self) -> comet_ml.Experiment:
        """
        Initialize and return a Comet ML experiment.

        Returns:
            comet_ml.Experiment: The initialized Comet ML experiment.

        Raises:
            ValueError: If the COMET_API_KEY environment variable is not set.
        """
        logger.info(f"Initializing CometML experiment for project: {self.project_name}")
        comet_api_key = os.getenv("COMET_API_KEY")
        if not comet_api_key:
            raise ValueError("COMET_API_KEY environment variable is not set.")
        return comet_ml.Experiment(
            api_key=comet_api_key,
            project_name=self.project_name,
            display_summary_level=0,
        )

    def create_artifact(self) -> comet_ml.Artifact:
        """
        Create and return a Comet ML artifact.

        Returns:
            comet_ml.Artifact: The created Comet ML artifact.
        """
        logger.info(f"Creating CometML artifact: {self.artifact_name}")
        self.artifact = comet_ml.Artifact(
            name=self.artifact_name,
            artifact_type="dataset",
            metadata={"task": "RNA caps classification"},
        )
        return self.artifact

    def upload_chunk(
        self, chunk_file: str, chunk_number: int, total_chunks: int
    ) -> None:
        """
        Upload a chunk of the dataset to the Comet ML artifact.

        Args:
            chunk_file (str): The path to the chunk file.
            chunk_number (int): The number of the current chunk.
            total_chunks (int): The total number of chunks.
        """
        with self.upload_lock:
            if self.artifact is None:
                logger.error(
                    "Artifact is not initialized. Call create_artifact() first."
                )
                return
            self.artifact.add(
                local_path_or_data=chunk_file,
                logical_path=os.path.basename(chunk_file),
                metadata={"chunk": chunk_number, "total_chunks": total_chunks},
            )
        logger.info(f"Added chunk to artifact: {os.path.basename(chunk_file)}")

    def create_targz_chunks(
        self, chunk_size: int = 200 * 1024 * 1024
    ) -> Tuple[List[str], str, int]:
        """
        Create tar.gz chunks of the dataset.

        Args:
            chunk_size (int, optional): The size of each chunk in bytes. Defaults to 20MB.

        Returns:
            Tuple[List[str], str, int]: A tuple containing the list of chunk files,
                                        the temporary directory path, and the total number of chunks.
        """
        logger.info("Creating tar.gz chunks of the dataset...")
        self.tmp_dir = tempfile.mkdtemp()
        logger.info(f"Temporary directory created at: {self.tmp_dir}")

        # Create a single tar.gz file of the entire dataset
        tar_path = os.path.join(self.tmp_dir, "dataset.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for root, _, files in os.walk(self.dataset_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.dataset_dir)
                    tar.add(file_path, arcname=arcname)

        # Split the tar.gz file into chunks
        chunk_number = 0
        with open(tar_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunk_file = os.path.join(
                    self.tmp_dir, f"dataset.tar.gz.{chunk_number:03d}"
                )
                with open(chunk_file, "wb") as chunk_f:
                    chunk_f.write(chunk)
                self.chunk_files.append(chunk_file)
                chunk_number += 1

        total_chunks = chunk_number
        logger.info(f"Created {total_chunks} tar.gz chunks")

        # Calculate hash of the original tar.gz file
        tar_hash = calculate_file_hash(tar_path)

        # Store tar hash
        hash_file_path = os.path.join(self.tmp_dir, "tar_hash.json")
        with open(hash_file_path, "w") as f:
            json.dump({"tar_hash": tar_hash}, f)
        logger.info(f"Tar hash stored in: {hash_file_path}")

        return self.chunk_files, self.tmp_dir, total_chunks

    def make_comet_artifacts(self) -> None:
        """
        Create and upload Comet ML artifacts.
        """
        self.create_artifact()
        self.chunk_files, self.tmp_dir, total_chunks = self.create_targz_chunks()

        # Upload chunks
        for i, chunk_file in enumerate(self.chunk_files):
            upload_thread = threading.Thread(
                target=self.upload_chunk, args=(chunk_file, i, total_chunks)
            )
            upload_thread.start()
            self.upload_threads.append(upload_thread)

        # Wait for all upload threads to complete
        for thread in self.upload_threads:
            thread.join()

        # Add tar hash to artifact
        hash_file_path = os.path.join(self.tmp_dir, "tar_hash.json")
        self.artifact.add(  # type: ignore
            local_path_or_data=hash_file_path,
            logical_path="tar_hash.json",
            metadata={"content": "Tar hash for integrity check"},
        )
        logger.info("Added tar hash to artifact")

    def log_artifacts_to_comet(self) -> Optional[str]:
        """
        Log artifacts to Comet ML.

        Returns:
            Optional[str]: The version of the logged artifact, or None if logging failed.
        """
        if self.experiment is not None and self.artifact is not None:
            logger.info("Logging artifact to CometML...")
            art = self.experiment.log_artifact(self.artifact)
            version = f"{art.version.major}.{art.version.minor}.{art.version.patch}"
            logger.info(f"Artifact logged successfully. Version: {version}")
            self.store_artifact_version_to_file(version)

            logger.info(
                "Artifact upload initiated. It will continue in the background."
            )

            # Clean up the temporary directory
            shutil.rmtree(self.tmp_dir)  # type: ignore
            logger.info(f"Temporary directory cleaned up: {self.tmp_dir}")

            return version
        return None

    def store_artifact_version_to_file(self, version: str) -> None:
        """
        Store the artifact version in a file.

        Args:
            version (str): The version of the artifact to store.
        """
        version_file = os.path.join(self.dataset_dir, "artifact_version.txt")
        with open(version_file, "w") as f:
            f.write(version)
        logger.info(f"Artifact version {version} written to {version_file}")

    def download_remote_dataset(self, version: str, max_retries: int = 3) -> None:
        """
        Download a remote dataset from Comet ML.

        Args:
            version (str): The version of the dataset to download.
            max_retries (int, optional): The maximum number of download attempts. Defaults to 3.

        Raises:
            Exception: If the download fails after the maximum number of retries.
        """
        logger.info(f"Downloading remote dataset v{version}...")

        for attempt in range(max_retries):
            try:
                art = self.experiment.get_artifact(
                    artifact_name=self.artifact_name, version_or_alias=version
                )

                tmp_dir = tempfile.mkdtemp()
                logger.info(f"Temporary directory for download created at: {tmp_dir}")
                art.download(tmp_dir)

                # Combine all chunks back into a single tar.gz file
                tar_path = os.path.join(tmp_dir, "dataset.tar.gz")
                with open(tar_path, "wb") as tar_file:
                    chunk_files = sorted(
                        [
                            f
                            for f in os.listdir(tmp_dir)
                            if f.startswith("dataset.tar.gz.")
                        ]
                    )
                    for chunk_file in chunk_files:
                        with open(os.path.join(tmp_dir, chunk_file), "rb") as chunk:
                            tar_file.write(chunk.read())

                # Verify tar.gz integrity
                with open(os.path.join(tmp_dir, "tar_hash.json")) as f:
                    original_hash = json.load(f)["tar_hash"]
                current_hash = calculate_file_hash(tar_path)
                if current_hash != original_hash:
                    raise ValueError("Tar file integrity check failed")  # noqa: TRY301

                # Extract the tar.gz file
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=self.dataset_dir)

                logger.info(
                    "Remote dataset downloaded, verified, and extracted successfully."
                )
                return

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")  # noqa: G003
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + random.uniform(
                        0, 1
                    )  # Exponential backoff with jitter
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Download failed.")
                    raise

            finally:
                # Clean up
                if "tmp_dir" in locals():
                    shutil.rmtree(tmp_dir)
                    logger.info(f"Temporary directory cleaned up: {tmp_dir}")

        raise Exception(  # noqa: TRY002
            "Failed to download and extract the dataset after maximum retries."
        )

    def end_comet_experiment(self) -> None:
        """
        End the Comet ML experiment.
        """
        logger.info("Ending CometML experiment...")
        self.experiment.end()


def upload_dataset_to_comet(dataset_dir: str, project_name: str) -> str:
    """
    Upload a dataset to Comet ML.

    Args:
        dataset_dir (str): The directory containing the dataset to upload.
        project_name (str): The name of the Comet ML project.

    Returns:
        str: The version of the uploaded dataset, or None if the upload failed.
    """
    comet_obj = CometArtifactManager(project_name=project_name, dataset_dir=dataset_dir)

    logger.info("Making Comet ML dataset artifacts for uploading...")
    comet_obj.make_comet_artifacts()

    logger.info("Logging artifacts to Comet ML...")
    version = comet_obj.log_artifacts_to_comet()

    if version:
        logger.info(
            f"Dataset version {version} logged to Comet ML successfully. Upload will continue in the background."
        )
        return version
    else:
        logger.error("Failed to log dataset to Comet ML.")
        return ""


def download_dataset_from_comet(
    dataset_dir: str, project_name: str, version: str
) -> None:
    """
    Download a dataset from Comet ML.

    Args:
        dataset_dir (str): The directory to download the dataset to.
        project_name (str): The name of the Comet ML project.
        version (str): The version of the dataset to download.
    """
    comet_obj = CometArtifactManager(project_name=project_name, dataset_dir=dataset_dir)
    comet_obj.download_remote_dataset(version)


if __name__ == "__main__":
    dataset_dir = "/home/valen/9-test-delete/dataset/"
    comet_project_name = "dataset2"
    # Upload dataset
    version = upload_dataset_to_comet(dataset_dir, comet_project_name)

    # Download dataset (uncomment to test)
    dataset_dir = "/home/valen/9-test-delete/dataset-extracted/"
    download_dataset_from_comet(dataset_dir, comet_project_name, version)
