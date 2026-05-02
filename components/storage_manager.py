import io
import json
import pickle
from model_configs import ModelConfig
from utils.logger import log
from google.cloud import storage


class StorageManager(ModelConfig):
    def __init__(self, data: dict):
        self.data = data
        super().__init__(data=self.data)  # Initialize ModelConfig with provided data
        self.output_dir = self.OUTPUT_DIR
        self.bucket_name = self.BUCKET_NAME
        self.storage_client = storage.Client()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_blob(self, relative_path: str):
        """Returns a GCS Blob object for the given relative path."""
        bucket = self.storage_client.bucket(self.bucket_name)
        full_path = f"{self.output_dir}/{relative_path}"
        return bucket.blob(full_path), full_path

    # ------------------------------------------------------------------ #
    #  ML Model  (pickle)                                                  #
    # ------------------------------------------------------------------ #

    def save_model(self, model, model_name: str) -> None:
        """
        Serialises a trained ML model with pickle and uploads it to GCS.

        Args:
            model:       Any pickle-serialisable object (sklearn, XGBoost, etc.)
            model_name:  Filename to use in GCS, e.g. 'my_model.pkl'
        """
        try:
            blob, full_path = self._get_blob(model_name)

            log.info(f"Serialising and uploading model to GCS: {full_path}")
            model_bytes = pickle.dumps(model)
            blob.upload_from_file(
                io.BytesIO(model_bytes),
                content_type="application/octet-stream",
            )
            log.success(f"✓ Model saved to gs://{self.bucket_name}/{full_path}")

        except Exception as e:
            log.error(f"Failed to save model '{model_name}': {e}")
            raise e

    def load_model(self, model_name: str):
        """
        Downloads and deserialises a pickled ML model from GCS.

        Args:
            model_name:  Filename in GCS, e.g. 'my_model.pkl'

        Returns:
            The deserialised model object, or None if not found.
        """
        try:
            blob, full_path = self._get_blob(model_name)

            if not blob.exists():
                log.warning(
                    f"No model found at gs://{self.bucket_name}/{full_path}. "
                    "Train and save the model first."
                )
                return None

            log.info(f"Downloading model from GCS: {full_path}")
            model_bytes = blob.download_as_bytes()
            model = pickle.loads(model_bytes)
            log.success(f"✓ Model loaded from gs://{self.bucket_name}/{full_path}")
            return model

        except Exception as e:
            log.error(f"Failed to load model '{model_name}': {e}")
            raise e

    # ------------------------------------------------------------------ #
    #  Metadata  (JSON)                                                    #
    # ------------------------------------------------------------------ #

    def save_metadata(self, metadata: dict, file_name: str) -> None:
        """
        Serialises a dict as JSON and uploads it to GCS.

        Args:
            metadata:   Python dict to persist.
            file_name:  Filename to use in GCS, e.g. 'model_metadata.json'
        """
        try:
            blob, full_path = self._get_blob(file_name)

            log.info(f"Uploading metadata to GCS: {full_path}")
            blob.upload_from_string(
                data=json.dumps(metadata, indent=4),
                content_type="application/json",
            )
            log.success(f"✓ Metadata saved to gs://{self.bucket_name}/{full_path}")

        except Exception as e:
            log.error(f"Failed to save metadata '{file_name}': {e}")
            raise e

    def load_metadata(self, file_name: str) -> dict:
        """
        Downloads and parses a JSON metadata file from GCS.

        Args:
            file_name:  Filename in GCS, e.g. 'model_metadata.json'

        Returns:
            Parsed dict, or {} if the file does not exist.
        """
        try:
            blob, full_path = self._get_blob(file_name)

            if not blob.exists():
                log.info(
                    f"No metadata found at gs://{self.bucket_name}/{full_path}. "
                    "Returning empty dict."
                )
                return {}

            log.info(f"Downloading metadata from GCS: {full_path}")
            data = blob.download_as_text()
            metadata = json.loads(data)
            log.success(f"✓ Metadata loaded from gs://{self.bucket_name}/{full_path}")
            return metadata

        except Exception as e:
            log.error(f"Failed to load metadata '{file_name}': {e}")
            raise e

    # ------------------------------------------------------------------ #
    #  Convenience: save / load both at once                             #
    # ------------------------------------------------------------------ #

    def save_model_and_metadata(
        self, model, model_name: str, metadata: dict, metadata_name: str
    ) -> None:
        """Saves a trained model and its metadata in a single call."""
        self.save_model(model, model_name)
        self.save_metadata(metadata, metadata_name)

    def load_model_and_metadata(self, model_name: str, metadata_name: str) -> tuple:
        """
        Loads a model and its metadata in a single call.

        Returns:
            (model, metadata) tuple.
            model    is None if not found.
            metadata is {}   if not found.
        """
        model = self.load_model(model_name)
        metadata = self.load_metadata(metadata_name)
        return model, metadata
