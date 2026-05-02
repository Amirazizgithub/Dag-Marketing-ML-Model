from google.cloud import bigquery
from model_configs import ModelConfig
from utils.logger import log
import pandas as pd


# Loading data from big query using the SQL query defined in ModelConfig.REVENUE_SQL_QUERY
class DataLoader(ModelConfig):
    """Utility class for loading and preprocessing data."""

    def __init__(self, data: dict):
        super().__init__(data=data)
        try:
            self.client = bigquery.Client(project=self.PROJECT_ID)
        except Exception as e:
            log.error(f"Error initializing BigQuery client: {e}")
            raise e

    def load_data(self):
        """Load data from BigQuery using the configured SQL query."""
        try:
            query = self.REVENUE_SQL_QUERY
            df = self.client.query(query).result().to_dataframe()
            log.success(f"Data loaded successfully with {len(df)} records.")
            df.drop_duplicates(inplace=True)  # Ensure no duplicate records
            log.info(f"Data after removing duplicates has {len(df)} records.")
            df[self.DATE_VARIABLE] = pd.to_datetime(
                df[self.DATE_VARIABLE], errors="coerce"
            )  # Ensure date column is datetime
            df.sort_values(
                by=self.DATE_VARIABLE, inplace=True
            )  # Ensure data is sorted by date
            return df
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise e


# df = DataLoader(data={}).load_data()
# log.info(f"Sample data:\n{df.head()}")
