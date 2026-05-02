from typing import Dict
from utils.logger import log


class ModelConfig:
    """Base class for model configurations."""

    def __init__(self, data: Dict):
        try:
            # --- PAYLOAD CONFIG ---
            log.info("Initializing ModelConfig with provided data...")
            self.CLIENT_NUMBER: str = data.get("client_number")
            self.DATE_VARIABLE: str = data.get("date_variable", "date")
            self.TARGET_VARIABLE: str = data.get("target_variable", "revenue")
            self.PROJECT_ID: str = data.get("project_id")
            self.TABLE_ID: str = data.get("table_id")
            self.REGION: str = data.get("region", "us-central1")
            self.START_DATE: str = data.get("start_date", "2026-01-01")
            self.END_DATE: str = data.get("end_date", "2026-06-30")
            self.MODEL_TYPE: str = data.get("model_type", "lightgbm")
            self.TEST_SIZE: int = data.get("test_size", 30)

            # --- HARD CODED CONFIG ---
            self.BUCKET_NAME: str = "dag-marketing-ml-model"
            self.OUTPUT_DIR: str = f"model_artifacts/{self.CLIENT_NUMBER}_meta_data"
            self.BASELINE_WINDOW: int = 30  # Number of days for baseline calculation
            self.SEED: int = 42

            # --- GOOGLE SQL QUERY ---
            self._REVENUE_BASE_SQL = """
            WITH
                all_date AS (
                    SELECT * FROM UNNEST(generate_date_array('{start_date}', '{end_date}'))
                    AS new_date
                ),
                add_stats AS (
                    SELECT
                    CAST(date(new_date) AS STRING) AS {date_variable},
                    ROUND(COALESCE(sum(Impressions), 0), 2) AS impressions,
                    ROUND(COALESCE(sum(Conversions), 0), 2) AS conversions,
                    ROUND(COALESCE(sum(Clicks), 0), 2) AS clicks,
                    ROUND(COALESCE(sum(Sessions), 0), 2) AS sessions,
                    ROUND(COALESCE(sum(New_Users), 0), 2) AS new_users,
                    ROUND(COALESCE(AVG(Bounce_Rate_pct), 0)) AS bounce_rate,
                    ROUND(COALESCE(sum(Clicks) * 100 / sum(Impressions), 0), 2) AS CTR,
                    ROUND(COALESCE(sum(Ad_Spend_INR), 0), 2) AS ad_spend,
                    ROUND(COALESCE(sum(Revenue_INR), 0), 2) AS {target_variable},
                    ROUND(COALESCE(safe_divide(sum(Ad_Spend_INR), sum(Clicks)), 0), 2) AS CPC,
                    ROUND(COALESCE(safe_divide(sum(Revenue_INR), sum(Ad_Spend_INR)), 0), 2)
                        AS roas,
                    ROUND(COALESCE(safe_divide(sum(Revenue_INR), sum(Conversions)), 0), 2)
                        AS AOV
                    FROM all_date ad
                    LEFT JOIN `{table_id}` pt
                    ON ad.new_date = (pt.date)
                    WHERE new_date BETWEEN '{start_date}' AND '{end_date}'
                    GROUP BY 1
                )
                SELECT * FROM add_stats ORDER BY {date_variable} ASC;
            """

            # Final SQL with clean formatting
            self.REVENUE_SQL_QUERY: str = self._REVENUE_BASE_SQL.format(
                start_date=self.START_DATE,
                end_date=self.END_DATE,
                table_id=self.TABLE_ID,
                date_variable=self.DATE_VARIABLE,
                target_variable=self.TARGET_VARIABLE,
            )
            log.info("ModelConfig initialized successfully.")
        except Exception as e:
            log.error(f"Error initializing ModelConfig: {e}")
            raise e
