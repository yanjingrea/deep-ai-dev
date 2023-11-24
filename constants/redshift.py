import pandas as pd
from rea_python.constants import OutputFormat, DBCopyMode
from rea_python.main.database import RedshiftHook
from rea_python.main.aws import get_secret

hook = RedshiftHook(
    iam_role_arn="arn:aws:iam::051694948699:role/prod-redshift-aws-access",
    via_s3_bucket='dev-misc-usage',
    via_s3_folder="redshift-copy",
)
hook.set_conn_from_uri(get_secret("prod/redshift/pipeline/db_conn_uri"))


def query_data(
        query
):
    return hook.execute_raw_query(
        query,
        output_format=OutputFormat.pandas
    )

def upload_data(
        df: pd.DataFrame,
        target_table

):
    return hook.copy_from_df(
        df,
        target_table=target_table,
        mode=DBCopyMode.DROP,
        ddl=None
    )
