"""
[PARTIAL DUPLICATE]:
	1. https://gitlab.com/reapl1/data/ds-real-estate-developer/-/blob/master/src/utils/common.py
	2. https://github.com/Real-Estate-Analytics/feature_pipelines/tree/main/common/utils/database.py
"""

from rea_python.main.database import RedshiftHook, PostgresHook, DBCopyMode
from rea_python.constants import OutputFormat
import boto3

import os
import tempfile

from typing import Dict, Union, Optional

from optimization.src.utils.ddl import DDL

from src.constants import S3_MODEL_BUCKET, S3_MODEL_PATH
import src.config as config

#-----------------------------------------------------------------------------------------------------------------------

# global RedshiftHook;
# will be initialized once, after calling `GetRedshiftHook` (see below)
#
RS_HOOK = None

# global PostgresHook;
# will be initialized once, after calling `GetPostgresHook` (see below)
#
PS_HOOK = None

#---------------------------------------------------

# database connection (Redshift)
#
def GetRedshiftHook():

    global RS_HOOK

    if RS_HOOK is None:

        RS_HOOK = RedshiftHook \
        (
            iam_role_arn="arn:aws:iam::051694948699:role/prod-redshift-aws-access",
            via_s3_bucket='dev-misc-usage',
            via_s3_folder='redshift-copy'
        )

        RS_HOOK.set_conn_from_uri(config.DATA_REDSHIFT_URI)

    return RS_HOOK

# database connection (Postgres)
#
def GetPostgresHook():

    global PS_HOOK

    if PS_HOOK is None:

        PS_HOOK = PostgresHook()
        PS_HOOK.set_conn_from_uri(config.DATA_POSTGRES_URI)

    return PS_HOOK

#-----------------------------------------------------------------------------------------------------------------------

# upload a dataframe to the database (using the `hook`)
#
#   df:             a DataFrame to copy data from
#
#   target_table:   a string following a format '{schema}.{table_name}'
#
#   partial_ddl:    [optional] column type specification;
#                   can be either a string (a fragment of a DDL code)
#                   or a dictionary {column_name: column_type_name} / DDL object
#                   where `column_type_name` is a type name specific to the SQL flavour of the target database
#                   (e.g. for Redshift: {'a': 'VARCHAR(256)', 'b': 'INTEGER', ...})
#
#                   Note: `partial_ddl` can specify types only for *some* of the columns in `df`;
#                         the rest of the column types will be inferred based on the data and the `hook`
#
#   mode:           [optional] table update mode (either DROP or APPEND from DBCopyMode)
#
#   hook:           [optional] database hook; if None provided, uses the global Redshift hook `RS_HOOK`
#
def push_table \
(
    df,
    target_table: str,
    partial_ddl: Optional[Union[str, DDL, Dict[str, str]]] = None,
    mode = DBCopyMode.DROP,
    hook = GetRedshiftHook()
):

    base_ddl = DDL.from_df(df, target_table, hook)

    if partial_ddl is None:
        ddl = base_ddl
    else:
        if isinstance(partial_ddl, str):
            ddl = DDL.from_str(partial_ddl)
            ddl.table_name = target_table
        elif isinstance(partial_ddl, dict):
            ddl = DDL.from_dict(partial_ddl)
        else:
            raise Exception(f'Unsupported type for `partial_ddl`: {type(partial_ddl)}')

        ddl = base_ddl.join(ddl)

    hook.copy_from_df \
    (
        df = df,
        mode = mode,
        target_table = target_table,
        ddl = str(ddl)
    )

# execute `query` and return the result as a DataFrame
# ('standard' or a GeoDataFrame, depending on the `output_format`)
#
def load_table \
(
    query,
    output_format = OutputFormat.pandas,
    hook = GetRedshiftHook()
):

    return hook.execute_raw_query \
    (
        query,
        output_format = output_format
    )

# just a [hook].drop_table wrapper
#
def drop_table \
(
    target_table: str,
    if_exists: bool = False,
    hook = GetRedshiftHook()
):
    hook.drop_table(target_table, if_exists = if_exists)

# load and execute a query from SQL located at `sql_path`
# given the `query_name`, `template_parameters` and a `hook`;
# (if `query_name` is None, executes the whole file)
#
def load_and_execute \
(
    sql_path,
    *,
    query_name = None,
    output_format = OutputFormat.raw,
    template_parameters = None,
    hook = GetRedshiftHook()
):

    if query_name is None:

        with open(sql_path, 'r') as f:
            data = hook.execute_raw_query \
            (
                f.read(),
                output_format = output_format,
                template_parameters = template_parameters
            )

        return data

    else:

        hook.load_queries_from_files([sql_path])

        return hook.execute_loaded_query \
        (
            query_name = query_name,
            output_format = output_format,
            template_parameters = template_parameters
        )

# load "stl_load_errors" table and return the most recent row
# (filtered by `target_table` if provided)
#
def get_last_sql_load_error \
(
    target_table: str = None,
    hook = GetRedshiftHook()
):

    columns = ['colname', 'type', 'raw_field_value', 'err_reason']
    stl_load_errors = load_table("select * from stl_load_errors", hook = hook)

    if target_table:
        idxs = stl_load_errors['filename'].str.match \
        (
            rf"s3://.+/{target_table.replace('.', '_')}"
        )

        stl_load_errors = stl_load_errors[idxs]

    last_error = stl_load_errors.sort_values \
    (
        'starttime',
        ascending = False
    ) \
    .iloc[0][columns] \
    .apply(str.strip)

    return last_error

#-----------------------------------------------------------------------------------------------------------------------

def upload_s3_object(s3_bucket, s3_key, local_path):
    s3 = boto3.resource('s3')
    s3.Bucket(s3_bucket).upload_file(local_path, s3_key)

def read_s3_contents(bucket_name, key):
    s3 = boto3.resource('s3')
    response = s3.Object(bucket_name, key).get()
    return response['Body'].read()

def get_model_path(model_type):

    try:
        suffix = '_' + '_'.join(map(str, model_type.VERSION))
    except:
        suffix = ''

    return S3_MODEL_PATH.format \
    (
        model_type = model_type.__name__,
        suffix = suffix
    )

# Load a demand model from S3
#
def LoadModel(model_type):

    return model_type.deserialize \
    (
        read_s3_contents \
        (
            S3_MODEL_BUCKET,
            get_model_path(model_type)
        )
    )

# upload a demand model to S3
#
def UploadModel(model):

    with tempfile.NamedTemporaryFile(delete = False) as f:
        f.write(model.serialize())

    upload_s3_object \
    (
        S3_MODEL_BUCKET,
        get_model_path(type(model)),
        f.name
    )

    os.remove(f.name)