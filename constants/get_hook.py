import pandas as pd
from rea_python.constants import OutputFormat, DBCopyMode
from rea_python.main.database import PostgresHook
from rea_python.main.aws import get_secret

hook = PostgresHook()
hook.set_conn_from_uri(get_secret("prod/data-staging-db/pipeline/db_conn_uri"))


