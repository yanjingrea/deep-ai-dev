from dataclasses import dataclass
from typing import Union, Optional, Literal

import pandas as pd
from rea_python.constants import OutputFormat, DBCopyMode
from rea_python.main.aws import get_secret
from rea_python.main.database import RedshiftHook, PostgresHook


def get_hook(
    source: Literal['redshift', 'postgres']
):
    if source == 'redshift':
        from constants.redshift import hook
    else:
        hook = PostgresHook()
        hook.set_conn_from_uri(get_secret("prod/data-staging-db/pipeline/db_conn_uri"))

    return hook


@dataclass
class Database:
    hook: Optional[Union[RedshiftHook, PostgresHook]] = None

    def __post_init__(self):
        if self.hook is None:
            self.__setattr__('hook', get_hook('redshift'))

    def query_data(
        self,
        query,
    ):
        return self.hook.execute_raw_query(
            query,
            output_format=OutputFormat.pandas
        )

    def upload_data(
        self,
        df: pd.DataFrame,
        target_table,
    ):
        return self.hook.copy_from_df(
            df,
            target_table=target_table,
            mode=DBCopyMode.DROP,
            ddl=None
        )
