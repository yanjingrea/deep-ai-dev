import json
import pandas as pd
import urllib.request


def read_singstat_id(table_id: str, frequency: str, query_filter='', return_columns: list = None):

    url = "https://tablebuilder.singstat.gov.sg/api/table/tabledata/" + table_id + query_filter

    # Add headers including a User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/58.0.3029.110 Safari/537.3',
        'Accept': 'application/json',
        # Include API key if required:
        # 'Authorization': 'Bearer YOUR_API_KEY_HERE'
    }

    req = urllib.request.Request(url, headers=headers)

    try:

        webURL = urllib.request.urlopen(req)
        json_format_data = json.loads(webURL.read())

        rows = json_format_data['Data']['row']

        data = pd.DataFrame()

        for idx, col in enumerate(rows):

            col_name = col['rowText'].lower().replace(' ', '_')

            if return_columns:
                if col_name not in return_columns:
                    continue

            col_data = pd.DataFrame(col['columns']).rename(columns={'value': col_name})

            if not col_data.empty:

                try:
                    col_data[col_name] = col_data[col_name].astype(int)

                except:
                    col_data[col_name] = pd.to_numeric(col_data[col_name].astype(float))

                if data.empty:

                    data = pd.concat([data, col_data])

                else:

                    data = pd.merge(data, col_data, on='key', how='outer')

        data = data.rename(columns={'key': frequency})

        if frequency == 'quarter':

            data.insert(
                1,
                'quarter_index',
                data['quarter'].apply(
                    lambda a: int(a[:4] + '0' + a[-2])
                )
            )

            frequency = 'quarter_index'

        data[frequency] = data[frequency].astype(int)

        if return_columns is not None:

            return data[[frequency] + return_columns]

        else:
            return data

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
