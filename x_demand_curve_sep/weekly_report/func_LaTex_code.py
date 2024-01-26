from datetime import datetime

td = datetime.today().date()

new_line = '\n  '

bracket = lambda string: '{' + string + '}'
begin = lambda key: r'\begin' + bracket(key)
end = lambda key: r'\end' + bracket(key)
textwidth = lambda width: str(width) + r'\textwidth'

def insert_image(
    image_path, width=1
):
    return (
            r'\includegraphics' + f'[width={textwidth(width)}]'
            r'{images/' + f'{td}/' + image_path + '}'
    )


def create_page(page_title, column1, column2):
    code = f"""
    {begin('frame')}{bracket(page_title)}
        {begin('columns')}
        {column1}
        
        {column2}
        {end('columns')}
    {end('frame')}
    """

    return code


def create_column(header, content: str, width=1):
    col_header = r'\colheader' + bracket(header)
    inserted_content = bracket(
        f"""
        {begin('center')}
            {insert_image(content, width)}
        {end('center')}
        """
    )

    code = f"""
    {begin('column')}{bracket(textwidth(0.45))}
        {col_header}
        {inserted_content}
    {end('column')}
    """

    return code
