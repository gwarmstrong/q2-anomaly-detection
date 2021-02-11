def column_value_splitter(table, metadata, metadata_column):
    values = metadata[metadata_column].dropna().unique()
    for val in values:
        sub_ids = metadata.loc[metadata[metadata_column] == val].index
        sub_table = table.filter(sub_ids, inplace=False, axis='sample')
        yield val, sub_ids, sub_table
