# Workflow folder for saving
workflow_folder = r'C:\Users\SinghDis\3D Objects\my-files\ml-driven-landscapes\workflow_2'

# Document fields for xml_parser
input_xml_fields = ['ETI', 'EAB', 'ICLM', 'ECLM', 'CPC']

# Column names for dataframes
fields = ['TI', 'AB', 'ICLM', 'CLMS', 'CTB', 'CPC', 'CTB_CPC']
text_fields_list = ['TI', 'AB', 'ICLM', 'CLMS']
token_type_list = ['word_tokens', 'phrase_tokens', 'sent_tokens']

# File path for xml_parser
input_xml_path = workflow_folder + r'\xml\input_xml.xml'

# File path for CPC definitions
cpc_defs_path = workflow_folder + r'\resources\cpc_defs.xlsx'

# # File paths for preprocessor
# parsed_xml_path = workflow_folder + r'\parquet\parsed_xml.parquet'
# parsed_xml_cpc_path = workflow_folder + r'\parquet\parsed_xml_cpc.parquet'
# bert_embeddings_path = workflow_folder + r'\parquet\bert_embeddings.parquet'

# File paths for preprocessor
parsed_xml_path = workflow_folder + r'\pickle\parsed_xml.pickle'
parsed_xml_cpc_path = workflow_folder + r'\pickle\parsed_xml_cpc.pickle'
preprocessed_df_path = workflow_folder + r'\pickle\preprocessed_df.pickle'
bert_embeddings_path = workflow_folder + r'\pickle\bert_embeddings.pickle'