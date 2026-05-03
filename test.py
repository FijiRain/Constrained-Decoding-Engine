from llm_sdk.llm_sdk import Small_LLM_Model


transformer = Small_LLM_Model()
prompt = "Le chat mange la"
print(transformer.get_path_to_vocab_file())
