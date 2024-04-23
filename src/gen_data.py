from BERT.text_data_synthesizer import synthesize_text_data, pretty_print_text_data

generated_bert_text = synthesize_text_data(100, asJson=False)
pretty_print_text_data(generated_bert_text)

print("Generated text data with {0} samples".format(len(generated_bert_text)))