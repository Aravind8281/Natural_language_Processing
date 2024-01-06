from googletrans import Translator
translator = Translator()
text_to_translate = "Hello, how are you?"
translated_text = translator.translate(text_to_translate, dest='es').text
print("Original Text:", text_to_translate)
print("Translated Text:", translated_text)
