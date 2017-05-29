import unicodedata

chinese_text = []
for c in extracted_text:
  if unicodedata.category(c) == 'Lo':
    chinese_text.append(c)
chinese_text = ''.join(chinese_text)
print(chinese_text)
