import ollama

res = ollama.chat(
	model="llava",
	messages=[
		{
			'role': 'user',
			'content': 'Describe this image:',
			'images': ['./art.jpg']
		}
	]
)

print(res['message']['content'])