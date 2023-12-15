import nlpcloud


client = nlpcloud.Client("bart-large-cnn", "afe272fbf2f34bfa0acc315b07a53403a4547398")

x = client.summarization("")
print(x)

