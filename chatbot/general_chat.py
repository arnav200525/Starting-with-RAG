import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"


class ChatBot:
    def __init__(self):
        self.history = []

    def chat(self, message):
        self.history.append(
            {"role": "user", "content": message}
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=self.history,
            temperature=0.7,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content

        self.history.append(
            {"role": "assistant", "content": answer}
        )

        return answer

    def reset(self):
        self.history = []


if __name__ == "__main__":
    bot = ChatBot()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "reset":
            bot.reset()
            print("Chat history cleared.")
            continue

        response = bot.chat(user_input)
        print("Bot:", response)