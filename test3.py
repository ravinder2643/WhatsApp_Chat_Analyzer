# pip install google.generativeai
# 
# If this gives error then execute
#
# 
# For Windows:
# # Create virtual environment
# python -m venv venv

# # Activate virtual environment
# venv\Scripts\activate
# 
# 
# For Linux/macOS:
# # Create virtual environment
# python3 -m venv venv

# # Activate virtual environment
# source venv/bin/activate


import google.generativeai as genai

class GenAIException(Exception):
    pass
class ChatBot:
    CHATBOT_NAME = "mpm"
    
    def __init__(self,api_key):
        self.genai = genai
        self.genai.configure(api_key=api_key)
        self.model = self.genai.GenerativeModel('gemini-pro')
        self.conversation = None
        self._conversation_history = []
        self.preload_conversation()

    def start_conversation(self):
        self.conversation = self.model.start_chat(history=self._conversation_history)
    
    def _generation_config(self,temperature):
        return genai.types.GenerationConfig(
            temperature = temperature
        )
    def clear_conversation(self):
        self.conversation = self.model.start_chat(history=[])
        
    def _construct_message(self,text, role='user'):
        return{
            'role':role,
            'parts':[text]
        }
    
    def preload_conversation(self,conversation_history=None):
        if isinstance(conversation_history,list):
            self._conversation_history = conversation_history
        else:
            self._conversation_history=[
                self._construct_message('From now on, return the output as a JSON object that can be loaded in Python with the key as \'text\'. For example, {"text":"<output goes here>"}'),
                self._construct_message('{"text": Sure, I can return the output as a regular JSON object with the key as `text`. Here is an example:{"text":"Your Output"}.','model')
            ]
    def send_prompt(self, prompt, temperature = 0.5):
        if not prompt:
            raise GenAIException('Empty Question')
        try:
            response = self.conversation.send_message(
                content=prompt,
                generation_config=self._generation_config(temperature),
            )
            response.resolve()
            return f'{response.text}\n'+'---'*20
        except Exception as e:
            raise GenAIException(e.message)
        
    @property
    def history(self):
        conversation_history=[
            {'role':message.role, 'text': message.parts[0].text} for message in self.conversation.history
        ]
        return conversation_history
        

api_key="AIzaSyAD8ssfwsrDuwZ43Ii8fBkQzDXhHP28O_o"
chatbot = ChatBot(api_key=api_key)
chatbot.start_conversation()       

print("Ask question . Type 'quit' to exit")

while True:
    user_input = input("You: ")
    if user_input.lower()=='quit':
        print("Exiting")
        break
    try:
        response = chatbot.send_prompt(user_input)
        print(f"{chatbot.CHATBOT_NAME}: {response}")
    except Exception as e:
        print(f"Error: {e}")

