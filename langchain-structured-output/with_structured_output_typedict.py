from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# schema
class Review(TypedDict):
  summary : str
  sentiment : str

structured_model = chat_model.with_structured_output(Review)

result = structured_model.invoke("I bought this bottle after my third cheap one dented within a month, and honestly, the build quality is noticeably better. The powder-coated matte finish survived being dropped on pavement twice with zero scratches, which impressed me more than it should have. Ice lasts an absurdly long time — I filled it at 7am and there were still cubes floating around 6pm, even sitting in a hot car. The leak-proof lid actually lives up to its name; I've tossed it in a bag with a laptop and nothing got wet. Two gripes: it doesnt fit in most cup holders (the base is comically wide), and the straw makes a whistling sound thats mortifying in quiet offices. Also, the 'midnight blue' looks basically black indoors. Would I buy it again? Yes, but only for gym days when I need 32oz of cold water. For desk use, get the smaller size.")

print(result)
print(type(result))
print(result['summary'])
print(result['sentiment'])