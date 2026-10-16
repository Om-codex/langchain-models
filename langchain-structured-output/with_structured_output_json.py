from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional,Literal, List  # Import List from typing
from pydantic import BaseModel, Field

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "description": "Write down all the key themes discussed in the review in a list",
      "type": "array",
      "items": { "type": "string" }
    },
    "summary": {
      "description": "A brief summary of the review",
      "type": "string"
    },
    "sentiment": {
      "description": "Return the sentiment of the review either positive, negative or neutral",
      "enum": ["pos", "neg"],
      "type": "string"
    },
    "pros": {
      "description": "Write down all the pros inside a review",
      "default": "null",
      "anyOf": [
        { "type": "array", "items": { "type": "string" } },
        { "type": "null" }
      ]
    },
    "cons": {
      "description": "Write down all the cons inside a review",
      "default": "null",
      "anyOf": [
        { "type": "array", "items": { "type": "string" } },
        { "type": "null" }
      ]
    },
    "name": {
      "description": "Write the name of the person who reviewed",
      "default": "null",
      "anyOf": [
        { "type": "string" },
        { "type": "null" }
      ]
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}
  

structured_model = chat_model.with_structured_output(json_schema)

review_text = """I’ve been testing the Sony WH-1000XM5s for about six months now, primarily using them for my 45-minute train commute and while working from home. I’ve also owned the Bose QC45 and the Apple AirPods Max, so I’m comparing these against the best in class.

Sound Quality
The sound signature is warm, slightly bass-heavy, but not muddy. Sony’s new Integrated Processor V1 does a great job of handling high-resolution audio.
Bass: Punchy and deep, perfect for hip-hop and electronic music.
Mids: Vocals are clear and forward. I can hear details in acoustic songs that I missed on my old headphones.
Treble: Crisp but not harsh. Cymbals and hi-hats don’t pierce the ears.
EQ: The companion app is fantastic. You can tweak the EQ extensively, and it saves presets for different genres.

Noise Cancellation (ANC)
This is where Sony shines. The ANC is exceptional.
Low Frequencies: It completely erases the rumble of the train engine and airplane cabin noise.
High Frequencies: It handles human voices better than the Bose QC45, though not perfectly. I can still hear snippets of conversations nearby, which is actually a safety feature for me.
Wind Noise: This is a weak point. If you walk outside on a breezy day, the wind can cause a "whooshing" sound in the headphones. Sony has added a "Wind Noise Structure Reduction" feature, but it’s not perfect.

Comfort & Design
Sony completely redesigned the headband and ear cups for the XM5s.
Fit: They feel lighter and softer than the XM4s. The synthetic leather ear cups are breathable, so my ears don’t get sweaty after 3 hours.
Clamping Force: Very light. I can wear them all day without a headache.
Portability: This is the biggest change. They do not fold. They only swivel flat. This means the carrying case is larger and less travel-friendly than the XM4s. If you backpack or travel light, this is a con.

Battery Life
ANC On: I’m getting about 30 hours per charge.
ANC Off: Up to 40 hours.
Quick Charge: 3 minutes of charging gives you 3 hours of playback. This has saved me multiple times when I forgot to charge overnight.

Call Quality
This was a major upgrade from the XM4s. The 4 beamforming microphones with AI-based noise reduction make my voice sound clear even in noisy cafes. People on the other end of the call rarely complain about background noise anymore.

✅ Pros
Best-in-Class ANC: Blocks out almost everything, especially low-frequency rumble.
Superior Sound: Rich, detailed audio with a great EQ app.
Comfort: Lightweight and soft ear cups make them ideal for all-day wear.
Call Quality: Crystal clear voice transmission thanks to improved mics.
Multipoint Connection: Seamlessly switches between my laptop and phone.

❌ Cons
No Foldable Design: The case is bulky and not ideal for small bags or pockets.
Price: They are expensive ($399 MSRP), though often on sale.
Wind Noise: Still struggles in windy outdoor conditions.
No Hi-Res LDAC Support on iPhone: Due to Apple’s limitations, you can’t get the full benefit of high-res audio if you’re an iPhone user (same as XM4s).

Verdict
The Sony WH-1000XM5s are the best all-around wireless headphones for most people. If you prioritize sound quality, noise cancellation, and comfort, these are hard to beat. However, if you travel frequently and need compact storage, the older XM4s might be a better value since they fold up smaller.

Would I buy them again? Yes. But I’d wait for a sale.
- Review by Harry Potter"""

result = structured_model.invoke(review_text)

print(result)