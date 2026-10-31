from langchain_text_splitters import CharacterTextSplitter

text = """The old lighthouse had not guided a ship in forty years, yet every evening at dusk, the keeper's granddaughter climbed the spiral stairs to wind the mechanism anyway. She never knew why her grandfather had insisted on it before he passed, only that he had whispered something about "keeping the rhythm" as though the light itself were a heartbeat rather than a beam. The townsfolk thought her eccentric, but no one stopped her. There was something comforting about seeing that pale sweep of gold cutting through the fog, even if no one out at sea could see it anymore.

Down in the village below, a baker named Elias rose each morning an hour before sunrise to shape his loaves by hand. He refused to buy the industrial mixers that his competitors used, claiming that machines made bread with no memory in it. His customers couldn't quite explain why his sourdough tasted like childhood or why eating it seemed to slow time down slightly, but they lined up for it regardless. On rainy days, when fewer people came, he would slice warm bread and leave paper bags on the doorsteps of houses where the lights stayed off too late at night.

Between the lighthouse and the bakery ran a single cobblestone road that curved like a question mark around the headland. Children dared each other to run its full length without looking back, convinced that at some point along the way you would glimpse a second version of yourself walking toward you from the opposite direction. Most saw nothing but saltgrass and gulls. One girl, however, swore she had seen herself carrying a suitcase packed with everything she owned, wearing a coat she hadn't yet bought, heading somewhere she couldn't name. She never spoke of it again after that summer.

Years later, when the shipping company finally decommissioned the route entirely, the town council voted to tear down the lighthouse for scrap. The vote passed narrowly, seven to five, and the demolition crew arrived on a Tuesday morning with ropes and pry bars. They found the door unlocked and the lamp still burning faintly, though no fuel had been delivered since spring. The foreman stood beneath it for a long while, then turned to his men and said they would come back tomorrow. They never did. A petition appeared within the week, signed by nearly everyone in town, including the baker and the granddaughter, who had begun teaching local children how to climb the stairs safely.

What saved the structure was not sentimentality but an argument no one expected: the migrating terns had begun nesting in its gallery again, returning after decades away as if consulting some internal calendar only birds possessed. Environmental law, indifferent to nostalgia, protected the site immediately. The lighthouse became a sanctuary, then a small museum, then a place where couples quietly got married in autumn when the crowds thinned and the light—restored now to working order—turned steadily over water that glittered like hammered tin.

Elias died in winter, peacefully, mid-loaf, hands dusted white with flour. The granddaughter inherited his recipe card, written in pencil so faded she needed a magnifying glass to read it. It contained no measurements, only instructions: knead until it feels like forgiveness, let it rise until it forgets it was ever flat, bake until the kitchen smells like someone is coming home. She followed them exactly, and the bread came out perfect, which she understood was the whole point. Some things, she decided, were not meant to be measured—only trusted.
"""

splitter = CharacterTextSplitter(
  chunk_size = 100,
  chunk_overlap = 0,
  separator = ''
)

result = splitter.split_text(text)
print(result)