"""
For each of the exercise you can run the following:
    python scripts/week6_gemini.py chat-temperatures
    python scripts/week6_gemini.py chat-example
    python scripts/week6_gemini.py gemini-chat
    python scripts/week6_gemini.py gemini-chat-theater
    python scripts/week6_gemini.py img-parse
"""

import json
import pathlib
from pathlib import Path

import polars as pl
import typer
from google import genai
from google.genai import types

app = typer.Typer(no_args_is_help=True)

FP = Path(__file__).parents[1].joinpath("output", "gemini")
FP.mkdir(exist_ok=True)


@app.command()
def chat_example():
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Explain how AI works"],
        config=types.GenerateContentConfig(temperature=0.1),
    )
    print(response.text)


@app.command()
def chat_temperatures():
    client = genai.Client()
    for t in [0.1, 0.5, 0.9]:
        print(f"Generating for temperature={t}")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["Explain how AI works in a few words"],
            config=types.GenerateContentConfig(temperature=0.1),
        )
        print(response.text)

        with open(file := FP / f"example-{str(t).split('.')[1]}.md", "w") as f:
            f.write(response.text)
        print(f"Saved output to {file}")


@app.command()
def chat_system():
    client = genai.Client()
    t = 0.5
    system_prompts = [
        "You are Geoffrey Hinton, the 'Godfather of Deep Learning'. Explain AI as if you were telling a bedtime story to children.",
        "You are Andrej Karpathy, former Tesla AI lead. Explain AI as if you were narrating a YouTube coding tutorial.",
        "You are Yann LeCun, Meta's Chief AI Scientist. Explain AI passionately while defending it against critics.",
    ]
    for i, prompt in enumerate(system_prompts):
        print(f"Generating for temperature={t}")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["Explain how AI works in a few words"],
            config=types.GenerateContentConfig(
                system_instruction=prompt,
                temperature=t,
            ),
        )
        print(response.text)

        with open(file := FP / f"example-system-prompt-{i}.md", "w") as f:
            f.write(response.text)
        print(f"Saved output to {file}")


@app.command()
def gemini_chat():
    turns = 10

    # Create two chat sessions
    client = genai.Client()
    chat20 = client.chats.create(model="gemini-2.0-flash")
    chat25 = client.chats.create(model="gemini-2.5-flash")

    dialogue = []

    # Start conversation: Gemini 2.0 greets Gemini 2.5
    first_message = "Suggest one cost-efficient policy change that could significantly improve Danish society. Keep your answer brief."
    print(" Gemini 2.0 ".center(80, "-"))
    print(first_message, "\n")
    dialogue.append(f"Gemini 2.0: {first_message}")

    # Gemini 2.5 replies
    response25 = chat25.send_message(first_message)
    print(" Gemini 2.5 ".center(80, "-"))
    print(response25.text, "\n")
    dialogue.append(f"Gemini 2.5: {response25.text}")

    # Continue conversation
    message = response25.text
    for _ in range(turns - 1):
        # Gemini 2.0 replies
        response20 = chat20.send_message(message)
        print(" Gemini 2.0 ".center(80, "-"))
        print(response20.text, "\n")
        dialogue.append(f"Gemini 2.0: {response20.text}")

        # Gemini 2.5 replies
        response25 = chat25.send_message(response20.text)
        print(" Gemini 2.5 ".center(80, "-"))
        print(response25.text, "\n")
        dialogue.append(f"Gemini 2.5: {response25.text}")

        # Next message comes from Gemini 2.5
        message = response25.text

    breakpoint()
    print(" End of conversation ".center(80, "*"))

    # Save conversation to file
    with open(file := "output/gemini-chat.md", "w") as f:
        f.write("\n\n".join(dialogue))
    print(f"Saved conversation to {file}")


@app.command()
def gemini_chat_theater():
    turns = 10

    # Create two chat sessions
    t = 0.5
    client = genai.Client()
    chat20 = client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a Danish politician. "
                "You are interested in the greater good of Denmark. "
                "But you are also interested in getting elected at the next election."
            ),
            temperature=t,
        ),
    )

    chat25 = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a junior consultant at a Big 3 consulting firm. "
                "Your goal is to persuade the politician to select you for the job."
            ),
            temperature=t,
        ),
    )

    dialogue = []

    # Start conversation: Gemini 2.0 greets Gemini 2.5
    first_message = "Suggest one cost-efficient policy change that could significantly improve Danish society. Keep your answer brief."
    print(" Gemini 2.0 ".center(80, "-"))
    print(first_message, "\n")
    dialogue.append(f"Gemini 2.0: {first_message}")

    # Gemini 2.5 replies
    response25 = chat25.send_message(first_message)
    print(" Gemini 2.5 ".center(80, "-"))
    print(response25.text, "\n")
    dialogue.append(f"Gemini 2.5: {response25.text}")

    # Continue conversation
    message = response25.text
    for _ in range(turns - 1):
        # Gemini 2.0 replies
        response20 = chat20.send_message(message)
        print(" Gemini 2.0 ".center(80, "-"))
        print(response20.text, "\n")
        dialogue.append(f"Gemini 2.0: {response20.text}")

        # Gemini 2.5 replies
        response25 = chat25.send_message(response20.text)
        print(" Gemini 2.5 ".center(80, "-"))
        print(response25.text, "\n")
        dialogue.append(f"Gemini 2.5: {response25.text}")

        # Next message comes from Gemini 2.5
        message = response25.text

    breakpoint()
    print(" End of conversation ".center(80, "*"))

    # Save conversation to file
    with open(file := "output/gemini-chat-theater.md", "w") as f:
        f.write("\n\n".join(dialogue))
    print(f"Saved conversation to {file}")


@app.command()
def img_parse():
    from google import genai
    from pydantic import BaseModel

    with open("data/table.png", "rb") as f:
        image_bytes = f.read()

    client = genai.Client()

    class Table(BaseModel):
        title: str
        headers: list[str]
        data: list[list[str]]

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),
            "Extract the tables from this image and return the data as a json",
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Table],
        },
    )

    data = response.text
    if data:
        dfs = []
        for table in json.loads(data):
            print(f"Table: {table['title']}")
            print(
                df := pl.DataFrame(
                    data=table["data"],
                    schema=[table["title"]] + (headers := table["headers"]),
                )
                .with_columns(pl.all().str.replace(",", "."))
                .with_columns(pl.col(headers).cast(pl.Float32))
            )
            dfs.append(df)

        with open("output/ft.json", "w") as file:
            file.write(data)


if __name__ == "__main__":
    app()
