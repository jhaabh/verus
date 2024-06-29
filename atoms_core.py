import os
import json
from bs4 import BeautifulSoup
import pandas as pd
from pydantic import BaseModel
from typing import List
import backoff
import openai
from litellm import completion
from prompts_and_data import SUMMARIES, SEED, SYSTEM_PROMPT_TEMPLATE
import instructor
from instructor import Mode
import zipfile
import shutil
from bs4 import BeautifulSoup
import requests

# Configuration
DEFAULT_MODEL = 'gpt-4o'
MAX_TOKENS = 2048
MAX_CHUNK_LENGTH = 1500

class Atom(BaseModel):
    subject: str
    topic: str
    entities: list[str]
    text: str

class SpecializedTerminology(BaseModel):
    term: str
    definition: str

class Atoms(BaseModel):
    atoms: list[Atom]
    specialized_terminologies: list[SpecializedTerminology]

# Utility functions
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def gen_from_messages(messages, model=DEFAULT_MODEL):
    return completion(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=messages
    )['choices'][0]['message']['content']

def append_data_jsonl(data, file_path):
    with open(file_path, 'a') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

# Core functionality
def extract_endnotes(xhtml_content):
    soup = BeautifulSoup(xhtml_content, 'html.parser')
    endnotes = []
    for li in soup.find_all('li', {'epub:type': 'endnote'}):
        endnote_id = int(li['id'].split('-')[-1])
        endnote_text = li.get_text(strip=True)
        endnotes.append((endnote_id, endnote_text))
    endnotes.sort(key=lambda x: x[0])
    return [endnote[1] for endnote in endnotes]

def process_paragraph(paragraph, para_num, book_num):
    text = paragraph.get_text()
    endnotes = []
    for link in paragraph.find_all('a', {'epub:type': 'noteref'}):
        endnote_id = link.get('href').split('#')[-1]
        endnote_num = int(endnote_id.split('-')[-1])
        text = text.replace(link.get_text(), f'[{endnote_num}]')
        endnotes.append(endnote_num)

    if len(text) <= MAX_CHUNK_LENGTH:
        return [{'text': text, 'endnotes': endnotes, 'para_num': para_num, 'book_num': book_num}]

    chunks = []
    current_chunk = ""
    current_endnotes = []
    statements = text.split('.')
    for statement in statements:
        statement = statement.strip()
        if not statement:
            continue
        if len(current_chunk) + len(statement) + 1 <= MAX_CHUNK_LENGTH:
            current_chunk += '. ' + statement if current_chunk else statement
        else:
            if current_chunk:
                chunk_index = len(chunks) + 1
                chunks.append({'text': current_chunk + '.',
                               'endnotes': current_endnotes,
                               'para_num': para_num,
                               'chunk_index': chunk_index,
                               'book_num': book_num})
            current_chunk = statement
            current_endnotes = []

        for endnote_num in endnotes:
            if f'[{endnote_num}]' in statement:
                current_endnotes.append(endnote_num)

    if current_chunk:
        chunk_index = len(chunks) + 1
        chunks.append({'text': current_chunk + '.',
                       'endnotes': current_endnotes,
                       'para_num': para_num,
                       'chunk_index': chunk_index,
                       'book_num': book_num})

    return chunks

def process_xhtml(xhtml_content, book_num):
    soup = BeautifulSoup(xhtml_content, 'html.parser')
    paragraphs = []
    for i, p in enumerate(soup.find_all('p'), start=1):
        chunks = process_paragraph(p, i, book_num)
        paragraphs.extend(chunks)
    return paragraphs

def download_and_setup_meditations_source():
    """
    Downloads the Meditations source from GitHub, extracts it, and sets up the directory structure.
    
    Returns:
    str: Path to the EPUB text directory
    """
    url = "https://github.com/standardebooks/marcus-aurelius_meditations_george-long/archive/refs/heads/master.zip"
    
    print("Downloading Meditations source...")
    response = requests.get(url)
    with open("meditations_snapshot.zip", "wb") as file:
        file.write(response.content)
    
    print("Extracting files...")
    with zipfile.ZipFile("meditations_snapshot.zip", "r") as zip_ref:
        zip_ref.extractall("meditations")
    
    print("Setting up directory structure...")
    src_path = os.path.join("meditations", "marcus-aurelius_meditations_george-long-master", "src")
    dst_path = os.path.join("meditations", "src")
    shutil.copytree(src_path, dst_path)
    
    epub_dir = os.path.join("meditations", "src", "epub", "text")
    
    print("Cleaning up...")
    os.remove("meditations_snapshot.zip")
    shutil.rmtree(os.path.join("meditations", "marcus-aurelius_meditations_george-long-master"))
    
    print(f"Setup complete. EPUB directory: {epub_dir}")
    return epub_dir

def extract_chunks(epub_dir=None):
    if epub_dir is None:
        epub_dir = download_and_setup_meditations_source()

    # Open endnotes
    with open(os.path.join(epub_dir, 'endnotes.xhtml'), 'r', encoding='utf-8') as file:
        endnotes_xhtml = file.read()
    endnotes_list = extract_endnotes(endnotes_xhtml)

    # Process each book and break it into paragraphs
    books_paragraphs = []
    for i in range(1, 13):
        book_file = f'book-{i}.xhtml'
        with open(os.path.join(epub_dir, book_file), 'r', encoding='utf-8') as file:
            book_xhtml = file.read()
        book_paragraphs = process_xhtml(book_xhtml, i)
        books_paragraphs.extend(book_paragraphs)

    return books_paragraphs, endnotes_list

def render_chunk_message(chunk):
    chunk_idx_text = f"Part: {chunk['chunk_index']}\n" if 'chunk_index' in chunk else ''
    input_text = f'''\
Book: {chunk['book_num']}
Paragraph: {chunk['para_num']}
{chunk_idx_text}
{chunk['text']}'''
    return input_text

def extract_atoms(all_chunks, file_path, seed_data, summaries, N=5):
    n = min(len(seed_data), N)
    l = len(all_chunks)

    chunks_processed = [(all_chunks[i], x) for i, x in enumerate(seed_data)]
    print(len(seed_data), len(chunks_processed))
    to_store = seed_data
    for i in range(n, l):
        start_idx   = i - n
        end_idx     = i
        examples    = chunks_processed[start_idx:end_idx]
        sample      = all_chunks[i]
        book_num    = sample['book_num']
        para_num    = sample['para_num']

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(book_number=book_num, chapter_summary=summaries[book_num-1])
        messages = [
            {'role': 'system', 'content': system_prompt}
        ]
        for chunk, output_chunk in examples:
            input_text = render_chunk_message(chunk)
            output_text = output_chunk['atoms_raw']
            messages.append({'role': 'user', 'content': input_text})
            messages.append({'role': 'assistant', 'content': output_text})
        messages.append({'role': 'user', 'content': render_chunk_message(sample)})
        atoms_raw = gen_from_messages(messages)

        output = {
            'book_num': book_num,
            'para_num': para_num,
            'chunk_index': sample.get('chunk_index', None),
            'atoms_raw': atoms_raw
        }
        chunks_processed.append((sample, output))
        print('Processed chunk number: ', i, book_num, para_num)

        to_store.append(output)
        if len(to_store) > 5:
            append_data_jsonl(to_store, file_path)
            to_store = []

def load_jsonl_to_dataframe(file_path, as_dataframe=False):
    data = []  # List to hold dictionaries
    try:
        with open(file_path, 'r') as file:
            for line in file:
                json_data = json.loads(line.strip())
                data.append(json_data)
    except FileNotFoundError:
        print(f"No file found at {file_path}")

    if as_dataframe:
        return pd.DataFrame(data) if data else pd.DataFrame()
    else:
        return data
    
def process_atoms(data, atoms_file_path, client):
    last_successful_idx = 0
    atoms_data = []
    for i in range(last_successful_idx, len(data)):
        x = data[i]
        raw = x['atoms_raw']
        atoms = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "user", "content": raw},
            ],
            max_tokens=2048,
            response_model=Atoms,
        )

        atoms_data.append(atoms)
        append_data_jsonl([atoms.dict()], atoms_file_path)
        print('Processed: ', i)

def setup_instructor():
    client = instructor.from_litellm(completion)
    return client

# Main execution function
def run_extraction(epub_dir, output_dir):
    chunks, _ = extract_chunks(epub_dir)
    sorted_chunks = sorted(chunks, key=lambda chunk: (
        chunk['book_num'],
        chunk['para_num'],
        chunk.get('chunk_index', 0)
    ))
    
    file_path = os.path.join(output_dir, 'chunks_processed.jsonl')
    atoms_file_path = os.path.join(output_dir, 'atoms_processed.jsonl')
    
    extract_atoms(sorted_chunks, file_path, SEED, SUMMARIES)
    
    data = load_jsonl_to_dataframe(file_path)
    process_atoms(data, atoms_file_path, setup_instructor())

if __name__ == "__main__":
    print("This script is not meant to be run directly. Please use it in a Colab notebook.")