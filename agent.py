from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated, Optional
import ast
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv

load_dotenv()

music_assistant_prompt = """
You are a knowledgeable music store assistant focused on helping customers discover and learn about music in our digital catalog. 
If you are unable to find playlists, songs, or albums associated with an artist, it is okay. Do not blame it on an issue with the database/system. Just inform the customer that the catalog does not have any playlists, songs, or albums associated with that artist, and ask if they would like to search for something else.

CORE RESPONSIBILITIES:
- Search and provide accurate information about songs, albums, artists, and playlists
- Offer relevant recommendations based on customer interests
- Handle music-related queries with attention to detail
- Help customers discover new music they might enjoy

SEARCH GUIDELINES:
1. Always perform thorough searches before concluding something is unavailable
2. If exact matches aren't found, try:
   - Checking for alternative spellings
   - Looking for similar artist names
   - Searching by partial matches
   - Checking different versions/remixes
3. When providing song lists:
   - Include the artist name with each song
   - Mention the album when relevant
   - Note if it's part of any playlists
   - Indicate if there are multiple versions
         
         Below are some examples of good responses to customer queries:
         
         "Query: How many songs do you have by James Brown
         Response: We have 20 songs by James Brown in our collection. Here are some of them:
            - "Please Please Please"
            - "Think"
            - "Night Train"
            - "Out Of Sight"
            - "Papa's Got A Brand New Bag Pt.1"
            - "I Got You (I Feel Good)"
            - "It's A Man's Man's Man's World"
            - "Cold Sweat"
            - "Say It Loud, I'm Black And I'm Proud Pt.1"
            - "Get Up (I Feel Like Being A) Sex Machine"
        and more. If you're looking for a specific track or need more information, feel free to ask
               
         "Query: Do you have the song dog eat dog? 
         Response: Yes, we do have the song "Dog Eat Dog" in our collection. It's by AC/DC. If you're interested in this track or need more information about it, feel free to ask!
"""

db = SQLDatabase.from_uri("sqlite:///chinook.db")
model = ChatOpenAI(model="gpt-4", temperature=0)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

albums = db._execute("select * from Album")
artists = db._execute("select * from Artist")
songs = db._execute("select * from Track")

artist_retriever = SKLearnVectorStore.from_texts(
    [a['Name'] for a in artists],
    OpenAIEmbeddings(), 
    metadatas=artists
).as_retriever()

song_retriever = SKLearnVectorStore.from_texts(
    [a['Name'] for a in songs],
    OpenAIEmbeddings(), 
    metadatas=songs
).as_retriever()

# helper functions
def get_customer_id_from_identifier(identifier: str) -> Optional[int]:
    """
    Retrieve Customer ID using an identifier, which can be a customer ID, email, or phone number.
    
    Args:
        identifier (str): The identifier can be customer ID, email, or phone.
    
    Returns:
        Optional[int]: The CustomerId if found, otherwise None.
    """
    if identifier.isdigit():
        return int(identifier)
    elif identifier[0] == "+":
        query = f"SELECT CustomerId FROM Customer WHERE Phone = '{identifier}';"
        result = db.run(query)
        formatted_result = ast.literal_eval(result)
        if formatted_result:
            return formatted_result[0][0]
    elif "@" in identifier:
        query = f"SELECT CustomerId FROM Customer WHERE Email = '{identifier}';"
        result = db.run(query)
        formatted_result = ast.literal_eval(result)
        if formatted_result:
            return formatted_result[0][0]
    return None 

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"]
            )
            for tc in tool_calls
        ]
    }

# music tools
@tool
def get_albums_by_artist(artist):
    """Get albums by an artist (or similar artists)."""
    docs = artist_retriever.get_relevant_documents(artist)
    artist_ids = ", ".join([str(d.metadata['ArtistId']) for d in docs])
    return db.run(f"SELECT Title, Name FROM Album LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId WHERE Album.ArtistId in ({artist_ids});", include_columns=True)

@tool
def get_tracks_by_artist(artist):
    """Get songs by an artist (or similar artists)."""
    docs = artist_retriever.get_relevant_documents(artist)
    artist_ids = ", ".join([str(d.metadata['ArtistId']) for d in docs])
    return db.run(f"SELECT Track.Name as SongName, Artist.Name as ArtistName FROM Album LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId LEFT JOIN Track ON Track.AlbumId = Album.AlbumId WHERE Album.ArtistId in ({artist_ids});", include_columns=True)

@tool
def get_songs_by_genre(genre: str):
    """
    Fetch songs from the database that match a specific genre.
    
    Args:
        genre (str): The genre of the songs to fetch.
    
    Returns:
        list[dict]: A list of songs that match the specified genre.
    """
    genre_id_query = f"SELECT GenreId FROM Genre WHERE Name LIKE '%{genre}%'"
    genre_ids = db.run(genre_id_query)
    if not genre_ids:
        return f"No songs found for the genre: {genre}"
    genre_ids = ast.literal_eval(genre_ids)
    genre_id_list = ", ".join(str(gid[0]) for gid in genre_ids)

    songs_query = f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName
        FROM Track
        LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.GenreId IN ({genre_id_list})
        GROUP BY Artist.Name
        LIMIT 8;
    """
    songs = db.run(songs_query, include_columns=True)
    if not songs:
        return f"No songs found for the genre: {genre}"
    formatted_songs = ast.literal_eval(songs)
    return [
        {"Song": song["SongName"], "Artist": song["ArtistName"]}
        for song in formatted_songs
    ]

@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return song_retriever.get_relevant_documents(song_title)

@tool
def get_playlists_by_song_and_artist(artist_name: str, song_name: str):
    """
    Fetch playlists from the database that contain a specific song by a given artist.
    
    Args:
        artist_name (str): The name of the artist.
        song_name (str): The name of the song.
    
    Returns:
        list[dict]: A list of playlists that contain the specified song by the artist.
    """
    artist_id_query = f"SELECT ArtistId FROM Artist WHERE Name LIKE '%{artist_name}%'"
    artist_ids = db.run(artist_id_query)
    
    if not artist_ids:
        return f"No artist found with the name: {artist_name}"
    artist_ids = ast.literal_eval(artist_ids)
    artist_id_list = ", ".join(str(aid[0]) for aid in artist_ids)

    track_id_query = f"""
        SELECT Track.TrackId
        FROM Track
        JOIN Album ON Track.AlbumId = Album.AlbumId
        WHERE Track.Name LIKE '%{song_name}%' AND Album.ArtistId IN ({artist_id_list})
    """
    track_ids = db.run(track_id_query)

    if not track_ids:
        return f"No song found with the name: {song_name} by artist: {artist_name}"
    track_ids = ast.literal_eval(track_ids)
    track_id_list = ", ".join(str(tid[0]) for tid in track_ids)

    playlist_query = f"""
        SELECT Playlist.Name as PlaylistName
        FROM Playlist
        JOIN PlaylistTrack ON Playlist.PlaylistId = PlaylistTrack.PlaylistId
        WHERE PlaylistTrack.TrackId IN ({track_id_list})
    """
    playlists = db.run(playlist_query, include_columns=True)
    
    if not playlists:
        return f"No playlists found containing the song: {song_name} by artist: {artist_name}"

    formatted_playlists = ast.literal_eval(playlists)
    return [
        {"Playlist": playlist["PlaylistName"]}
        for playlist in formatted_playlists
    ]
music_tools = [get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre, check_for_songs, get_playlists_by_song_and_artist]

def music_assistant(state: State) -> dict:
    music_assistant_with_tools = model.bind_tools(music_tools)
    result = music_assistant_with_tools.invoke([SystemMessage(content=music_assistant_prompt)] + state["messages"])
    return {"messages": state["messages"] + [result]}

def tool_node(state: State):
    return ToolNode(music_tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

builder = StateGraph(State)
builder.add_node("music_assistant", music_assistant)
builder.add_node("music_assistant_tools", tool_node)
builder.add_edge(START, "music_assistant")
builder.add_conditional_edges(
    "music_assistant",
    should_continue,
    {
        "continue": "music_assistant_tools",
        "end": END,
    }
)
builder.add_edge("music_assistant_tools", "music_assistant")
graph = builder.compile()
# result = graph.invoke({"messages": [HumanMessage(content="have any songs by coldplay?")]})
# print(result["messages"], "result from running graph")
