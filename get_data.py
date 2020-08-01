import lyricsgenius as genius
import config
import pandas as pd

access_token = config.access_token
artist = "Coldplay"

api = genius.Genius(access_token)
query = api.search_artist(artist, max_songs=2)

df = pd.DataFrame()


print(query.songs)
for song in query.songs:
    df = df.append(song.to_dict(), ignore_index=True)

df.to_json('data/{}.json'.format(artist))

